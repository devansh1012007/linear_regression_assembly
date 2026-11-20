/* -----------------------------------------------------------------------------
 * ols.h — the ONLY public header of asm-ols (C ABI).
 * Phase 0/1 subset: error codes, kernel prototypes, simple-fit structs.
 * The full ols_problem_t/ols_result_t arrive in Phase 2 (see docs/adr).
 * Everything under src/ is NASM; this header is the contract surface.
 * ---------------------------------------------------------------------------*/
#ifndef OLS_H
#define OLS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes (negative). Validation happens once in the API layer; kernels
 * trust their contracts (asm-project-delivery §2). */
#define OLS_OK                  0
#define OLS_E_ARG              -1
#define OLS_E_ALIGN            -2
#define OLS_E_SIZE_OVERFLOW    -3
#define OLS_E_WORKSPACE        -4
#define OLS_E_UNDERDETERMINED  -5
#define OLS_E_NONFINITE        -6
#define OLS_E_SINGULAR         -7
#define OLS_E_ILLCOND          -8

/* simple_fit flags */
#define OLS_F_STATS            1u   /* compute sse/sst/r2 (extra pass) */

/* ISA dispatch result */
#define OLS_ISA_SSE2           0
#define OLS_ISA_AVX2           1
#define OLS_ISA_AVX512         2

/* ---- Phase-1 kernels (C-callable, allocation-free, syscall-free) -----------
 * All take (const double *x, ...) with any 8-byte-aligned pointer; none reads
 * or writes outside the exact [ptr, ptr + n*8) ranges (guard-page enforced). */

/* Sum with Neumaier compensation: returns fl(sum x[i] + correction). */
double ols_sum_f64_sse2 (const double *x, size_t n);
double ols_sum_f64_avx2 (const double *x, size_t n);

/* Unconstrained dot product with Neumaier compensation. */
double ols_dot_f64_sse2 (const double *a, const double *b, size_t n);
double ols_dot_f64_avx2 (const double *a, const double *b, size_t n);

/* axpy: y[i] += alpha * x[i] elementwise (c = 0). Writes y[0..n) only. */
void   ols_axpy_f64_sse2 (double alpha, const double *x, double *y, size_t n);
void   ols_axpy_f64_avx2 (double alpha, const double *x, double *y, size_t n);

/* Shifted sums for simple regression: one pass over (x, y) computing
 *   n, s(x-cx), s(y-cy), s((x-cx)^2), s((x-cx)(y-cy)), s((y-cy)^2)
 * (all Neumaier-compensated), with pivots cx=piv[0], cy=piv[1].
 * Outputs are 6 doubles written to out[0..6). Reads x[0..n), y[0..n),
 * piv[0..2) only. Returns OLS_OK or OLS_E_NONFINITE if any input is not
 * finite. Pivoting bounds the centering cancellation by |mean - c|. */
int32_t ols_centered_sums_f64_sse2 (const double *x, const double *y,
                                    size_t n, double *out /*6*/,
                                    const double *piv /*2*/);

/* Simple (p=1) regression with intercept. One compensated pass for centered
 * sums, closed-form slope/intercept, optional residual pass for stats.
 * out layout (7 doubles): [0]=b1 slope, [1]=b0 intercept, [2]=sse,
 * [3]=sst, [4]=r2, [5]=xbar, [6]=ybar. With flags&OLS_F_STATS==0, [2..5]
 * are written as 0. Reads x[0..n), y[0..n); writes out[0..7) only.
 * n == 0 -> OLS_E_ARG; n == 1 or degenerate Sxx -> OLS_E_SINGULAR. */
int32_t ols_simple_fit_f64_sse2 (const double *x, const double *y,
                                 size_t n, uint32_t flags, double *out /*7*/);

/* Simple (p=1) regression WITHOUT intercept: b = sum(x*y)/sum(x*x)
 * (uncentered; NIST NoInt1/NoInt2 semantics, uncentered R^2 = 1 - SSE/y'y).
 * out layout (5 doubles): [0]=b, [1]=sse, [2]=ySqNorm, [3]=r2, [4]=n.
 * n == 0 -> OLS_E_ARG; sum_xx == 0 -> OLS_E_SINGULAR. */
int32_t ols_simple_fit_noint_f64_sse2 (const double *x, const double *y,
                                       size_t n, uint32_t flags, double *out /*5*/);

/* CPU feature checks (skills/asm-x86-64-engineering §6, extended). */
int32_t ols_cpu_has_avx2_fma (void);
int32_t ols_cpu_has_avx512   (void);

/* Harness support (test-only exports; NOT part of the stable API): */
uint64_t abi_probe_call (void *fn, uint64_t a1, uint64_t a2, uint64_t a3,
                         uint64_t a4, uint32_t *corrupt_mask);

#ifdef __cplusplus
}
#endif
#endif /* OLS_H */
