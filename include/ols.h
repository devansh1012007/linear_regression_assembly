/* -----------------------------------------------------------------------------
 * ols.h — the ONLY public header of asm-ols (C ABI).
 * Phase 0/1: kernels + simple fits. Phase 2+: the ols_fit problem/result API.
 * Everything under src/ is NASM; this header is the contract surface.
 * Struct layouts are FROZEN (versioned); asm uses the offsets in abi.inc,
 * and tests/py/struct_check.py verifies the two never drift.
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
#define OLS_E_RANGE           -9      /* parser: value outside representable envelope */

/* simple_fit flags (own namespace, Phase-1 kernels) */
#define OLS_F_STATS            1u   /* compute sse/sst/r2 (extra pass) */

/* ols_fit flags (separate namespace) */
#define OLS_FIT_INTERCEPT      1u   /* center; fit b0 */
#define OLS_FIT_REPRODUCIBLE   2u   /* fixed-order merges (bitwise for 1..N threads) */
#define OLS_FIT_DOT2           4u   /* double-double accumulation (COULD item) */
#define OLS_FIT_SKIP_STATS     8u   /* skip residual pass + se/t (solves only) */

/* ols_fit methods */
#define OLS_AUTO               0
#define OLS_GRAM_CHOL          1
#define OLS_GRAM_CSNE          2
#define OLS_QR                 3
#define OLS_QR_PIVOT           4
#define OLS_CG                 5   /* reserved; currently OLS_E_ARG */

/* ISA dispatch ids. OLS_ISA_AVX512 is a CPU feature id only: the library
 * ships no AVX-512 kernel yet, so ols_init clamps an AVX-512 request (and
 * auto-detection) to AVX2 — ols_isa() never reports capability that no
 * kernel implements (review finding §5). */
#define OLS_ISA_SSE2           0
#define OLS_ISA_AVX2           1
#define OLS_ISA_AVX512         2

/* ---- Phase-1 kernels (C-callable, allocation-free, syscall-free) ----------- */
double ols_sum_f64_sse2 (const double *x, size_t n);
double ols_sum_f64_avx2 (const double *x, size_t n);
double ols_dot_f64_sse2 (const double *a, const double *b, size_t n);
double ols_dot_f64_avx2 (const double *a, const double *b, size_t n);
void   ols_axpy_f64_sse2 (double alpha, const double *x, double *y, size_t n);
void   ols_axpy_f64_avx2 (double alpha, const double *x, double *y, size_t n);

/* Shifted sums: out = [n, s(x-cx), s(y-cy), s((x-cx)^2), s((x-cx)(y-cy),
 * s((y-cy)^2)], Neumaier-compensated; pivots bound the centering
 * cancellation by |mean - c| (numerical-ols §2). */
int32_t ols_centered_sums_f64_sse2 (const double *x, const double *y,
                                    size_t n, double *out /*6*/,
                                    const double *piv /*2*/);
int32_t ols_simple_fit_f64_sse2 (const double *x, const double *y,
                                 size_t n, uint32_t flags, double *out /*7*/);
int32_t ols_simple_fit_noint_f64_sse2 (const double *x, const double *y,
                                       size_t n, uint32_t flags,
                                       double *out /*5*/);
int32_t ols_quad_fit_f64_sse2 (const double *x, const double *y,
                               size_t n, uint32_t flags, double *out /*7*/);

/* ---- Phase-2: multiple regression via the fused Gram engine ---------------- */

/* Caller-built argument block for the Gram kernels: ONE register argument at
 * the ABI boundary (the 10-field signature otherwise spills to stack args at
 * every call site; the struct keeps asm offsets fixed and testable). */
typedef struct {
  const double *X;           /* 0  : n x p row-major */
  size_t ldx8;               /* 8  : row stride of X in BYTES (ldx*8), >= p*8 */
  const double *y;           /* 16 : n */
  const double *w;           /* 24 : n weights, NULL = unweighted */
  size_t n;                  /* 32 */
  size_t p;                  /* 40 */
  const double *c;           /* 48 : q = p+1 pivots */
  double *G;                 /* 56 : q*q out (upper triangle) */
  double *s;                 /* 64 : q out (shifted column sums) */
  double *Wout;              /* 72 : weight total out */
  void *scratch;             /* 80 : caller workspace, 64-byte aligned */
  size_t scratch_bytes;      /* 88 : >= ols_gram_scratch_bytes(q) */
} ols_gram_args_t;           /* size 96 (v2: scratch fields added, see ADR-0005) */

/* Fused pivot-shifted Gram of Z = [X | y]: S = sum w_i (z_i-c)(z_i-c)' (upper
 * triangle, row-major q x q), s = sum w_i (z_i - c), W = sum w_i. G and s are
 * OVERWRITTEN (zeroed first). AVX2 panel kernel or SSE2 scalar reference. */
int32_t ols_gram_f64_avx2 (const ols_gram_args_t *A);
int32_t ols_gram_f64_sse2 (const ols_gram_args_t *A);   /* ignores scratch */

/* Scratch bytes the AVX2 Gram kernel needs for a given q = p+1 (packed rows +
 * column-major G scratch). ols_workspace_size includes it automatically. */
size_t ols_gram_scratch_bytes (size_t q);

/* Caller-built argument block for the residual kernels. */
typedef struct {
  const double *X;           /* 0  */
  size_t ldx8;               /* 8  */
  const double *y;           /* 16 */
  const double *w;           /* 24 : NULL = unweighted */
  size_t n;                  /* 32 */
  size_t p;                  /* 40 */
  double b0;                 /* 48 : intercept (0 when no intercept) */
  const double *beta;        /* 56 : p slopes */
  double *sse_out;           /* 64 */
  double *xtr_out;           /* 72 : p accumulators (may be NULL) */
} ols_resid_args_t;          /* size 80 */

/* Fused residual pass: r_i = y_i - b0 - x_i'beta; accumulates SSE (Neumaier)
 * and X'r (p accumulators). *sse_out and xtr_out[0..p) are OVERWRITTEN. */
int32_t ols_residual_f64_sse2 (const ols_resid_args_t *A);
int32_t ols_residual_f64_avx2 (const ols_resid_args_t *A);

/* Cholesky of a symmetric positive-definite q x q row-major matrix, in place
 * (upper triangle of A becomes R with A = R'R; strictly lower untouched).
 * Compensated dots. Fails OLS_E_SINGULAR if pivot <= eps_rel * max_diag. */
int32_t ols_chol_f64_sse2 (double *A, size_t q, double eps_rel, size_t stride);

/* Triangular solves on R (q x q upper, row-major): solves R R' x = b
 * (two back-substitutions), x written over b. */
int32_t ols_chol_solve_f64_sse2 (const double *R, size_t q, double *b, size_t stride);

/* Explicit inverse of the SPD matrix A = R'R (q x q row-major) into B_out
 * (may not alias A). Uses the factor in A's upper triangle (from ols_chol). */
int32_t ols_chol_inv_f64_sse2 (const double *A, size_t q, double *B_out, size_t stride);

/* Column means of the first `nb` rows of [X|y] (weighted, compensated):
 * the pivot vector c for the Gram pass. */
int32_t ols_pivot_means_f64_sse2 (const double *X, size_t ldx, const double *y,
                                  const double *w /*nullable*/, size_t nb,
                                  size_t p, double *c /*q*/);

/* ---- problem / result structs (layout frozen; see abi.inc) ---------------- */
typedef struct {
  uint32_t struct_size;      /* 0  : sizeof(ols_problem_t) */
  uint32_t version;          /* 4  : 1 */
  const double *X;           /* 8  : n x p row-major */
  const double *y;           /* 16 : n */
  const double *w;           /* 24 : n weights, NULL = unweighted */
  size_t n;                  /* 32 */
  size_t p;                  /* 40 */
  size_t ldx;                /* 48 : row stride of X in doubles, >= p */
  uint32_t method;           /* 56 : OLS_* */
  uint32_t flags;            /* 60 : OLS_FIT_* */
  double lambda;             /* 64 : ridge (0 = none) */
  double rcond;              /* 72 : rank tolerance (0 = default) */
  uint32_t threads;          /* 80 : 1 = single (core never spawns) */
  uint32_t _pad;             /* 84 */
} ols_problem_t;             /* size 88 */

typedef struct {
  double *beta;              /* 0  : p slopes */
  double intercept;          /* 8  */
  double sse;                /* 16 */
  double sst;                /* 24 */
  double r2;                 /* 32 */
  double r2_adj;             /* 40 */
  double sigma2;             /* 48 */
  double cond_est;           /* 56 : lower bound (maxR/minR)^2 */
  double *se;                /* 64 : p+1 (se[0..p) for beta, se[p] intercept) */
  int32_t rank;              /* 72 */
  int32_t method_used;       /* 76 */
  int32_t isa_used;          /* 80 : OLS_ISA_* */
  int32_t _pad;              /* 84 */
  double *t;                /* 88 : p+1 t-statistics (beta_j/se_j, t[p]=b0/se0);
                               NULL = skip (append-only v2; size 96) */
} ols_result_t;             /* size 96 (v2: t appended; see ADR-0006) */

int32_t ols_fit (const ols_problem_t *prob, ols_result_t *res,
                 void *workspace, size_t ws_bytes);
size_t ols_workspace_size (size_t n, size_t p, uint32_t method, uint32_t flags);

/* CPU features / dispatch (asm-x86-64-engineering §6). ols_cpu_has_avx512
 * reports the CPU feature truthfully; ols_init/ols_isa select only
 * implemented kernels (SSE2, AVX2) — see src/core/dispatch.asm. */
int32_t ols_cpu_has_avx2_fma (void);
int32_t ols_cpu_has_avx512   (void);
int32_t ols_init (int32_t force_isa);   /* auto caps at AVX2; 2 clamps to 1 */
int32_t ols_isa (void);                 /* returns 0 or 1 only, for now */

/* ABI introspection for the struct-drift test: packs sizeof/offsets. */
uint64_t ols_abi_offsets (void);

/* Harness support (test-only exports; NOT part of the stable API): */
uint64_t abi_probe_call (void *fn, uint64_t a1, uint64_t a2, uint64_t a3,
                         uint64_t a4, uint32_t *corrupt_mask);

#ifdef __cplusplus
}
#endif
#endif /* OLS_H */
