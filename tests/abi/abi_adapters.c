/* abi_adapters.c — integer-arg wrappers so the ABI probe (which passes 4
 * integer args) can call kernels whose first argument is a double.
 * The adapter loads the double from a pointer-sized bit pattern in rdi,
 * then tail-calls the kernel with the remaining 3 args shifted. */
#include <stdint.h>
#include "ols.h"

typedef uint64_t u64;

/* probe entry: (u64 alpha_bits, const double *x, double *y, u64 n) */
u64 abi_probe_axpy_sse2(u64 alpha_bits, const double *x, double *y, u64 n) {
    double alpha;
    __builtin_memcpy(&alpha, &alpha_bits, 8);
    ols_axpy_f64_sse2(alpha, x, y, n);
    return 0;
}
u64 abi_probe_axpy_avx2(u64 alpha_bits, const double *x, double *y, u64 n) {
    double alpha;
    __builtin_memcpy(&alpha, &alpha_bits, 8);
    ols_axpy_f64_avx2(alpha, x, y, n);
    return 0;
}
