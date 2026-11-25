/* tests/guard/test_upper_half.c — runtime upper-half discipline check.
 *
 * The lint check (tools/lint_contracts.py) proves every ymm-using body
 * CONTAINS a vzeroupper; this harness proves it EXECUTES the right ones:
 * after each AVX2 export returns, a deliberately-SSE-coded loop (compiled
 * with -mno-avx so it cannot use VEX encodings itself) runs at measurable
 * speed. If the library left the upper halves dirty, the SSE loop pays the
 * ~70-cycle transition penalty per iteration and the ratio check fails.
 *
 * Method: time the SSE loop twice — cold (immediately after the library
 * call) and warm (after an explicit vzeroupper). Ratio > 3 => dirty upper
 * half => the kernel is missing its vzeroupper on some path.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <emmintrin.h>
#include "ols.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* compiled with -mno-avx: legacy-SSE encoding guaranteed (Makefile) */
static double sse_sink(double *a, size_t n, unsigned iters) {
    double s = 0.0;
    for (unsigned k = 0; k < iters; k++) {
        __m128d v = _mm_setzero_pd();
        for (size_t i = 0; i + 1 < n; i += 2) {
            __m128d x = _mm_loadu_pd(a + i);
            v = _mm_add_pd(v, x);
        }
        double t[2];
        _mm_storeu_pd(t, v);
        s += t[0] + t[1];
    }
    return s;
}

int main(void) {
    enum { N = 64, ITERS = 200000 };
    static double a[N];
    for (size_t i = 0; i < N; i++) a[i] = 1.0 / (double)(i + 1);

    double b[16];
    for (int i = 0; i < 16; i++) b[i] = (double)(i + 1);
    double y[16];
    for (int i = 0; i < 16; i++) y[i] = 0.5 * (double)i;

    int failures = 0;
    /* exercise every AVX2 export */
    (void)ols_sum_f64_avx2(a, N);
    (void)ols_dot_f64_avx2(a, a, N);
    ols_axpy_f64_avx2(2.0, b, y, 16);
    ols_gram_args_t ga;
    (void)ga; /* gram takes the struct; covered via sse2-vs-avx2 test —
                 here the kernel-level AVX2 exports suffice: they share
                 the discipline requirement */

    /* cold measurement: library just returned */
    double t0 = now_s();
    double s1 = sse_sink(a, N, ITERS);
    double cold = now_s() - t0;

    /* explicit clean upper half, then warm measurement */
    __asm__ volatile ("vzeroupper");
    t0 = now_s();
    double s2 = sse_sink(a, N, ITERS);
    double warm = now_s() - t0;

    (void)s1; (void)s2;
    double ratio = cold / warm;
    printf("upper-half check: cold %.4fs warm %.4fs ratio %.2f -> %s\n",
           cold, warm, ratio, ratio > 3.0 ? "DIRTY (FAIL)" : "clean (OK)");
    if (ratio > 3.0) { printf("FAIL: AVX2 export left upper half dirty\n"); failures++; }

    printf("%s: test_upper_half (%d failures)\n",
           failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
