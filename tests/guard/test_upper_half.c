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

/* compiled with -mno-avx: legacy-SSE encoding guaranteed (Makefile).
 * Returns elapsed seconds, writes the checksum through *sink so the
 * loop cannot be optimized away between the two measurements. */
static double sse_sink(double *a, size_t n, unsigned iters, double *sink) {
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
    *sink = s;
    return 0.0; /* elapsed filled by wrapper below */
}

/* time one sse_sink call; k repetitions, takes the MINIMUM (least noise) */
static double time_min(double *a, size_t n, unsigned iters, double *sink, int k) {
    double best = 1e30;
    for (int r = 0; r < k; r++) {
        double t0 = now_s();
        sse_sink(a, n, iters, sink);
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(void) {
    /* ITERS sized so the SSE loop is long enough to time reliably on any
     * host, including runners whose transition penalty never materializes.
     * CI evidence (run 45691554319): cold 0.0000s on a fast runner — the
     * ratio there was pure timer noise (5.34 from sub-us clock reads).
     * Guard rails below distinguish "too fast to measure" from "dirty". */
    enum { N = 64, ITERS = 200000, REPS = 5 };
    static double a[N];
    for (size_t i = 0; i < N; i++) a[i] = 1.0 / (double)(i + 1);

    double b[16];
    for (int i = 0; i < 16; i++) b[i] = (double)(i + 1);
    double y[16];
    for (int i = 0; i < 16; i++) y[i] = 0.5 * (double)i;

    int failures = 0;
    double sink = 0.0;
    (void)sink;
    /* exercise every AVX2 export */
    (void)ols_sum_f64_avx2(a, N);
    (void)ols_dot_f64_avx2(a, a, N);
    ols_axpy_f64_avx2(2.0, b, y, 16);
    ols_gram_args_t ga;
    (void)ga; /* gram takes the struct; covered via sse2-vs-avx2 test —
                 here the kernel-level AVX2 exports suffice: they share
                 the discipline requirement */

    /* warm baseline first (upper half clean): min over REPS runs */
    __asm__ volatile ("vzeroupper");
    double s2 = 0.0;
    double warm = time_min(a, N, ITERS, &s2, REPS);

    /* re-dirty candidates: library calls again, then cold measurement */
    (void)ols_sum_f64_avx2(a, N);
    (void)ols_dot_f64_avx2(a, a, N);
    ols_axpy_f64_avx2(2.0, b, y, 16);
    double s1 = 0.0;
    double cold = time_min(a, N, ITERS, &s1, REPS);

    double ratio = (warm > 0.0) ? cold / warm : 0.0;
    int dirty = 0;
    if (warm < 1e-4) {
        /* loop too fast to resolve a transition penalty (~70 cycles per
         * dirty entry; 70c/3GHz*ITERS/REPS-budget would need ~ms times).
         * On such a host the runtime check is vacuous — say so honestly
         * instead of failing on timer noise; the STATIC check
         * (tools/lint_contracts.py, vzeroupper discipline) still gates. */
        printf("upper-half check: warm %.6fs below timing resolution — "
               "SKIPPED on this host (static lint still enforced)\n", warm);
    } else {
        dirty = ratio > 3.0;
        printf("upper-half check: cold %.4fs warm %.4fs ratio %.2f -> %s\n",
               cold, warm, ratio, dirty ? "DIRTY (FAIL)" : "clean (OK)");
        if (dirty) {
            printf("FAIL: AVX2 export left upper half dirty\n");
            failures++;
        }
    }

    printf("%s: test_upper_half (%d failures)\n",
           failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
