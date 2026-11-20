/* machine_bench.c — Phase-0 characterization kernels (simd-kernel-opt §1).
 * Modes:
 *   fma <nacc>            : independent vfmadd231pd chains, prints flops/s best-of
 *   load                  : ymm loads from an L1-resident buffer, prints B/s
 *   stream <bytes> <iters>: sequential ymm read over a working set, prints B/s
 *   depadd <iters>        : dependent add chain, prints iters/s (clock proxy)
 * All results go to stdout; the Python driver collects medians.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static double sink = 0.0;

#define FMA_ITER 100000
/* Accumulators as 16 named __m256d via a macro so gcc keeps them in ymm
 * registers (an indexed acc[] array gets spilled to memory because its
 * address escapes; that turned this into a load-bound loop — see perf-log). */
#define FMA_CHAIN(N) \
    __attribute__((noinline)) static double fma_bench_##N(void) { \
        __m256d a0 = _mm256_set1_pd(1.0000001), b0 = _mm256_set1_pd(0.9999999); \
        __m256d r0 = _mm256_set1_pd(1.0); \
        __m256d r1 = _mm256_set1_pd(1.1), r2 = _mm256_set1_pd(1.2), r3 = _mm256_set1_pd(1.3); \
        __m256d r4 = _mm256_set1_pd(1.4), r5 = _mm256_set1_pd(1.5), r6 = _mm256_set1_pd(1.6), r7 = _mm256_set1_pd(1.7); \
        __m256d r8 = _mm256_set1_pd(1.8), r9 = _mm256_set1_pd(1.9), r10 = _mm256_set1_pd(2.0), r11 = _mm256_set1_pd(2.1); \
        __m256d r12 = _mm256_set1_pd(2.2), r13 = _mm256_set1_pd(2.3), r14 = _mm256_set1_pd(2.4), r15 = _mm256_set1_pd(2.5); \
        (void)r1; (void)r2; (void)r3; (void)r4; (void)r5; (void)r6; (void)r7; \
        (void)r8; (void)r9; (void)r10; (void)r11; (void)r12; (void)r13; (void)r14; (void)r15; \
        for (int it = 0; it < FMA_ITER; it++) { \
            r0 = _mm256_fmadd_pd(a0, b0, r0); \
            if (N > 1)  r1 = _mm256_fmadd_pd(a0, b0, r1); \
            if (N > 2)  r2 = _mm256_fmadd_pd(a0, b0, r2); \
            if (N > 3)  r3 = _mm256_fmadd_pd(a0, b0, r3); \
            if (N > 4)  r4 = _mm256_fmadd_pd(a0, b0, r4); \
            if (N > 5)  r5 = _mm256_fmadd_pd(a0, b0, r5); \
            if (N > 6)  r6 = _mm256_fmadd_pd(a0, b0, r6); \
            if (N > 7)  r7 = _mm256_fmadd_pd(a0, b0, r7); \
            if (N > 8)  r8 = _mm256_fmadd_pd(a0, b0, r8); \
            if (N > 9)  r9 = _mm256_fmadd_pd(a0, b0, r9); \
            if (N > 10) r10 = _mm256_fmadd_pd(a0, b0, r10); \
            if (N > 11) r11 = _mm256_fmadd_pd(a0, b0, r11); \
            if (N > 12) r12 = _mm256_fmadd_pd(a0, b0, r12); \
            if (N > 13) r13 = _mm256_fmadd_pd(a0, b0, r13); \
            if (N > 14) r14 = _mm256_fmadd_pd(a0, b0, r14); \
            if (N > 15) r15 = _mm256_fmadd_pd(a0, b0, r15); \
        } \
        __m256d s = _mm256_add_pd(r0, _mm256_setzero_pd()); \
        if (N > 1)  s = _mm256_add_pd(s, r1); \
        if (N > 2)  s = _mm256_add_pd(s, r2); \
        if (N > 3)  s = _mm256_add_pd(s, r3); \
        if (N > 4)  s = _mm256_add_pd(s, r4); \
        if (N > 5)  s = _mm256_add_pd(s, r5); \
        if (N > 6)  s = _mm256_add_pd(s, r6); \
        if (N > 7)  s = _mm256_add_pd(s, r7); \
        if (N > 8)  s = _mm256_add_pd(s, r8); \
        if (N > 9)  s = _mm256_add_pd(s, r9); \
        if (N > 10) s = _mm256_add_pd(s, r10); \
        if (N > 11) s = _mm256_add_pd(s, r11); \
        if (N > 12) s = _mm256_add_pd(s, r12); \
        if (N > 13) s = _mm256_add_pd(s, r13); \
        if (N > 14) s = _mm256_add_pd(s, r14); \
        if (N > 15) s = _mm256_add_pd(s, r15); \
        __m128d lo = _mm256_castpd256_pd128(s); \
        __m128d hi = _mm256_extractf128_pd(s, 1); \
        __m128d ss = _mm_add_pd(lo, hi); \
        sink += _mm_cvtsd_f64(ss) + _mm_cvtsd_f64(_mm_unpackhi_pd(ss, ss)); \
        return (double)FMA_ITER * N * 8.0; \
    }
FMA_CHAIN(1) FMA_CHAIN(2) FMA_CHAIN(3) FMA_CHAIN(4)
FMA_CHAIN(5) FMA_CHAIN(6) FMA_CHAIN(7) FMA_CHAIN(8)
FMA_CHAIN(9) FMA_CHAIN(10) FMA_CHAIN(11) FMA_CHAIN(12)
FMA_CHAIN(13) FMA_CHAIN(14) FMA_CHAIN(15) FMA_CHAIN(16)

typedef double (*fma_fn)(void);
static fma_fn fma_fns[17] = {NULL, fma_bench_1, fma_bench_2, fma_bench_3, fma_bench_4,
                             fma_bench_5, fma_bench_6, fma_bench_7, fma_bench_8,
                             fma_bench_9, fma_bench_10, fma_bench_11, fma_bench_12,
                             fma_bench_13, fma_bench_14, fma_bench_15, fma_bench_16};

__attribute__((noinline))
static double load_bench(double *buf, long nbuf_doubles) {
    /* 16 ymm loads + cheap adds per iteration; L1-resident buffer */
    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
    const __m256d *v = (const __m256d *)buf;
    long nchunks = nbuf_doubles / 4;         /* ymm chunks */
    double bytes = 0;
    long it = 0;
    for (; it < 300000; it++) {
        long base = (it * 16) % (nchunks - 16);
        for (int k = 0; k < 16; k++) {
            __m256d x = _mm256_load_pd((const double *)(v + base + k));
            s0 = _mm256_add_pd(s0, x);
            s1 = _mm256_add_pd(s1, x);
        }
        bytes += 16 * 32;
    }
    sink += _mm256_cvtsd_f64(_mm256_add_pd(s0, s1));
    return bytes;
}

__attribute__((noinline))
static double stream_read(double *buf, size_t bytes) {
    /* sequential read; 4 ymm per step with a single dep-chain sum (bandwidth,
     * not FMA, is the target) */
    __m256d s = _mm256_setzero_pd();
    const __m256d *v = (const __m256d *)buf;
    size_t chunks = bytes / 32;
    for (size_t i = 0; i + 3 < chunks; i += 4) {
        __m256d a = _mm256_load_pd((const double *)(v + i + 0));
        __m256d b = _mm256_load_pd((const double *)(v + i + 1));
        __m256d c = _mm256_load_pd((const double *)(v + i + 2));
        __m256d d = _mm256_load_pd((const double *)(v + i + 3));
        s = _mm256_add_pd(s, _mm256_add_pd(_mm256_add_pd(a, b), _mm256_add_pd(c, d)));
    }
    sink += _mm256_cvtsd_f64(s);
    return (double)((chunks / 4) * 4 * 32);
}

__attribute__((noinline))
static unsigned long depadd(unsigned long n) {
    /* raw dependent add chain in asm: gcc folds any C-level carry chain
     * (measured: 6.7e15 "ops/s" before this fix). add rax, imm8 is 1 cy
     * latency, 4/cy throughput but serially dependent -> 1 op/cy. */
    unsigned long x = 1;
    __asm__ volatile (
        "1: addq $1, %[x]\n"
        "   decq %[n]\n"
        "   jnz 1b\n"
        : [x] "+r" (x), [n] "+r" (n)
        :
        : "cc");
    return x;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "mode required\n"); return 2; }
    if (!strcmp(argv[1], "fma")) {
        if (argc < 3) return 2;
        int nacc = atoi(argv[2]);
        double best = 1e300;
        double flops = 0;
        for (int r = 0; r < 5; r++) {
            double t0 = now_s();
            flops = fma_fns[nacc]();
            double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        printf("%.17g\n", flops / best);
    } else if (!strcmp(argv[1], "load")) {
        long n = 4096;                        /* 32 KiB */
        double *buf = aligned_alloc(64, n * 8);
        for (long i = 0; i < n; i++) buf[i] = (double)i * 1e-7;
        double best = 1e300, by = 0;
        for (int r = 0; r < 3; r++) {
            double t0 = now_s();
            by = load_bench(buf, n);
            double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        printf("%.17g\n", by / best);
        free(buf);
    } else if (!strcmp(argv[1], "stream")) {
        if (argc < 4) return 2;
        size_t bytes = strtoull(argv[2], NULL, 10);
        int reps = atoi(argv[3]);
        double *buf = aligned_alloc(64, bytes);
        for (size_t i = 0; i < bytes / 8; i++) buf[i] = (double)i * 1e-9;
        double best = 1e300, by = 0;
        for (int r = 0; r < reps; r++) {
            double t0 = now_s();
            by = stream_read(buf, bytes);
            double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        printf("%.17g\n", by / best);
        free(buf);
    } else if (!strcmp(argv[1], "depadd")) {
        if (argc < 3) return 2;
        unsigned long n = strtoull(argv[2], NULL, 10);
        double best = 1e300;
        unsigned long x = 0;
        for (int r = 0; r < 3; r++) {
            double t0 = now_s();
            x = depadd(n);
            double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        if (x == 42) fprintf(stderr, ".");
        printf("%.17g\n", (double)n / best);  /* dependent ops per second */
    } else {
        return 2;
    }
    return 0;
}
