/* load_bench.c — re-measure L1 load / store throughput with INDEPENDENT chains.
 * The Phase-0 number (56.12 GB/s = 15.7 B/cyc) looks like a load->add
 * dependency artifact; asm-benchmarking rule: measure, don't assume.
 * Variants (all L1-resident, 8 KB working set, pinned via taskset upstream):
 *   ld1  : 1 ymm load + dependent add   (reproduces the old methodology)
 *   ld8  : 8 independent ymm loads + 8 independent adds (throughput)
 *   ld16 : 16 independent ymm loads
 *   st1  : 1 ymm store per iter (independent)
 *   st2  : 2 independent ymm stores
 *   rmw  : mem-dest FMA (vfmadd231pd [mem]) — the Gram kernel's core op
 *   rmw8 : 8 independent mem-dest FMAs to 8 distinct lines
 * Reports GB/s and ops/cycle using CLOCK_MONOTONIC_RAW + calibrated clock.
 * Build: gcc -O2 -mavx2 -mfma load_bench.c -o load_bench
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define REPS 200
#define ITERS 200000          /* per rep; 8KB buffer stays L1-resident */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* dependent-chain clock calibration: 1 add/cycle */
static double calibrate_clock(void) {
    volatile long x = 0;
    long n = 400000000L;
    double t0 = now_s();
    for (long i = 0; i < n; i++) x = x + 1;
    double t1 = now_s();
    (void)x;
    return (double)n / (t1 - t0);   /* adds (cycles-ish) per second */
}

__attribute__((noinline))
static double bench_ld1(__m256d *buf, long iters) {
    __m256d a = _mm256_setzero_pd();
    for (long i = 0; i < iters; i++)
        a = _mm256_add_pd(a, _mm256_loadu_pd((double*)(buf + (i & 255))));
    /* sink */
    volatile double s = _mm256_cvtsd_f64(a);
    return s;
}

__attribute__((noinline))
static double bench_ld8(__m256d *buf, long iters) {
    __m256d a0=_mm256_setzero_pd(),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    for (long i = 0; i < iters; i += 8) {
        a0=_mm256_add_pd(a0,_mm256_loadu_pd((double*)(buf+((i+0)&255))));
        a1=_mm256_add_pd(a1,_mm256_loadu_pd((double*)(buf+((i+1)&255))));
        a2=_mm256_add_pd(a2,_mm256_loadu_pd((double*)(buf+((i+2)&255))));
        a3=_mm256_add_pd(a3,_mm256_loadu_pd((double*)(buf+((i+3)&255))));
        a4=_mm256_add_pd(a4,_mm256_loadu_pd((double*)(buf+((i+4)&255))));
        a5=_mm256_add_pd(a5,_mm256_loadu_pd((double*)(buf+((i+5)&255))));
        a6=_mm256_add_pd(a6,_mm256_loadu_pd((double*)(buf+((i+6)&255))));
        a7=_mm256_add_pd(a7,_mm256_loadu_pd((double*)(buf+((i+7)&255))));
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a0,a1),
                              _mm256_add_pd(a2,a3)),
                              _mm256_add_pd(_mm256_add_pd(a4,a5),
                              _mm256_add_pd(a6,a7)));
    volatile double v = _mm256_cvtsd_f64(s);
    return v;
}

__attribute__((noinline))
static double bench_st1(__m256d *buf, __m256d v, long iters) {
    for (long i = 0; i < iters; i++)
        _mm256_storeu_pd((double*)(buf + (i & 255)), v);
    volatile double s = _mm256_cvtsd_f64(v);
    return s;
}

__attribute__((noinline))
static double bench_st2(__m256d *buf, __m256d v, long iters) {
    for (long i = 0; i < iters; i += 2) {
        _mm256_storeu_pd((double*)(buf + (i & 255)), v);
        _mm256_storeu_pd((double*)(buf + ((i+1) & 255)), v);
    }
    volatile double s = _mm256_cvtsd_f64(v);
    return s;
}

__attribute__((noinline))
static double bench_rmw1(__m256d *buf, __m256d b, long iters) {
    __m256d a = _mm256_setzero_pd();
    for (long i = 0; i < iters; i++)
        a = _mm256_fmadd_pd(b, a, _mm256_loadu_pd((double*)(buf + (i & 255))));
    volatile double s = _mm256_cvtsd_f64(a);
    return s;
}

/* 8 INDEPENDENT mem-dest FMAs per iter, 8 distinct 64B lines */
__attribute__((noinline))
static double bench_rmw8(__m256d *buf, __m256d b, long iters) {
    __m256d a0=_mm256_setzero_pd(),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    for (long i = 0; i < iters; i += 8) {
        a0=_mm256_fmadd_pd(b,a0,_mm256_loadu_pd((double*)(buf+((i+0)&255))));
        a1=_mm256_fmadd_pd(b,a1,_mm256_loadu_pd((double*)(buf+((i+1)&255))));
        a2=_mm256_fmadd_pd(b,a2,_mm256_loadu_pd((double*)(buf+((i+2)&255))));
        a3=_mm256_fmadd_pd(b,a3,_mm256_loadu_pd((double*)(buf+((i+3)&255))));
        a4=_mm256_fmadd_pd(b,a4,_mm256_loadu_pd((double*)(buf+((i+4)&255))));
        a5=_mm256_fmadd_pd(b,a5,_mm256_loadu_pd((double*)(buf+((i+5)&255))));
        a6=_mm256_fmadd_pd(b,a6,_mm256_loadu_pd((double*)(buf+((i+6)&255))));
        a7=_mm256_fmadd_pd(b,a7,_mm256_loadu_pd((double*)(buf+((i+7)&255))));
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a0,a1),
                              _mm256_add_pd(a2,a3)),
                              _mm256_add_pd(_mm256_add_pd(a4,a5),
                              _mm256_add_pd(a6,a7)));
    /* write results back so the RMW lines are real */
    _mm256_storeu_pd((double*)(buf + 0), s);
    volatile double v = _mm256_cvtsd_f64(s);
    return v;
}

int main(void) {
    __m256d *buf = aligned_alloc(64, 256 * 32 + 64);
    for (int i = 0; i < 256; i++)
        buf[i] = _mm256_set_pd(i * 4 + 3, i * 4 + 2, i * 4 + 1, i * 4 + 0);
    __m256d v = _mm256_set1_pd(1.0000001);

    double cyc_s = calibrate_clock();
    double sink = 0;

    struct { const char *name; int bytes_per_iter; int ops_per_iter; } cfg[] = {
        {"ld1_dep ", 32, 1},
        {"ld8_ind ", 256, 8},
        {"st1     ", 32, 1},
        {"st2     ", 64, 2},
        {"rmw1_dep", 32, 1},
        {"rmw8_ind", 256, 8},
    };
    double med[6]; double samples[REPS];
    for (int c = 0; c < 6; c++) {
        for (int r = 0; r < REPS; r++) {
            double t0 = now_s();
            switch (c) {
            case 0: sink += bench_ld1(buf, ITERS); break;
            case 1: sink += bench_ld8(buf, ITERS); break;
            case 2: sink += bench_st1(buf, v, ITERS); break;
            case 3: sink += bench_st2(buf, v, ITERS); break;
            case 4: sink += bench_rmw1(buf, v, ITERS); break;
            case 5: sink += bench_rmw8(buf, v, ITERS); break;
            }
            double t1 = now_s();
            samples[r] = t1 - t0;
        }
        /* median */
        for (int i = 1; i < REPS; i++) {
            double k = samples[i]; int j = i - 1;
            while (j >= 0 && samples[j] > k) { samples[j+1] = samples[j]; j--; }
            samples[j+1] = k;
        }
        med[c] = samples[REPS/2];
        double bps = (double)cfg[c].bytes_per_iter * ITERS / med[c];
        double ops = (double)cfg[c].ops_per_iter * ITERS / med[c];
        printf("%s : %8.1f GB/s  %6.2f ymm-ops/cyc  (%.1f B/cyc)\n",
               cfg[c].name, bps / 1e9, ops / cyc_s, bps / cyc_s);
    }
    printf("clock: %.3f GHz; sink=%g\n", cyc_s / 1e9, sink);
    free(buf);
    return 0;
}
