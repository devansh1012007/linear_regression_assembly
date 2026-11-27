/*
 * avx512_probe.c — stretch-triage hardware check (PLAN §6/§10 AVX-512 stretch item).
 *
 * Pure-C micro-bench (no asm, no nasm needed): measures, with >= 15 reps each,
 * median + p10/p90:
 *   1. FMA peak, zmm (64-bit lanes), 8 independent accumulators, operands in L1.
 *   2. FMA peak, ymm (AVX2), same structure — must reproduce machine.json's 57.62.
 *   3. Load bandwidth L1 (zmm loads, 2 streams + sum, checksum sink).
 *   4. DRAM bandwidth single-thread (streaming sum over 256 MiB) and 2-thread
 *      (pthreads, private 256 MiB buffers each; host RAM is 4 GiB, so 2 x 256 MiB
 *      is safe).
 *   5. Emits raw samples; the Python wrapper computes stats, ratios and p*.
 *
 * Timing: __rdtsc (via rdtscp + lfence where available) AND
 * clock_gettime(CLOCK_MONOTONIC_RAW); each rep is sized >= ~0.2 s wall so
 * TSC-vs-wall drift is visible and small.
 *
 * Safety: if AVX-512 instructions fault (SIGILL: some hypervisors advertise
 * CPUID bits but do not enable the state), the SIGILL handler prints
 * "SIGILL <benchmark-name>" and exits 3 — the wrapper reports it honestly.
 *
 * Dead-code elimination: every benchmark accumulates into a volatile sink;
 * the final sink value is printed as "checksum <hex>" and checked by the
 * wrapper (the repo's checksum-sink idiom).
 *
 * Build (from the wrapper):
 *   gcc -O2 -mavx512f -mavx512dq -mfma -o build/bench/avx512_probe bench/kernels/avx512_probe.c -lpthread
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <stdint.h>
#include <unistd.h>
#include <sched.h>
#include <immintrin.h>
#include <pthread.h>

static const char *cur_bench = "startup";
static volatile double g_sink = 0.0;

static void on_sigill(int sig) {
    (void)sig;
    /* async-signal-safe: write() would be safer, but we only print a tombstone */
    fprintf(stderr, "SIGILL during '%s' (AVX-512 state not actually enabled?)\n",
            cur_bench);
    fprintf(stdout, "FAULT SIGILL %s\n", cur_bench);
    fflush(stdout);
    _exit(3);
}

static void install_sigill(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sigill;
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint64_t now_tsc(void) {
    unsigned lo, hi;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi) :: "rcx");
    return ((uint64_t)hi << 32) | lo;
}
static void fence_tsc(void) {
    __asm__ __volatile__("lfence" ::: "memory");
}

/* Pin: single-thread benches on vCPU 1 (skill §2: avoid core 0); the 2-thread
 * DRAM bench explicitly spreads workers over vCPUs 0 and 1 (an external
 * taskset -c 1 would serialize both threads onto one vCPU and halve it). */
static void pin_to(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof set, &set) != 0)
        printf("INFO pin_to(%d) failed (continuing unpinned)\n", cpu);
}

/* ------------------------------------------------------------------ */
/* 1+2. FMA peak. 8 independent loop-carried accumulators (matches      */
/* machine.json's accumulator-count table, where 8 ymm accs = 57.61     */
/* GFLOP/s = 2 FMA ports); va/vb stay in registers. Flops counted per   */
/* call by the caller.                                                  */
/* ------------------------------------------------------------------ */

/* zmm: 8 lanes of f64 per FMA (16 flops); ymm: 4 lanes (8 flops) */

static double fma_zmm_body(size_t loops) {
    const double a = 1.0000001, b = 1.00000003; /* |a*b| stays ~1, no inf/NaN */
    __m512d acc0 = _mm512_set1_pd(1.0), acc1 = _mm512_set1_pd(2.0);
    __m512d acc2 = _mm512_set1_pd(3.0), acc3 = _mm512_set1_pd(4.0);
    __m512d acc4 = _mm512_set1_pd(5.0), acc5 = _mm512_set1_pd(6.0);
    __m512d acc6 = _mm512_set1_pd(7.0), acc7 = _mm512_set1_pd(8.0);
    __m512d va = _mm512_set1_pd(a), vb = _mm512_set1_pd(b);
    for (size_t i = 0; i < loops; i++) {
        /* 8 unrolled FMAs per accumulator: independent regs, operands in L1 */
        acc0 = _mm512_fmadd_pd(va, vb, acc0);
        acc1 = _mm512_fmadd_pd(va, vb, acc1);
        acc2 = _mm512_fmadd_pd(va, vb, acc2);
        acc3 = _mm512_fmadd_pd(va, vb, acc3);
        acc4 = _mm512_fmadd_pd(va, vb, acc4);
        acc5 = _mm512_fmadd_pd(va, vb, acc5);
        acc6 = _mm512_fmadd_pd(va, vb, acc6);
        acc7 = _mm512_fmadd_pd(va, vb, acc7);
    }
    __m512d s = _mm512_add_pd(_mm512_add_pd(acc0, acc1),
                              _mm512_add_pd(acc2, acc3));
    __m512d t = _mm512_add_pd(_mm512_add_pd(acc4, acc5),
                              _mm512_add_pd(acc6, acc7));
    __m512d u = _mm512_add_pd(s, t);
    double d[8] __attribute__((aligned(64)));
    _mm512_store_pd(d, u);
    return d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7];
}

static double fma_ymm_body(size_t loops) {
    const double a = 1.0000001, b = 1.00000003; /* |a*b| stays ~1, no inf/NaN */
    __m256d acc0 = _mm256_set1_pd(1.0), acc1 = _mm256_set1_pd(2.0);
    __m256d acc2 = _mm256_set1_pd(3.0), acc3 = _mm256_set1_pd(4.0);
    __m256d acc4 = _mm256_set1_pd(5.0), acc5 = _mm256_set1_pd(6.0);
    __m256d acc6 = _mm256_set1_pd(7.0), acc7 = _mm256_set1_pd(8.0);
    __m256d va = _mm256_set1_pd(a), vb = _mm256_set1_pd(b);
    for (size_t i = 0; i < loops; i++) {
        acc0 = _mm256_fmadd_pd(va, vb, acc0);
        acc1 = _mm256_fmadd_pd(va, vb, acc1);
        acc2 = _mm256_fmadd_pd(va, vb, acc2);
        acc3 = _mm256_fmadd_pd(va, vb, acc3);
        acc4 = _mm256_fmadd_pd(va, vb, acc4);
        acc5 = _mm256_fmadd_pd(va, vb, acc5);
        acc6 = _mm256_fmadd_pd(va, vb, acc6);
        acc7 = _mm256_fmadd_pd(va, vb, acc7);
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(acc0, acc1),
                              _mm256_add_pd(acc2, acc3));
    __m256d t = _mm256_add_pd(_mm256_add_pd(acc4, acc5),
                              _mm256_add_pd(acc6, acc7));
    __m256d u = _mm256_add_pd(s, t);
    double d[4] __attribute__((aligned(32)));
    _mm256_store_pd(d, u);
    return d[0] + d[1] + d[2] + d[3];
}

/* flops per body call: loops * 8 accs * lanes * 2 */
#define FMA_ZMM_FLOPS(loops) ((double)(loops) * 8.0 * 8.0 * 2.0)
#define FMA_YMM_FLOPS(loops) ((double)(loops) * 8.0 * 4.0 * 2.0)

/* Supplementary: 16 independent zmm accumulators — checks whether the
 * 8-acc variant (task-prescribed) saturates the 2 zmm FMA ports or is
 * limited by FMA latency x accumulator count. Written with explicit
 * named variables (NOT an acc[] array: -O2 keeps the array on the stack
 * and the loop becomes memory-bound — a compiler artifact, not hardware).
 * Not used for the headline ratio; reported so the 8-acc number can be
 * interpreted. 16 accs + va + vb = 18 of 32 zmm registers. */
static double fma_zmm16_body(size_t loops) {
    const double a = 1.0000001, b = 1.00000003;
    __m512d a0 = _mm512_set1_pd(1.0), a1 = _mm512_set1_pd(2.0);
    __m512d a2 = _mm512_set1_pd(3.0), a3 = _mm512_set1_pd(4.0);
    __m512d a4 = _mm512_set1_pd(5.0), a5 = _mm512_set1_pd(6.0);
    __m512d a6 = _mm512_set1_pd(7.0), a7 = _mm512_set1_pd(8.0);
    __m512d a8 = _mm512_set1_pd(9.0), a9 = _mm512_set1_pd(10.0);
    __m512d a10 = _mm512_set1_pd(11.0), a11 = _mm512_set1_pd(12.0);
    __m512d a12 = _mm512_set1_pd(13.0), a13 = _mm512_set1_pd(14.0);
    __m512d a14 = _mm512_set1_pd(15.0), a15 = _mm512_set1_pd(16.0);
    __m512d va = _mm512_set1_pd(a), vb = _mm512_set1_pd(b);
    for (size_t i = 0; i < loops; i++) {
        a0 = _mm512_fmadd_pd(va, vb, a0);
        a1 = _mm512_fmadd_pd(va, vb, a1);
        a2 = _mm512_fmadd_pd(va, vb, a2);
        a3 = _mm512_fmadd_pd(va, vb, a3);
        a4 = _mm512_fmadd_pd(va, vb, a4);
        a5 = _mm512_fmadd_pd(va, vb, a5);
        a6 = _mm512_fmadd_pd(va, vb, a6);
        a7 = _mm512_fmadd_pd(va, vb, a7);
        a8 = _mm512_fmadd_pd(va, vb, a8);
        a9 = _mm512_fmadd_pd(va, vb, a9);
        a10 = _mm512_fmadd_pd(va, vb, a10);
        a11 = _mm512_fmadd_pd(va, vb, a11);
        a12 = _mm512_fmadd_pd(va, vb, a12);
        a13 = _mm512_fmadd_pd(va, vb, a13);
        a14 = _mm512_fmadd_pd(va, vb, a14);
        a15 = _mm512_fmadd_pd(va, vb, a15);
    }
    __m512d s = _mm512_add_pd(_mm512_add_pd(a0, a1), _mm512_add_pd(a2, a3));
    __m512d t = _mm512_add_pd(_mm512_add_pd(a4, a5), _mm512_add_pd(a6, a7));
    __m512d u = _mm512_add_pd(_mm512_add_pd(a8, a9), _mm512_add_pd(a10, a11));
    __m512d v = _mm512_add_pd(_mm512_add_pd(a12, a13), _mm512_add_pd(a14, a15));
    __m512d w = _mm512_add_pd(_mm512_add_pd(s, t), _mm512_add_pd(u, v));
    double d[8] __attribute__((aligned(64)));
    _mm512_store_pd(d, w);
    return d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7];
}
#define FMA_ZMM16_FLOPS(loops) ((double)(loops) * 16.0 * 8.0 * 2.0)

/* ------------------------------------------------------------------ */
/* 3. L1 load bandwidth: zmm loads from a working set that fits L1     */
/* (machine.json: L1 32 KiB) — 2 x 16 KiB streams, checksum sink.      */
/* ------------------------------------------------------------------ */
#define L1_N 2048 /* doubles per stream = 16 KiB; 2 streams = 32 KiB = L1 */

static double load_l1_body(size_t loops) {
    static double a[L1_N] __attribute__((aligned(64)));
    static double b[L1_N] __attribute__((aligned(64)));
    if (a[0] == 0.0) { for (int i = 0; i < L1_N; i++) { a[i] = 1.0 + i * 1e-9; b[i] = 2.0 + i * 1e-9; } }
    __m512d s = _mm512_setzero_pd();
    for (size_t i = 0; i < loops; i++) {
        __m512d sa = _mm512_setzero_pd(), sb = _mm512_setzero_pd();
        for (int j = 0; j < L1_N; j += 16) { /* 16 doubles per iter of j = 2 zmm per stream */
            sa = _mm512_add_pd(sa, _mm512_load_pd(a + j));
            sa = _mm512_add_pd(sa, _mm512_load_pd(a + j + 8));
            sb = _mm512_add_pd(sb, _mm512_load_pd(b + j));
            sb = _mm512_add_pd(sb, _mm512_load_pd(b + j + 8));
        }
        s = _mm512_add_pd(s, _mm512_add_pd(sa, sb));
    }
    double d[8] __attribute__((aligned(64)));
    _mm512_store_pd(d, s);
    return d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7];
}
/* bytes per body call: loops * 2 arrays * L1_N * 8 */
#define L1_BYTES(loops) ((double)(loops) * 2.0 * (double)L1_N * 8.0)

/* ymm variant of the same 2-stream L1 load loop — apples-to-apples with
 * load_l1_body; 32 B loads vs 64 B loads: ADR-0012's re-open condition is
 * about LOAD throughput, and this measures whether 64 B loads actually move
 * 2x the bytes per load-uop on this host (they should: 2 load ports). */
static double load_l1_ymm_body(size_t loops) {
    static double a[L1_N] __attribute__((aligned(64)));
    static double b[L1_N] __attribute__((aligned(64)));
    if (a[0] == 0.0) { for (int i = 0; i < L1_N; i++) { a[i] = 1.0 + i * 1e-9; b[i] = 2.0 + i * 1e-9; } }
    __m256d s = _mm256_setzero_pd();
    for (size_t i = 0; i < loops; i++) {
        __m256d sa = _mm256_setzero_pd(), sb = _mm256_setzero_pd();
        for (int j = 0; j < L1_N; j += 16) { /* 16 doubles per iter = 4 ymm per stream */
            sa = _mm256_add_pd(sa, _mm256_load_pd(a + j));
            sa = _mm256_add_pd(sa, _mm256_load_pd(a + j + 4));
            sa = _mm256_add_pd(sa, _mm256_load_pd(a + j + 8));
            sa = _mm256_add_pd(sa, _mm256_load_pd(a + j + 12));
            sb = _mm256_add_pd(sb, _mm256_load_pd(b + j));
            sb = _mm256_add_pd(sb, _mm256_load_pd(b + j + 4));
            sb = _mm256_add_pd(sb, _mm256_load_pd(b + j + 8));
            sb = _mm256_add_pd(sb, _mm256_load_pd(b + j + 12));
        }
        s = _mm256_add_pd(s, _mm256_add_pd(sa, sb));
    }
    double d[4] __attribute__((aligned(32)));
    _mm256_store_pd(d, s);
    return d[0] + d[1] + d[2] + d[3];
}

/* ------------------------------------------------------------------ */
/* 4. DRAM streaming bandwidth: sum over a 256 MiB array.              */
/* ------------------------------------------------------------------ */
#define DRAM_DOUBLES (256u * 1024u * 1024u / 8u) /* 256 MiB / 8 B = 33,554,432 */

typedef struct { double *buf; size_t n; int passes; int cpu; double out; } dram_arg_t;

static double dram_body(const double *buf, size_t n, int passes) {
    __m512d s0 = _mm512_setzero_pd(), s1 = _mm512_setzero_pd();
    __m512d s2 = _mm512_setzero_pd(), s3 = _mm512_setzero_pd();
    for (int p = 0; p < passes; p++) {
        size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            s0 = _mm512_add_pd(s0, _mm512_load_pd(buf + i));
            s1 = _mm512_add_pd(s1, _mm512_load_pd(buf + i + 8));
            s2 = _mm512_add_pd(s2, _mm512_load_pd(buf + i + 16));
            s3 = _mm512_add_pd(s3, _mm512_load_pd(buf + i + 24));
        }
        for (; i < n; i++) {
            __m512d t = _mm512_set1_pd(buf[i]); /* scalar tail kept in a zmm op */
            s0 = _mm512_add_pd(s0, t);
        }
    }
    __m512d s = _mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3));
    double d[8] __attribute__((aligned(64)));
    _mm512_store_pd(d, s);
    return d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7];
}

static void *dram_thread(void *p) {
    dram_arg_t *A = (dram_arg_t *)p;
    pin_to(A->cpu);
    A->out = dram_body(A->buf, A->n, A->passes);
    return NULL;
}

/* How many whole passes over the buffer make ~target_s (>=1 pass)? */
static int dram_calibrate(const double *buf, size_t n, double target_s) {
    double t0 = now_s();
    g_sink += dram_body(buf, n, 1);
    double dt = now_s() - t0;
    int passes = (int)(target_s / dt) + 1;
    if (passes < 1) passes = 1;
    if (passes > 4096) passes = 4096;
    return passes;
}

/* ------------------------------------------------------------------ */
/* Runner: reps of a body, prints SAMPLE lines with wall + tsc.        */
/* ------------------------------------------------------------------ */
typedef double (*body_fn)(size_t);

/* Calibrate loop count so one rep ~= target seconds. Returns loops. */
static size_t calibrate(body_fn fn, double flops_or_bytes_per_loop,
                        double target_s, const char *metric) {    (void)metric;
    size_t loops = 1000;
    for (int it = 0; it < 12; it++) {
        double t0 = now_s();
        g_sink += fn(loops);
        double dt = now_s() - t0;
        if (dt > 0.05) {
            double scale = target_s / dt;
            if (scale > 4.0) { loops *= 4; continue; }      /* still too fast */
            if (scale > 1.3) { loops = (size_t)(loops * scale); }
            return loops;
        }
        loops *= 4;
        if (loops > (size_t)1 << 40) return loops;
    }
    (void)flops_or_bytes_per_loop;
    return loops;
}

/* Generic timed rep loop for compute/L1 benchmarks. */
static void run_bench(const char *name, body_fn fn, size_t loops,
                      double flops, double bytes, int reps) {
    printf("BENCH %s loops %zu reps %d flops_per_call %.1f bytes_per_call %.1f\n",
           name, loops, reps, flops, bytes);
    for (int r = 0; r < reps; r++) {
        fence_tsc();
        double t0 = now_s();
        uint64_t c0 = now_tsc();
        g_sink += fn(loops);
        uint64_t c1 = now_tsc();
        double t1 = now_s();
        printf("SAMPLE %s %d wall_s %.9f tsc %llu\n", name, r, t1 - t0,
               (unsigned long long)(c1 - c0));
    }
}

int main(int argc, char **argv) {
    int reps = (argc > 1) ? atoi(argv[1]) : 15;
    if (reps < 3) reps = 3;
    install_sigill();
    setvbuf(stdout, NULL, _IOLBF, 0);

    printf("PROBE gcc-built avx512 microbench\n");
    printf("INFO reps %d\n", reps);

    /* --- TSC frequency: wall time over a long TSC interval --- */
    {
        fence_tsc();
        double w0 = now_s();
        uint64_t c0 = now_tsc();
        struct timespec req = {0, 200 * 1000 * 1000}; /* 200 ms */
        nanosleep(&req, NULL);
        uint64_t c1 = now_tsc();
        double w1 = now_s();
        printf("TSC ghz %.6f over %.4f s\n",
               (double)(c1 - c0) / (w1 - w0) / 1e9, w1 - w0);
    }

    /* --- 1. zmm FMA peak --- */
    cur_bench = "fma_zmm";
    {
        size_t loops = calibrate((body_fn)fma_zmm_body, FMA_ZMM_FLOPS(1), 0.2, "gflops");
        run_bench("fma_zmm", (body_fn)fma_zmm_body, loops,
                  FMA_ZMM_FLOPS(loops), 0.0, reps);
    }

    /* --- 2. ymm FMA peak (reproduce machine.json 57.62) --- */
    cur_bench = "fma_ymm";
    {
        size_t loops = calibrate((body_fn)fma_ymm_body, FMA_YMM_FLOPS(1), 0.2, "gflops");
        run_bench("fma_ymm", (body_fn)fma_ymm_body, loops,
                  FMA_YMM_FLOPS(loops), 0.0, reps);
    }

    /* --- 1b. supplementary: zmm FMA with 16 accumulators --- */
    cur_bench = "fma_zmm16";
    {
        size_t loops = calibrate((body_fn)fma_zmm16_body, FMA_ZMM16_FLOPS(1), 0.2, "gflops");
        run_bench("fma_zmm16", (body_fn)fma_zmm16_body, loops,
                  FMA_ZMM16_FLOPS(loops), 0.0, reps);
    }

    /* --- 3. L1 load bandwidth (zmm 64B loads) --- */
    cur_bench = "load_l1_zmm";
    {
        size_t loops = calibrate(load_l1_body, L1_BYTES(1), 0.2, "gbps");
        run_bench("load_l1_zmm", load_l1_body, loops,
                  0.0, L1_BYTES(loops), reps);
    }

    /* --- 3b. L1 load bandwidth (ymm 32B loads, same working set) --- */
    cur_bench = "load_l1_ymm";
    {
        size_t loops = calibrate(load_l1_ymm_body, L1_BYTES(1), 0.2, "gbps");
        run_bench("load_l1_ymm", load_l1_ymm_body, loops,
                  0.0, L1_BYTES(loops), reps);
    }

    /* --- 4a. DRAM 1 thread --- */
    cur_bench = "dram_1t";
    {
        pin_to(1);
        double *buf = NULL;
        if (posix_memalign((void **)&buf, 64, (size_t)DRAM_DOUBLES * 8) != 0) {
            printf("ERROR alloc dram\n"); return 2;
        }
        /* fill once with a cheap LCG pattern (touches every page once) */
        uint64_t st = 0x9E3779B97F4A7C15ull;
        for (size_t i = 0; i < DRAM_DOUBLES; i++) {
            st = st * 6364136223846793005ull + 1442695040888963407ull;
            buf[i] = (double)(st >> 40) * (1.0 / 16777216.0);
        }
        int passes = dram_calibrate(buf, DRAM_DOUBLES, 0.2);
        printf("BENCH dram_1t bytes_per_call %zu passes %d reps %d\n",
               (size_t)DRAM_DOUBLES * 8 * (size_t)passes, passes, reps);
        for (int r = 0; r < reps; r++) {
            fence_tsc();
            double t0 = now_s();
            uint64_t c0 = now_tsc();
            g_sink += dram_body(buf, DRAM_DOUBLES, passes);
            uint64_t c1 = now_tsc();
            double t1 = now_s();
            printf("SAMPLE dram_1t %d wall_s %.9f tsc %llu\n", r, t1 - t0,
                   (unsigned long long)(c1 - c0));
        }
        free(buf);
    }

    /* --- 4b. DRAM 2 threads (private buffers: 2 x 256 MiB, safe in 4 GiB) --- */
    cur_bench = "dram_2t";
    {
        double *bufs[2] = {NULL, NULL};
        pthread_t th[2];
        dram_arg_t args[2];
        for (int t = 0; t < 2; t++) {
            if (posix_memalign((void **)&bufs[t], 64, (size_t)DRAM_DOUBLES * 8) != 0) {
                printf("ERROR alloc dram 2t\n"); return 2;
            }
            uint64_t st = 0x1234567 + (uint64_t)t;
            for (size_t i = 0; i < DRAM_DOUBLES; i++) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                bufs[t][i] = (double)(st >> 40) * (1.0 / 16777216.0);
            }
            args[t].buf = bufs[t]; args[t].n = DRAM_DOUBLES; args[t].passes = 0;
            args[t].cpu = t; args[t].out = 0.0;
        }
        /* calibrate with both buffers already faulted in */
        args[0].passes = dram_calibrate(bufs[0], DRAM_DOUBLES, 0.2);
        args[1].passes = args[0].passes;
        printf("BENCH dram_2t bytes_per_call %zu passes %d reps %d\n",
               (size_t)DRAM_DOUBLES * 8 * (size_t)args[0].passes * 2,
               args[0].passes, reps);
        for (int r = 0; r < reps; r++) {
            fence_tsc();
            double t0 = now_s();
            uint64_t c0 = now_tsc();
            for (int t = 0; t < 2; t++)
                if (pthread_create(&th[t], NULL, dram_thread, &args[t]) != 0) {
                    printf("ERROR pthread\n"); return 2;
                }
            for (int t = 0; t < 2; t++) pthread_join(th[t], NULL);
            for (int t = 0; t < 2; t++) g_sink += args[t].out;
            uint64_t c1 = now_tsc();
            double t1 = now_s();
            printf("SAMPLE dram_2t %d wall_s %.9f tsc %llu\n", r, t1 - t0,
                   (unsigned long long)(c1 - c0));
        }
        free(bufs[0]); free(bufs[1]);
    }

    cur_bench = "done";
    printf("CHECKSUM %.17g\n", (double)g_sink);
    printf("DONE\n");
    return 0;
}
