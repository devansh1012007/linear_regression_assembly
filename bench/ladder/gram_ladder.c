/* bench/ladder/gram_ladder.c — Phase 5: baseline ladder for the Gram pass.
 *
 * Rungs (same workload: upper-triangle G = Z'Z over n rows, q columns,
 * double, row-major Z):
 *   1. naive-O2  — scalar triple loop, -O2
 *   2. naive-O3  — scalar triple loop, -O3 (auto-vectorized by gcc)
 *   3. intrin    — hand AVX2+FMA intrinsics, 4-column blocking
 *   4. asm-v3    — the library kernel (ols_gram_f64_avx2), pivot zeros
 *   5. numpy     — measured from Python (driver prints instructions; the
 *                  python side appends Z'Z timing)
 * Output: one line per (rung, n, q): ns/row + GB/s. argv: n q [reps].
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ols.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void naive_gram(const double *Z, size_t n, size_t q, double *G) {
    memset(G, 0, q * q * sizeof *G);
    for (size_t i = 0; i < n; i++) {
        const double *z = Z + i * q;
        for (size_t a = 0; a < q; a++) {
            double za = z[a];
            if (za == 0.0) continue;
            for (size_t b = a; b < q; b++)
                G[a * q + b] += za * z[b];
        }
    }
}

#include <immintrin.h>
static void intrin_gram(const double *Z, size_t n, size_t q, double *G) {
    memset(G, 0, q * q * sizeof *G);
    /* 4-column outer blocking: for each column block, accumulate the
     * q x 4 stripe of G over all rows (8-row inner tiles) */
    for (size_t a0 = 0; a0 < q; a0 += 4) {
        size_t aw = a0 + 4 <= q ? 4 : q - a0;
        for (size_t i = 0; i < n; i++) {
            const double *z = Z + i * q;
            for (size_t a = a0; a < a0 + aw; a++) {
                __m256d va = _mm256_broadcast_sd(z + a);
                /* b runs over a..q; full stripes then a scalar tail */
                size_t b = a;
                for (; b + 4 <= q; b += 4) {
                    __m256d vb = _mm256_loadu_pd(z + b);
                    __m256d g = _mm256_loadu_pd(G + a * q + b);
                    g = _mm256_fmadd_pd(va, vb, g);
                    _mm256_storeu_pd(G + a * q + b, g);
                }
                for (; b < q; b++)
                    G[a * q + b] += z[a] * z[b];
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s n q [reps]\n", argv[0]);
        return 2;
    }
    size_t n = strtoull(argv[1], NULL, 10);
    size_t q = strtoull(argv[2], NULL, 10);
    int reps = argc > 3 ? atoi(argv[3]) : 5;
    size_t p = q - 1;

    double *Z = aligned_alloc(64, n * q * sizeof *Z);
    double *G = aligned_alloc(64, q * q * sizeof *G);
    double *Gref = aligned_alloc(64, q * q * sizeof *G);
    srand(42);
    for (size_t i = 0; i < n * q; i++)
        Z[i] = (double)rand() / RAND_MAX - 0.5;

    /* correctness cross-check first (intrin vs naive) */
    naive_gram(Z, n, q, Gref);
    intrin_gram(Z, n, q, G);
    double dmax = 0;
    for (size_t i = 0; i < q * q; i++) {
        double d = fabs(G[i] - Gref[i]);
        if (d > dmax) dmax = d;
    }

    /* asm kernel setup: c = 0 pivots => G is the plain Z'Z upper triangle.
     * The kernel walks y as a CONTIGUOUS vector (stride 8) — the first
     * draft aliased y into Z's last column, which the kernel then read
     * with the wrong stride (maxdiff 81 at n=1e5). Copy the column out. */
    ols_gram_args_t ga;
    memset(&ga, 0, sizeof ga);
    double *c = calloc(q, sizeof *c);
    double *s = calloc(q, sizeof *c);
    double W = 0;
    double *ycol = malloc(n * sizeof *ycol);
    for (size_t i = 0; i < n; i++)
        ycol[i] = Z[i * q + p];
    ga.X = Z;
    ga.ldx8 = q * 8;
    ga.y = ycol;
    ga.n = n;
    ga.p = p;
    ga.c = c;
    ga.G = aligned_alloc(64, q * q * 8);
    ga.s = s;
    ga.Wout = &W;
    size_t sb = ols_gram_scratch_bytes(q);
    ga.scratch = aligned_alloc(64, sb);
    ga.scratch_bytes = sb;

    struct { const char *name; void (*fn)(const double*, size_t, size_t, double*); } rungs[] = {
        {"naive-O2", naive_gram}, {"intrin", intrin_gram}};
    /* run each C rung */
    for (unsigned r = 0; r < 2; r++) {
        double best = 1e30;
        for (int k = 0; k < reps; k++) {
            double t0 = now_s();
            rungs[r].fn(Z, n, q, G);
            double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        printf("%s n=%zu q=%zu %.2f ns/row %.2f GB/s\n",
               rungs[r].name, n, q, best / n * 1e9,
               n * q * 8 / best / 1e9);
    }
    /* asm rung */
    {
        double best = 1e30;
        int32_t rc = 0;
        for (int k = 0; k < reps; k++) {
            double t0 = now_s();
            rc = ols_gram_f64_avx2(&ga);
            double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        /* correctness: kernel G vs naive upper triangle */
        double dk = 0;
        for (size_t a = 0; a < q; a++)
            for (size_t b = a; b < q; b++) {
                double d = fabs(ga.G[a * q + b] - Gref[a * q + b]);
                if (d > dk) dk = d;
            }
        printf("asm-v3 n=%zu q=%zu %.2f ns/row %.2f GB/s (rc=%d, maxdiff %.2e)\n",
               n, q, best / n * 1e9, n * q * 8 / best / 1e9, rc, dk);
    }
    printf("# intrin-vs-naive maxdiff %.3e; numpy: run the python driver\n", dmax);
    return 0;
}
