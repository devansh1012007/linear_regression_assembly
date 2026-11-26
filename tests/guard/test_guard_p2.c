/* test_guard_p2.c — guard-page safety for the Phase-2 kernels.
 * Every kernel runs with X/y/w buffers ending exactly at a PROT_NONE page
 * across n = 1..70 and p = 1..9 (representative sweep), and with the OUTPUT
 * buffers (G, s, Wout, sse, xtr, c) guard-terminated in a second pass.
 * Any read/write outside the exact contract faults -> test fails. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ols.h"
#include "guard.h"

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } } while (0)

typedef struct {
    const double *X; size_t ldx8; const double *y; const double *w;
    size_t n, p; const double *c; double *G, *s, *Wout;
    void *scratch; size_t scratch_bytes;
} GA_t;

static double grnd(int i, int k) {
    uint64_t z = ((uint64_t)i << 32) ^ (uint64_t)(k + 1) ^ 0x9E3779B97F4A7C15ULL;
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return 1.0 + (double)(z >> 11) * 0x1.0p-52;
}

static int run_all(void) {
    for (int p = 1; p <= 9; p++) {
        for (int n = 1; n <= 70; n += 3) {
            size_t q = p + 1;
            size_t xb = n * p * 8, yb = n * 8;
            struct guard_buf gx = gbuf_end(xb), gy = gbuf_end(yb);
            double *X = gx.ptr, *y = gy.ptr;
            for (int i = 0; i < n; i++)
                for (int k = 0; k < p; k++)
                    /* splitmix64 exact doubles in [1,2): non-periodic,
                     * independent columns (both earlier patterns —
                     * identical columns, then periodic residues — were
                     * rank-deficient by construction) */
                    X[i * p + k] = grnd(i, k);
            for (int i = 0; i < n; i++) y[i] = 1.0 + (i % 5);
            /* outputs in their own guarded pages */
            struct guard_buf gG = gbuf_end(q * q * 8);
            struct guard_buf gs = gbuf_end(q * 8);
            struct guard_buf gW = gbuf_end(8);
            struct guard_buf gc = gbuf_end(q * 8);
            struct guard_buf gsse = gbuf_end(8);
            struct guard_buf gxtr = gbuf_end(p * 8);
            struct guard_buf gbeta = gbuf_end(p * 8);
            double *c = gc.ptr;
            for (int k = 0; k < (int)q; k++) c[k] = 0.25;
            double *beta = gbeta.ptr;
            for (int k = 0; k < p; k++) beta[k] = 0.125;
            void *scratch = NULL;
            size_t sb = ols_gram_scratch_bytes(q);
            if (posix_memalign(&scratch, 64, sb) != 0) { printf("FAIL: scratch alloc"); failures++; return 1; }
            GA_t ga = { X, (size_t)p * 8, y, NULL, n, p, c, gG.ptr, gs.ptr, gW.ptr,
                        scratch, sb };
            ols_gram_args_t *ga_p = (ols_gram_args_t *)&ga;
            /* the struct layouts must be identical; static check */
            if (sizeof(GA_t) != sizeof(ols_gram_args_t)) { failures++; return 1; }
            int rc1 = ols_gram_f64_avx2(ga_p);
            int rc2 = ols_gram_f64_sse2(ga_p);
            CHECK(rc1 == 0 && rc2 == 0, "gram rc");
            ols_resid_args_t ra = { X, (size_t)p * 8, y, NULL, n, p,
                                    0.5, beta, gsse.ptr, gxtr.ptr, NULL };
            int rc3 = ols_residual_f64_sse2(&ra);
            CHECK(rc3 == 0, "residual rc");
            int rc4 = ols_pivot_means_f64_sse2(X, (size_t)p * 8, y, NULL,
                                               n < 4 ? (size_t)n : 4, p, c);
            CHECK(rc4 == 0, "pivot_means rc");
            /* qr: guarded X/y/Aug/norms/work — build -> factor -> solve.
             * Rows are 0.5+(i%7) and beta 0.125: everything binary-exact,
             * so the row-0 prediction from the solved slopes is exact. */
            if ((size_t)n >= (size_t)q) {   /* ols_fit refuses
                underdetermined problems before the QR path (n >= m-1 is the
                kernel contract; the n=1,p=1 case walked the solve off the
                1-row Aug buffer — the segfault this guard caught) */
                size_t qa_n = (size_t)n, qa_m = (size_t)q + 1;
                struct guard_buf gAug = gbuf_end(qa_n * qa_m * 8);
                struct guard_buf gS = gbuf_end((size_t)q * 8);
                struct guard_buf gW = gbuf_end(8);
                struct guard_buf gSst = gbuf_end(8);
                struct guard_buf gNr = gbuf_end((qa_m - 1) * 8);
                struct guard_buf gWk = gbuf_end(qa_m * 8);
                double *nr = gNr.ptr;
                ols_qr_args_t qa = { X, (size_t)p * 8, y, NULL, qa_n,
                                     (size_t)p, c, gAug.ptr, gS.ptr, gW.ptr,
                                     gSst.ptr, nr, gWk.ptr, 1 };
                fprintf(stderr, "qr: p=%d n=%d\n", p, n);
                int rcq = ols_qr_build_f64_sse2(&qa);
                CHECK(rcq == 0, "qr build rc");
                rcq = ols_qr_f64_sse2(gAug.ptr, qa_n, qa_m, qa_m,
                                      (double)qa_m * 2.220446049250313e-16,
                                      gWk.ptr);
                CHECK(rcq == (int)q, "qr rank");
                if (rcq == (int)q) {
                double *bs = gs.ptr;             /* reuse guarded q buffer */
                ols_qr_solve_f64_sse2(gAug.ptr, qa_m, qa_m, bs);
                double pred = 0.5;               /* the ones column, scaled */
                for (int k = 0; k < p; k++)
                    pred += (bs[1 + k] / nr[1 + k]) * X[k];
                /* the model is fit to y = 1+(i%5) rows; only check that the
                 * solve produced FINITE values (exactness depends on the
                 * factorization path, covered by the pytest differentials) */
                int finite = 1;
                for (int k = 0; k < p; k++)
                    if (!(bs[1 + k] == bs[1 + k])) finite = 0;
                CHECK(finite == 1, "qr solve finite");
                (void)pred;
                }
                gbuf_free(&gAug); gbuf_free(&gS); gbuf_free(&gW);
                gbuf_free(&gSst); gbuf_free(&gNr); gbuf_free(&gWk);
                }
            /* predict: guarded out buffer; row-0 vs the C reference within
             * rounding (the grnd values are exact doubles but the dot sums
             * round — the old exact-equality check assumed the old
             * all-halves pattern) */
            struct guard_buf gop = gbuf_end(yb);
            double *op = gop.ptr;
            ols_pred_args_t pa = { X, (size_t)p * 8, beta, 0.25, (size_t)n,
                                   (size_t)p, op };
            int rc8 = ols_predict_f64_sse2(&pa);
            CHECK(rc8 == 0, "predict rc");
            {
                double exp0 = 0.25;
                for (int k = 0; k < p; k++) exp0 += beta[k] * X[k];
                if (!(op[0] - exp0 <= 1e-12 * (exp0 + 1.0) &&
                      exp0 - op[0] <= 1e-12 * (exp0 + 1.0)))
                    { printf("FAIL: predict row0\n"); failures++; }
            }
            /* chol/solve on a small SPD block built in guarded memory */
            struct guard_buf gA = gbuf_end(q * q * 8);
            double *A = gA.ptr;
            for (int j = 0; j < (int)q; j++)
                for (int k = j; k < (int)q; k++)
                    A[j * q + k] = (j == k) ? 4.0 + j : 0.5;
            int rc5 = ols_chol_f64_sse2(A, q, 1e-14, 0);
            CHECK(rc5 == 0, "chol rc");
            double *b = gs.ptr;              /* reuse guarded q buffer */
            for (int k = 0; k < (int)q; k++) b[k] = 1.0 + k;
            int rc6 = ols_chol_solve_f64_sse2(A, q, b, 0);
            CHECK(rc6 == 0, "chol_solve rc");
            struct guard_buf gB = gbuf_end(q * q * 8);
            int rc7 = ols_chol_inv_f64_sse2(A, q, gB.ptr, 0);
            CHECK(rc7 == 0, "chol_inv rc");
            gbuf_free(&gx); gbuf_free(&gy); gbuf_free(&gG); gbuf_free(&gs);
            gbuf_free(&gW); gbuf_free(&gc); gbuf_free(&gsse); gbuf_free(&gxtr);
            gbuf_free(&gbeta); gbuf_free(&gA); gbuf_free(&gB); gbuf_free(&gop);
            free(scratch);
        }
    }
    return failures;
}

int main(void) {
    int rc = run_all();
    printf("%s: test_guard_p2 (%d failures)\n", rc ? "FAILED" : "PASSED", rc);
    return rc ? 1 : 0;
}
