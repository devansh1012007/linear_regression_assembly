/* src/thread/ols_mt.c — Phase 4c: row-partitioned multithreaded Gram.
 *
 * The CORE (the NASM files under src/core) stays syscall-free; this is the
 * orchestration layer the Makefile compiles into libols (pthreads, like
 * libc — the freestanding CLI does not link it).
 *
 * Scheme (ADR-0008): T worker threads each run ols_gram_f64_avx2 on a
 * contiguous row slice into their OWN G_t/s_t/W_t buffers (the SAME pivots
 * c for every slice), then the caller-side merge folds the partials in
 * FIXED INDEX ORDER (t = 0, 1, ... T-1) — never completion order. Each
 * S_t is centered on the same c, so the merge is a pure elementwise sum;
 * the fixed order makes the float addition sequence — and therefore the
 * result — bitwise identical for any thread count and any scheduling.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "ols.h"

#define MT_MAX_THREADS 256

typedef struct {
    ols_gram_args_t args;      /* slice-specific copy */
    int32_t rc;
} mt_slice_t;

typedef struct {
    size_t t;                  /* this worker's slice index */
    mt_slice_t *slice;
} mt_job_t;

static void *mt_worker(void *arg_) {
    mt_job_t *job = (mt_job_t *)arg_;
    job->slice->rc = ols_gram_f64_avx2(&job->slice->args);
    return NULL;
}

int32_t ols_gram_mt (const ols_gram_args_t *A, size_t threads,
                     void *scratch, size_t scratch_bytes) {
    if (threads <= 1 || A->n < 2 * threads) {
        if (A->scratch_bytes < ols_gram_scratch_bytes(A->p + 1))
            return OLS_E_WORKSPACE;
        return ols_gram_f64_avx2(A);
    }
    size_t T = threads;
    if (T > MT_MAX_THREADS) T = MT_MAX_THREADS;
    if (scratch_bytes < ols_gram_mt_scratch_bytes(A->p + 1, T))
        return OLS_E_WORKSPACE;

    const size_t q = A->p + 1;
    double *partials = (double *)scratch;
    const size_t per = q * q + q + 1;
    size_t pbytes = ((per * 8 * T) + 63) & ~(size_t)63;
    char *kscr = (char *)scratch + pbytes;
    size_t ks_each = (ols_gram_scratch_bytes(q) + 63) & ~(size_t)63;

    mt_slice_t *slices = calloc(T, sizeof *slices);
    mt_job_t *jobs = calloc(T, sizeof *jobs);
    if (!slices || !jobs) { free(slices); free(jobs); return -ENOMEM; }

    size_t base_n = A->n / T, rem = A->n % T, row = 0;
    for (size_t t = 0; t < T; t++) {
        size_t cnt = base_n + (t < rem ? 1 : 0);
        ols_gram_args_t *sa = &slices[t].args;
        *sa = *A;
        sa->n = cnt;
        sa->X = (const double *)((const char *)A->X + row * A->ldx8);
        if (A->y) sa->y = A->y + row;
        if (A->w) sa->w = A->w + row;
        sa->G = partials + t * per;
        sa->s = partials + t * per + q * q;
        sa->Wout = partials + t * per + q * q + q;
        sa->scratch = kscr + t * ks_each;
        sa->scratch_bytes = ols_gram_scratch_bytes(q);
        jobs[t].t = t;
        jobs[t].slice = &slices[t];
        row += cnt;
    }

    pthread_t th[MT_MAX_THREADS];
    size_t spawned = 0;
    for (size_t t = 1; t < T; t++) {
        if (pthread_create(&th[t], NULL, mt_worker, &jobs[t]) == 0)
            spawned++;
        else
            jobs[t].slice->rc = ols_gram_f64_avx2(&jobs[t].slice->args);
    }
    /* slice 0 on the main thread */
    mt_worker(&jobs[0]);
    for (size_t t = 1; t < T; t++)
        if (th[t] && spawned) pthread_join(th[t], NULL);

    int32_t rc = 0;
    for (size_t t = 0; t < T; t++)
        if (slices[t].rc) { rc = slices[t].rc; break; }
    if (rc) { free(slices); free(jobs); return rc; }

    /* fixed-order merge (index order, never completion order) */
    memset(A->G, 0, q * q * 8);
    memset(A->s, 0, q * 8);
    double W = 0.0;
    for (size_t t = 0; t < T; t++) {
        const double *Gt = partials + t * per;
        const double *st = Gt + q * q;
        for (size_t i = 0; i < q * q; i++) A->G[i] += Gt[i];
        for (size_t i = 0; i < q; i++) A->s[i] += st[i];
        W += *(Gt + q * q + q);
    }
    if (A->Wout) *A->Wout = W;
    free(slices);
    free(jobs);
    return 0;
}

size_t ols_gram_mt_scratch_bytes (size_t q, size_t threads) {
    const size_t per = q * q + q + 1;
    size_t T = threads > MT_MAX_THREADS ? MT_MAX_THREADS : threads;
    size_t pbytes = ((per * 8 * T) + 63) & ~(size_t)63;
    size_t ks = (ols_gram_scratch_bytes(q) + 63) & ~(size_t)63;
    return pbytes + T * ks;
}
