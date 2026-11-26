/* cli/asm_ols.c — the asm-ols command-line front end (Phase 4a).
 *
 * The CORE (src/) stays pure NASM: allocation-free, syscall-free. This CLI
 * is the product layer on top: dataset loading (.olsb binary / CSV), method
 * selection, human-readable report + machine-readable JSON. It links only
 * against libols + libc (a static libc-free build is the Phase-4c target
 * once the loaders move to raw syscalls; the split keeps that possible).
 *
 * Usage:
 *   asm-ols fit  <dataset.olsb|dataset.csv> [--target NAME] [--features A,B,..]
 *                [--method auto|chol|csne|qr] [--no-intercept] [--lambda L]
 *                [--rcond R] [--json] [--data-only]
 *   asm-ols info <dataset>
 *
 * .olsb format (little-endian, u32 magic 0x4F4C5342 "OLSB", version 1):
 *   header: magic u32 | version u32 | n_rows u64 | n_cols u64
 *           | col_name_off u64 | data_off u64
 *   column-name block (at col_name_off): n_cols NUL-terminated names
 *   data block (at data_off): n_rows * n_cols f64, row-major
 *   trailer: CRC32 (zlib polynomial) of the data block, u32
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <getopt.h>
#include <errno.h>
#include <math.h>
#include "ols.h"

#define OLSB_MAGIC 0x4F4C5342u
#define OLSB_VERSION 1u

typedef struct {
    uint64_t n_rows, n_cols;
    char **names;          /* n_cols */
    double *data;          /* n_rows * n_cols */
} dataset_t;

static uint32_t crc32_buf(const unsigned char *p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(c & 1)));
    }
    return c ^ 0xFFFFFFFFu;
}

static void ds_free(dataset_t *d) {
    for (uint64_t j = 0; j < d->n_cols; j++) free(d->names[j]);
    free(d->names);
    free(d->data);
    memset(d, 0, sizeof *d);
}

/* ---- .olsb ---------------------------------------------------------------- */
static int load_olsb(const char *path, dataset_t *d) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "asm-ols: cannot open %s: %s\n", path, strerror(errno)); return -1; }
    uint32_t hdr[6];   /* magic, version, n_rows lo/hi, n_cols lo/hi */
    unsigned char raw[48];
    if (fread(raw, 1, 48, f) != 48) { fprintf(stderr, "asm-ols: short header\n"); goto fail; }
    memcpy(&hdr[0], raw, 4); memcpy(&hdr[1], raw + 4, 4);
    uint64_t n_rows, n_cols, name_off, data_off, trailer_off;
    memcpy(&n_rows, raw + 8, 8); memcpy(&n_cols, raw + 16, 8);
    memcpy(&name_off, raw + 24, 8); memcpy(&data_off, raw + 32, 8);
    memcpy(&trailer_off, raw + 40, 8);
    if (hdr[0] != OLSB_MAGIC) { fprintf(stderr, "asm-ols: bad magic (not .olsb)\n"); goto fail; }
    if (hdr[1] != OLSB_VERSION) { fprintf(stderr, "asm-ols: version %u != 1\n", hdr[1]); goto fail; }
    if (!n_rows || !n_cols) { fprintf(stderr, "asm-ols: empty dataset\n"); goto fail; }
    if (n_rows > (1u<<28) || n_cols > 4096) { fprintf(stderr, "asm-ols: dimensions out of range\n"); goto fail; }
    d->n_rows = n_rows; d->n_cols = n_cols;
    d->names = calloc(n_cols, sizeof *d->names);
    d->data = malloc((size_t)n_rows * n_cols * sizeof(double));
    if (!d->names || !d->data) { fprintf(stderr, "asm-ols: out of memory\n"); goto fail; }
    /* names: NUL-terminated block at name_off */
    if (fseek(f, (long)name_off, SEEK_SET)) goto fail;
    {
        size_t nb = (size_t)(data_off - name_off);
        char *nb_buf = malloc(nb + 1);
        if (!nb_buf || fread(nb_buf, 1, nb, f) != nb) { free(nb_buf); fprintf(stderr, "asm-ols: bad name block\n"); goto fail; }
        nb_buf[nb] = 0;
        char *p = nb_buf;
        for (uint64_t j = 0; j < n_cols; j++) {
            if (p >= nb_buf + nb) { free(nb_buf); fprintf(stderr, "asm-ols: too few names\n"); goto fail; }
            d->names[j] = strdup(p);
            p += strlen(p) + 1;
        }
        free(nb_buf);
    }
    /* data + CRC trailer */
    size_t db = (size_t)n_rows * n_cols * 8;
    if (fseek(f, (long)data_off, SEEK_SET)) goto fail;
    if (fread(d->data, 1, db, f) != db) { fprintf(stderr, "asm-ols: short data block\n"); goto fail; }
    uint32_t crc_stored;
    if (fseek(f, (long)trailer_off, SEEK_SET) || fread(&crc_stored, 4, 1, f) != 1) {
        fprintf(stderr, "asm-ols: missing CRC trailer\n"); goto fail;
    }
    if (crc32_buf((const unsigned char *)d->data, db) != crc_stored) {
        fprintf(stderr, "asm-ols: CRC mismatch — file corrupt\n"); goto fail;
    }
    fclose(f);
    return 0;
fail:
    fclose(f);
    return -1;
}

/* ---- CSV (RFC-4180-ish: header row required, comma separator, no quoted
 * commas in v1 — documented limitation, parse errors are line numbers) ---- */
static int load_csv(const char *path, dataset_t *d) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "asm-ols: cannot open %s: %s\n", path, strerror(errno)); return -1; }
    char line[65536];
    if (!fgets(line, sizeof line, f)) { fprintf(stderr, "asm-ols: empty CSV\n"); goto fail; }
    /* count columns from the header */
    uint64_t n_cols = 1;
    for (char *p = line; *p; p++) if (*p == ',') n_cols++;
    d->n_cols = n_cols;
    d->names = calloc(n_cols, sizeof *d->names);
    if (!d->names) goto fail;
    {
        char *p = line;
        for (uint64_t j = 0; j < n_cols; j++) {
            char *comma = strchr(p, ',');
            if (comma) *comma = 0;
            /* trim \r\n */
            char *e = p + strlen(p);
            while (e > p && (e[-1] == '\n' || e[-1] == '\r')) *--e = 0;
            d->names[j] = strdup(p);
            if (comma) p = comma + 1;
        }
    }
    size_t cap = 1024;
    d->data = malloc(cap * n_cols * sizeof(double));
    if (!d->data) goto fail;
    uint64_t row = 0;
    unsigned lineno = 1;
    while (fgets(line, sizeof line, f)) {
        lineno++;
        /* skip blank lines */
        char *e = line + strlen(line);
        while (e > line && (e[-1] == '\n' || e[-1] == '\r')) *--e = 0;
        if (!line[0]) continue;
        if ((row + 1) * n_cols > cap) {
            cap *= 2;
            double *nd = realloc(d->data, cap * n_cols * sizeof(double));
            if (!nd) goto fail;
            d->data = nd;
        }
        char *p = line, *end = NULL;
        for (uint64_t j = 0; j < n_cols; j++) {
            errno = 0;
            double v = strtod(p, &end);
            if (end == p || (j + 1 < n_cols && *end != ',')) {
                fprintf(stderr, "asm-ols: CSV parse error at line %u, field %llu\n",
                        lineno, (unsigned long long)j);
                goto fail;
            }
            if (!isfinite(v)) {
                fprintf(stderr, "asm-ols: non-finite value at line %u field %llu (rejected per contract)\n",
                        lineno, (unsigned long long)j);
                goto fail;
            }
            d->data[row * n_cols + j] = v;
            p = (*end == ',') ? end + 1 : end;
        }
        if (*p) { fprintf(stderr, "asm-ols: extra fields at line %u\n", lineno); goto fail; }
        row++;
    }
    d->n_rows = row;
    if (!row) { fprintf(stderr, "asm-ols: no data rows\n"); goto fail; }
    fclose(f);
    return 0;
fail:
    fclose(f);
    return -1;
}

static int load_any(const char *path, dataset_t *d) {
    size_t L = strlen(path);
    if (L >= 5 && !strcmp(path + L - 5, ".olsb")) return load_olsb(path, d);
    return load_csv(path, d);
}

/* ---- fit ------------------------------------------------------------------ */
static const char *method_name(int m) {
    switch (m) {
    case OLS_GRAM_CHOL: return "Gram+Cholesky";
    case OLS_GRAM_CSNE: return "Gram+CSNE";
    case OLS_QR:        return "Householder QR";
    default:            return "?";
    }
}

static int col_index(const dataset_t *d, const char *name) {
    for (uint64_t j = 0; j < d->n_cols; j++)
        if (!strcmp(d->names[j], name)) return (int)j;
    return -1;
}

static int cmd_fit(dataset_t *d, const char *target,
                   char **feat, int n_feat, int method, int intercept,
                   double lambda, double rcond, int json,
                   double conf_level) {
    int ti = col_index(d, target);
    if (ti < 0) { fprintf(stderr, "asm-ols: no column named '%s'\n", target); return 1; }
    if ((uint64_t)n_feat > d->n_cols - 1) { fprintf(stderr, "asm-ols: too many features\n"); return 1; }
    uint64_t n = d->n_rows, p = (n_feat > 0) ? (uint64_t)n_feat : d->n_cols - 1;
    /* build X row-major (contiguous) + y */
    double *X = malloc((size_t)n * p * 8);
    double *y = malloc((size_t)n * 8);
    double *beta = malloc((size_t)p * 8);
    double *se = malloc((size_t)(p + 1) * 8);
    double *t = malloc((size_t)(p + 1) * 8);
    if (!X || !y || !beta || !se || !t) { fprintf(stderr, "asm-ols: oom\n"); return 1; }
    uint64_t fi = 0;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t dst = 0;
        for (uint64_t j = 0; j < d->n_cols; j++) {
            if ((int)j == ti) continue;
            if (n_feat > 0) {
                int keep = 0;
                for (int k = 0; k < n_feat; k++)
                    if (col_index(d, feat[k]) == (int)j) keep = 1;
                if (!keep) continue;
            }
            X[i * p + dst++] = d->data[i * d->n_cols + j];
        }
        fi = dst;
        y[i] = d->data[i * d->n_cols + ti];
    }
    (void)fi;
    ols_problem_t prob = { .struct_size = sizeof prob, .version = 1,
                           .X = X, .y = y, .w = NULL, .n = n, .p = p,
                           .ldx = p, .method = (uint32_t)method,
                           .flags = intercept ? OLS_FIT_INTERCEPT : 0,
                           .lambda = lambda, .rcond = rcond, .threads = 1 };
    ols_result_t res = { .beta = beta, .se = se, .t = t, .rank = -1 };
    size_t ws_bytes = ols_workspace_size(n, p, method, prob.flags);
    void *ws = malloc(ws_bytes);
    int32_t rc = ols_fit(&prob, &res, ws, ws_bytes);

    if (rc != OLS_OK) {
        fprintf(stderr, "asm-ols: fit failed: %s (rc %d)\n",
                rc == OLS_E_SINGULAR ? "SINGULAR (rank-deficient design; try QR or drop a column)"
                : rc == OLS_E_UNDERDETERMINED ? "UNDERDETERMINED (n <= p + intercept)"
                : rc == OLS_E_ILLCOND ? "ILL-CONDITIONED (rcond exceeded; inspect cond_est)"
                : "argument/validation error", rc);
        free(ws); free(X); free(y); free(beta); free(se); free(t);
        return 1;
    }
    const uint64_t dof = n - p - (intercept ? 1u : 0u);
    if (json) {
        printf("{\"n\": %llu, \"p\": %llu, \"method\": \"%s\", \"intercept\": %s,\n",
               (unsigned long long)n, (unsigned long long)p, method_name(res.method_used),
               intercept ? "true" : "false");
        printf(" \"coefficients\": {");
        uint64_t k = 0;
        for (uint64_t j = 0; j < d->n_cols; j++) {
            if ((int)j == ti) continue;
            if (n_feat > 0) {
                int keep = 0;
                for (int q = 0; q < n_feat; q++)
                    if (col_index(d, feat[q]) == (int)j) keep = 1;
                if (!keep) continue;
            }
            printf("%s\"%s\": %.17g", k ? ", " : "", d->names[j], beta[k]);
            k++;
        }
        printf("},\n \"intercept_value\": %.17g,\n", res.intercept);
        printf(" \"sse\": %.17g, \"sst\": %.17g, \"r2\": %.17g, \"r2_adj\": %.17g,\n",
               res.sse, res.sst, res.r2, res.r2_adj);
        /* inference (phase 7): two-sided t p-values per coefficient +
         * intercept, model F upper-tail p. df conventions match fit.asm:
         * t dof = n - p - intercept; F df = (p, n - p - intercept).
         * Rank-deficient fits zero se/t -> NaN p-values; JSON has no NaN
         * so those emit null (the %.17g of NaN would print "nan"). */
        double f_p = ols_f_pvalue_f64_sse2(res.f_stat, (double)p, (double)dof);
        printf(" \"sigma2\": %.17g, \"cond_est\": %.17g, \"rank\": %d,\n",
               res.sigma2, res.cond_est, res.rank);
        if (f_p == f_p && f_p >= 0.0 && f_p <= 1.0)
            printf(" \"f_stat\": %.17g, \"f_pvalue\": %.17g,\n", res.f_stat, f_p);
        else
            printf(" \"f_stat\": %.17g, \"f_pvalue\": null,\n", res.f_stat);
        printf(" \"p_values\": {");
        {
            uint64_t k = 0;
            for (uint64_t j = 0; j < d->n_cols; j++) {
                if ((int)j == ti) continue;
                if (n_feat > 0) {
                    int keep = 0;
                    for (int q = 0; q < n_feat; q++)
                        if (col_index(d, feat[q]) == (int)j) keep = 1;
                    if (!keep) continue;
                }
                double pv = ols_t_pvalue_f64_sse2(t[k], (double)dof);
                const char *v = (pv == pv && pv >= 0.0 && pv <= 1.0)
                                ? NULL : "null";
                if (v) printf("%s\"%s\": %s", k ? ", " : "", d->names[j], v);
                else   printf("%s\"%s\": %.17g", k ? ", " : "", d->names[j], pv);
                k++;
            }
            if (intercept) {
                double pv = ols_t_pvalue_f64_sse2(t[p], (double)dof);
                if (pv == pv && pv >= 0.0 && pv <= 1.0)
                    printf(", \"(intercept)\": %.17g", pv);
                else
                    printf(", \"(intercept)\": null");
            }
        }
        if (conf_level > 0.0) {
            printf("},\n \"confint\": {");
            uint64_t k2 = 0;
            for (uint64_t j = 0; j < d->n_cols; j++) {
                if ((int)j == ti) continue;
                if (n_feat > 0) {
                    int keep = 0;
                    for (int q = 0; q < n_feat; q++)
                        if (col_index(d, feat[q]) == (int)j) keep = 1;
                    if (!keep) continue;
                }
                double lo, hi;
                ols_confint_f64_sse2(beta[k2], se[k2], (double)dof,
                                     conf_level, &lo, &hi);
                printf("%s\"%s\": [%.17g, %.17g]",
                       k2 ? ", " : "", d->names[j], lo, hi);
                k2++;
            }
            if (intercept) {
                double lo, hi;
                ols_confint_f64_sse2(res.intercept, se[p], (double)dof,
                                     conf_level, &lo, &hi);
                printf(", \"(intercept)\": [%.17g, %.17g]", lo, hi);
            }
            printf("},\n \"conf_level\": %.17g,\n", conf_level);
        } else {
            printf("},\n");
        }
        printf(" \"f_df\": [%llu, %llu]}\n",
               (unsigned long long)p, (unsigned long long)dof);
    } else {
        printf("Model: multiple linear regression\n");
        printf("Observations: %llu  Predictors: %llu  Solver: %s%s\n",
               (unsigned long long)n, (unsigned long long)p,
               method_name(res.method_used),
               lambda > 0 ? " (ridge)" : "");
        printf("\nCoefficients:\n");
        if (conf_level > 0.0) {
            double lo0, hi0;
            ols_confint_f64_sse2(res.intercept, se[p], (double)dof,
                                 conf_level, &lo0, &hi0);
            if (intercept)
                printf("  %-24s %15.6g  (se %.4g, t %.3g)  CI%.0f%% [%.6g, %.6g]\n",
                       "(intercept)", res.intercept, se[p], t[p],
                       100.0*conf_level, lo0, hi0);
        } else if (intercept) {
            printf("  %-24s %15.6g  (se %.4g, t %.3g)\n",
                   "(intercept)", res.intercept, se[p], t[p]);
        }
        uint64_t k = 0;
        for (uint64_t j = 0; j < d->n_cols; j++) {
            if ((int)j == ti) continue;
            if (n_feat > 0) {
                int keep = 0;
                for (int q = 0; q < n_feat; q++)
                    if (col_index(d, feat[q]) == (int)j) keep = 1;
                if (!keep) continue;
            }
            if (conf_level > 0.0) {
                double lo, hi;
                ols_confint_f64_sse2(beta[k], se[k], (double)dof,
                                     conf_level, &lo, &hi);
                printf("  %-24s %15.6g  (se %.4g, t %.3g)  CI%.0f%% [%.6g, %.6g]\n",
                       d->names[j], beta[k], se[k], t[k],
                       100.0*conf_level, lo, hi);
            } else {
                printf("  %-24s %15.6g  (se %.4g, t %.3g)\n",
                       d->names[j], beta[k], se[k], t[k]);
            }
            k++;
        }
        printf("\nR2: %.6f  Adj R2: %.6f  F: %.4g  RMSE: %.6g\n",
               res.r2, res.r2_adj, res.f_stat,
               res.sse > 0 ? sqrt(res.sse / (double)(n - p - (intercept ? 1 : 0))) : 0.0);
        printf("cond est (lower bound): %.4g  rank: %d\n", res.cond_est, res.rank);
    }
    free(ws); free(X); free(y); free(beta); free(se); free(t);
    return 0;
}

static int cmd_info(dataset_t *d, const char *path) {
    printf("dataset: %s\n  rows: %llu  cols: %llu\n  columns:",
           path, (unsigned long long)d->n_rows, (unsigned long long)d->n_cols);
    for (uint64_t j = 0; j < d->n_cols && j < 12; j++)
        printf(" %s", d->names[j]);
    if (d->n_cols > 12) printf(" ... (+%llu)", (unsigned long long)(d->n_cols - 12));
    printf("\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: asm-ols fit <data.olsb|data.csv> --target NAME [--features A,B,..]\n"
            "                   [--method auto|chol|csne|qr] [--no-intercept]\n"
            "                   [--lambda L] [--rcond R] [--json]\n"
            "                   [--confint[=LEVEL]]   LEVEL in (0,1), default 0.95\n"
            "       asm-ols info <data>\n");
        return 2;
    }
    const char *cmd = argv[1], *path = argv[2];
    dataset_t ds = {0};
    if (load_any(path, &ds)) return 1;
    if (!strcmp(cmd, "info")) {
        int r = cmd_info(&ds, path);
        ds_free(&ds);
        return r;
    }
    if (strcmp(cmd, "fit")) {
        fprintf(stderr, "asm-ols: unknown command '%s'\n", cmd);
        ds_free(&ds);
        return 2;
    }
    const char *target = NULL, *feats = NULL, *method_s = "auto";
    int intercept = 1, json = 0;
    double conf_level = 0.0;   /* 0 = off; --confint -> 0.95 */
    double lambda = 0.0, rcond = 0.0;
    static const struct option lo[] = {
        {"target", required_argument, 0, 't'},
        {"features", required_argument, 0, 'f'},
        {"method", required_argument, 0, 'm'},
        {"no-intercept", no_argument, 0, 'n'},
        {"lambda", required_argument, 0, 'l'},
        {"rcond", required_argument, 0, 'c'},
        {"json", no_argument, 0, 'j'},
        {"confint", optional_argument, 0, 'C'},
        {0, 0, 0, 0}};
    int o;
    while ((o = getopt_long(argc - 2, argv + 2, "", lo, NULL)) != -1) {
        switch (o) {
        case 't': target = optarg; break;
        case 'f': feats = optarg; break;
        case 'm': method_s = optarg; break;
        case 'n': intercept = 0; break;
        case 'l': lambda = strtod(optarg, NULL); break;
        case 'c': rcond = strtod(optarg, NULL); break;
        case 'j': json = 1; break;
        case 'C':
            if (optarg) {
                char *end = NULL;
                conf_level = strtod(optarg, &end);
                if (!optarg[0] || (end && *end) ||
                    !(conf_level > 0.0 && conf_level < 1.0)) {
                    fprintf(stderr, "asm-ols: --confint level must be in (0,1)\n");
                    return 2;
                }
            } else {
                conf_level = 0.95;
            }
            break;
        default: ds_free(&ds); return 2;
        }
    }
    if (!target) {
        fprintf(stderr, "asm-ols: --target is required for fit\n");
        ds_free(&ds);
        return 2;
    }
    int method = OLS_AUTO;
    if (!strcmp(method_s, "chol")) method = OLS_GRAM_CHOL;
    else if (!strcmp(method_s, "csne")) method = OLS_GRAM_CSNE;
    else if (!strcmp(method_s, "qr")) method = OLS_QR;
    else if (strcmp(method_s, "auto")) {
        fprintf(stderr, "asm-ols: unknown method '%s'\n", method_s);
        ds_free(&ds);
        return 2;
    }
    char **feat = NULL;
    int n_feat = 0;
    if (feats) {
        n_feat = 1;
        for (const char *p = feats; *p; p++) if (*p == ',') n_feat++;
        feat = calloc((size_t)n_feat, sizeof *feat);
        char *tmp = strdup(feats);
        char *save = NULL;
        for (int k = 0; k < n_feat; k++)
            feat[k] = strtok_r(k ? NULL : tmp, ",", &save);
    }
    int r = cmd_fit(&ds, target, feat, n_feat, method, intercept, lambda,
                    rcond, json, conf_level);
    free(feat);
    ds_free(&ds);
    return r;
}
