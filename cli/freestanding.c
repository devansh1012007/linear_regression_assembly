/* freestanding.c — libc-free `asm-ols` CLI: raw Linux syscalls, _start.
 *
 * PLAN Phase-4 / MoSCoW MUST ("binary I/O + CLI, Linux, libc-free");
 * gate G4 clause 4: "static binary has no dynamic section and a
 * non-executable stack".
 *
 * Scope (deliberate, documented): a fit-only tool over .olsb files with
 * the same JSON field names as cli/asm_ols.c (the libc CLI remains the
 * full-featured one: CSV, methods, confint, p-values). This binary
 * exists to prove the production libols.a needs no libc: it links
 * with -nostdlib -static against the same build/lib core objects and
 * calls only ols_init, ols_workspace_size and ols_fit.
 *
 * Numbers print via hand-rolled %.17g (see out_num below). Verified
 * against the libc CLI by tests/py/freestanding_check.py.
 */
typedef unsigned long u64; typedef long s64;
typedef unsigned int u32; typedef int s32;
typedef unsigned short u16; typedef unsigned char u8;

/* ---- raw syscalls (x86-64) ----------------------------------------------- */
static s64 sys(s64 n, s64 a, s64 b, s64 c, s64 d, s64 e, s64 f) {
    s64 r;
    register s64 r10 __asm__("r10") = d;
    register s64 r8  __asm__("r8")  = e;
    register s64 r9  __asm__("r9")  = f;
    __asm__ __volatile__ ("syscall" : "=a"(r)
                  : "a"(n), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8), "r"(r9)
                  : "rcx", "r11", "memory");
    return r;
}
#define SYS_read 0
#define SYS_write 1
#define SYS_open 2
#define SYS_close 3
#define SYS_mmap 9
#define SYS_exit 60

static void sys_exit(int code) { sys(SYS_exit, code, 0, 0, 0, 0, 0); __builtin_unreachable(); }
static s64 sys_read(int fd, void *b, u64 n) { return sys(SYS_read, fd, (s64)b, n, 0, 0, 0); }
static s64 sys_write(int fd, const void *b, u64 n) { return sys(SYS_write, fd, (s64)b, n, 0, 0, 0); }
static s64 sys_open(const char *p, s32 flags, s32 mode) { return sys(SYS_open, (s64)p, flags, mode, 0, 0, 0); }
static s64 sys_close(int fd) { return sys(SYS_close, fd, 0, 0, 0, 0, 0); }
static void *sys_mmap_anon(u64 bytes) {
    s64 r = sys(SYS_mmap, 0, bytes, 0x3 /*PROT_READ|WRITE*/, 0x22 /*MAP_PRIVATE|ANON*/, -1, 0);
    return (r < 0 && r > -4096) ? (void*)0 : (void*)r;
}

/* ---- tiny libc replacements ---------------------------------------------- */
static u64 str_len(const char *s) { u64 n = 0; while (s[n]) n++; return n; }
static int str_eq(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return *a == *b;
}
static void mem_copy(void *d, const void *s, u64 n) {
    u8 *dd = d; const u8 *ss = s;
    while (n--) *dd++ = *ss++;
}
static void mem_zero(void *d, u64 n) { u8 *p = d; while (n--) *p++ = 0; }

static void out_str(const char *s) { sys_write(1, s, str_len(s)); }
static void out_err(const char *s) { sys_write(2, s, str_len(s)); }
static void out_ch(char c) { sys_write(1, &c, 1); }
static void out_mem(const char *s, u64 n) { sys_write(1, s, n); }

/* dtoa: print v with %.17g semantics via exact integer digits: rescale to
 * [1e16,1e17) (17 significant digits, exact in the 64-bit x87 mantissa),
 * round to integer, format, re-apply the exponent. */
static void fmt_u64(u64 v) {  /* decimal, no padding */
    char b[20]; int n = 0;
    do { b[n++] = '0' + (int)(v % 10); v /= 10; } while (v);
    while (n--) out_ch(b[n]);
}
/* ---- %.17g via __float128 (libgcc, not libc) -----------------------------
 * The JSON stats must round-trip: 17 significant digits, correctly
 * rounded. Double/long-double rescale loops measurably lose the 17th
 * digit (sweeps: 2%..100% mismatches vs libc). __float128 carries a
 * 113-bit mantissa (~34 digits): scaling v into [1e16,1e17) accumulates
 * < 1e-30 relative error, so the final round-to-integer is exact to
 * well below the 0.5-ulp needed at digit 17. libgcc.a is linked with
 * -nostdlib (allowed; it is not libc) — the binary stays static, no
 * dynamic section. Acceptance: 300k-case sweep vs libc %.17g in the
 * out_num unit harness (scripts), plus freestanding_check.py. */
static void out_num(double v) {
    if (v != v) { out_str("null"); return; }              /* JSON-safe NaN */
    if (v == v / 2 && v != 0) { out_str("null"); return; } /* +/-inf */
    if (v == 0) { out_str("0"); return; }
    if (v < 0) { out_ch('-'); v = -v; }

    __float128 q = (__float128)v;
    int e10 = 0;
    /* decimal exponent: normalize to [1e16, 1e17) one decade at a time
     * (each step correctly rounded in 113-bit; cumulative error ~1e-31) */
    while (q >= (__float128)1e17) { q /= (__float128)10; e10++; }
    while (q < (__float128)1e16)  { q *= (__float128)10; e10--; }
    /* round-half-EVEN (libc g-precision semantics). 112/300000 cases in
     * the sweep sit exactly on .5 and half-up sent them the wrong way. */
    unsigned long long fl = (unsigned long long)q;
    __float128 fr = q - (__float128)fl;
    unsigned long long m;
    if (fr > (__float128)0.5) m = fl + 1;
    else if (fr < (__float128)0.5) m = fl;
    else m = (fl & 1ull) ? fl + 1 : fl;      /* exact .5 -> nearest even */
    if (m >= 100000000000000000ull) { m /= 10; e10++; }

    char d[19];
    int nd = 17;
    for (int i = 16; i >= 0; i--) { d[i] = (char)('0' + (int)(m % 10)); m /= 10; }
    d[17] = 0; d[18] = 0;
    while (nd > 1 && d[nd-1] == '0') nd--;   /* %.17g strips trailing 0s */

    int exp = e10 + 16;
    if (exp < -4 || exp >= 17) {
        out_ch(d[0]);
        if (nd > 1) { out_ch('.'); out_mem(d + 1, nd - 1); }
        out_ch('e');
        if (exp >= 0) out_ch('+'); else { out_ch('-'); exp = -exp; }
        if (exp < 10) out_ch('0');
        fmt_u64((u64)exp);
    } else if (exp >= nd - 1) {
        out_mem(d, nd);
        for (int i = nd - 1; i < exp; i++) out_ch('0');
    } else {
        int ip = exp + 1;
        if (ip <= 0) {
            out_str("0.");
            for (int i = 0; i < -ip; i++) out_ch('0');
            out_mem(d, nd);
        } else {
            out_mem(d, ip);
            if (nd > ip) { out_ch('.'); out_mem(d + ip, nd - ip); }
        }
    }
}

/* ---- production library API (libols.a, no libc needed) -------------------
 * Struct layouts COPIED EXACTLY from include/ols.h (frozen; abi.inc):
 * problem = 88 bytes, result = 104 bytes. Do not reorder. */
typedef struct {
    u32 struct_size;         /* 0  */
    u32 version;             /* 4  */
    const double *X;         /* 8  */
    const double *y;         /* 16 */
    const double *w;         /* 24 */
    u64 n;                   /* 32 */
    u64 p;                   /* 40 */
    u64 ldx;                 /* 48 */
    u32 method;              /* 56 */
    u32 flags;               /* 60 */
    double lambda;           /* 64 */
    double rcond;            /* 72 */
    u32 threads;             /* 80 */
    u32 _pad;                /* 84 */
} fs_problem;                /* 88 */
typedef struct {
    double *beta;            /* 0  */
    double intercept;        /* 8  */
    double sse;              /* 16 */
    double sst;              /* 24 */
    double r2;               /* 32 */
    double r2_adj;           /* 40 */
    double sigma2;           /* 48 */
    double cond_est;         /* 56 */
    double *se;              /* 64 */
    s32 rank;                /* 72 */
    s32 method_used;         /* 76 */
    s32 isa_used;            /* 80 */
    s32 _pad;                /* 84 */
    double *t;               /* 88 */
    double f_stat;           /* 96 */
} fs_result;                 /* 104 */
extern s32 ols_init(s32 force_isa);
extern u64 ols_workspace_size(u64 n, u64 p, u32 method, u32 flags);
extern s32 ols_fit(const fs_problem *prob, fs_result *res,
                       void *workspace, u64 ws_bytes);

/* ---- .olsb loader (same format as cli/asm_ols.c) -------------------------- */
typedef struct {
    u64 n_rows, n_cols;
    char (*names)[64];      /* fixed 64/name: freestanding, no malloc */
    double *data;
} fs_dataset;

static u32 rd_u32(const u8 *p) { return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24); }
static u64 rd_u64(const u8 *p) { return (u64)rd_u32(p) | ((u64)rd_u32(p + 4) << 32); }

static s64 load_olsb(const char *path, fs_dataset *d) {
    s64 fd = sys_open(path, 0 /*O_RDONLY*/, 0);
    if (fd < 0) { out_err("asm-ols-fs: cannot open file\n"); return -1; }
    u8 hdr[48];
    if (sys_read(fd, hdr, 48) != 48) { out_err("asm-ols-fs: short header\n"); sys_close(fd); return -1; }
    if (rd_u32(hdr) != 0x4F4C5342u) { out_err("asm-ols-fs: bad magic\n"); sys_close(fd); return -1; }
    if (rd_u32(hdr + 4) != 1) { out_err("asm-ols-fs: version != 1\n"); sys_close(fd); return -1; }
    u64 n = rd_u64(hdr + 8), nc = rd_u64(hdr + 16);
    u64 name_off = rd_u64(hdr + 24), data_off = rd_u64(hdr + 32);
    if (!n || !nc || nc > 4096) { out_err("asm-ols-fs: bad dims\n"); sys_close(fd); return -1; }
    d->n_rows = n; d->n_cols = nc;
    d->data = sys_mmap_anon(n * nc * 8 + 8);
    d->names = sys_mmap_anon(nc * 64);
    if (!d->data || !d->names) { out_err("asm-ols-fs: oom\n"); sys_close(fd); return -1; }
    /* names: read whole block then split (block is small: <= 64KB) */
    u64 nb = data_off - name_off;
    u8 *nbuf = sys_mmap_anon(nb + 1);
    if (!nbuf) { out_err("asm-ols-fs: oom\n"); sys_close(fd); return -1; }
    /* seek via read-forward is avoided: mmap the whole file instead */
    sys_close(fd);
    fd = sys_open(path, 0, 0);
    u64 total = data_off + n * nc * 8;      /* trailer not needed (no CRC check in fs build) */
    u8 *whole = sys_mmap_anon(total + 16);
    if (!whole) { out_err("asm-ols-fs: oom\n"); return -1; }
    u64 got = 0;
    while (got < total) {
        s64 r = sys_read(fd, whole + got, total - got);
        if (r <= 0) break;
        got += r;
    }
    sys_close(fd);
    if (got < data_off + n * nc * 8) { out_err("asm-ols-fs: short file\n"); return -1; }
    mem_copy(d->data, whole + data_off, n * nc * 8);
    mem_copy(nbuf, whole + name_off, nb);
    nbuf[nb] = 0;
    u64 p = 0;
    for (u64 j = 0; j < nc; j++) {
        u64 k = 0;
        while (p + k < nb && nbuf[p + k] && k < 63) { d->names[j][k] = nbuf[p + k]; k++; }
        d->names[j][k] = 0;
        p += k + 1;
    }
    return 0;
}

/* ---- _start ---------------------------------------------------------------- */
static void usage(void) {
    out_err("usage: asm-ols-fs fit <data.olsb> --target NAME [--features A,B,..] [--no-intercept] [--json]\n");
}

/* _start in pure asm: at process entry rsp points at argc. Read it,
 * re-anchor rsp 16-byte aligned (ABI for the call), and jump into the C
 * entry with SysV convention (argc in rdi, argv in rsi) so the compiler
 * owns a normal frame from there on. (First draft read rsp from C after
 * the prologue had already moved it — argc came back garbage.) */
__asm__(
    ".globl _start\n"
    "_start:\n"
    "    xorl  %ebp, %ebp\n"          /* mark outermost frame */
    "    movq  (%rsp), %rdi\n"        /* argc */
    "    leaq  8(%rsp), %rsi\n"       /* argv */
    "    andq  $-16, %rsp\n"          /* align for the call */
    "    call  fs_main\n"
    "    movl  %eax, %edi\n"          /* unreachable: fs_main exits */
    "    movl  $60, %eax\n"
    "    syscall\n"
);

static void __attribute__((used)) fs_main(u64 argc, char **argv) {
    if (argc < 3) { usage(); sys_exit(2); }
    if (!str_eq(argv[1], "fit")) { usage(); sys_exit(2); }

    const char *path = argv[2], *target = 0, *feats = 0;
    int intercept = 1, json = 0;
    for (u64 i = 3; i < argc; i++) {
        if (str_eq(argv[i], "--target") && i + 1 < argc) target = argv[++i];
        else if (str_eq(argv[i], "--features") && i + 1 < argc) feats = argv[++i];
        else if (str_eq(argv[i], "--no-intercept")) intercept = 0;
        else if (str_eq(argv[i], "--json")) json = 1;
        else { usage(); sys_exit(2); }
    }
    if (!target) { usage(); sys_exit(2); }

    /* -1 = AUTO dispatch (same as the libc CLI's lazy ols_isa() init).
     * First draft passed 0, which FORCE-SELECTS SSE2: the AVX2 Gram never
     * ran and every result differed ~1e-15 from the libc CLI — the
     * documented sse2-vs-avx2 accumulation-order contract, not a bug. */
    if (ols_init(-1) < 0) { out_err("asm-ols-fs: ols_init failed\n"); sys_exit(1); }

    fs_dataset d;
    if (load_olsb(path, &d)) sys_exit(1);

    int ti = -1;
    for (u64 j = 0; j < d.n_cols; j++)
        if (str_eq(d.names[j], target)) { ti = (int)j; break; }
    if (ti < 0) { out_err("asm-ols-fs: no column named 'target'\n"); sys_exit(1); }

    /* feature list -> indices (max 64 features, fs scope) */
    int fidx[64], nf = 0;
    if (feats) {
        const char *p = feats;
        while (*p && nf < 64) {
            char name[64]; u64 k = 0;
            while (*p && *p != ',' && k < 63) name[k++] = *p++;
            name[k] = 0;
            if (*p == ',') p++;
            int idx = -1;
            for (u64 j = 0; j < d.n_cols; j++)
                if (str_eq(d.names[j], name)) { idx = (int)j; break; }
            if (idx < 0 || idx == ti) { out_err("asm-ols-fs: bad --features entry\n"); sys_exit(2); }
            fidx[nf++] = idx;
        }
        if (!nf) { out_err("asm-ols-fs: bad --features\n"); sys_exit(2); }
        /* duplicates */
        for (int a = 0; a < nf; a++)
            for (int b = a + 1; b < nf; b++)
                if (fidx[a] == fidx[b]) { out_err("asm-ols-fs: duplicate feature\n"); sys_exit(2); }
    }
    u64 p = nf ? (u64)nf : d.n_cols - 1;
    if (d.n_rows <= p + 1) { out_err("asm-ols-fs: underdetermined\n"); sys_exit(1); }

    /* X row-major + y (contiguous copy, threads=1) */
    double *X = sys_mmap_anon(d.n_rows * p * 8 + 16);
    double *y = sys_mmap_anon(d.n_rows * 8 + 16);
    if (!X || !y) { out_err("asm-ols-fs: oom\n"); sys_exit(1); }
    for (u64 i = 0; i < d.n_rows; i++) {
        u64 dst = 0;
        for (u64 j = 0; j < d.n_cols; j++) {
            if ((int)j == ti) continue;
            if (nf) {
                int keep = 0;
                for (int q = 0; q < nf; q++) if (fidx[q] == (int)j) keep = 1;
                if (!keep) continue;
            }
            X[i * p + dst++] = d.data[i * d.n_cols + j];
        }
        y[i] = d.data[i * d.n_cols + ti];
    }

    double *beta = sys_mmap_anon((p + 1) * 8 * 3 + 16);
    double *se = beta + p + 1, *t = se + p + 1;
    fs_problem prob; mem_zero(&prob, sizeof prob);
    prob.struct_size = sizeof prob; prob.version = 1;
    prob.X = X; prob.y = y; prob.n = d.n_rows; prob.p = (u32)p;
    prob.ldx = (u32)p; prob.method = 0 /*AUTO*/;
    prob.flags = intercept ? 1u : 0u;
    prob.threads = 1;
    fs_result res; mem_zero(&res, sizeof res);
    res.beta = beta; res.se = se; res.t = t; res.rank = -1;

    u64 ws_bytes = ols_workspace_size(d.n_rows, p, 0, prob.flags);
    void *ws = sys_mmap_anon(ws_bytes + 64);
    if (!ws) { out_err("asm-ols-fs: oom\n"); sys_exit(1); }
    s32 rc = ols_fit(&prob, &res, ws, ws_bytes);
    if (rc != 0) { out_err("asm-ols-fs: fit failed (rc)\n"); sys_exit(1); }

    if (!json) {
        out_str("Model: multiple linear regression (freestanding build)\n");
        out_str("Observations: "); fmt_u64(d.n_rows); out_str("  Predictors: "); fmt_u64(p); out_ch('\n');
    }
    out_str("{\"n\": "); fmt_u64(d.n_rows);
    out_str(", \"p\": "); fmt_u64(p);
    out_str(", \"intercept\": "); out_str(intercept ? "true" : "false");
    out_str(",\n \"coefficients\": {");
    u64 k = 0;
    for (u64 j = 0; j < d.n_cols; j++) {
        if ((int)j == ti) continue;
        if (nf) {
            int keep = 0;
            for (int q = 0; q < nf; q++) if (fidx[q] == (int)j) keep = 1;
            if (!keep) continue;
        }
        if (k) out_str(", ");
        out_ch('"');
        /* names are validated ASCII-safe at build time by to_olsb.py; the
         * fs build still escapes the two JSON-breaking chars */
        for (const char *s = d.names[j]; *s; s++) {
            if (*s == '"') out_str("\\\"");
            else if (*s == '\\') out_str("\\\\");
            else out_ch(*s);
        }
        out_str("\": "); out_num(beta[k]);
        k++;
    }
    out_str("},\n \"intercept_value\": "); out_num(res.intercept);
    out_str(", \"sse\": "); out_num(res.sse);
    out_str(", \"sst\": "); out_num(res.sst);
    out_str(", \"r2\": "); out_num(res.r2);
    out_str(", \"r2_adj\": "); out_num(res.r2_adj);
    out_str(", \"sigma2\": "); out_num(res.sigma2);
    out_str(", \"cond_est\": "); out_num(res.cond_est);
    out_str(", \"rank\": "); fmt_u64((u64)res.rank);
    out_str("}\n");
    sys_exit(0);
}
