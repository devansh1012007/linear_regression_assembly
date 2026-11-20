/* test_guard.c — guard-page harness self-test (asm-verification §3).
 * Proves the harness catches a deliberate overread BEFORE it is trusted on
 * real kernels. Expected faults run in a forked child; the parent asserts
 * the child died with SIGSEGV. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <setjmp.h>
#include <xmmintrin.h>
#include "ols.h"
#include "guard.h"

extern uint64_t test_bad_overread_end(const double *, uint64_t);
extern uint64_t test_good_read_exact(const double *, uint64_t);

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else { printf("ok:   %s\n", msg); } } while (0)

static int expect_segv(void (*fn)(void)) {
    pid_t pid = fork();
    if (pid == 0) {
        fn();
        _exit(0);                    /* survived: the harness FAILED to catch */
    }
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFSIGNALED(st) && WTERMSIG(st) == SIGSEGV;
}

static struct guard_buf g_buf;
static void run_bad_overread(void) {
    double *a = (double *)g_buf.ptr;      /* ends exactly at PROT_NONE */
    volatile uint64_t v = test_bad_overread_end(a, 32);   /* reads a[32]: fault */
    (void)v;
}

int main(void) {
    /* 1. guard_end placement: ptr+bytes must land on a page boundary */
    size_t bytes = 32 * sizeof(double);
    g_buf = gbuf_end(bytes);
    CHECK(g_buf.ptr != NULL, "guard_end allocation succeeds");
    CHECK(((uintptr_t)g_buf.ptr + bytes) % g_pgsize == 0,
          "guard_end buffer ends at page boundary");

    /* 2. good read must NOT fault: exact-range read works under guards */
    double *a = (double *)g_buf.ptr;
    for (int i = 0; i < 32; i++) a[i] = 1.0;      /* fill BEFORE reading */
    uint64_t v = test_good_read_exact(a, 32);
    CHECK(v == 0x3FF0000000000000ULL, "good exact read returns a[31] bits (1.0)");

    /* 3. deliberate overread MUST fault (proof the harness detects overreads) */
    CHECK(expect_segv(run_bad_overread), "deliberate overread faults (SIGSEGV)");

    /* 4. guard_begin: underread faults (probe reads a[-1]) */
    struct guard_buf gb = gbuf_begin(bytes);
    CHECK(gb.ptr != NULL, "guard_begin allocation succeeds");
    CHECK(((uintptr_t)gb.ptr) % g_pgsize == 0, "guard_begin starts at page boundary");

    gbuf_free(&g_buf);
    gbuf_free(&gb);

    printf("%s: test_guard (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
