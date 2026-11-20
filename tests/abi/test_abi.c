/* test_abi.c — ABI probe self-test (asm-verification §2).
 * Proves the probe catches seeded bad functions BEFORE trusting it on the
 * real kernels: a probe you have not seen fail is not a probe.
 * Also checks DF, MXCSR preservation and the vzeroupper discipline.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <xmmintrin.h>   /* _mm_getcsr */
#include "ols.h"

extern uint64_t test_bad_clobber_r12(uint64_t, uint64_t, uint64_t, uint64_t);
extern uint64_t test_bad_clobber_rbx_r14(uint64_t, uint64_t, uint64_t, uint64_t);
extern uint64_t test_good_leaf(uint64_t, uint64_t, uint64_t, uint64_t);
extern uint64_t test_bad_no_vzeroupper(uint64_t, uint64_t, uint64_t, uint64_t);

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else { printf("ok:   %s\n", msg); } } while (0)

/* --- vzeroupper discipline check (asm-verification §2b) -------------------
 * Performed by tests/py/test_pyramid.py via ctypes on the AVX2 kernels:
 * ymm upper halves are filled with a pattern, the kernel is called, and the
 * uppers are read back; must be zero after any ymm-using callee. */

int main(void) {
    uint32_t mask = 0xFFFFFFFFu;

    /* 1. good leaf: mask must be 0 and rax must come through */
    uint64_t r = abi_probe_call((void *)test_good_leaf, 10, 20, 30, 40, &mask);
    CHECK(mask == 0, "good leaf -> no clobber (mask 0)");
    CHECK(r == 100, "good leaf -> return value preserved");

    /* 2. r12 clobber -> bit 4 */
    r = abi_probe_call((void *)test_bad_clobber_r12, 1, 2, 3, 4, &mask);
    CHECK(mask == 4, "bad r12 clobber detected (mask 4)");
    CHECK(r == 42, "bad fn return value still forwarded");

    /* 3. rbx+r14 clobber -> mask 1|16 = 17 */
    r = abi_probe_call((void *)test_bad_clobber_rbx_r14, 1, 2, 3, 4, &mask);
    CHECK(mask == 17, "bad rbx+r14 clobber detected (mask 17)");

    /* 4. MXCSR unchanged across a good call (compare before/after) */
    unsigned mx0 = _mm_getcsr();
    abi_probe_call((void *)test_good_leaf, 0, 0, 0, 0, &mask);
    unsigned mx1 = _mm_getcsr();
    CHECK(mx0 == mx1, "MXCSR preserved by good leaf");

    /* 5. DF still clear (cld) after calls */
    unsigned long df;
    __asm__ ("pushfq\n\tpopq %0\n\tandq $0x400, %0" : "=r" (df));
    CHECK(df == 0, "direction flag clear after calls");

    printf("%s: test_abi (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
