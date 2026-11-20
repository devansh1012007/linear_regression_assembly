---
name: asm-verification
description: Test strategy and verified harness code for assembly kernels and numerical results in asm-ols (ABI register probe, guard-page overread tests, shape/alignment matrices, special values, property tests, NIST accuracy tables, fuzzing, mutation testing, CI). Load before writing tests, reviewing a kernel, fixing a bug, or claiming that something is correct.
---

# Verifying assembly and numerics

Assembly fails silently and non-locally: a clobbered callee-saved register crashes the *caller* 10 000 instructions later; a one-element overread works for months until the page layout changes. The test suite is the real product; the asm is what it guards.

## 1. The test pyramid (all required before a kernel is "done")
1. **Reference differential** — every exported function vs a C reference and a Python/mpmath reference over a full shape matrix: `n in {0,1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65,100,127,128,129,1000,1023,1024,1025,random}`, `p in {1..40, 63, 64, 65}`, row stride `ldx in {p, p+1, p+3, padded}`, pointer misalignment `offset in 0..63` bytes (in 8-byte steps and a few odd steps if the contract allows).
2. **ABI conformance** (section 2): callee-saved registers, stack alignment, DF, MXCSR, vzeroupper.
3. **Memory safety** (section 3): no read before/after inputs, no write outside outputs.
4. **Special values**: NaN/Inf/denormal/huge (1e300)/tiny (1e-300)/negative zero. Contract decides the behavior (reject with `OLS_E_NONFINITE` by default); the test pins it.
5. **Properties** (section 4).
6. **Accuracy**: NIST StRD table, kappa sweep (see `numerical-ols-algorithms`).
7. **Cross-ISA**: scalar SSE2 path vs AVX2 vs AVX-512 within the documented error bound (use `OLS_FORCE_ISA`).
8. **Determinism**: reproducible mode is bitwise identical for 1..N threads; otherwise results are compared within bounds only.
9. **Fuzzing** of every parser (binary header, CSV): random/mutated inputs must produce an error code, never a crash or hang; differential fuzz CSV numbers against Python `float()`.
10. **Test the tests** (section 5): mutation testing.

## 2. ABI probe — verified code (SysV)
Calls any function with canaries in rbx, rbp, r12-r15 and reports which changed. Return value in rax and xmm0 is preserved for the caller. Port to Win64 by also canarying rdi/rsi and xmm6-xmm15.
```nasm
; uint64_t abi_probe_call(void *fn, u64 a1, u64 a2, u64 a3, u64 a4, uint32_t *corrupt_mask)
default rel
%macro CHK 3                          ; reg, canary, bit
    mov     rcx, %2
    cmp     %1, rcx
    je      %%ok
    or      r8d, %3
%%ok:
%endmacro
section .text
global abi_probe_call:function
abi_probe_call:
    push    rbp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8                    ; keeps rsp 16-byte aligned at the call; holds the out pointer
    mov     [rsp], r9
    mov     rax, rdi
    mov     rdi, rsi
    mov     rsi, rdx
    mov     rdx, rcx
    mov     rcx, r8
    mov     rbx, 0x1111111111111111
    mov     rbp, 0x2222222222222222
    mov     r12, 0x3333333333333333
    mov     r13, 0x4444444444444444
    mov     r14, 0x5555555555555555
    mov     r15, 0x6666666666666666
    call    rax
    xor     r8d, r8d
    CHK     rbx, 0x1111111111111111, 1
    CHK     rbp, 0x2222222222222222, 2
    CHK     r12, 0x3333333333333333, 4
    CHK     r13, 0x4444444444444444, 8
    CHK     r14, 0x5555555555555555, 16
    CHK     r15, 0x6666666666666666, 32
    mov     r9, [rsp]
    mov     [r9], r8d
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
section .note.GNU-stack noalloc noexec nowrite progbits
```
Self-test it first: a deliberately bad function (`mov r12, 42; ret`) must yield mask 4; a good one mask 0. A probe you have not seen fail is not a probe. The test runner calls **every exported symbol** through it (arguments from a table; symbols with more than 4 args get a small adapter).
Additional ABI checks: (a) after the call, `DF == 0` and `MXCSR` equals its pre-call value; (b) **vzeroupper check**: fill ymm0-15 upper halves with a nonzero pattern before the call, read them back (`vextractf128`) after: all upper halves must be zero for any function that used ymm.

## 3. Guard-page buffers — verified harness (C)
ASan does **not** instrument assembly, so use page protection. Place an input so it ends exactly at a `PROT_NONE` page (overread faults) and also so it begins right after one (underread faults). Outputs get canary bytes around them to detect overwrites.
```c
#define _GNU_SOURCE
#include <sys/mman.h>
#include <stdint.h>
/* [p, p+bytes) ends exactly at a PROT_NONE page: any read/write past the end faults. */
static void *guard_end(size_t bytes) {
    size_t pg = 4096, pages = (bytes + pg - 1) / pg + 1;
    char *base = mmap(0, (pages + 1) * pg, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    mprotect(base + pages * pg, pg, PROT_NONE);
    return base + pages * pg - bytes;
}
/* Mirror image for underread: guard_begin() puts a PROT_NONE page right before the buffer. */
```
Run every kernel at `n = 0..300` (and every shape in the matrix) with all inputs from `guard_end` and `guard_begin`. Verified on the dot kernel (n=1..300). Unaligned-by-contract kernels must also pass with the misaligned `offset` sweep. A kernel that *needs* an overread allowance (e.g. reading up to 31 bytes past the last row) must say so in its contract and the harness allocates exactly that allowance.

## 4. Property tests (hypothesis-style; seeds logged; failures shrunk to minimal shapes)
- Scale: `fit(X, c*y).b == c * fit(X, y).b`; `fit(c*X, y).b == fit(X, y).b / c` (within bounds).
- Shift with intercept: `fit(X + t, y).b == fit(X, y).b` (slopes), intercept shifts by `-t'b`.
- Row permutation invariance within bounds (and bitwise in reproducible mode with fixed blocks).
- Normal-equation orthogonality: `||X' r|| / (||X|| ||r||) <= c * u * kappa` for the returned residual.
- Duplicate-row weighting: duplicating a row equals weight 2.
- Ridge limit: `lambda -> 0` recovers OLS; `lambda -> inf` drives slopes to 0.
- Method agreement: all methods agree within their bounds on well-conditioned data.
- Monotonic error: error vs kappa never *improves* when kappa grows (catches accidental lucky tolerances).

## 5. Test the tests: mutation testing
Script `tools/mutate.py`: for each kernel, apply one mutation at a time to a copy of the `.asm` (off-by-one in a tail bound, `jb` -> `jbe`, swap an operand, delete a `vzeroupper`, change a displacement `+32` -> `+24`, drop an accumulator from the final reduction), rebuild, run the suite. **Every mutant must be killed.** Surviving mutants are test gaps: write the missing test, then re-run. Report the mutation score in the README.

## 6. Tolerance policy
Tolerances come from error-bound theory (see `numerical-ols-algorithms`), carry a recorded constant, and live in one file (`tests/tolerances.py`). No test is ever relaxed, skipped, or special-cased to go green without a written root-cause report in `docs/`. Flaky tests are bugs: seeds are fixed and logged.

## 7. CI (`make verify` locally == CI)
Matrix: {Linux x86-64 native} x {`OLS_FORCE_ISA=sse2, avx2, avx512 if present`} x {debug asm build, release build}. Also under `qemu-x86_64 -cpu Haswell` and `-cpu SandyBridge` (no AVX2: dispatcher must pick the SSE2 path) to test dispatch honestly. Valgrind (memcheck) may be used on the harness for uninitialized reads (supports AVX2, not AVX-512). Artifacts: JUnit-style results, NIST table, kappa-sweep plot, mutation score.

## 8. Review checklist for the independent reviewer agent
Contract header present and true? Every pointer access justified against the stated memory contract? Tails correct for n mod 16, mod 4, mod 1? Callee-saved and Win64 xmm6-15 handled? vzeroupper on every exit path? Accumulators independent (no VEX.128 upper-zero accident)? Overflow checks on size math? Does a test exist that fails if each line of this function is wrong (mutation evidence)? Are numeric claims in docs generated from files in `results/`?
