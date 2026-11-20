---
name: asm-x86-64-engineering
description: Rules, ABI tables, pitfalls and verified templates for writing x86-64 NASM code (SysV + Win64) in the asm-ols project. Load before writing or reviewing ANY .asm/.inc file, including function prologues, macros, CPUID dispatch, tail handling and debugging.
---

# x86-64 assembly engineering (NASM)

Everything under `src/` follows this skill. The snippets marked **[verified]** were assembled with NASM 2.16 and executed against a C harness (dot kernel for n=0..300 with guard pages, CPUID check, clobber probe). Re-run them in CI; do not trust them on faith after you edit them.

## 1. Toolchain and flags
- Assemble: `nasm -f elf64 -g -F dwarf -Wall -Werror file.asm` (Win64 later: `-f win64 -F cv8`).
- Link static, no libc, for the CLI: `ld -static -z noexecstack -z relro -z now -o asm-ols ...`. Harness builds link the core into a `.so`/`.a` and use gcc.
- Every `.asm` file ends with `section .note.GNU-stack noalloc noexec nowrite progbits` (otherwise the linker warns and may mark the stack executable). **[verified]**
- Inspect: `objdump -d -M intel`, `readelf -lW` (check GNU_STACK is RW, not RWE), `nm`, `nasm -l file.lst`.
- NASM emits no `.cfi_*` unwind info. Leaf kernels don't need it; if call-graph profiling or gdb backtraces through asm frames matter, record that trade-off in an ADR (GAS with `.intel_syntax noprefix` is the alternative).

## 2. ABI cheat-sheet
| | SysV AMD64 (Linux/macOS) | Microsoft x64 (Windows) |
|---|---|---|
| int args | rdi, rsi, rdx, rcx, r8, r9 | rcx, rdx, r8, r9 (5th+ at [rsp+40] on entry) |
| float args | xmm0-7 | xmm0-3 (shared position with int args) |
| return | rax / xmm0 | rax / xmm0 |
| callee-saved | rbx, rbp, r12-r15 | rbx, rbp, rdi, rsi, r12-r15, **xmm6-xmm15 (low 128 bits)** |
| shadow space | none | 32 bytes caller-allocated |
| red zone | 128 B below rsp (leaf only) | none |
| stack at call | rsp 16-byte aligned before `call` (so rsp = 8 mod 16 at entry) | same |
| flags | DF must be clear at call/ret | same |

Rules: kernels reference arguments only through `ARG1..ARG6` from `abi.inc` (below). A kernel that touches xmm6-xmm15/ymm6-ymm15 must save/restore xmm6-15 under `ABI_WIN64`. Prefer kernels that fit in `ymm0-ymm5` + caller-saved GPRs when practical; register-hungry microkernels (12+ accumulators) get an explicit Win64 save/restore macro.

```nasm
; abi.inc   [verified: assembles for elf64 and win64]
%ifidn __OUTPUT_FORMAT__, win64
  %define ARG1 rcx
  %define ARG2 rdx
  %define ARG3 r8
  %define ARG4 r9
  %define ABI_WIN64 1
%else
  %define ARG1 rdi
  %define ARG2 rsi
  %define ARG3 rdx
  %define ARG4 rcx
  %define ARG5 r8
  %define ARG6 r9
%endif
```

## 3. Function contract header (mandatory on every global symbol)
```nasm
; double ols_dot_f64_avx2(const double *a, const double *b, size_t n)
; Args    : SysV rdi, rsi, rdx | Win64 rcx, rdx, r8 (use ARGn)
; Returns : xmm0
; Requires: AVX2 + FMA3 (checked by dispatcher, never by this function)
; Memory  : reads a[0..n), b[0..n) only. Any alignment. Never reads past the end (guard-page tested).
; Clobbers: rax, rcx, ymm0-ymm5. Preserves callee-saved GPRs, rsp, DF, MXCSR. Ends with vzeroupper.
; Numerics: 4x4 lane partial sums -> reduction order differs from sequential; error <= gamma_n * sum|a_i b_i|
; Cost    : predicted cycles/element from the machine model (load-port bound: 2 loads per FMA) goes here BEFORE benchmarking; measured value goes in docs/perf-log.md
```
A CI script fails the build if a `global` symbol lacks a header with Args/Returns/Requires/Memory/Clobbers.

## 4. Verified kernel template (dot product, AVX2+FMA3, SysV)
```nasm
default rel
section .text
global ols_dot_f64_avx2:function
align 32
ols_dot_f64_avx2:
    vxorpd  ymm0, ymm0, ymm0
    vxorpd  ymm1, ymm1, ymm1
    vxorpd  ymm2, ymm2, ymm2
    vxorpd  ymm3, ymm3, ymm3
    vxorpd  xmm5, xmm5, xmm5          ; scalar-tail accumulator (see pitfall 1)
    xor     eax, eax
    mov     rcx, rdx
    and     rcx, -16                  ; elements handled by the 16-wide main loop
    jz      .tail4
.main:
    vmovupd ymm4, [rdi + rax*8]
    vfmadd231pd ymm0, ymm4, [rsi + rax*8]
    vmovupd ymm4, [rdi + rax*8 + 32]
    vfmadd231pd ymm1, ymm4, [rsi + rax*8 + 32]
    vmovupd ymm4, [rdi + rax*8 + 64]
    vfmadd231pd ymm2, ymm4, [rsi + rax*8 + 64]
    vmovupd ymm4, [rdi + rax*8 + 96]
    vfmadd231pd ymm3, ymm4, [rsi + rax*8 + 96]
    add     rax, 16
    cmp     rax, rcx
    jb      .main
.tail4:
    lea     rcx, [rax + 4]
    cmp     rcx, rdx
    ja      .tail1
    vmovupd ymm4, [rdi + rax*8]
    vfmadd231pd ymm0, ymm4, [rsi + rax*8]
    mov     rax, rcx
    jmp     .tail4
.tail1:
    cmp     rax, rdx
    jae     .reduce
    vmovsd  xmm4, [rdi + rax*8]
    vfmadd231sd xmm5, xmm4, [rsi + rax*8]
    inc     rax
    jmp     .tail1
.reduce:
    vaddpd  ymm0, ymm0, ymm1
    vaddpd  ymm2, ymm2, ymm3
    vaddpd  ymm0, ymm0, ymm2
    vextractf128 xmm1, ymm0, 1
    vaddpd  xmm0, xmm0, xmm1
    vhaddpd xmm0, xmm0, xmm0
    vaddsd  xmm0, xmm0, xmm5
    vzeroupper
    ret
section .note.GNU-stack noalloc noexec nowrite progbits
```
This is a *structure* template, not a fast kernel: 4 accumulators cannot saturate the FMA pipes (see `simd-kernel-optimization`). Its job is to show tail handling without overreads and the reduction pattern. Copy-paste of whole kernels is banned; generate unrolled variants with `%macro`/`%rep`/`%assign`.

## 5. Pitfalls (each one has bitten real code)
1. **VEX.128/scalar writes zero the upper half of the destination ymm.** `vfmadd231sd xmm0, ...` destroys `ymm0[255:128]`. Keep scalar/tail accumulators in a *different* register from vector accumulators.
2. **SSE/AVX transition penalties.** Always `vzeroupper` before `ret` from any function that touched ymm, and before calling legacy-SSE code. Never mix legacy-SSE (non-VEX) and VEX instructions in one function.
3. **Legacy SSE memory operands must be 16-byte aligned; VEX ones need not.** Use `vmovupd`/VEX forms; reserve `vmovapd` for data you *proved* aligned (and assert it in the contract).
4. **`cpuid` clobbers rbx** (callee-saved). Push/pop it. **[verified in section 6]**
5. **Feature detection must check the OS**, not only CPUID bits: `OSXSAVE` and `XGETBV` (XCR0 bits 1,2 for YMM; bits 5,6,7 additionally for AVX-512). A CPU can advertise AVX2 while the OS/hypervisor has not enabled the state.
6. **32-bit register writes zero-extend to 64 bits; 8/16-bit writes don't.** Use `movzx`; never rely on stale upper bits of an argument (callers may leave garbage in the upper half of 32-bit args).
7. **Stack alignment.** `rsp` must be 16-byte aligned *at the `call`*. Count pushes. If you spill ymm with `vmovaps`, align to 32 explicitly (`and rsp, -32` with a saved frame pointer).
8. **`div`/`idiv` in loops, `loop`, `rep movs` for short copies, `vhaddpd` in hot loops** are all slower than their alternatives. Reduce horizontally once, outside the loop.
9. **False dependencies:** zero the destination (`vxorpd`) before `vcvtsi2sd`/`vsqrtsd`-style partial writes if its previous value is not otherwise dead.
10. **Signed/unsigned loop bounds:** use unsigned compares (`jb/ja/jae`) for sizes; a size_t `n` with the high bit set must fail validation in the API layer, not wrap in a loop.
11. **Overflow in size math:** `n*p*8` must be computed with `mul` and checked via carry/overflow before any allocation or indexing.
12. **MXCSR:** kernels never change rounding mode or FTZ/DAZ. If the CLI sets them it does so explicitly and documents it.

## 6. Verified CPU-feature check (AVX2 + FMA3 + OS support)
```nasm
default rel
section .text
global ols_cpu_has_avx2_fma:function
ols_cpu_has_avx2_fma:                 ; returns eax = 1/0
    push    rbx
    mov     eax, 1
    xor     ecx, ecx
    cpuid
    and     ecx, (1<<12)|(1<<27)|(1<<28)       ; FMA, OSXSAVE, AVX
    cmp     ecx, (1<<12)|(1<<27)|(1<<28)
    jne     .no
    xor     ecx, ecx
    xgetbv                                     ; edx:eax = XCR0
    and     eax, 6                             ; SSE + YMM state enabled by OS
    cmp     eax, 6
    jne     .no
    mov     eax, 7
    xor     ecx, ecx
    cpuid
    bt      ebx, 5                             ; AVX2
    jnc     .no
    mov     eax, 1
    pop     rbx
    ret
.no:
    xor     eax, eax
    pop     rbx
    ret
section .note.GNU-stack noalloc noexec nowrite progbits
```
AVX-512 (optional path): CPUID.7:EBX bits 16 (F), 17 (DQ), 30 (BW), 31 (VL) **and** XCR0 bits 5,6,7 set. A dispatcher selects a variant at init into a table of function pointers (`ols_gram_f64` etc.); an env var `OLS_FORCE_ISA=sse2|avx2|avx512` overrides it for testing. Never prefer AVX-512 without a measurement on the machine in question.

## 7. Macros and generated code
- Tile sizes are `%assign`ed constants; the microkernel is a macro expanded with `%rep` over rows/cols so that changing the tile is a one-line edit and cannot desynchronize 12 hand-copied FMAs.
- Local labels: `.name` under a global label, `%%name` inside macros.
- Name symbols `ols_<op>_<type>_<isa>`; internal helpers `ols__...` with `:hidden`. Comments explain *why* (cycle accounting, hazards), not *what*.

## 8. Debug workflow
- gdb: `layout asm`, `tui reg vector`, `p $ymm0.v4_double`, `x/8fg $rsi`, `watch *(double*)$rsp`; build with `-g -F dwarf` so `list` shows asm lines.
- Wrong numbers: first shrink to the smallest failing (n, p, offset), then compare with the scalar asm path, then with the C reference; the first diverging accumulator tells you which tile/lane is wrong.
- Crashes in tails: re-run under the guard-page allocator (see `asm-verification`); ASan does **not** instrument assembly.
- Clobber bugs: run the ABI probe (see `asm-verification`) on every exported symbol.
