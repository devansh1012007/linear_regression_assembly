; ------------------------------------------------------------------------------
; dispatch.asm — ISA selection (the one canonical ols_init/ols_isa).
; Kernel selection itself lives at the fit call site (fit.asm branches on
; ols_isa() to pick the AVX2 or SSE2 Gram kernel); this module owns the
; policy. AUTO caps at AVX2 and explicit AVX-512 requests are clamped to
; AVX2: no AVX-512 kernel exists yet, and ols_isa() must never advertise
; capability the library does not implement (review finding §5).
; ols_cpu_has_avx512 stays a truthful CPU feature report for the harness.
; OLS_FORCE_ISA=sse2|avx2|avx512 overrides detection (testing only) — env
; reading needs libc, so the override is a variable the HARNESS sets before
; calling ols_init; the core itself stays syscall-free.
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"

extern ols_cpu_has_avx2_fma

section .data align=8
ols__isa_selected:  dd -1                   ; -1 = uninitialized
global ols__isa_selected:data 4

section .text

; int32_t ols_init(int32_t force_isa)
; Args    : edi = -1 (auto) | OLS_ISA_SSE2(0) | OLS_ISA_AVX2(1) | OLS_ISA_AVX512(2)
; Returns : eax = isa selected (0 or 1 — AVX-512 requests clamp to 1 until an
;           AVX-512 kernel exists)
; Requires: force_isa values other than -1..2 are treated as -1 (auto)
; Memory  : writes the 4-byte global ols__isa_selected (idempotent)
; Clobbers: rax, rcx, rbx(saved via callee), rdx
; Notes   : idempotent and thread-safe by construction (plain aligned store of
;           the same value every time).
global ols_init:function
ols_init:
    cmp     edi, -1
    je      .auto
    cmp     edi, OLS_ISA_AVX512
    ja      .auto                        ; out of range -> auto
    cmp     edi, OLS_ISA_AVX2
    jbe     .forced
    mov     edi, OLS_ISA_AVX2            ; clamp: no AVX-512 kernel yet
.forced:
    mov     [ols__isa_selected], edi
    mov     eax, edi
    ret
.auto:
    call    ols_cpu_has_avx2_fma wrt ..plt
    test    eax, eax
    jz      .sse2
    mov     dword [ols__isa_selected], OLS_ISA_AVX2
    mov     eax, OLS_ISA_AVX2
    ret
.sse2:
    mov     dword [ols__isa_selected], OLS_ISA_SSE2
    xor     eax, eax
    ret

; int32_t ols_isa(void)
; Args    : none
; Returns : eax = last selected ISA (calls ols_init(-1) first if uninitialized)
; Requires: n/a
; Memory  : reads/writes ols__isa_selected
; Clobbers: rax, rcx, rdx, rbx(saved via callee)
global ols_isa:function
ols_isa:
    mov     eax, [ols__isa_selected]
    cmp     eax, -1
    jne     .done
    mov     edi, -1
    call    ols_init
.done:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
