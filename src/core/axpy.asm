; ------------------------------------------------------------------------------
; axpy.asm — y += alpha * x, elementwise, both ISAs.
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"
%include "src/macros.inc"

section .text

; ==============================================================================
; void ols_axpy_f64_sse2(double alpha, const double *x, double *y, size_t n)
; Args    : xmm0 = alpha, rdi = x, rsi = y, rdx = n
; Returns : nothing
; Requires: SSE2 (baseline)
; Memory  : reads x[0..n); reads then writes y[0..n) ONLY (8-byte granularity).
;           Any 8-byte alignment.
; Clobbers: rax, rdx, xmm0-xmm4. Preserves callee-saved GPRs, rsp, DF, MXCSR.
; Numerics: y[i] = fl(y[i] + fl(alpha*x[i])); no compensation (axpy is not a
;           reduction); documented per contract.
; Cost    : PREDICTED (perf-log #5): ~1.5-2 cy/el scalar (mulsd+addsd chain is
;           per-element independent -> pipelined; load/store bound 2/cy).
; ==============================================================================
global ols_axpy_f64_sse2:function
ols_axpy_f64_sse2:
    xor     eax, eax
    test    rdx, rdx
    jz      .done
.loop:
    movsd   xmm1, [rdi + rax*8]     ; x[i]
    mulsd   xmm1, xmm0              ; alpha * x[i]
    addsd   xmm1, [rsi + rax*8]     ; + y[i]
    movsd   [rsi + rax*8], xmm1     ; store
    inc     rax
    cmp     rax, rdx
    jb      .loop
.done:
    ret

; ==============================================================================
; void ols_axpy_f64_avx2(double alpha, const double *x, double *y, size_t n)
; Args    : xmm0 = alpha, rdi = x, rsi = y, rdx = n
; Returns : nothing
; Requires: AVX2 + FMA3
; Memory  : reads x[0..n); reads/writes y[0..n) only. Any 8-byte alignment.
;           Guard-page tested (write side via canary pages).
; Clobbers: rax, rcx, ymm0-ymm4. vzeroupper on exit.
; Numerics: y[i] = fma(alpha, x[i], y[i]) — ONE rounding, unlike the sse2
;           path's two. Documented difference; differential tests compare
;           against np-y + alpha*np-x with per-element bound 1 ulp.
; Cost    : PREDICTED (perf-log #6): 1 load + 1 FMA + 1 store per 4 el;
;           store-port bound ~1 cy/4el = 0.25 cy/el in L1.
; ==============================================================================
global ols_axpy_f64_avx2:function
align 32
ols_axpy_f64_avx2:
    vbroadcastsd ymm0, xmm0         ; alpha to all lanes
    xor     eax, eax
    mov     rcx, rdx
    and     rcx, -4                 ; main loop handles 4 at a time (1x ymm)
    jz      .tail4
.main:
    vmovupd ymm1, [rdi + rax*8]     ; x
    vmovupd ymm2, [rsi + rax*8]     ; y
    vfmadd231pd ymm2, ymm1, ymm0    ; y += alpha * x
    vmovupd [rsi + rax*8], ymm2
    add     rax, 4
    cmp     rax, rcx
    jb      .main
    ; fall through to .tail4 (n mod 4 leftovers)
.tail4:
    lea     rcx, [rax + 4]
    cmp     rcx, rdx
    ja      .tail1
    vmovupd ymm1, [rdi + rax*8]
    vmovupd ymm2, [rsi + rax*8]
    vfmadd231pd ymm2, ymm1, ymm0
    vmovupd [rsi + rax*8], ymm2
    mov     rax, rcx
    jmp     .tail4
.tail1:
    cmp     rax, rdx
    jae     .done
    vmovsd  xmm1, [rdi + rax*8]
    vmovsd  xmm2, [rsi + rax*8]
    vfmadd231sd xmm2, xmm1, xmm0
    vmovsd  [rsi + rax*8], xmm2
    inc     rax
    jmp     .tail1
.done:
    vzeroupper
    ret
section .note.GNU-stack noalloc noexec nowrite progbits
