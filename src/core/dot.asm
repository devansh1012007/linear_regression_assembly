; ------------------------------------------------------------------------------
; dot.asm — dot products.
;   ols_dot_f64_avx2 : AVX2+FMA3. Structure from skills/asm-x86-64-engineering
;                      §4 (verified template: 16-wide main + 4-wide cleanup +
;                      scalar tail, no overreads). Copied as the Phase-0
;                      starting point per that skill's "re-run in CI" rule.
;   ols_dot_f64_sse2 : scalar SSE2 reference with Neumaier-compensated sum.
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"
%include "src/macros.inc"

section .text

; ==============================================================================
; double ols_dot_f64_avx2(const double *a, const double *b, size_t n)
; Args    : rdi = a, rsi = b, rdx = n
; Returns : xmm0 = sum a[i]*b[i]
; Requires: AVX2 + FMA3 (dispatcher guarantees; never checked here)
; Memory  : reads a[0..n), b[0..n) only. Any 8-byte alignment. Never reads
;           past the end (guard-page tested for n=0..300).
; Clobbers: rax, rcx, ymm0-ymm5. Preserves callee-saved GPRs, rsp, DF, MXCSR.
;           Ends with vzeroupper.
; Numerics: 4 vector accumulators x 4 lanes, fixed reduction order; error
;           <= gamma_{2n} * sum|a_i b_i| (uncompensated; the compensated
;           variant is the sse2 path and the Phase-6 Dot2 COULD item).
; Cost    : PREDICTED (perf-log #1, written before benchmark): 2 loads + 4
;           FMAs per 16 doubles; load-port bound at 2x32B/cy -> ~0.125 cy/el
;           in L1 for FMA throughput 2/cy; realistic ~0.25-0.5 cy/el. Measured
;           value goes to docs/perf-log.md, never re-typed by hand.
; ==============================================================================
global ols_dot_f64_avx2:function
align 32
ols_dot_f64_avx2:
    vxorpd  ymm0, ymm0, ymm0
    vxorpd  ymm1, ymm1, ymm1
    vxorpd  ymm2, ymm2, ymm2
    vxorpd  ymm3, ymm3, ymm3
    vxorpd  xmm5, xmm5, xmm5          ; scalar-tail accumulator (separate reg:
                                       ; VEX.128 writes zero the upper half)
    xor     eax, eax
    mov     rcx, rdx
    and     rcx, -16                  ; elements handled by the 16-wide loop
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

; ==============================================================================
; double ols_dot_f64_sse2(const double *a, const double *b, size_t n)
; Args    : rdi = a, rsi = b, rdx = n
; Returns : xmm0 = Neumaier-compensated sum of fl(a[i]*b[i])
; Requires: SSE2 only (x86-64 baseline)
; Memory  : reads a[0..n), b[0..n) only (scalar ops read exactly 8 bytes).
;           Any 8-byte alignment.
; Clobbers: rax, rdx, xmm0-xmm7. Preserves callee-saved GPRs, rsp, DF, MXCSR.
; Numerics: products fl(a*b) round once; the SUM is compensated (Neumaier
;           1974). Error ~ u*sum|a_i b_i| + gamma_1*sum|a_i b_i|; reference
;           path for cross-checking the AVX2 kernel.
; Cost    : PREDICTED (perf-log #2, before benchmark): addsd latency chain
;           (t = s + x) ~4 cy -> ~4-6 cy/el. Accuracy reference, not speed.
; Register map: xmm0=s, xmm1=c, xmm2=|.|mask, xmm3=x, xmm4=t, xmm5/6=|s|/|x|,
;               xmm7=selected err. Branch-free select via andpd/andnpd/orpd
;               (blendvpd is SSE4.1; we are SSE2-baseline by contract).
; ==============================================================================
global ols_dot_f64_sse2:function
ols_dot_f64_sse2:
    xor     eax, eax                ; i = 0
    pxor    xmm0, xmm0              ; s = 0
    pxor    xmm1, xmm1              ; c = 0
    test    rdx, rdx
    jz      .done
    mov     rcx, 0x7FFFFFFFFFFFFFFF
    movq    xmm2, rcx               ; lane0 = abs mask
    punpcklqdq xmm2, xmm2           ; duplicate to lane1 (cmppd is lane-wise)
.loop:
    movsd   xmm3, [rdi + rax*8]     ; a[i]
    mulsd   xmm3, [rsi + rax*8]     ; x = fl(a[i]*b[i])
    movapd  xmm4, xmm0
    addsd   xmm4, xmm3              ; t = s + x
    movapd  xmm5, xmm0
    andpd   xmm5, xmm2              ; |s|
    movapd  xmm6, xmm3
    andpd   xmm6, xmm2              ; |x|
    cmpnltpd xmm5, xmm6             ; mask(lane-wise) = |s| >= |x| ? ~0 : 0
    movapd  xmm7, xmm0
    subsd   xmm7, xmm4              ; s - t
    addsd   xmm7, xmm3              ; e1 = (s - t) + x
    subsd   xmm3, xmm4              ; x - t
    addsd   xmm3, xmm0              ; e2 = (x - t) + s
    andpd   xmm7, xmm5              ; e1 & mask
    andnpd  xmm5, xmm3              ; ~mask & e2
    orpd    xmm7, xmm5              ; err = |s|>=|x| ? e1 : e2
    addsd   xmm1, xmm7              ; c += err
    movapd  xmm0, xmm4              ; s = t
    inc     rax
    cmp     rax, rdx
    jb      .loop
.done:
    addsd   xmm0, xmm1              ; return s + c
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
