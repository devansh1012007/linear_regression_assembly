; ------------------------------------------------------------------------------
; sums.asm — compensated sums and centered statistics.
;   ols_sum_f64_sse2          : scalar Neumaier sum (accuracy reference)
;   ols_sum_f64_avx2          : 4-lane compensated sum, scalar tail
;   ols_centered_sums_f64_sse2: one-pass (n, sx, sy, sxx, sxy, syy), all
;                               Neumaier-compensated; simple-regression engine
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"
%include "src/macros.inc"

section .text

; ==============================================================================
; double ols_sum_f64_sse2(const double *x, size_t n)
; Args    : rdi = x, rsi = n
; Returns : xmm0 = Neumaier-compensated sum
; Requires: SSE2 (baseline)
; Memory  : reads x[0..n) only (scalar 8-byte reads). Any 8-byte alignment.
; Clobbers: rax, rdx, xmm0-xmm7. Preserves callee-saved GPRs, rsp, DF, MXCSR.
; Numerics: Neumaier 1974; |err| <= 2u*sum|x_i| + O(u^2).
; Cost    : PREDICTED (perf-log #3, before benchmark): dependent addsd chain
;           ~4 cy/el (addsd latency 4 on most cores); reference path.
; Register map: xmm0=s xmm1=c xmm2=abs-mask xmm3=x xmm4=t xmm5=|s| xmm6=|x|
;               xmm7=selected err
; ==============================================================================
global ols_sum_f64_sse2:function
ols_sum_f64_sse2:
    xor     eax, eax
    pxor    xmm0, xmm0              ; s
    pxor    xmm1, xmm1              ; c
    test    rsi, rsi
    jz      .done
    mov     rdx, 0x7FFFFFFFFFFFFFFF
    movq    xmm2, rdx
    punpcklqdq xmm2, xmm2           ; abs mask in both lanes
.loop:
    movsd   xmm3, [rdi + rax*8]     ; x
    movapd  xmm4, xmm0
    addsd   xmm4, xmm3              ; t = s + x
    movapd  xmm5, xmm0
    andpd   xmm5, xmm2              ; |s|
    movapd  xmm6, xmm3
    andpd   xmm6, xmm2              ; |x|
    cmpnltpd xmm5, xmm6             ; lane-mask = |s| >= |x| ? ~0 : 0
    movapd  xmm7, xmm0
    subsd   xmm7, xmm4              ; s - t
    addsd   xmm7, xmm3              ; e1 = (s-t)+x
    subsd   xmm3, xmm4              ; x - t
    addsd   xmm3, xmm0              ; e2 = (x-t)+s
    andpd   xmm7, xmm5
    andnpd  xmm5, xmm3
    orpd    xmm7, xmm5              ; err = mask ? e1 : e2
    addsd   xmm1, xmm7              ; c += err
    movapd  xmm0, xmm4              ; s = t
    inc     rax
    cmp     rax, rsi
    jb      .loop
.done:
    addsd   xmm0, xmm1
    ret

; ==============================================================================
; double ols_sum_f64_avx2(const double *x, size_t n)
; Args    : rdi = x, rsi = n
; Returns : xmm0 = compensated sum (4 lanes Neumaier + scalar tail)
; Requires: AVX2 + FMA3
; Memory  : reads x[0..n) only. Any 8-byte alignment. Guard-page tested.
; Clobbers: rax, rcx, rdx, xmm0-xmm7, ymm0-ymm7. vzeroupper on exit.
; Numerics: per-lane Neumaier; the 4 lane channels + scalar tail channel are
;           merged by sequential Neumaier steps into one (s,c) pair. Error
;           class u*sum|x| (like scalar), but NOT bitwise equal to scalar
;           (different association order) — documented, tested within bounds.
; Cost    : PREDICTED (perf-log #4, before benchmark): ~1.5-2 cy per 4 el
;           (~0.4-0.5 cy/el); 1 load/iter not binding at 2 loads/cy.
; Register map (main loop): ymm0=s ymm1=c ymm2=mask ymm3=x ymm4=t ymm5=e1
;               ymm6=e2 ymm7=abs-mask
; ==============================================================================
global ols_sum_f64_avx2:function
align 32
ols_sum_f64_avx2:
    vxorpd  ymm0, ymm0, ymm0        ; s (4 lanes)
    vxorpd  ymm1, ymm1, ymm1        ; c (4 lanes)
    vxorpd  xmm8, xmm8, xmm8        ; s_tail (xmm8/9: safe, no ymm aliasing)
    vxorpd  xmm9, xmm9, xmm9        ; c_tail
    xor     eax, eax
    mov     rcx, rsi
    and     rcx, -4                 ; main loop handles 4 at a time (1x ymm)
    jz      .tail
    ; ymm7 = abs mask per qword (all ones >> 1)
    vpcmpeqd ymm7, ymm7, ymm7
    vpsrlq  ymm7, ymm7, 1
.main:
    vmovupd ymm3, [rdi + rax*8]
    vaddpd  ymm4, ymm0, ymm3        ; t = s + x
    vandpd  ymm2, ymm0, ymm7        ; |s|
    vandpd  ymm5, ymm3, ymm7        ; |x|
    vcmpnltpd ymm2, ymm2, ymm5      ; mask = |s| >= |x|
    vsubpd  ymm5, ymm0, ymm4
    vaddpd  ymm5, ymm5, ymm3        ; e1 = (s-t)+x
    vsubpd  ymm6, ymm3, ymm4
    vaddpd  ymm6, ymm6, ymm0        ; e2 = (x-t)+s
    vblendvpd ymm5, ymm6, ymm5, ymm2    ; err = mask ? e1 : e2
    vaddpd  ymm1, ymm1, ymm5        ; c += err
    vmovapd ymm0, ymm4              ; s = t
    add     rax, 4
    cmp     rax, rcx
    jb      .main
    ; fall through to .tail (handles n mod 4 leftover elements)
.tail:
    cmp     rax, rsi
    jae     .reduce
    movsd   xmm3, [rdi + rax*8]     ; x
    movapd  xmm4, xmm8
    addsd   xmm4, xmm3              ; t = s_tail + x
    mov     rdx, 0x7FFFFFFFFFFFFFFF
    movq    xmm2, rdx
    punpcklqdq xmm2, xmm2           ; abs mask (scalar lanes)
    movapd  xmm5, xmm8
    andpd   xmm5, xmm2              ; |s|
    movapd  xmm6, xmm3
    andpd   xmm6, xmm2              ; |x|
    cmpnltpd xmm5, xmm6             ; mask
    movapd  xmm7, xmm8
    subsd   xmm7, xmm4
    addsd   xmm7, xmm3              ; e1
    subsd   xmm3, xmm4
    addsd   xmm3, xmm8              ; e2
    andpd   xmm7, xmm5
    andnpd  xmm5, xmm3
    orpd    xmm7, xmm5              ; err
    addsd   xmm9, xmm7              ; c_tail += err
    movapd  xmm8, xmm4              ; s_tail = t
    inc     rax
    cmp     rax, rsi
    jb      .tail
.reduce:
    ; --- merge 4 lane channels (s_k, c_k) pairwise, then with scalar tail ---
    ; spill lanes to stack slots, then run 4 scalar Neumaier merge steps into
    ; a single accumulator. Fixed order: lane0 <- lane1 <- lane2 <- lane3 <- tail.
    sub     rsp, 8 * 8              ; 8 scratch qwords, keeps 16-alignment
    ; dump lane s and c
    vmovups [rsp + 0],  ymm0        ; s0..s3
    vmovups [rsp + 32], ymm1        ; c0..c3
    ; master (s, c) starts at lane 0
    movsd   xmm0, [rsp + 0]         ; s
    movsd   xmm1, [rsp + 32]        ; c
    ; abs mask once
    mov     rdx, 0x7FFFFFFFFFFFFFFF
    movq    xmm2, rdx
    punpcklqdq xmm2, xmm2
    ; merge k = 1,2,3: (s,c) += (s_k, c_k)  [Neumaier step, c-k folded in x]
    %assign MERGE_K 1
    %rep 3
        movsd   xmm3, [rsp + 8*MERGE_K]          ; x = s_k
        addsd   xmm3, [rsp + 32 + 8*MERGE_K]     ; x = s_k + c_k (compensated k)
        movapd  xmm4, xmm0
        addsd   xmm4, xmm3                       ; t = s + x
        movapd  xmm5, xmm0
        andpd   xmm5, xmm2                       ; |s|
        movapd  xmm6, xmm3
        andpd   xmm6, xmm2                       ; |x|
        cmpnltpd xmm5, xmm6
        movapd  xmm7, xmm0
        subsd   xmm7, xmm4
        addsd   xmm7, xmm3                       ; e1
        subsd   xmm3, xmm4
        addsd   xmm3, xmm0                       ; e2
        andpd   xmm7, xmm5
        andnpd  xmm5, xmm3
        orpd    xmm7, xmm5
        addsd   xmm1, xmm7                       ; c += err
        movapd  xmm0, xmm4                       ; s = t
    %assign MERGE_K MERGE_K + 1
    %endrep
    ; merge scalar tail (xmm8=s_tail, xmm9=c_tail)
    movapd  xmm3, xmm8
    addsd   xmm3, xmm9                            ; x = s_tail + c_tail
    movapd  xmm4, xmm0
    addsd   xmm4, xmm3                            ; t
    movapd  xmm5, xmm0
    andpd   xmm5, xmm2
    movapd  xmm6, xmm3
    andpd   xmm6, xmm2
    cmpnltpd xmm5, xmm6
    movapd  xmm7, xmm0
    subsd   xmm7, xmm4
    addsd   xmm7, xmm3                            ; e1
    subsd   xmm3, xmm4
    addsd   xmm3, xmm0                            ; e2
    andpd   xmm7, xmm5
    andnpd  xmm5, xmm3
    orpd    xmm7, xmm5
    addsd   xmm1, xmm7
    movapd  xmm0, xmm4
    addsd   xmm0, xmm1                            ; result = s + c
    add     rsp, 8 * 8
    vzeroupper
    ret
section .note.GNU-stack noalloc noexec nowrite progbits
