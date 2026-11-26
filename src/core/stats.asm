; ------------------------------------------------------------------------------
; stats.asm — fused residual pass (Phase 2): SSE + X'r in one sweep.
;   ols_residual_f64_sse2(A) / ols_residual_f64_avx2(A)
; A = ols_resid_args_t (include/ols.h; RA_* offsets in abi.inc).
; Per row: r = w*(y - b0 - x'beta); SSE accumulates r^2 (Neumaier);
; X'r accumulates x_j*r per column (plain; CSNE's correction is small —
; numerical-ols §4).
;
; Register plan (verified line-by-line against the sequence below):
;   live across rows: X=rbx ldx8=r12 y=r13 n=r14 p=r15, mask=xmm7
;   per row: w=xmm6 rhat/r=xmm0 d=xmm1 x2=xmm6(after r saved to xmm4)
;            Neumaier temps xmm2,xmm3,xmm5; k loop k=rcx Xrow=r9
;   frame: [0]sse_s [8]sse_c [16]b0 [24]beta [32]w [40]sseOut [48..)xtr
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"
%include "src/macros.inc"

section .text

; ==============================================================================
; int32_t ols_residual_f64_sse2(const ols_resid_args_t *A)
; Args    : rdi = A
; Returns : eax = OLS_OK | OLS_E_NONFINITE
; Requires: SSE2. *sse_out OVERWRITTEN; xtr_out[0..p) and *rsum_out
;           OVERWRITTEN if non-NULL.
; Memory  : reads A once; X rows [0..n) cols [0..p) at stride A->ldx8,
;           y[0..n), w[0..n) if non-NULL, beta[0..p); writes *sse_out,
;           xtr_out[0..p), *rsum_out (= sum w_i r_i, CSNE input).
; Clobbers: rax,rcx,rdx,rsi,rdi,r8-r11 + rbx,rbp,r12-r15 (pushed/popped).
; Numerics: SSE Neumaier; X'r plain adds.
; Cost    : PREDICTED (perf-log #16): ~2p+14 ops/row -> q=8 ~20 cy/row.
; ==============================================================================
global ols_residual_f64_sse2:function
ols_residual_f64_sse2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8
    mov     rbx, [rdi+RA_X]
    mov     r12, [rdi+RA_LDX8]
    mov     r13, [rdi+RA_Y]
    mov     r14, [rdi+RA_N]
    mov     r15, [rdi+RA_P]
    lea     rax, [r15*8]
    lea     rdx, [rax + 96]         ; xtr at [rsp+56..) + 56 ctrl; tail holds
    add     rdx, 15                 ; rsum_s/c and the rsum_out ptr
    and     rdx, -16
    sub     rsp, rdx
    movsd   xmm0, [rdi+RA_B0]
    movsd   [rsp+16], xmm0
    mov     rax, [rdi+RA_BETA]
    mov     [rsp+24], rax
    mov     rax, [rdi+RA_W]
    mov     [rsp+32], rax
    mov     rax, [rdi+RA_SSE]
    mov     [rsp+40], rax
    mov     rax, [rdi+RA_XTR]
    mov     [rsp+48], rax           ; xtr_out
    mov     rax, [rdi+RA_RSUM]
    mov     [rsp+80+r15*8], rax     ; rsum_out ptr (tail slot)
    pxor    xmm0, xmm0
    movsd   [rsp+0], xmm0
    movsd   [rsp+8], xmm0
    xor     rcx, rcx
.zx:
    cmp     rcx, r15
    jae     .zx_done
    movsd   [rsp + 56 + rcx*8], xmm0
    inc     rcx
    jmp     .zx
.zx_done:
    ; rsum accumulators (sum w*r — the CSNE correction input) at the tail
    lea     rcx, [rsp + 56]
    lea     rcx, [rcx + r15*8]      ; rsum_s
    pxor    xmm0, xmm0
    movsd   [rcx], xmm0
    movsd   [rcx+8], xmm0
    mov     rax, 0x7FFFFFFFFFFFFFFF
    movq    xmm7, rax
    punpcklqdq xmm7, xmm7
    xor     r8, r8                  ; i
.row:
    cmp     r8, r14
    jae     .rows_done
    ; 1. w
    mov     rax, [rsp+32]
    test    rax, rax
    jz      .w1
    movsd   xmm6, [rax + r8*8]
    jmp     .w2
.w1:
    mov     eax, 1
    cvtsi2sd xmm6, eax
.w2:
    ; 2. finite(w)
    movapd  xmm5, xmm6
    subsd   xmm5, xmm6
    ucomisd xmm5, xmm5
    jp      .nonfinite
    ; 3. rhat = y[i] - b0
    movsd   xmm0, [r13 + r8*8]
    subsd   xmm0, [rsp+16]
    ; 4. X row
    mov     rax, r8
    imul    rax, r12
    mov     r9, rbx
    add     r9, rax
    ; 5-6. d = sum Xrow[k]*beta[k]
    pxor    xmm1, xmm1
    xor     rcx, rcx
.dot:
    cmp     rcx, r15
    jae     .dot_done
    movsd   xmm2, [r9 + rcx*8]
    mov     rax, [rsp+24]
    mulsd   xmm2, [rax + rcx*8]
    addsd   xmm1, xmm2
    inc     rcx
    jmp     .dot
.dot_done:
    ; 7. r = w*(rhat - d)
    subsd   xmm0, xmm1
    mulsd   xmm0, xmm6
    ; 8. finite(r)
    movapd  xmm5, xmm0
    subsd   xmm5, xmm0
    ucomisd xmm5, xmm5
    jp      .nonfinite
    ; 9. x2 = r*r  (r preserved in xmm4 for the X'r pass; w is already
    ; folded into r, so square r itself — the old `mulsd xmm6, xmm4` used
    ; the stale w register and returned sse = r when w == 1)
    movapd  xmm4, xmm0
    movapd  xmm6, xmm4
    mulsd   xmm6, xmm4              ; xmm6 = r^2
    ; 10. Neumaier(sse_s, sse_c, x2): t=xmm2 |s|=xmm3 |x|... use xmm5 as |.|
    movsd   xmm0, [rsp+0]           ; s
    movapd  xmm2, xmm0
    addsd   xmm2, xmm6              ; t
    movapd  xmm3, xmm0
    andpd   xmm3, xmm7              ; |s|
    movapd  xmm5, xmm6
    andpd   xmm5, xmm7              ; |x|
    cmpnltpd xmm3, xmm5             ; mask
    movapd  xmm9, xmm0
    subsd   xmm9, xmm2
    addsd   xmm9, xmm6              ; e1
    subsd   xmm6, xmm2
    addsd   xmm6, xmm0              ; e2
    andpd   xmm9, xmm3
    andnpd  xmm3, xmm6
    orpd    xmm9, xmm3              ; err
    movsd   xmm10, [rsp+8]
    addsd   xmm10, xmm9
    movsd   [rsp+8], xmm10
    movsd   [rsp+0], xmm2           ; s = t
    ; 10b. Neumaier(rsum_s, rsum_c, r): sum w*r — CSNE needs the weighted
    ; residual sum (xtr accumulates X*(w*r) from the same weighted r).
    ; r is copied to xmm6 (dead after the sse block) so xmm4 survives for
    ; the xtr pass below.
    movapd  xmm6, xmm4
    lea     rax, [rsp + 56]
    lea     rax, [rax + r15*8]      ; rsum_s
    movsd   xmm0, [rax]
    movapd  xmm2, xmm0
    addsd   xmm2, xmm6              ; t
    movapd  xmm3, xmm0
    andpd   xmm3, xmm7              ; |s|
    movapd  xmm5, xmm6
    andpd   xmm5, xmm7              ; |x|
    cmpnltpd xmm3, xmm5             ; mask
    movapd  xmm9, xmm0
    subsd   xmm9, xmm2
    addsd   xmm9, xmm6              ; e1
    subsd   xmm6, xmm2
    addsd   xmm6, xmm0              ; e2
    andpd   xmm9, xmm3
    andnpd  xmm3, xmm6
    orpd    xmm9, xmm3              ; err
    addsd   xmm9, [rax+8]
    movsd   [rax+8], xmm9           ; c += err
    movsd   [rax], xmm2             ; s = t
    ; 11. xtr[k] += Xrow[k]*r  (r in xmm4)
    xor     rcx, rcx
.xtr:
    cmp     rcx, r15
    jae     .xtr_done
    movsd   xmm2, [r9 + rcx*8]
    mulsd   xmm2, xmm4
    movsd   xmm3, [rsp + 56 + rcx*8]
    addsd   xmm3, xmm2
    movsd   [rsp + 56 + rcx*8], xmm3
    inc     rcx
    jmp     .xtr
.xtr_done:
    inc     r8
    jmp     .row
.rows_done:
    ; epilogue: *sseOut = s + c; xtr/rsum copied out if pointers non-NULL
    mov     rax, [rsp+40]
    movsd   xmm0, [rsp+0]
    addsd   xmm0, [rsp+8]
    movsd   [rax], xmm0
    mov     rax, [rsp+80+r15*8]     ; rsum_out
    test    rax, rax
    jz      .no_rsum
    lea     rcx, [rsp + 56]
    lea     rcx, [rcx + r15*8]      ; rsum_s
    movsd   xmm0, [rcx]
    addsd   xmm0, [rcx+8]
    movsd   [rax], xmm0
.no_rsum:
    mov     rax, [rsp+48]
    test    rax, rax
    jz      .no_xtr
    xor     rcx, rcx
.cp:
    cmp     rcx, r15
    jae     .no_xtr
    movsd   xmm0, [rsp + 56 + rcx*8]
    movsd   [rax + rcx*8], xmm0
    inc     rcx
    jmp     .cp
.no_xtr:
    xor     eax, eax
    jmp     .out
.nonfinite:
    mov     eax, OLS_E_NONFINITE
.out:
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ==============================================================================
; int32_t ols_pivot_means_f64_sse2(const double *X, size_t ldx8,
;                                  const double *y, const double *w,
;                                  size_t nb, size_t p, double *c)
; Weighted compensated column means of the first nb rows of [X|y] -> c[0..q).
; Args    : rdi=X rsi=ldx8 rdx=y rcx=w(NULL ok) r8=nb r9=p, [rbp+16]=c
; Returns : eax = OLS_OK | OLS_E_NONFINITE
; Requires: SSE2; nb >= 1.
; Memory  : reads X rows [0..nb) cols [0..p), y[0..nb), w[0..nb) if non-NULL;
;           writes c[0..q).
; Clobbers: caller-saved + rbx,rbp,r12-r15 (pushed/popped).
; Numerics: per column Neumaier (s,sc) over x = w*z; mean = s/W_total where
;           W_total is the Neumaier weight sum (not nb) — correct for weights.
; Cost    : PREDICTED (perf-log #18): ~(q+8) ops/row * nb rows.
; Frame: s[q] at [rsp), sc[q] at [rsp + q*8); W in frame at [rsp + 2*q*8].
; ==============================================================================
global ols_pivot_means_f64_sse2:function
ols_pivot_means_f64_sse2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8
    mov     rbx, rdi                ; X
    mov     r12, rsi                ; ldx8
    mov     r13, rdx                ; y
    mov     r14, r8                 ; nb
    mov     r15, r9
    inc     r15                     ; q
    mov     r10, [rbp+16]           ; c
    mov     r11, rcx                ; w (NULL ok) — r11 is caller-saved but we
    ; make no calls, so it survives; still, stash it in the frame for safety.
    ; frame: s[q] + sc[q] + W(2) + wp(1), 16-aligned
    lea     rax, [r15*8]
    lea     rdx, [rax*2 + 32]
    add     rdx, 15
    and     rdx, -16
    sub     rsp, rdx
    mov     [rsp + rax*2 + 0], r11  ; w ptr  (at 2*q*8)
    mov     r11, rax
    add     r11, r11                ; 2*q*8 (offset of the ctrl tail)
    ; zero s, sc, W
    mov     rdx, rsp
    add     rdx, rax                ; sc base (rsp + q*8)
    pxor    xmm0, xmm0
    xor     rcx, rcx
.zc:
    cmp     rcx, r15
    jae     .zc_done
    movsd   [rsp + rcx*8], xmm0
    movsd   [rdx + rcx*8], xmm0
    inc     rcx
    jmp     .zc
.zc_done:
    movsd   [rsp + r11 + 8], xmm0   ; W_s
    movsd   [rsp + r11 + 16], xmm0  ; W_c
    mov     rax, 0x7FFFFFFFFFFFFFFF
    movq    xmm7, rax
    punpcklqdq xmm7, xmm7
    lea     rdx, [rsp + r15*8]      ; sc base (rsp + q*8)
    xor     r8, r8                  ; i
.pm_row:
    cmp     r8, r14
    jae     .pm_rows_done
    ; w -> xmm6
    mov     rax, [rsp + r11]
    test    rax, rax
    jz      .pmw1
    movsd   xmm6, [rax + r8*8]
    jmp     .pmw2
.pmw1:
    mov     eax, 1
    cvtsi2sd xmm6, eax
.pmw2:
    movapd  xmm9, xmm6
    subsd   xmm9, xmm6
    ucomisd xmm9, xmm9
    jp      .pm_nonfinite
    ; keep w in xmm4 (freed after the W update)
    movapd  xmm4, xmm6
    ; W Neumaier
    movsd   xmm0, [rsp + r11 + 8]  ; W_s
    movapd  xmm1, xmm0
    addsd   xmm1, xmm6              ; t
    movapd  xmm2, xmm0
    andpd   xmm2, xmm7
    movapd  xmm3, xmm6
    andpd   xmm3, xmm7
    cmpnltpd xmm2, xmm3
    movapd  xmm5, xmm0
    subsd   xmm5, xmm1
    addsd   xmm5, xmm6              ; e1
    subsd   xmm6, xmm1
    addsd   xmm6, xmm0              ; e2
    andpd   xmm5, xmm2
    andnpd  xmm2, xmm6
    orpd    xmm5, xmm2
    movsd   xmm9, [rsp + r11 + 16]
    addsd   xmm9, xmm5
    movsd   [rsp + r11 + 16], xmm9
    movsd   [rsp + r11 + 8], xmm1
    ; X row
    mov     rax, r8
    imul    rax, r12
    mov     r9, rbx
    add     r9, rax
    xor     rcx, rcx                ; k
.pm_k:
    cmp     rcx, r15
    jae     .pm_k_done
    mov     rsi, r15
    dec     rsi                     ; p
    cmp     rcx, rsi
    jb      .pm_from_x
    movsd   xmm6, [r13 + r8*8]      ; y[i]
    jmp     .pm_z
.pm_from_x:
    movsd   xmm6, [r9 + rcx*8]
.pm_z:
    movapd  xmm9, xmm6
    subsd   xmm9, xmm6
    ucomisd xmm9, xmm9
    jp      .pm_nonfinite
    mulsd   xmm6, xmm4              ; x = w*z[k]
    ; Neumaier into (s[k], sc[k])
    movsd   xmm0, [rsp + rcx*8]     ; s
    movapd  xmm1, xmm0
    addsd   xmm1, xmm6              ; t
    movapd  xmm2, xmm0
    andpd   xmm2, xmm7
    movapd  xmm3, xmm6
    andpd   xmm3, xmm7
    cmpnltpd xmm2, xmm3
    movapd  xmm5, xmm0
    subsd   xmm5, xmm1
    addsd   xmm5, xmm6              ; e1
    subsd   xmm6, xmm1
    addsd   xmm6, xmm0              ; e2
    andpd   xmm5, xmm2
    andnpd  xmm2, xmm6
    orpd    xmm5, xmm2
    ; sc[k] += err ; s[k] = t   (sc base in rdx)
    movsd   xmm9, [rdx + rcx*8]
    addsd   xmm9, xmm5
    movsd   [rdx + rcx*8], xmm9
    movsd   [rsp + rcx*8], xmm1
    inc     rcx
    jmp     .pm_k
.pm_k_done:
    inc     r8
    jmp     .pm_row
.pm_rows_done:
    ; means: c[k] = (s[k] + sc[k]) / (W_s + W_c)
    movsd   xmm8, [rsp + r11 + 8]
    addsd   xmm8, [rsp + r11 + 16]  ; W total
    xor     rcx, rcx
.pm_out:
    cmp     rcx, r15
    jae     .pm_out_done
    movsd   xmm0, [rsp + rcx*8]
    movsd   xmm1, [rdx + rcx*8]
    addsd   xmm0, xmm1
    divsd   xmm0, xmm8
    movsd   [r10 + rcx*8], xmm0
    inc     rcx
    jmp     .pm_out
.pm_out_done:
    xor     eax, eax
    jmp     .pm_out_exit
.pm_nonfinite:
    mov     eax, OLS_E_NONFINITE
.pm_out_exit:
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ==============================================================================
; int32_t ols_predict_f64_sse2(const ols_pred_args_t *A)
; Args    : rdi = A
; Returns : eax = OLS_OK | OLS_E_NONFINITE
; Requires: SSE2. out[0..n) OVERWRITTEN.
; Memory  : reads A once; X rows [0..n) cols [0..p) at stride A->ldx8,
;           beta[0..p); writes out[0..n).
; Clobbers: rax,rcx,rdx,rsi,rdi,r8-r11 + rbx,rbp,r12-r15 (pushed/popped).
; Numerics: per-row Neumaier-compensated dot + b0; p=0 -> out = b0;
;           NaN/inf results rejected (the x-x==x parity check catches both).
; Cost    : PREDICTED (perf-log #24): ~p+8 ops/row -> q=8 ~15 cy/row
;           (the residual pass minus y/w loads and sse/xtr accumulation).
; ==============================================================================
global ols_predict_f64_sse2:function
ols_predict_f64_sse2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 40                 ; [0] b0 [8] out [16] ldx8 [24] beta
    and     rsp, -16
    mov     rbx, [rdi+PA_X]
    mov     r12, [rdi+PA_LDX8]
    mov     r13, [rdi+PA_BETA]
    movsd   xmm0, [rdi+PA_B0]
    movsd   [rsp+0], xmm0
    mov     rax, [rdi+PA_OUT]
    mov     [rsp+8], rax
    mov     r14, [rdi+PA_N]
    mov     r15, [rdi+PA_P]
    mov     rax, 0x7FFFFFFFFFFFFFFF
    movq    xmm7, rax
    punpcklqdq xmm7, xmm7
    xor     r8, r8                  ; i
.row:
    cmp     r8, r14
    jae     .rows_done
    ; per-row Neumaier dot: s (xmm0) c (xmm1), term (xmm2)
    pxor    xmm0, xmm0
    pxor    xmm1, xmm1
    mov     rax, r8
    imul    rax, r12
    lea     r9, [rbx + rax]         ; row i
    xor     rcx, rcx
.dot:
    cmp     rcx, r15
    jae     .dot_done
    movsd   xmm2, [r9 + rcx*8]
    movsd   xmm3, [r13 + rcx*8]
    mulsd   xmm2, xmm3              ; term
    movapd  xmm4, xmm0
    addsd   xmm4, xmm2              ; t
    movapd  xmm5, xmm0
    andpd   xmm5, xmm7              ; |s|
    movapd  xmm6, xmm2
    andpd   xmm6, xmm7              ; |x|
    cmpnltpd xmm5, xmm6             ; mask
    movapd  xmm8, xmm0
    subsd   xmm8, xmm4
    addsd   xmm8, xmm2              ; e1
    subsd   xmm2, xmm4
    addsd   xmm2, xmm0              ; e2
    andpd   xmm8, xmm5
    andnpd  xmm5, xmm2
    orpd    xmm8, xmm5              ; err
    addsd   xmm1, xmm8
    movapd  xmm0, xmm4              ; s = t
    inc     rcx
    jmp     .dot
.dot_done:
    ; yhat = b0 + s + c; reject NaN/inf
    movsd   xmm2, [rsp+0]           ; b0
    addsd   xmm2, xmm0
    addsd   xmm2, xmm1
    movapd  xmm3, xmm2
    subsd   xmm3, xmm2
    ucomisd xmm3, xmm3
    jp      .nonfinite
    mov     rax, [rsp+8]
    movsd   [rax + r8*8], xmm2
    inc     r8
    jmp     .row
.rows_done:
    xor     eax, eax
    jmp     .out
.nonfinite:
    mov     eax, OLS_E_NONFINITE
.out:
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
