; ------------------------------------------------------------------------------
; chol.asm — stride-aware Cholesky factor, triangular solves, inverse.
;   ols_chol_f64_sse2(A, q, eps, stride)       : in-place upper factor R
;   ols_chol_solve_f64_sse2(R, q, b, stride)   : solve R R' x = b (b <- x)
;   ols_chol_inv_f64_sse2(A, q, B, stride)     : explicit (R'R)^-1 -> B
; stride = row stride in DOUBLES (0 -> q, i.e. contiguous). This lets
; ols_fit factor the leading p x p block of the q-wide Gram in place.
; All inner products Neumaier-compensated (numerical-ols §3).
;
; Register plans (fixed for the whole file):
;   chol : A=rbx  order=r12  stride=r13  j=r14  k=r15      rcx,rdx,rax scratch
;   solve: R=rbx  order=r12  stride=r13  b=r14  j=r15      rcx,rdx,rax scratch
;   inv  : A=rbx  order=r12  stride=r13  B=r14  col=r15    rcx,rdx,rax,r8 scratch
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"
%include "src/macros.inc"

section .text

; ==============================================================================
; int32_t ols_chol_f64_sse2(double *A, size_t q, double eps_rel, size_t stride)
; Args    : rdi = A, rsi = q, xmm0 = eps_rel, rdx = stride (doubles, 0 -> q)
; Returns : eax = OLS_OK | OLS_E_SINGULAR
; Requires: SSE2; A's upper triangle (within the stride) holds the SPD matrix.
; Memory  : reads/writes A's upper triangle only; strictly-lower untouched.
; Clobbers: rax,rcx,rdx,rsi,rdi,r8-r11 + rbx,rbp,r12-r15 (pushed/popped).
; Numerics: right-looking; pivot dots Neumaier. Singular if pivot <= eps *
;           max(input diag) or NaN.
; Cost    : PREDICTED (perf-log #13): q^3/3 scalar flops, ~1.5-2 cy/flop.
; ==============================================================================
global ols_chol_f64_sse2:function
ols_chol_f64_sse2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 40                 ; [0] threshold [8] eps
    and     rsp, -16
    mov     rbx, rdi                ; A
    mov     r12, rsi                ; order
    mov     r13, rdx                ; stride
    test    r13, r13
    jnz     .st_ok
    mov     r13, rsi
.st_ok:
    movsd   [rsp+8], xmm0           ; eps
    ; ---- max input diag + threshold ----
    pxor    xmm0, xmm0
    xor     rcx, rcx
.maxd:
    cmp     rcx, r12
    jae     .maxd_done
    mov     rax, rcx
    imul    rax, r13
    add     rax, rcx
    movsd   xmm1, [rbx + rax*8]
    ucomisd xmm1, xmm0
    jbe     .maxd_next
    movapd  xmm0, xmm1
.maxd_next:
    inc     rcx
    jmp     .maxd
.maxd_done:
    mulsd   xmm0, [rsp+8]
    movsd   [rsp+0], xmm0           ; threshold
    ; abs mask
    mov     rax, 0x7FFFFFFFFFFFFFFF
    movq    xmm7, rax
    punpcklqdq xmm7, xmm7
    ; ---- factor ----
    xor     r14, r14                ; j
.fj:
    cmp     r14, r12
    jae     .fok
    ; pivot dot (Neumaier): sum R[k][j]^2, k < j
    pxor    xmm0, xmm0              ; s
    pxor    xmm1, xmm1              ; c
    xor     rcx, rcx                ; k
.fpd:
    cmp     rcx, r14
    jae     .fpd_done
    mov     rdx, rcx
    imul    rdx, r13
    add     rdx, r14
    movsd   xmm3, [rbx + rdx*8]     ; R[k][j]
    mulsd   xmm3, xmm3
    movapd  xmm4, xmm0
    addsd   xmm4, xmm3              ; t
    movapd  xmm5, xmm0
    andpd   xmm5, xmm7
    movapd  xmm6, xmm3
    andpd   xmm6, xmm7
    cmpnltpd xmm5, xmm6
    movapd  xmm8, xmm0
    subsd   xmm8, xmm4
    addsd   xmm8, xmm3              ; e1
    subsd   xmm3, xmm4
    addsd   xmm3, xmm0              ; e2
    andpd   xmm8, xmm5
    andnpd  xmm5, xmm3
    orpd    xmm8, xmm5
    addsd   xmm1, xmm8
    movapd  xmm0, xmm4
    inc     rcx
    jmp     .fpd
.fpd_done:
    addsd   xmm0, xmm1
    mov     rdx, r14
    imul    rdx, r13
    add     rdx, r14
    movsd   xmm2, [rbx + rdx*8]     ; A[j][j]
    subsd   xmm2, xmm0              ; pivot
    movsd   xmm4, [rsp+0]           ; threshold
    ucomisd xmm2, xmm4
    jbe     .fsing
    sqrtsd  xmm2, xmm2
    movsd   [rbx + rdx*8], xmm2     ; R[j][j]
    ; trailing: k = j+1..q-1
    lea     r15, [r14+1]
.fk:
    cmp     r15, r12
    jae     .fj_next
    ; dot (Neumaier): sum R[i][j]*R[i][k], i < j
    pxor    xmm0, xmm0
    pxor    xmm1, xmm1
    xor     rcx, rcx                ; i
.fpd2:
    cmp     rcx, r14
    jae     .fpd2_done
    mov     rdx, rcx
    imul    rdx, r13
    mov     rax, rdx
    add     rax, r14
    movsd   xmm3, [rbx + rax*8]     ; R[i][j]
    mov     rax, rdx
    add     rax, r15
    movsd   xmm9, [rbx + rax*8]     ; R[i][k]
    mulsd   xmm3, xmm9
    movapd  xmm4, xmm0
    addsd   xmm4, xmm3
    movapd  xmm5, xmm0
    andpd   xmm5, xmm7
    movapd  xmm6, xmm3
    andpd   xmm6, xmm7
    cmpnltpd xmm5, xmm6
    movapd  xmm8, xmm0
    subsd   xmm8, xmm4
    addsd   xmm8, xmm3
    subsd   xmm3, xmm4
    addsd   xmm3, xmm0
    andpd   xmm8, xmm5
    andnpd  xmm5, xmm3
    orpd    xmm8, xmm5
    addsd   xmm1, xmm8
    movapd  xmm0, xmm4
    inc     rcx
    jmp     .fpd2
.fpd2_done:
    addsd   xmm0, xmm1
    mov     rdx, r14
    imul    rdx, r13
    add     rdx, r15
    movsd   xmm4, [rbx + rdx*8]     ; A[j][k]
    subsd   xmm4, xmm0
    mov     rax, r14
    imul    rax, r13
    add     rax, r14
    movsd   xmm5, [rbx + rax*8]     ; R[j][j]
    divsd   xmm4, xmm5
    movsd   [rbx + rdx*8], xmm4     ; R[j][k]
    inc     r15
    jmp     .fk
.fj_next:
    inc     r14
    jmp     .fj
.fok:
    xor     eax, eax
    jmp     .fout
.fsing:
    mov     eax, OLS_E_SINGULAR
.fout:
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ==============================================================================
; int32_t ols_chol_solve_f64_sse2(const double *R, size_t q, double *b,
;                                 size_t stride)
; Args    : rdi = R, rsi = q, rdx = b, rcx = stride (doubles, 0 -> q)
; Returns : eax = OLS_OK | OLS_E_SINGULAR (zero/NaN diag)
; Memory  : reads R upper; reads/writes b[0..q).
; Clobbers: caller-saved + rbx,rbp,r12-r15 (pushed/popped).
; ==============================================================================
; Args    : rdi = R, rsi = q, rdx = b, rcx = stride (doubles, 0 -> q)
; Returns : eax = OLS_OK | OLS_E_SINGULAR (zero/NaN diag)
; Requires: SSE2; R's upper triangle holds the factor.
; Memory  : reads R upper; reads/writes b[0..q).
; Clobbers: caller-saved + rbx,rbp,r12-r15 (pushed/popped), xmm0-xmm15.
; Numerics: two back-substitutions; diag guards on both passes.
; Cost    : perf-log #14.
global ols_chol_solve_f64_sse2:function
ols_chol_solve_f64_sse2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 24
    mov     rbx, rdi
    mov     r12, rsi
    mov     r13, rcx
    test    r13, r13
    jnz     .sst
    mov     r13, rsi
.sst:
    mov     r14, rdx                ; b
    ; forward: R' z = b
    xor     r15, r15                ; j
.sfw:
    cmp     r15, r12
    jae     .sfw_done
    pxor    xmm0, xmm0
    xor     rcx, rcx                ; i
.sfw_i:
    cmp     rcx, r15
    jae     .sfw_i_done
    mov     rax, rcx
    imul    rax, r13
    add     rax, r15
    movsd   xmm1, [rbx + rax*8]     ; R[i][j]
    mulsd   xmm1, [r14 + rcx*8]
    addsd   xmm0, xmm1
    inc     rcx
    jmp     .sfw_i
.sfw_i_done:
    movsd   xmm2, [r14 + r15*8]
    subsd   xmm2, xmm0
    mov     rax, r15
    imul    rax, r13
    add     rax, r15
    movsd   xmm3, [rbx + rax*8]     ; R[j][j]
    ucomisd xmm3, xmm3
    jp      .ssing
    pxor    xmm4, xmm4
    ucomisd xmm3, xmm4
    je      .ssing
    divsd   xmm2, xmm3
    movsd   [r14 + r15*8], xmm2
    inc     r15
    jmp     .sfw
.sfw_done:
    ; backward: R x = z, j from q-1 down to 0 (exit on the sign of the
    ; size_t cursor after dec: jb never fired for 0xFFFF.. — the q=1 bug)
    mov     r15, r12
    dec     r15
.sbw:
    test    r15, r15
    js      .sdone
    pxor    xmm0, xmm0
    lea     rcx, [r15+1]
.sbw_k:
    cmp     rcx, r12
    jae     .sbw_k_done
    mov     rax, r15
    imul    rax, r13
    add     rax, rcx
    movsd   xmm1, [rbx + rax*8]     ; R[j][k]
    mulsd   xmm1, [r14 + rcx*8]
    addsd   xmm0, xmm1
    inc     rcx
    jmp     .sbw_k
.sbw_k_done:
    movsd   xmm2, [r14 + r15*8]
    subsd   xmm2, xmm0
    mov     rax, r15
    imul    rax, r13
    add     rax, r15
    movsd   xmm3, [rbx + rax*8]
    ucomisd xmm3, xmm3
    jp      .ssing
    pxor    xmm4, xmm4
    ucomisd xmm3, xmm4
    je      .ssing
    divsd   xmm2, xmm3
    movsd   [r14 + r15*8], xmm2
    dec     r15
    jmp     .sbw
.sdone:
    xor     eax, eax
    jmp     .sout
.ssing:
    mov     eax, OLS_E_SINGULAR
.sout:
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ==============================================================================
; int32_t ols_chol_inv_f64_sse2(const double *A, size_t q, double *B,
;                               size_t stride)
; Args    : rdi = A (R in upper triangle), rsi = q, rdx = B_out, rcx = stride
; Returns : eax = OLS_OK | OLS_E_SINGULAR
; Memory  : reads A upper; writes B[0..q*q) CONTIGUOUS (B is always dense).
;           Frame: v buffer at [rsp+32), q doubles, allocated dynamically.
; Clobbers: caller-saved + rbx,rbp,r12-r15 (pushed/popped).
; Numerics: column j: R'w = e_j, Rv = w, B[:,j] = v (plain dots; the inverse's
;           conditioning is reported via cond_est by the caller).
; ==============================================================================
; Args    : rdi = A (R in upper triangle), rsi = q, rdx = B_out, rcx = stride
; Returns : eax = OLS_OK | OLS_E_SINGULAR
; Requires: SSE2; B_out must not alias A; A's upper triangle holds R.
; Memory  : reads A upper; writes B[0..q*q) CONTIGUOUS. Frame: v at [rsp+32).
; Clobbers: caller-saved + rbx,rbp,r12-r15 (pushed/popped), xmm0-xmm15.
; Numerics: column j: R'w = e_j, Rv = w, B[:,j] = v.
; Cost    : perf-log #15.
global ols_chol_inv_f64_sse2:function
ols_chol_inv_f64_sse2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8
    mov     rbx, rdi                ; A
    mov     r12, rsi                ; q
    mov     r13, rcx
    test    r13, r13
    jnz     .ist
    mov     r13, rsi
.ist:
    mov     r14, rdx                ; B
    lea     rax, [r12*8]
    add     rax, 47                 ; 32 (v offset) + 15 (align)
    and     rax, -16
    sub     rsp, rax                ; v at [rsp+32)
    xor     r15, r15                ; col j
.icol:
    cmp     r15, r12
    jae     .idone
    ; e_j -> v
    xor     rcx, rcx
.ie:
    cmp     rcx, r12
    jae     .ie_done
    xor     eax, eax
    cmp     rcx, r15
    jne     .izero
    mov     eax, 1
.izero:
    cvtsi2sd xmm0, eax
    movsd   [rsp + 32 + rcx*8], xmm0
    inc     rcx
    jmp     .ie
.ie_done:
    ; forward R' w = e_j
    xor     r8, r8                  ; i
.ifw:
    cmp     r8, r12
    jae     .ifw_done
    pxor    xmm0, xmm0
    xor     rcx, rcx
.ifw_k:
    cmp     rcx, r8
    jae     .ifw_kdone
    mov     rax, rcx
    imul    rax, r13
    add     rax, r8
    movsd   xmm1, [rbx + rax*8]
    mulsd   xmm1, [rsp + 32 + rcx*8]
    addsd   xmm0, xmm1
    inc     rcx
    jmp     .ifw_k
.ifw_kdone:
    movsd   xmm2, [rsp + 32 + r8*8]
    subsd   xmm2, xmm0
    mov     rax, r8
    imul    rax, r13
    add     rax, r8
    movsd   xmm3, [rbx + rax*8]
    ucomisd xmm3, xmm3
    jp      .ising
    pxor    xmm4, xmm4
    ucomisd xmm3, xmm4
    je      .ising
    divsd   xmm2, xmm3
    movsd   [rsp + 32 + r8*8], xmm2
    inc     r8
    jmp     .ifw
.ifw_done:
    ; backward R v = w (same underflow fix as the solve)
    mov     r8, r12
    dec     r8
.ibw:
    test    r8, r8
    js      .ibw_done
    pxor    xmm0, xmm0
    lea     rcx, [r8+1]
.ibw_k:
    cmp     rcx, r12
    jae     .ibw_kdone
    mov     rax, r8
    imul    rax, r13
    add     rax, rcx
    movsd   xmm1, [rbx + rax*8]
    mulsd   xmm1, [rsp + 32 + rcx*8]
    addsd   xmm0, xmm1
    inc     rcx
    jmp     .ibw_k
.ibw_kdone:
    movsd   xmm2, [rsp + 32 + r8*8]
    subsd   xmm2, xmm0
    mov     rax, r8
    imul    rax, r13
    add     rax, r8
    movsd   xmm3, [rbx + rax*8]
    ucomisd xmm3, xmm3
    jp      .ising
    pxor    xmm4, xmm4
    ucomisd xmm3, xmm4
    je      .ising
    divsd   xmm2, xmm3
    movsd   [rsp + 32 + r8*8], xmm2
    dec     r8
    jmp     .ibw
.ibw_done:
    ; B[k][j] = v[k] (B CONTIGUOUS)
    xor     rcx, rcx
.istore:
    cmp     rcx, r12
    jae     .istore_done
    mov     rax, rcx
    imul    rax, r12
    add     rax, r15
    movsd   xmm0, [rsp + 32 + rcx*8]
    movsd   [r14 + rax*8], xmm0
    inc     rcx
    jmp     .istore
.istore_done:
    inc     r15
    jmp     .icol
.idone:
    xor     eax, eax
    jmp     .iout
.ising:
    mov     eax, OLS_E_SINGULAR
.iout:
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
section .note.GNU-stack noalloc noexec nowrite progbits
