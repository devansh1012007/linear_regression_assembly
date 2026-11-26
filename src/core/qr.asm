; ------------------------------------------------------------------------------
; qr.asm — Householder QR for the OLS accuracy path (ADR-0007).
;   ols_qr_build_f64_sse2(const ols_qr_args_t *A)   : weighted, pivot-shifted,
;       column-equilibrated augmented copy [1|X-c|y-c] (intercept) or [X|y],
;       plus pivot-shifted sums S, W and the SST byproduct.
;   ols_qr_f64_sse2(Aug, n, m, stride, rcond, work) : in-place Householder on
;       the leading m-1 design columns (reflected alpha; reflectors applied
;       to ALL m columns); returns the rank in eax (j = first deficient
;       column, dc = full rank; the caller decides what rank < dc means).
;   ols_qr_solve_f64_sse2(Aug, m, stride, b_out)    : R back-substitution for
;       the m-1 scaled coefficients (the SSE tail sum is computed by the
;       caller: sse = sum_{i>=dc} Aug[i][m-1]^2 — the residual TAIL norm^2,
;       NOT R_dd^2; only dc rows of R exist, see mp_qr_ols).
; Design columns arrive EQUILIBRATED (unit 2-norm), so |R_jj| in (0,1] and
; the rcond test is an absolute threshold on the equilibrated diagonal.
; SSE2-scalar arithmetic (sd ops); the Householder update is two contiguous
; row passes (w accumulation + rank-1 update). Vectorization is Phase-5
; baseline-ladder scope, not Phase-3 scope (ADR-0007).
; Register plans (fixed per kernel; the s/Syy accumulators are updated
; THROUGH MEMORY so xmm5 (sw) / xmm6 (w) survive every inner block):
;   build : rbx=A  r14=i  r15=j; ctrl+accumulators on rsp
;   qr    : rbx=Aug r12=n r13=m r14=stride r15=j; scalars on rsp
;   solve : rbx=Aug r12=m r13=stride r14=b_out r15=i
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"
%include "src/macros.inc"

section .text

; ==============================================================================
; int32_t ols_qr_build_f64_sse2(const ols_qr_args_t *A)
; Args    : rdi = A
; Returns : eax = OLS_OK | OLS_E_NONFINITE
; Requires: SSE2. Aug[0..n) x [0..m) OVERWRITTEN (m = p+1+intercept, stride m);
;           S[0..q) OVERWRITTEN (pivot-shifted sums, raw units, Neumaier);
;           *Wout, *sst_out OVERWRITTEN; norms[0..m-1) = design-column norms.
; Memory  : reads X rows [0..n) cols [0..p) at stride A->ldx8, y[0..n),
;           w[0..n) if non-NULL, c[0..q); writes Aug/S/Wout/sst_out/norms.
; Clobbers: rax,rcx,rdx,rsi,rdi,r8-r11 + rbx,rbp,r12-r15 (pushed/popped).
; Numerics: column SSQ plain (non-negative sums; norms only steer the
;           arbitrary equilibration); shifted sums + Syy Neumaier;
;           no-intercept SST is the RAW sum(w*y^2) (no -s^2/W subtraction —
;           same convention as the chol path's uncentered G[p][p]).
; Cost    : PREDICTED (perf-log #23): ~3p+16 ops/row -> q=9 ~40 cy/row.
; Frame: ctrl 15 slots ([0..112); ACC base ptr at [rsp+112], ACC at [rsp+120):
;   ACC = s pairs (2q) | ssq (dc) | W | Syy_s | Syy_c
;   ctrl: [0]n [8]p [16]ic [24]m [32]dc [40]c [48]S [56]Wout [64]sst_out
;         [72]norms [80]w [88]ldx8 [96]X [104]y
; ==============================================================================
global ols_qr_build_f64_sse2:function
ols_qr_build_f64_sse2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    mov     rbx, rdi                ; A
    ; ---- allocate: ctrl(15) + acc(2q + dc + 3) doubles, 16-aligned ----
    mov     rax, [rbx+QA_P]
    mov     rdx, [rbx+QA_INTERCEPT]
    lea     rcx, [rax + rax]
    lea     rcx, [rcx + 2]          ; 2q
    add     rcx, rdx                ; + ic
    add     rcx, rax                ; + p (dc)
    add     rcx, 18                 ; + 3 acc + 15 ctrl slots
    lea     rcx, [rcx*8]
    add     rcx, 15
    and     rcx, -16
    sub     rsp, rcx
    and     rsp, -16
    ; ---- ctrl slots ----
    mov     rax, [rbx+QA_N]
    mov     [rsp+0], rax
    mov     rax, [rbx+QA_P]
    mov     [rsp+8], rax
    mov     rax, [rbx+QA_INTERCEPT]
    mov     [rsp+16], rax
    mov     rcx, [rsp+8]
    mov     rdx, [rsp+16]
    lea     rcx, [rcx + rdx + 1]    ; m = p + 1 + ic
    mov     [rsp+24], rcx
    dec     rcx
    mov     [rsp+32], rcx           ; dc = m - 1
    mov     rax, [rbx+QA_C]
    mov     [rsp+40], rax
    mov     rax, [rbx+QA_S]
    mov     [rsp+48], rax
    mov     rax, [rbx+QA_WOUT]
    mov     [rsp+56], rax
    mov     rax, [rbx+QA_SST]
    mov     [rsp+64], rax
    mov     rax, [rbx+QA_NORMS]
    mov     [rsp+72], rax
    mov     rax, [rbx+QA_W]
    mov     [rsp+80], rax
    mov     rax, [rbx+QA_LDX8]
    mov     [rsp+88], rax
    mov     rax, [rbx+QA_X]
    mov     [rsp+96], rax
    mov     rax, [rbx+QA_Y]
    mov     [rsp+104], rax
    lea     rax, [rsp + 120]
    mov     [rsp+112], rax          ; ACC base
    ; ---- zero accumulators (2q + dc + 3 doubles) ----
    pxor    xmm0, xmm0
    mov     rcx, rax
    mov     rdx, [rsp+8]
    lea     r8, [rdx + rdx + 2]     ; 2q
    add     r8, [rsp+32]            ; + dc
    add     r8, 3                   ; + W + Syy pair
    xor     r10, r10
.zero:
    cmp     r10, r8
    jae     .zero_done
    movsd   [rcx + r10*8], xmm0
    inc     r10
    jmp     .zero
.zero_done:
    mov     rax, 0x7FFFFFFFFFFFFFFF
    movq    xmm7, rax
    punpcklqdq xmm7, xmm7           ; abs mask (dedicated)
    ; ssq base offset (doubles) = 2q; W offset = 2q + dc; Syy = W + 8
    xor     r14, r14                ; i
.row:
    cmp     r14, [rsp+0]
    jae     .rows_done
    ; w -> xmm6, sw -> xmm5
    mov     rax, [rsp+80]
    test    rax, rax
    jz      .w1
    movsd   xmm6, [rax + r14*8]
    jmp     .w2
.w1:
    mov     eax, 1
    cvtsi2sd xmm6, eax
.w2:
    movapd  xmm1, xmm6
    subsd   xmm1, xmm1
    ucomisd xmm1, xmm1
    jp      .nonfinite
    sqrtsd  xmm5, xmm6
    ; W += w (plain; non-negative sums; W slot = ACC + (2q + dc))
    mov     rcx, [rsp+112]
    mov     rdx, [rsp+8]
    lea     rdx, [rdx + rdx + 2]    ; 2q
    add     rdx, [rsp+32]           ; + dc
    lea     rcx, [rcx + rdx*8]
    movsd   xmm1, [rcx]
    addsd   xmm1, xmm6
    movsd   [rcx], xmm1
    ; row pointers (recomputed per row)
    mov     rax, r14
    imul    rax, [rsp+24]           ; i * m
    mov     rdx, [rbx+QA_AUG]
    lea     r12, [rdx + rax*8]      ; Aug row i
    mov     rax, r14
    imul    rax, [rsp+88]           ; i * ldx8 (bytes)
    mov     rdx, [rsp+96]
    lea     r9, [rdx + rax]         ; X row
    ; ones column: Aug[i][0] = sw; ssq[0] += w
    cmp     qword [rsp+16], 0
    je      .no_ones
    movsd   [r12], xmm5
    mov     rcx, [rsp+112]
    mov     rdx, [rsp+8]
    lea     rdx, [rdx + rdx + 2]    ; 2q doubles
    lea     rcx, [rcx + rdx*8]      ; ssq base
    movsd   xmm1, [rcx]
    addsd   xmm1, xmm6
    movsd   [rcx], xmm1
.no_ones:
    ; ---- X columns j = 0..p-1 ----
    xor     r15, r15
.xj:
    cmp     r15, [rsp+8]
    jae     .xj_done
    movsd   xmm0, [r9 + r15*8]      ; x_ij
    mov     rcx, [rsp+40]
    subsd   xmm0, [rcx + r15*8]     ; v = x - c[j]
    movapd  xmm1, xmm0
    subsd   xmm1, xmm1
    ucomisd xmm1, xmm1
    jp      .nonfinite
    mulsd   xmm0, xmm5              ; a = sw * v
    mov     rcx, [rsp+16]           ; ic
    lea     rdx, [r15 + rcx]
    movsd   [r12 + rdx*8], xmm0     ; Aug[i][j+ic]
    ; ssq[j+ic] += a^2 (plain)
    mov     rcx, [rsp+112]
    mov     rdx, [rsp+8]
    lea     rdx, [rdx + rdx + 2]
    lea     rcx, [rcx + rdx*8]      ; ssq base
    mov     rdx, [rsp+16]
    lea     r8, [r15 + rdx]
    movsd   xmm1, [rcx + r8*8]
    mulsd   xmm0, xmm0
    addsd   xmm1, xmm0
    movsd   [rcx + r8*8], xmm1
    ; s[col] += w*v (Neumaier through memory; col = j+ic)
    movsd   xmm0, [r9 + r15*8]      ; x (reload)
    mov     rcx, [rsp+40]
    subsd   xmm0, [rcx + r15*8]     ; v
    mulsd   xmm0, xmm6              ; w*v
    mov     rcx, [rsp+112]
    mov     rdx, [rsp+16]
    lea     r8, [r15 + rdx]          ; col = j + ic in Aug, but S is indexed
    ; by DATA column j (X cols 0..p-1, y = p) — the ones column has no slot
    mov     r8, r15
    lea     r8, [r8 + r8]
    lea     rcx, [rcx + r8*8]        ; &s_pair[j] (2*j doubles)
    movsd   xmm1, [rcx]             ; s_s
    movapd  xmm2, xmm1
    addsd   xmm2, xmm0              ; t
    movsd   xmm3, [rcx]
    andpd   xmm3, xmm7              ; |s|
    movapd  xmm4, xmm0
    andpd   xmm4, xmm7              ; |v|
    cmpnltpd xmm3, xmm4             ; mask
    movsd   xmm4, [rcx]
    subsd   xmm4, xmm2
    addsd   xmm4, xmm0              ; e
    andpd   xmm4, xmm3
    movsd   xmm1, [rcx+8]
    addsd   xmm1, xmm4
    movsd   [rcx+8], xmm1
    movsd   [rcx], xmm2
    inc     r15
    jmp     .xj
.xj_done:
    ; ---- y column: yv = y[i] - c[p]; Aug[i][p+ic] = sw*yv ----
    mov     rdx, [rsp+104]
    movsd   xmm0, [rdx + r14*8]     ; y_i
    mov     rcx, [rsp+40]
    mov     r8, [rsp+8]             ; p
    subsd   xmm0, [rcx + r8*8]      ; yv
    movapd  xmm1, xmm0
    subsd   xmm1, xmm1
    ucomisd xmm1, xmm1
    jp      .nonfinite
    mov     rcx, [rsp+16]
    lea     rdx, [r8 + rcx]
    mulsd   xmm0, xmm5
    movsd   [r12 + rdx*8], xmm0
    ; s[p] += w*yv (Neumaier; yv lost — recompute below)
    mov     rdx, [rsp+104]
    movsd   xmm0, [rdx + r14*8]
    mov     rcx, [rsp+40]
    mov     r8, [rsp+8]
    subsd   xmm0, [rcx + r8*8]      ; yv
    mulsd   xmm0, xmm6              ; w*yv
    mov     rcx, [rsp+112]
    mov     rdx, [rsp+8]
    lea     rcx, [rcx + rdx*8]
    lea     rcx, [rcx + rdx*8]       ; &s_pair[p] (2*p doubles)
    movsd   xmm1, [rcx]
    movapd  xmm2, xmm1
    addsd   xmm2, xmm0
    movsd   xmm3, [rcx]
    andpd   xmm3, xmm7
    movapd  xmm4, xmm0
    andpd   xmm4, xmm7
    cmpnltpd xmm3, xmm4
    movsd   xmm4, [rcx]
    subsd   xmm4, xmm2
    addsd   xmm4, xmm0
    andpd   xmm4, xmm3
    movsd   xmm1, [rcx+8]
    addsd   xmm1, xmm4
    movsd   [rcx+8], xmm1
    movsd   [rcx], xmm2
    ; Syy += w*yv^2 (Neumaier): value = (w*yv)*yv; xmm0 = w*yv intact
    mov     rdx, [rsp+104]
    movsd   xmm4, [rdx + r14*8]
    mov     rcx, [rsp+40]
    mov     r8, [rsp+8]
    subsd   xmm4, [rcx + r8*8]      ; yv
    mulsd   xmm0, xmm4              ; w*yv^2
    mov     rcx, [rsp+112]
    mov     rdx, [rsp+8]
    lea     rax, [rdx + rdx + 2]    ; 2q
    add     rax, [rsp+32]           ; + dc
    lea     rcx, [rcx + rax*8]      ; W slot; Syy_s at +8, Syy_c at +16
    movsd   xmm1, [rcx+8]
    movapd  xmm2, xmm1
    addsd   xmm2, xmm0
    movsd   xmm3, [rcx+8]
    andpd   xmm3, xmm7
    movapd  xmm4, xmm0
    andpd   xmm4, xmm7
    cmpnltpd xmm3, xmm4
    movsd   xmm4, [rcx+8]
    subsd   xmm4, xmm2
    addsd   xmm4, xmm0
    andpd   xmm4, xmm3
    movsd   xmm1, [rcx+16]
    addsd   xmm1, xmm4
    movsd   [rcx+16], xmm1
    movsd   [rcx+8], xmm2
    inc     r14
    jmp     .row
.rows_done:
    ; ---- S[j] = s_s + s_c for j < q ----
    mov     rcx, [rsp+112]
    mov     rdx, [rsp+48]
    xor     r10, r10
    mov     r8, [rsp+8]
    inc     r8                      ; q
.sout:
    cmp     r10, r8
    jae     .sout_done
    lea     rax, [r10 + r10]
    movsd   xmm0, [rcx + rax*8]
    addsd   xmm0, [rcx + rax*8 + 8]
    movsd   [rdx + r10*8], xmm0
    inc     r10
    jmp     .sout
.sout_done:
    ; ---- *Wout = W (ACC + (2q + dc) doubles) ----
    mov     rcx, [rsp+112]
    mov     rdx, [rsp+8]
    lea     rax, [rdx + rdx + 2]    ; 2q
    add     rax, [rsp+32]           ; + dc
    lea     rcx, [rcx + rax*8]
    mov     rdx, [rsp+56]
    movsd   xmm0, [rcx]
    movsd   [rdx], xmm0
    ; ---- norms + column scaling (ssq base = ACC + 2q) ----
    mov     rcx, [rsp+112]
    mov     rdx, [rsp+8]
    lea     rdx, [rdx + rdx + 2]
    lea     r8, [rcx + rdx*8]       ; ssq base
    xor     r15, r15
.nrm:
    cmp     r15, [rsp+32]           ; j vs dc
    jae     .nrm_done
    movsd   xmm0, [r8 + r15*8]
    sqrtsd  xmm0, xmm0
    mov     rdx, [rsp+72]
    movsd   [rdx + r15*8], xmm0     ; norms[j]
    pxor    xmm1, xmm1
    ucomisd xmm0, xmm1
    jbe     .nrm_next               ; zero column: leave unscaled (the
    ; factorization reports rank-deficient there)
    mov     rax, 1
    cvtsi2sd xmm1, rax
    divsd   xmm1, xmm0              ; inv
    xor     r14, r14
.scale:
    cmp     r14, [rsp+0]
    jae     .nrm_next
    mov     rax, r14
    imul    rax, [rsp+24]           ; i * m
    mov     rcx, [rbx+QA_AUG]
    lea     rcx, [rcx + rax*8]
    movsd   xmm2, [rcx + r15*8]
    mulsd   xmm2, xmm1
    movsd   [rcx + r15*8], xmm2
    inc     r14
    jmp     .scale
.nrm_next:
    inc     r15
    jmp     .nrm
.nrm_done:
    ; ---- sst: ic ? Syy - s[p]^2/W : Syy ----
    mov     rcx, [rsp+112]
    mov     rdx, [rsp+8]
    lea     rax, [rdx + rdx + 2]    ; 2q
    add     rax, [rsp+32]           ; + dc
    lea     rax, [rax + 1]          ; Syy_s offset
    lea     rcx, [rcx + rax*8]
    movsd   xmm0, [rcx]
    addsd   xmm0, [rcx+8]           ; Syy
    cmp     qword [rsp+16], 0
    je      .sst_raw
    mov     rcx, [rsp+48]
    mov     rdx, [rsp+8]
    movsd   xmm1, [rcx + rdx*8]     ; s[p]
    mulsd   xmm1, xmm1
    mov     rcx, [rsp+112]
    mov     rdx, [rsp+8]
    lea     rax, [rdx + rdx + 2]
    add     rax, [rsp+32]
    lea     rcx, [rcx + rax*8]      ; W slot
    divsd   xmm1, [rcx]
    subsd   xmm0, xmm1
.sst_raw:
    mov     rdx, [rsp+64]
    movsd   [rdx], xmm0
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
; size_t ols_qr_f64_sse2(double *Aug, size_t n, size_t m, size_t stride,
;                        double rcond, double *work)
; Args    : rdi = Aug (n x m row-major, stride in doubles), rsi = n, rdx = m,
;           rcx = stride, xmm0 = rcond, r8 = work (m doubles, caller-provided)
; Returns : eax = rank (j at the first deficient column; dc = m-1 = full rank)
;           | OLS_E_NONFINITE
; Requires: SSE2; Aug equilibrated by ols_qr_build (|R_jj| in (0,1] then).
; Memory  : reads/writes Aug; reads/writes work[0..m).
; Clobbers: rax,rcx,rdx,rsi,rdi,r8-r11 + rbx,rbp,r12-r15 (pushed/popped).
; Numerics: reflected alpha = -sign(x0)*||x|| (no cancellation); beta =
;           1/(||x||*(||x||+|x0|)) via the v'v identity; SSQ plain
;           (equilibrated values); update = w pass + rank-1 pass over
;           contiguous row segments (i == j hoisted out of both loops).
; Cost    : PREDICTED (perf-log #23): ~2n(m-j) mul/add per column j, scalar
;           with strided column reads — Phase-5 vectorizes; this phase's win
;           condition is LRE, not ns.
; Frame: [0] rcond [8] work [16] dc [24] beta_h [32] alpha [40] v0 [48] nrm
; ==============================================================================
global ols_qr_f64_sse2:function
ols_qr_f64_sse2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 64
    and     rsp, -16
    mov     rbx, rdi                ; Aug
    mov     r12, rsi                ; n
    mov     r13, rdx                ; m
    mov     r14, rcx                ; stride
    mov     [rsp+8], r8             ; work
    movsd   [rsp+0], xmm0           ; rcond
    mov     rax, rdx
    dec     rax
    mov     [rsp+16], rax           ; dc = m - 1
    mov     rax, 0x7FFFFFFFFFFFFFFF
    movq    xmm6, rax
    punpcklqdq xmm6, xmm6           ; abs mask
    mov     rax, 0x8000000000000000
    movq    xmm7, rax
    punpcklqdq xmm7, xmm7           ; sign mask
    xor     r15, r15                ; j
.col:
    cmp     r15, [rsp+16]
    jae     .full_rank
    ; ---- column norm (rows j..n-1, strided, plain) ----
    pxor    xmm0, xmm0
    mov     rcx, r15
.cnrm:
    cmp     rcx, r12
    jae     .cnrm_done
    mov     rax, rcx
    imul    rax, r14
    lea     rax, [rax + r15]
    movsd   xmm1, [rbx + rax*8]
    mulsd   xmm1, xmm1
    addsd   xmm0, xmm1
    inc     rcx
    jmp     .cnrm
.cnrm_done:
    movapd  xmm1, xmm0
    subsd   xmm1, xmm1
    ucomisd xmm1, xmm1
    jp      .nonfinite
    sqrtsd  xmm0, xmm0              ; nrm
    movsd   [rsp+48], xmm0
    ucomisd xmm0, [rsp+0]           ; nrm <= rcond -> deficient at j
    jbe     .rank_def
    ; ---- alpha = -sign(x0) * nrm ----
    mov     rax, r15
    imul    rax, r14
    lea     rax, [rax + r15]
    movsd   xmm1, [rbx + rax*8]     ; x0
    movapd  xmm2, xmm1
    andpd   xmm2, xmm7              ; sign bit of x0
    andpd   xmm0, xmm6              ; |nrm|
    orpd    xmm0, xmm2
    xorpd   xmm0, xmm7              ; alpha
    movsd   [rsp+32], xmm0
    subsd   xmm1, xmm0              ; v0 = x0 - alpha
    movsd   [rsp+40], xmm1
    ; beta_h = 1 / (nrm * (nrm + |x0|))
    movsd   xmm1, [rsp+48]
    mov     rax, r15
    imul    rax, r14
    lea     rax, [rax + r15]
    movsd   xmm2, [rbx + rax*8]
    andpd   xmm2, xmm6              ; |x0|
    addsd   xmm2, xmm1
    mulsd   xmm2, xmm1
    mov     rax, 1
    cvtsi2sd xmm1, rax
    divsd   xmm1, xmm2
    movsd   [rsp+24], xmm1
    ; ---- zero work[k], k = j+1..m-1 ----
    mov     rcx, [rsp+8]
    lea     rdx, [r15 + 1]
.zero_w:
    cmp     rdx, r13
    jae     .zero_w_done
    pxor    xmm0, xmm0
    movsd   [rcx + rdx*8], xmm0
    inc     rdx
    jmp     .zero_w
.zero_w_done:
    ; ---- w pass ----
    mov     rax, r15
    imul    rax, r14
    lea     r9, [rbx + rax*8]       ; row j
    movsd   xmm4, [rsp+40]          ; v0
    lea     rdx, [r15 + 1]
.w_j:
    cmp     rdx, r13
    jae     .w_j_done
    movsd   xmm0, [r9 + rdx*8]
    mulsd   xmm0, xmm4
    mov     rcx, [rsp+8]
    movsd   xmm1, [rcx + rdx*8]
    addsd   xmm1, xmm0
    movsd   [rcx + rdx*8], xmm1
    inc     rdx
    jmp     .w_j
.w_j_done:
    lea     r10, [r15 + 1]          ; i
.w_i:
    cmp     r10, r12
    jae     .w_done
    mov     rax, r10
    imul    rax, r14
    lea     r9, [rbx + rax*8]
    movsd   xmm4, [r9 + r15*8]      ; v_i
    lea     rdx, [r15 + 1]
.w_ik:
    cmp     rdx, r13
    jae     .w_ik_done
    movsd   xmm0, [r9 + rdx*8]
    mulsd   xmm0, xmm4
    mov     rcx, [rsp+8]
    movsd   xmm1, [rcx + rdx*8]
    addsd   xmm1, xmm0
    movsd   [rcx + rdx*8], xmm1
    inc     rdx
    jmp     .w_ik
.w_ik_done:
    inc     r10
    jmp     .w_i
.w_done:
    ; ---- update pass ----
    movsd   xmm5, [rsp+24]          ; beta_h
    mov     rax, r15
    imul    rax, r14
    lea     r9, [rbx + rax*8]       ; row j
    movsd   xmm4, [rsp+40]
    mulsd   xmm4, xmm5              ; f_j = beta_h * v0
    lea     rdx, [r15 + 1]
.u_j:
    cmp     rdx, r13
    jae     .u_j_done
    mov     rcx, [rsp+8]
    movsd   xmm0, [rcx + rdx*8]
    mulsd   xmm0, xmm4
    movsd   xmm1, [r9 + rdx*8]
    subsd   xmm1, xmm0
    movsd   [r9 + rdx*8], xmm1
    inc     rdx
    jmp     .u_j
.u_j_done:
    lea     r10, [r15 + 1]
.u_i:
    cmp     r10, r12
    jae     .u_done
    mov     rax, r10
    imul    rax, r14
    lea     r9, [rbx + rax*8]
    movsd   xmm4, [r9 + r15*8]
    mulsd   xmm4, xmm5              ; f_i
    lea     rdx, [r15 + 1]
.u_ik:
    cmp     rdx, r13
    jae     .u_ik_done
    mov     rcx, [rsp+8]
    movsd   xmm0, [rcx + rdx*8]
    mulsd   xmm0, xmm4
    movsd   xmm1, [r9 + rdx*8]
    subsd   xmm1, xmm0
    movsd   [r9 + rdx*8], xmm1
    inc     rdx
    jmp     .u_ik
.u_ik_done:
    inc     r10
    jmp     .u_i
.u_done:
    ; R_jj = alpha (below-diagonal entries already hold v_i = x_i)
    movsd   xmm0, [rsp+32]
    mov     rax, r15
    imul    rax, r14
    lea     rax, [rax + r15]
    movsd   [rbx + rax*8], xmm0
    inc     r15
    jmp     .col
.full_rank:
    mov     rax, [rsp+16]           ; dc
    jmp     .out
.rank_def:
    mov     rax, r15                ; j
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
; void ols_qr_solve_f64_sse2(const double *Aug, size_t m, size_t stride,
;                            double *b_out)
; Args    : rdi = Aug (factored), rsi = m, rdx = stride (doubles),
;           rcx = b_out (m-1 doubles: the scaled coefficients)
; Returns : none (b_out[0..m-1) written)
; Requires: SSE2; Aug factored by ols_qr_f64_sse2 with full rank.
; Memory  : reads Aug upper triangle rows [0..m-1) + column m-1 rows [0..m-1)
;           (the transformed y), writes b_out[0..m-1).
; Clobbers: rax,rcx,rdx,rsi,rdi,r8-r11 + rbx,rbp,r12-r15 (pushed/popped).
; Numerics: plain back-substitution on the equilibrated R; b[i] =
;           (z_i - sum_{k=i+1..dc-1} R_ik b_k) / R_ii.
; Cost    : PREDICTED (perf-log #23): m^2/2 fused ops, negligible.
; ==============================================================================
global ols_qr_solve_f64_sse2:function
ols_qr_solve_f64_sse2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    mov     rbx, rdi
    mov     r12, rsi                ; m
    mov     r13, rdx                ; stride
    mov     r14, rcx                ; b_out
    mov     r15, r12
    sub     r15, 2                  ; i = dc - 1 = m - 2 (the first draft
    ; started at dc and wrote b_out[dc] out of bounds)
.bsub:
    cmp     r15, 0
    jl      .bsub_done              ; SIGNED exit: `jb` reads i=-1 as
    ; 0xFFFF... (unsigned above zero) and the loop walks off the stack —
    ; the segfault the C-driver SIGSEGV handler pinned to solve+0x64
    mov     rax, r15
    imul    rax, r13
    lea     r9, [rbx + rax*8]       ; row i
    mov     rax, r12
    dec     rax                     ; m-1 (y column)
    movsd   xmm0, [r9 + rax*8]      ; z_i
    mov     rax, r12
    dec     rax                     ; dc
    lea     r10, [r15 + 1]
.bk:
    cmp     r10, rax                ; k < dc
    jae     .bk_done
    movsd   xmm1, [r9 + r10*8]      ; R_ik
    movsd   xmm2, [r14 + r10*8]     ; b[k]
    mulsd   xmm1, xmm2
    subsd   xmm0, xmm1
    inc     r10
    jmp     .bk
.bk_done:
    divsd   xmm0, [r9 + r15*8]      ; / R_ii
    movsd   [r14 + r15*8], xmm0
    dec     r15
    jmp     .bsub
.bsub_done:
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ==============================================================================
; size_t ols_qr_pivot_f64_sse2(double *Aug, size_t n, size_t m, size_t stride,
;                              double rcond, double *work, size_t *perm)
; Args    : rdi = Aug, rsi = n, rdx = m, rcx = stride, xmm0 = rcond,
;           r8 = work (m doubles), r9 = perm (m-1 size_t, OUT: perm[j] =
;           original column index now sitting at position j)
; Returns : eax = rank (first deficient column index; m-1 = full rank)
; Requires: SSE2; Aug EQUILIBRATED by ols_qr_build (norms precomputed);
;           n >= m-1; perm has m-1 slots.
; Memory  : reads/writes Aug (column swaps), work[0..m), perm[0..m-1).
; Clobbers: rax,rcx,rdx,rsi,rdi,r8-r11 + rbx,rbp,r12-r15 (pushed/popped).
; Numerics: at each step j, pick the column with max residual norm among
;           j..m-2 (norms vector maintained by the ||c||^2 -= R_jk^2
;           update — exact for equilibrated columns), swap it into position
;           j (Aug columns, norms, perm), then the same reflected
;           Householder as ols_qr_f64_sse2. Rank = first j with
;           norms[j] <= rcond (the equilibrated absolute threshold).
; Cost    : PREDICTED (perf-log #25): the swap pass is O(n) per column —
;           same order as the Householder update; ~20-30% over the
;           unpivoted factor at these sizes. Accuracy phase; LRE is the
;           win condition (ADR-0007 successor).
; Frame: [0] rcond [8] work [16] dc [24] beta_h [32] alpha [40] v0
;        [48] nrm [56] perm [64] best_j [72] best_nrm
; ==============================================================================
global ols_qr_pivot_f64_sse2:function
ols_qr_pivot_f64_sse2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 96
    and     rsp, -16
    mov     rbx, rdi                ; Aug
    mov     r12, rsi                ; n
    mov     r13, rdx                ; m
    mov     r14, rcx                ; stride
    mov     [rsp+8], r8             ; work
    mov     [rsp+56], r9            ; perm
    movsd   [rsp+0], xmm0           ; rcond
    mov     rax, rdx
    dec     rax
    mov     [rsp+16], rax           ; dc = m-1
    mov     rax, 0x7FFFFFFFFFFFFFFF
    movq    xmm6, rax
    punpcklqdq xmm6, xmm6           ; abs mask
    mov     rax, 0x8000000000000000
    movq    xmm7, rax
    punpcklqdq xmm7, xmm7           ; sign mask
    xor     r15, r15                ; j
.pcol:
    cmp     r15, [rsp+16]
    jae     .pfull_rank
    ; ---- pick max-norm column among j..dc-1 (norms are recomputed EXACTLY
    ; from the current Aug column — O(n*(dc-j)) total; simpler and more
    ; robust than maintaining an incremental norms vector, at these p).
    ; rcond < 0 pins column 0 (intercept problems: the caller's intercept
    ; functional assumes the ones column stays at position 0) ----
    pxor    xmm0, xmm0              ; best_nrm = 0
    mov     [rsp+64], r15           ; best_j = j
    mov     rcx, r15
    cmp     r15, 0
    jne     .pscan
    ; j == 0: rcond < 0 pins column 0 (skip the scan); rcond >= 0 scans.
    ; (The first draft's polarity was inverted here too — pinning never
    ; engaged and the ones column could be swapped away.)
    pxor    xmm8, xmm8
    comisd  xmm8, [rsp+0]           ; compare 0 vs rcond
    ja      .pscan_pinned           ; 0 > rcond  (rcond < 0): PIN column j
    jmp     .pscan                  ; rcond >= 0: normal scan
.pscan_pinned:
    mov     [rsp+64], r15           ; best_j = j (no scan, no swap)
    ; compute column j's norm for the rank check only
    pxor    xmm1, xmm1
    mov     r8, r15
.pscan_pinned_n:
    cmp     r8, r12
    jae     .pscan_pinned_done
    mov     rax, r8
    imul    rax, r14
    lea     rax, [rax + r15]        ; column j (the pinned position)
    movsd   xmm2, [rbx + rax*8]
    mulsd   xmm2, xmm2
    addsd   xmm1, xmm2
    inc     r8
    jmp     .pscan_pinned_n
.pscan_pinned_done:
    sqrtsd  xmm0, xmm1
    movsd   [rsp+72], xmm0
    jmp     .prank_check
.pscan:
    cmp     rcx, [rsp+16]
    jae     .pscan_done
    pxor    xmm1, xmm1              ; ssq
    mov     r8, r15                 ; rows j..n-1
.pnorm:
    cmp     r8, r12
    jae     .pnorm_done
    mov     rax, r8
    imul    rax, r14
    lea     rdx, [rax + rcx]
    movsd   xmm2, [rbx + rdx*8]
    mulsd   xmm2, xmm2
    addsd   xmm1, xmm2
    inc     r8
    jmp     .pnorm
.pnorm_done:
    sqrtsd  xmm1, xmm1
    ucomisd xmm1, xmm0
    jbe     .pscan_next
    movapd  xmm0, xmm1
    mov     [rsp+64], rcx
.pscan_next:
    inc     rcx
    jmp     .pscan
.pscan_done:
    movsd   [rsp+72], xmm0          ; best_nrm
.prank_check:
    ; rank check BEFORE the swap: deficient at j. rcond < 0 (pin mode):
    ; threshold = |rcond|; rcond >= 0: threshold = rcond. (The first draft
    ; had the jae/negate polarity INVERTED — a pinned -1.0 threshold
    ; compared against -1.0 and nothing was ever rank-deficient.)
    movsd   xmm1, [rsp+0]
    pxor    xmm2, xmm2
    comisd  xmm2, xmm1              ; compare 0 vs rcond
    ja      .prank_neg              ; 0 > rcond (rcond < 0): |rcond| threshold
    jmp     .prank_abs              ; rcond >= 0: threshold as-is
.prank_neg:
    mov     rax, 0x8000000000000000
    movq    xmm3, rax
    xorpd   xmm1, xmm3
.prank_abs:
    ucomisd xmm0, xmm1
    jbe     .prank_def
    ; ---- swap columns best_j -> j (Aug rows, perm) ----
    mov     rcx, [rsp+64]
    cmp     rcx, r15
    je      .pno_swap
    xor     r8, r8                  ; row 0: the ENTIRE columns swap — the
                                    ; rows above j hold each column's R
                                    ; history; swapping only j..n-1 (the
                                    ; first draft) corrupted the solve
.pswap:
    cmp     r8, r12
    jae     .pswap_done
    mov     rax, r8
    imul    rax, r14
    lea     rdx, [rax + r15]
    lea     rsi, [rax + rcx]
    movsd   xmm0, [rbx + rdx*8]
    movsd   xmm1, [rbx + rsi*8]
    movsd   [rbx + rdx*8], xmm1
    movsd   [rbx + rsi*8], xmm0
    inc     r8
    jmp     .pswap
.pswap_done:
    ; perm swap (size_t)
    mov     rax, [rsp+56]
    mov     rdx, [rax + r15*8]
    mov     r8, [rax + rcx*8]
    mov     [rax + r15*8], r8
    mov     [rax + rcx*8], rdx
.pno_swap:
    ; ---- Householder on column j (identical to ols_qr_f64_sse2) ----
    pxor    xmm0, xmm0
    mov     rcx, r15
.pcnrm:
    cmp     rcx, r12
    jae     .pcnrm_done
    mov     rax, rcx
    imul    rax, r14
    lea     rax, [rax + r15]
    movsd   xmm1, [rbx + rax*8]
    mulsd   xmm1, xmm1
    addsd   xmm0, xmm1
    inc     rcx
    jmp     .pcnrm
.pcnrm_done:
    movapd  xmm1, xmm0
    subsd   xmm1, xmm1
    ucomisd xmm1, xmm1
    jp      .pnonfinite
    sqrtsd  xmm0, xmm0
    movsd   [rsp+48], xmm0
    ; alpha = -sign(x0)*nrm
    mov     rax, r15
    imul    rax, r14
    lea     rax, [rax + r15]
    movsd   xmm1, [rbx + rax*8]
    movapd  xmm2, xmm1
    andpd   xmm2, xmm7
    andpd   xmm0, xmm6
    orpd    xmm0, xmm2
    xorpd   xmm0, xmm7
    movsd   [rsp+32], xmm0
    subsd   xmm1, xmm0
    movsd   [rsp+40], xmm1
    ; beta_h = 1/(nrm*(nrm+|x0|))
    movsd   xmm1, [rsp+48]
    mov     rax, r15
    imul    rax, r14
    lea     rax, [rax + r15]
    movsd   xmm2, [rbx + rax*8]
    andpd   xmm2, xmm6
    addsd   xmm2, xmm1
    mulsd   xmm2, xmm1
    mov     rax, 1
    cvtsi2sd xmm1, rax
    divsd   xmm1, xmm2
    movsd   [rsp+24], xmm1
    ; zero work
    mov     rcx, [rsp+8]
    lea     rdx, [r15 + 1]
.pzero_w:
    cmp     rdx, r13
    jae     .pzero_w_done
    pxor    xmm0, xmm0
    movsd   [rcx + rdx*8], xmm0
    inc     rdx
    jmp     .pzero_w
.pzero_w_done:
    ; w pass
    mov     rax, r15
    imul    rax, r14
    lea     r9, [rbx + rax*8]
    movsd   xmm4, [rsp+40]
    lea     rdx, [r15 + 1]
.pw_j:
    cmp     rdx, r13
    jae     .pw_j_done
    movsd   xmm0, [r9 + rdx*8]
    mulsd   xmm0, xmm4
    mov     rcx, [rsp+8]
    movsd   xmm1, [rcx + rdx*8]
    addsd   xmm1, xmm0
    movsd   [rcx + rdx*8], xmm1
    inc     rdx
    jmp     .pw_j
.pw_j_done:
    lea     r10, [r15 + 1]
.pw_i:
    cmp     r10, r12
    jae     .pw_done
    mov     rax, r10
    imul    rax, r14
    lea     r9, [rbx + rax*8]
    movsd   xmm4, [r9 + r15*8]
    lea     rdx, [r15 + 1]
.pw_ik:
    cmp     rdx, r13
    jae     .pw_ik_done
    movsd   xmm0, [r9 + rdx*8]
    mulsd   xmm0, xmm4
    mov     rcx, [rsp+8]
    movsd   xmm1, [rcx + rdx*8]
    addsd   xmm1, xmm0
    movsd   [rcx + rdx*8], xmm1
    inc     rdx
    jmp     .pw_ik
.pw_ik_done:
    inc     r10
    jmp     .pw_i
.pw_done:
    ; update pass
    movsd   xmm5, [rsp+24]
    mov     rax, r15
    imul    rax, r14
    lea     r9, [rbx + rax*8]
    movsd   xmm4, [rsp+40]
    mulsd   xmm4, xmm5
    lea     rdx, [r15 + 1]
.pu_j:
    cmp     rdx, r13
    jae     .pu_j_done
    mov     rcx, [rsp+8]
    movsd   xmm0, [rcx + rdx*8]
    mulsd   xmm0, xmm4
    movsd   xmm1, [r9 + rdx*8]
    subsd   xmm1, xmm0
    movsd   [r9 + rdx*8], xmm1
    inc     rdx
    jmp     .pu_j
.pu_j_done:
    lea     r10, [r15 + 1]
.pu_i:
    cmp     r10, r12
    jae     .pu_done
    mov     rax, r10
    imul    rax, r14
    lea     r9, [rbx + rax*8]
    movsd   xmm4, [r9 + r15*8]
    mulsd   xmm4, xmm5
    lea     rdx, [r15 + 1]
.pu_ik:
    cmp     rdx, r13
    jae     .pu_ik_done
    mov     rcx, [rsp+8]
    movsd   xmm0, [rcx + rdx*8]
    mulsd   xmm0, xmm4
    movsd   xmm1, [r9 + rdx*8]
    subsd   xmm1, xmm0
    movsd   [r9 + rdx*8], xmm1
    inc     rdx
    jmp     .pu_ik
.pu_ik_done:
    inc     r10
    jmp     .pu_i
.pu_done:
    movsd   xmm0, [rsp+32]
    mov     rax, r15
    imul    rax, r14
    lea     rax, [rax + r15]
    movsd   [rbx + rax*8], xmm0
    inc     r15
    jmp     .pcol
.pfull_rank:
    mov     rax, [rsp+16]
    jmp     .pout
.prank_def:
    mov     rax, r15
    jmp     .pout
.pnonfinite:
    mov     eax, OLS_E_NONFINITE
.pout:
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
