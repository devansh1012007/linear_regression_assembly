; ------------------------------------------------------------------------------
; simple.asm — Phase-1 regression solvers (scalar SSE2, compensated).
;   ols_centered_sums_f64_sse2    : one-pass SHIFTED stats (n, sx, sy, sxx, sxy, syy)
;   ols_simple_fit_f64_sse2       : simple regression with intercept
;   ols_simple_fit_noint_f64_sse2 : through-origin regression (NoInt1/2)
;   ols_quad_fit_f64_sse2         : quadratic fit (Pontius), 3-pass centered
;
; Numerics (numerical-ols-algorithms §2): accumulation is SHIFTED by caller
; pivots and Neumaier-compensated; centering happens once at the end in
; double. Raw (unshifted) second moments lose all digits when |mean| >> std:
; the cancellation in sxx - sx*sx/n is intrinsic to the formula (rounding of
; the sx*sx product at 1e-16 relative), not an accumulation error — measured
; and documented in docs/reports/phase0-1/. The shift bounds the error by
; |mean - c| instead of |mean|.
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"
%include "src/macros.inc"

section .text

; NEUMAIER_STEP s, c, xin, m, t, u, v
;   (s, c) <- Neumaier(s, c, xin).  m = abs mask (both lanes, PRESERVED).
;   t, u, v: three scratch registers (clobbered). xin: CLOBBERED (e2 in it).
;   Why branch-free: data-dependent branches mispredict on mixed signs.
;   Dataflow (needs t, mask, e1, e2 live simultaneously -> 3 scratch + xin):
;     t  = s + xin
;     u  = |s| ; v = |xin| ; u = (|s| >= |xin|) ? ~0 : 0     [mask]
;     e1 = (s - t) + xin    [in v]
;     e2 = (xin - t) + s    [in xin]
;     err = (u & e1) | (~u & e2) ; c += err ; s = t
%macro NEUMAIER_STEP 7
    movapd  %4, %1                  ; t = s
    addsd   %4, %3                  ; t = s + xin
    movapd  %5, %1                  ; u = s
    andpd   %5, %6                  ; u = |s|
    movapd  %7, %3                  ; v = xin
    andpd   %7, %6                  ; v = |xin|
    cmpnltpd %5, %7                 ; u = mask (|s| >= |xin|)
    movapd  %7, %1                  ; v = s
    subsd   %7, %4                  ; v = s - t
    addsd   %7, %3                  ; v = e1 = (s - t) + xin
    subsd   %3, %4                  ; xin = xin - t
    addsd   %3, %1                  ; xin = e2 = (xin - t) + s
    andpd   %7, %5                  ; e1 & mask
    andnpd  %5, %3                  ; ~mask & e2
    orpd    %7, %5                  ; err
    addsd   %2, %7                  ; c += err
    movapd  %1, %4                  ; s = t
%endmacro

%macro FINITE_CHECK 4
    movapd  %2, %1
    subsd   %2, %1
    pxor    %3, %3
    cmpneqpd %2, %3
    movmskpd eax, %2
    test    eax, eax
    jnz     %4
%endmacro

; ==============================================================================
; int32_t ols_centered_sums_f64_sse2(const double *x, const double *y,
;                                    size_t n, double *out /*6*/,
;                                    const double *piv /*2*/)
; SHIFTED accumulation: out = [n, s(x-cx), s(y-cy), s((x-cx)^2),
; s((x-cx)(y-cy)), s((y-cy)^2)] with cx=piv[0], cy=piv[1].
; Args    : rdi = x, rsi = y, rdx = n, rcx = out, r8 = piv
; Returns : eax = OLS_OK(0) | OLS_E_NONFINITE(-6)
; Requires: SSE2. n == 0 allowed (out = zeros, piv not read).
; Memory  : reads x[0..n), y[0..n), piv[0..2); writes out[0..6).
; Clobbers: rax, rcx, rdx, r8, r9, r10, r11, xmm0-xmm15. Uses the 128-byte
;           red zone (leaf function, no calls): 3 product spill slots.
;           Preserves callee-saved GPRs, rsp, DF, MXCSR.
; Numerics: every channel Neumaier-compensated; shift subtraction exact for
;           Sterbenz-close operands, else 1 rounding.
; Cost    : PREDICTED (perf-log #7): ~2 loads + 2 subs + 3 muls + 5 compensated
;           steps (~55 uops) per element -> ~13-16 cy/el throughput-bound.
; ==============================================================================
global ols_centered_sums_f64_sse2:function
ols_centered_sums_f64_sse2:
    ; out[0..6) = 0 (also the n == 0 result)
    pxor    xmm0, xmm0
    movups  [rcx], xmm0
    movups  [rcx+16], xmm0
    movups  [rcx+32], xmm0
    test    rdx, rdx
    jz      .ok
    ; pivots spilled to the red zone: cx @ [rsp-48], cy @ [rsp-56]
    movsd   xmm0, [r8]
    movsd   [rsp-48], xmm0
    movsd   xmm0, [r8+8]
    movsd   [rsp-56], xmm0
    pxor    xmm0, xmm0              ; s  channel A (x')
    pxor    xmm1, xmm1              ; c  A
    pxor    xmm2, xmm2              ; s  B (y')
    pxor    xmm3, xmm3              ; c  B
    pxor    xmm4, xmm4              ; s  C (x'^2)
    pxor    xmm5, xmm5              ; c  C
    pxor    xmm6, xmm6              ; s  D (x'y')
    pxor    xmm7, xmm7              ; c  D
    pxor    xmm8, xmm8              ; s  E (y'^2)
    pxor    xmm9, xmm9              ; c  E
    mov     r10, 0x7FFFFFFFFFFFFFFF
    movq    xmm10, r10
    punpcklqdq xmm10, xmm10         ; abs mask
    xor     r9, r9                  ; i
.loop:
    movsd   xmm11, [rdi + r9*8]     ; x
    movsd   xmm12, [rsi + r9*8]     ; y
    FINITE_CHECK xmm11, xmm15, xmm13, .nonfinite
    FINITE_CHECK xmm12, xmm15, xmm14, .nonfinite
    movsd   xmm13, [rsp-48]
    subsd   xmm11, xmm13            ; x' = x - cx
    movsd   xmm14, [rsp-56]
    subsd   xmm12, xmm14            ; y' = y - cy
    ; spill products to the red zone BEFORE the steps clobber xmm11/12
    movapd  xmm15, xmm11
    mulsd   xmm15, xmm11            ; x'^2
    movsd   [rsp-8], xmm15
    movapd  xmm15, xmm11
    mulsd   xmm15, xmm12            ; x'y'
    movsd   [rsp-16], xmm15
    mulsd   xmm12, xmm12            ; y'^2  (xmm12 now holds the product)
    movsd   [rsp-24], xmm12
    ; five sequential steps; inputs reloaded fresh into xmm12 each time.
    ; temps per step: t=xmm15, |s|→xmm11, |x|→xmm12(input); e1 lands in xmm11
    ; after andpd selection — the macro's 3 temps map to (xmm15, xmm11, xmm12)
    ; where the LAST is the input itself (macro handles xin clobbering).
    movsd   xmm12, [rdi + r9*8]
    subsd   xmm12, [rsp-48]
    NEUMAIER_STEP xmm0, xmm1, xmm12, xmm10, xmm15, xmm13, xmm14
    movsd   xmm12, [rsi + r9*8]
    subsd   xmm12, [rsp-56]
    NEUMAIER_STEP xmm2, xmm3, xmm12, xmm10, xmm15, xmm13, xmm14
    movsd   xmm12, [rsp-8]
    NEUMAIER_STEP xmm4, xmm5, xmm12, xmm10, xmm15, xmm13, xmm14
    movsd   xmm12, [rsp-16]
    NEUMAIER_STEP xmm6, xmm7, xmm12, xmm10, xmm15, xmm13, xmm14
    movsd   xmm12, [rsp-24]
    NEUMAIER_STEP xmm8, xmm9, xmm12, xmm10, xmm15, xmm13, xmm14
    inc     r9
    cmp     r9, rdx
    jb      .loop
    ; out = [n, sA+cA, sB+cB, sC+cC, sD+cD, sE+cE]
    cvtsi2sd xmm11, rdx
    movsd   [rcx], xmm11
    movapd  xmm12, xmm0
    addsd   xmm12, xmm1
    movsd   [rcx+8], xmm12
    movapd  xmm12, xmm2
    addsd   xmm12, xmm3
    movsd   [rcx+16], xmm12
    movapd  xmm12, xmm4
    addsd   xmm12, xmm5
    movsd   [rcx+24], xmm12
    movapd  xmm12, xmm6
    addsd   xmm12, xmm7
    movsd   [rcx+32], xmm12
    movapd  xmm12, xmm8
    addsd   xmm12, xmm9
    movsd   [rcx+40], xmm12
.ok:
    xor     eax, eax
    ret
.nonfinite:
    mov     eax, OLS_E_NONFINITE
    ret

; ==============================================================================
; int32_t ols_simple_fit_f64_sse2(const double *x, const double *y,
;                                 size_t n, uint32_t flags, double *out /*7*/)
; Simple regression with intercept, pivot-shifted:
;   c = (x[0], y[0]); shifted sums; Sxx = sxx - sx'^2/n (small correction);
;   b1 = Sxy/Sxx; xbar = cx + sx'/n; ybar = cy + sy'/n; b0 = ybar - b1*xbar.
; Args    : rdi = x, rsi = y, rdx = n, ecx = flags, r8 = out
; Returns : eax = OLS_OK | OLS_E_ARG (n==0) | OLS_E_NONFINITE | OLS_E_SINGULAR
; Requires: SSE2
; Memory  : reads x[0..n), y[0..n); writes out[0..7) ONLY. Errors zero out.
; Clobbers: caller-saved GPRs, xmm0-xmm15; pushes rbx, rbp, r12-r15.
; Numerics: stats via explicit residual pass (never the cancelling identity).
; Cost    : PREDICTED (perf-log #8): sums ~13-16 cy/el + residual ~7 cy/el.
; Frame (rsp after sub, 16-aligned): [0..48) sums, [48] b1, [56] b0, [64] n_f64
; ==============================================================================
global ols_simple_fit_f64_sse2:function
ols_simple_fit_f64_sse2:
    test    rdx, rdx
    jz      .earg
    push    rbp
    mov     rbp, rsp
    push    rbx                        ; x
    push    r12                        ; y
    push    r13                        ; n
    push    r14                        ; flags
    push    r15                        ; out
    sub     rsp, 88                    ; frame: sums@0..48 b1@48 b0@56 n@64 piv@72..88
    mov     rbx, rdi
    mov     r12, rsi
    mov     r13, rdx
    mov     r14d, ecx
    mov     r15, r8
    ; pivots = (x[0], y[0]) stored at [rsp+72]
    movsd   xmm0, [rdi]
    movsd   [rsp+72], xmm0
    movsd   xmm0, [rsi]
    movsd   [rsp+80], xmm0
    mov     rdi, rbx
    mov     rsi, r12
    mov     rdx, r13
    lea     rcx, [rsp]
    lea     r8, [rsp+72]
    call    ols_centered_sums_f64_sse2
    test    eax, eax
    jnz     .zero_ret_code
    cvtsi2sd xmm5, r13
    movsd   [rsp+64], xmm5            ; n_f64
    ; Sxx = sxx - sx'^2/n ; Sxy = sxy - sx'*sy'/n  (shifted: tiny corrections)
    movsd   xmm0, [rsp+8]              ; sx'
    movsd   xmm1, [rsp+16]             ; sy'
    movsd   xmm2, [rsp+24]             ; sxx
    movsd   xmm3, [rsp+32]             ; sxy
    movapd  xmm6, xmm0
    mulsd   xmm6, xmm0
    divsd   xmm6, xmm5
    subsd   xmm2, xmm6                 ; Sxx
    movapd  xmm6, xmm0
    mulsd   xmm6, xmm1
    divsd   xmm6, xmm5
    subsd   xmm3, xmm6                 ; Sxy
    pxor    xmm7, xmm7
    comisd  xmm2, xmm7
    jbe     .singular
    divsd   xmm3, xmm2                 ; b1
    movsd   [rsp+48], xmm3
    ; xbar = cx + sx'/n ; ybar = cy + sy'/n  (divide FIRST, then add pivot)
    movsd   xmm0, [rsp+8]
    divsd   xmm0, xmm5
    addsd   xmm0, [rsp+72]             ; + cx
    movsd   xmm1, [rsp+16]
    divsd   xmm1, xmm5
    addsd   xmm1, [rsp+80]             ; + cy
    movsd   [r15+40], xmm0
    movsd   [r15+48], xmm1
    movapd  xmm6, xmm3
    mulsd   xmm6, xmm0
    subsd   xmm1, xmm6                 ; b0
    movsd   [rsp+56], xmm1
    movsd   [r15], xmm3                ; b1
    movsd   [r15+8], xmm1             ; b0
    pxor    xmm7, xmm7
    movsd   [r15+16], xmm7
    movsd   [r15+24], xmm7
    movsd   [r15+32], xmm7
    test    r14d, OLS_F_STATS
    jz      .nostats
    ; residual pass with (b1, b0) from frame
    movsd   xmm0, [rsp+48]             ; b1
    movsd   xmm1, [rsp+56]             ; b0
    pxor    xmm2, xmm2
    pxor    xmm3, xmm3
    mov     r10, 0x7FFFFFFFFFFFFFFF
    movq    xmm4, r10
    punpcklqdq xmm4, xmm4
    xor     r8, r8
.stats_loop:
    movsd   xmm5, [rbx + r8*8]
    mulsd   xmm5, xmm0
    addsd   xmm5, xmm1
    movsd   xmm6, [r12 + r8*8]
    subsd   xmm6, xmm5
    mulsd   xmm6, xmm6
    NEUMAIER_STEP xmm2, xmm3, xmm6, xmm4, xmm5, xmm7, xmm8
    inc     r8
    cmp     r8, r13
    jb      .stats_loop
    addsd   xmm2, xmm3
    movsd   [r15+16], xmm2             ; sse
    ; sst = syy_shifted - sy'^2/n + correction? SST is translation-invariant:
    ; s((y-cy)^2) - s(y-cy)^2/n is EXACTLY the centered SST.
    movsd   xmm5, [rsp+16]             ; sy'
    movsd   xmm6, [rsp+40]             ; syy'
    movsd   xmm7, [rsp+64]             ; n
    movapd  xmm8, xmm5
    mulsd   xmm8, xmm5
    divsd   xmm8, xmm7
    subsd   xmm6, xmm8                 ; sst
    movsd   [r15+24], xmm6
    pxor    xmm8, xmm8
    comisd  xmm6, xmm8
    jbe     .r2_one
    movapd  xmm8, xmm2
    divsd   xmm8, xmm6
    mov     r10d, 1
    cvtsi2sd xmm9, r10d
    subsd   xmm9, xmm8
    movsd   [r15+32], xmm9
    jmp     .nostats
.r2_one:
    mov     r10d, 1
    cvtsi2sd xmm9, r10d
    movsd   [r15+32], xmm9
.nostats:
    xor     eax, eax
    jmp     .epilogue
.earg:
    mov     eax, OLS_E_ARG
    ret
.zero_ret_code:
    pxor    xmm7, xmm7
    %assign ZI 0
    %rep 7
    movsd   [r15 + ZI], xmm7
    %assign ZI ZI + 8
    %endrep
    jmp     .epilogue_keep_rax
.singular:
    mov     eax, OLS_E_SINGULAR
    pxor    xmm7, xmm7
    %assign ZI 0
    %rep 7
    movsd   [r15 + ZI], xmm7
    %assign ZI ZI + 8
    %endrep
    jmp     .epilogue_keep_rax
.epilogue:
    xor     eax, eax
.epilogue_keep_rax:
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ==============================================================================
; int32_t ols_simple_fit_noint_f64_sse2(const double *x, const double *y,
;                                       size_t n, uint32_t flags,
;                                       double *out /*5*/)
; Through-origin regression (NIST NoInt1/NoInt2): b = s(xy)/s(xx) with pivots
; (0, 0) — no intercept means no centering; the sums stay raw but are
; Neumaier-compensated. R^2 uncentered: 1 - sse/y'y.
; Args    : rdi = x, rsi = y, rdx = n, ecx = flags, r8 = out
; Returns : eax = OLS_OK | OLS_E_ARG | OLS_E_NONFINITE | OLS_E_SINGULAR
; Requires: SSE2
; Memory  : reads x[0..n), y[0..n); writes out[0..5). Errors zero out.
; Clobbers: caller-saved GPRs, xmm0-xmm15; pushes rbx, rbp, r12-r15.
; Numerics: sxx == 0 (all-zero x) -> OLS_E_SINGULAR; sxx overflow (x=1e300
;           constant) -> inf/inf = NaN is REJECTED as singular too: the
;           contract says any non-positive-or-NaN Sxx is singular.
; Cost    : PREDICTED (perf-log #9): two passes ~10-12 cy/el.
; ==============================================================================
global ols_simple_fit_noint_f64_sse2:function
ols_simple_fit_noint_f64_sse2:
    test    rdx, rdx
    jz      .earg
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 88
    mov     rbx, rdi
    mov     r12, rsi
    mov     r13, rdx
    mov     r14d, ecx
    mov     r15, r8
    ; pivots (0, 0)
    pxor    xmm0, xmm0
    movsd   [rsp+72], xmm0
    movsd   [rsp+80], xmm0
    mov     rdi, rbx
    mov     rsi, r12
    mov     rdx, r13
    lea     rcx, [rsp]
    lea     r8, [rsp+72]
    call    ols_centered_sums_f64_sse2
    test    eax, eax
    jnz     .zero5_ret_code
    movsd   xmm3, [rsp+32]             ; sxy
    movsd   xmm2, [rsp+24]             ; sxx
    ; singular if sxx is not > 0 (covers 0, negative, NaN: comisd unordered
    ; sets CF and ZF -> jbe taken for NaN as well)
    pxor    xmm7, xmm7
    comisd  xmm2, xmm7
    jbe     .singular
    divsd   xmm3, xmm2                 ; b
    ; reject NaN slope ONLY (x ~ 1e300 constants: sxx = sxy = +inf, b =
    ; inf/inf = NaN). NaN is unordered with itself -> PF set -> jp taken.
    ; Negative slopes are legal; do NOT use jbe against zero here (that bug
    ; shipped for one commit and was caught by the mutation baseline).
    comisd  xmm3, xmm3
    jp      .singular
    movsd   [rsp+48], xmm3
    movsd   [r15], xmm3
    pxor    xmm7, xmm7
    movsd   [r15+8], xmm7
    movsd   [r15+16], xmm7
    movsd   [r15+24], xmm7
    cvtsi2sd xmm9, r13
    movsd   [r15+32], xmm9
    test    r14d, OLS_F_STATS
    jz      .nostats
    pxor    xmm2, xmm2
    pxor    xmm3, xmm3
    mov     r10, 0x7FFFFFFFFFFFFFFF
    movq    xmm4, r10
    punpcklqdq xmm4, xmm4
    movsd   xmm1, [rsp+48]             ; b
    xor     r8, r8
.stats_loop:
    movsd   xmm5, [rbx + r8*8]
    mulsd   xmm5, xmm1
    movsd   xmm6, [r12 + r8*8]
    subsd   xmm6, xmm5
    mulsd   xmm6, xmm6
    NEUMAIER_STEP xmm2, xmm3, xmm6, xmm4, xmm5, xmm7, xmm8
    inc     r8
    cmp     r8, r13
    jb      .stats_loop
    addsd   xmm2, xmm3
    movsd   [r15+8], xmm2              ; sse
    movsd   xmm6, [rsp+40]             ; syy (raw = shifted by 0)
    movsd   [r15+16], xmm6             ; ySqNorm
    pxor    xmm8, xmm8
    comisd  xmm6, xmm8
    jbe     .r2_one
    divsd   xmm2, xmm6
    mov     r10d, 1
    cvtsi2sd xmm9, r10d
    subsd   xmm9, xmm2
    movsd   [r15+24], xmm9
    jmp     .nostats
.r2_one:
    mov     r10d, 1
    cvtsi2sd xmm9, r10d
    movsd   [r15+24], xmm9
.nostats:
    xor     eax, eax
    jmp     .epilogue
.earg:
    mov     eax, OLS_E_ARG
    ret
.zero5_ret_code:
    pxor    xmm7, xmm7
    %assign ZI 0
    %rep 5
    movsd   [r15 + ZI], xmm7
    %assign ZI ZI + 8
    %endrep
    jmp     .epilogue_keep_rax
.singular:
    mov     eax, OLS_E_SINGULAR
    pxor    xmm7, xmm7
    %assign ZI 0
    %rep 5
    movsd   [r15 + ZI], xmm7
    %assign ZI ZI + 8
    %endrep
    jmp     .epilogue_keep_rax
.epilogue:
    xor     eax, eax
.epilogue_keep_rax:
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ==============================================================================
; int32_t ols_quad_fit_f64_sse2(const double *x, const double *y,
;                               size_t n, uint32_t flags, double *out /*7*/)
; Quadratic fit y = b0 + b1 x + b2 x^2 (NIST Pontius).
; Same three-pass structure as before, with pivot shifting on x (cx = x[0])
; to keep Sxx etc. well-conditioned; z = (x-cx)^2 accumulated directly.
; The solve uses the SHIFTED variable u = x - cx: fit y = a0 + a1 u + a2 u^2,
; then b1 = a1 - 2 a2 cx, b2 = a2, b0 = a0. (Exact algebraic identities.)
; Args    : rdi = x, rsi = y, rdx = n, ecx = flags, r8 = out
; Returns : eax = OLS_OK | OLS_E_ARG (n<3) | OLS_E_NONFINITE | OLS_E_SINGULAR
; Requires: SSE2
; Memory  : reads x[0..n), y[0..n); writes out[0..7) only.
;           out = [b0, b1, b2, sse, sst, r2, 0]
; Clobbers: caller-saved GPRs, xmm0-xmm15; pushes rbx, rbp, r12-r15.
; Cost    : PREDICTED (perf-log #10): three compensated passes ~25-30 cy/el.
; Frame (rsp): 0..24 spare; sums S1@32..56 (su, sy, sz? see below);
;   +32 su +40 sy +48 sz +56 suu +64 suz +72 szz +80 suy +88 szy +96 syy
;   +104 n_f64 +112 a0 +120 a1 +128 a2
; ==============================================================================
global ols_quad_fit_f64_sse2:function
ols_quad_fit_f64_sse2:
    cmp     rdx, 3
    jb      .earg
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 152
    and     rsp, -16
    mov     rbx, rdi
    mov     r12, rsi
    mov     r13, rdx
    mov     r14d, ecx
    mov     r15, r8
    ; cx -> [rsp+136]
    movsd   xmm0, [rdi]
    movsd   [rsp+136], xmm0
    cvtsi2sd xmm0, r13
    movsd   [rsp+104], xmm0            ; n_f64
    ; ---- P1: su, sy, sz where u = x-cx, z = u^2 ----
    pxor    xmm0, xmm0
    pxor    xmm1, xmm1
    pxor    xmm2, xmm2
    pxor    xmm3, xmm3
    pxor    xmm4, xmm4
    pxor    xmm5, xmm5
    mov     r10, 0x7FFFFFFFFFFFFFFF
    movq    xmm10, r10
    punpcklqdq xmm10, xmm10
    movsd   xmm14, [rsp+136]           ; cx
    xor     r8, r8
.p1:
    movsd   xmm11, [rbx + r8*8]        ; x
    FINITE_CHECK xmm11, xmm13, xmm15, .nonfinite
    movsd   xmm12, [r12 + r8*8]        ; y
    FINITE_CHECK xmm12, xmm13, xmm15, .nonfinite
    subsd   xmm11, xmm14               ; u
    movapd  xmm15, xmm11
    mulsd   xmm15, xmm11               ; z = u^2
    movsd   [rsp-8], xmm15             ; red-zone spill of z
    NEUMAIER_STEP xmm0, xmm1, xmm11, xmm10, xmm13, xmm15, xmm9
    movsd   xmm12, [r12 + r8*8]
    NEUMAIER_STEP xmm2, xmm3, xmm12, xmm10, xmm13, xmm15, xmm9
    movsd   xmm11, [rsp-8]
    NEUMAIER_STEP xmm4, xmm5, xmm11, xmm10, xmm13, xmm15, xmm9
    inc     r8
    cmp     r8, r13
    jb      .p1
    movapd  xmm6, xmm0
    addsd   xmm6, xmm1
    movsd   [rsp+32], xmm6             ; su
    movapd  xmm6, xmm2
    addsd   xmm6, xmm3
    movsd   [rsp+40], xmm6             ; sy
    movapd  xmm6, xmm4
    addsd   xmm6, xmm5
    movsd   [rsp+48], xmm6             ; sz
    ; ---- P2: suu, suz, szz ----
    pxor    xmm0, xmm0
    pxor    xmm1, xmm1
    pxor    xmm2, xmm2
    pxor    xmm3, xmm3
    pxor    xmm4, xmm4
    pxor    xmm5, xmm5
    xor     r8, r8
.p2:
    movsd   xmm11, [rbx + r8*8]
    subsd   xmm11, xmm14               ; u
    movapd  xmm15, xmm11
    mulsd   xmm15, xmm11               ; z
    movapd  xmm12, xmm11
    mulsd   xmm12, xmm11               ; u*u
    movsd   [rsp-8], xmm15             ; spill z
    movsd   [rsp-16], xmm12            ; spill u^2
    NEUMAIER_STEP xmm0, xmm1, xmm12, xmm10, xmm13, xmm15, xmm9
    movsd   xmm11, [rbx + r8*8]
    subsd   xmm11, xmm14
    movsd   xmm12, [rsp-8]
    mulsd   xmm11, xmm12               ; u*z
    NEUMAIER_STEP xmm2, xmm3, xmm11, xmm10, xmm13, xmm15, xmm9
    movsd   xmm12, [rsp-8]
    mulsd   xmm12, xmm12               ; z*z
    NEUMAIER_STEP xmm4, xmm5, xmm12, xmm10, xmm13, xmm15, xmm9
    inc     r8
    cmp     r8, r13
    jb      .p2
    movapd  xmm6, xmm0
    addsd   xmm6, xmm1
    movsd   [rsp+56], xmm6             ; suu
    movapd  xmm6, xmm2
    addsd   xmm6, xmm3
    movsd   [rsp+64], xmm6             ; suz
    movapd  xmm6, xmm4
    addsd   xmm6, xmm5
    movsd   [rsp+72], xmm6             ; szz
    ; ---- P3: suy, szy, syy ----
    pxor    xmm0, xmm0
    pxor    xmm1, xmm1
    pxor    xmm2, xmm2
    pxor    xmm3, xmm3
    pxor    xmm4, xmm4
    pxor    xmm5, xmm5
    xor     r8, r8
.p3:
    movsd   xmm11, [rbx + r8*8]
    subsd   xmm11, xmm14               ; u
    movsd   xmm12, [r12 + r8*8]        ; y
    movapd  xmm15, xmm11
    mulsd   xmm15, xmm11               ; z
    mulsd   xmm11, xmm12               ; u*y
    movsd   [rsp-8], xmm15
    NEUMAIER_STEP xmm0, xmm1, xmm11, xmm10, xmm13, xmm15, xmm9
    movsd   xmm11, [rsp-8]
    movsd   xmm12, [r12 + r8*8]
    mulsd   xmm11, xmm12               ; z*y
    NEUMAIER_STEP xmm2, xmm3, xmm11, xmm10, xmm13, xmm15, xmm9
    movsd   xmm12, [r12 + r8*8]
    mulsd   xmm12, xmm12               ; y*y
    NEUMAIER_STEP xmm4, xmm5, xmm12, xmm10, xmm13, xmm15, xmm9
    inc     r8
    cmp     r8, r13
    jb      .p3
    movapd  xmm6, xmm0
    addsd   xmm6, xmm1
    movsd   [rsp+80], xmm6             ; suy
    movapd  xmm6, xmm2
    addsd   xmm6, xmm3
    movsd   [rsp+88], xmm6             ; szy
    movapd  xmm6, xmm4
    addsd   xmm6, xmm5
    movsd   [rsp+96], xmm6             ; syy
    ; ---- centered Gram in u (centering corrections are small) ----
    movsd   xmm9, [rsp+104]            ; n_f64
    movsd   xmm0, [rsp+32]             ; su
    movsd   xmm1, [rsp+40]             ; sy
    movsd   xmm2, [rsp+48]             ; sz
    ; Suu = suu - su^2/n
    movsd   xmm3, [rsp+56]
    movapd  xmm12, xmm0
    mulsd   xmm12, xmm0
    divsd   xmm12, xmm9
    subsd   xmm3, xmm12
    ; Suz = suz - su*sz/n
    movsd   xmm4, [rsp+64]
    movapd  xmm12, xmm0
    mulsd   xmm12, xmm2
    divsd   xmm12, xmm9
    subsd   xmm4, xmm12
    ; Szz = szz - sz^2/n
    movsd   xmm5, [rsp+72]
    movapd  xmm12, xmm2
    mulsd   xmm12, xmm2
    divsd   xmm12, xmm9
    subsd   xmm5, xmm12
    ; Suy = suy - su*sy/n
    movsd   xmm6, [rsp+80]
    movapd  xmm12, xmm0
    mulsd   xmm12, xmm1
    divsd   xmm12, xmm9
    subsd   xmm6, xmm12
    ; Szy = szy - sz*sy/n
    movsd   xmm7, [rsp+88]
    movapd  xmm12, xmm2
    mulsd   xmm12, xmm1
    divsd   xmm12, xmm9
    subsd   xmm7, xmm12
    ; det = Suu*Szz - Suz^2
    movapd  xmm12, xmm3
    mulsd   xmm12, xmm5
    movapd  xmm13, xmm4
    mulsd   xmm13, xmm4
    subsd   xmm12, xmm13
    pxor    xmm14, xmm14
    comisd  xmm12, xmm14
    jbe     .singular
    ; a1 = (Szz*Suy - Suz*Szy)/det
    movapd  xmm13, xmm5
    mulsd   xmm13, xmm6
    movapd  xmm14, xmm4
    mulsd   xmm14, xmm7
    subsd   xmm13, xmm14
    divsd   xmm13, xmm12
    movsd   [rsp+120], xmm13           ; a1
    ; a2 = (Suu*Szy - Suz*Suy)/det
    movapd  xmm14, xmm3
    mulsd   xmm14, xmm7
    movapd  xmm15, xmm4
    mulsd   xmm15, xmm6
    subsd   xmm14, xmm15
    divsd   xmm14, xmm12
    movsd   [rsp+128], xmm14           ; a2
    ; a0 = ybar - a1*ubar - a2*zbar
    movapd  xmm0, xmm1                 ; sy
    divsd   xmm0, xmm9                 ; ybar
    movapd  xmm1, xmm3                 ; (recompute ubar from su)
    movsd   xmm1, [rsp+32]
    divsd   xmm1, xmm9                 ; ubar
    movsd   xmm2, [rsp+48]
    divsd   xmm2, xmm9                 ; zbar
    movsd   xmm3, [rsp+120]            ; a1
    movsd   xmm4, [rsp+128]            ; a2
    movapd  xmm5, xmm3
    mulsd   xmm5, xmm1
    subsd   xmm0, xmm5
    movapd  xmm5, xmm4
    mulsd   xmm5, xmm2
    subsd   xmm0, xmm5                 ; a0
    movsd   [rsp+112], xmm0
    ; back to original variable (FULL expansion):
    ;   y = a0 + a1(x-cx) + a2(x-cx)^2
    ;     = (a0 - a1*cx + a2*cx^2) + (a1 - 2*a2*cx)*x + a2*x^2
    ; b0 = a0 - a1*cx + a2*cx^2 ; b1 = a1 - 2*a2*cx ; b2 = a2
    movsd   xmm2, [rsp+136]            ; cx
    movapd  xmm5, xmm4
    addsd   xmm5, xmm4                 ; 2*a2
    mulsd   xmm5, xmm2                 ; 2*a2*cx
    movsd   xmm3, [rsp+120]            ; a1
    subsd   xmm3, xmm5                 ; b1 = a1 - 2 a2 cx
    movsd   xmm0, [rsp+112]            ; a0
    movsd   xmm5, [rsp+120]            ; a1
    mulsd   xmm5, xmm2                 ; a1*cx
    subsd   xmm0, xmm5                 ; a0 - a1*cx
    movapd  xmm5, xmm2
    mulsd   xmm5, xmm2                 ; cx^2
    mulsd   xmm5, xmm4                 ; a2*cx^2
    addsd   xmm0, xmm5                 ; b0 = a0 - a1*cx + a2*cx^2
    movsd   [r15], xmm0                ; b0
    movsd   [r15+8], xmm3              ; b1
    movsd   [r15+16], xmm4             ; b2
    pxor    xmm7, xmm7
    movsd   [r15+24], xmm7
    movsd   [r15+32], xmm7
    movsd   [r15+40], xmm7
    movsd   [r15+48], xmm7
    test    r14d, OLS_F_STATS
    jz      .nostats
    ; residual pass with b0,b1,b2 in xmm0,xmm1,xmm2
    movapd  xmm1, xmm3                 ; b1
    movapd  xmm2, xmm4                 ; b2
    pxor    xmm3, xmm3
    pxor    xmm4, xmm4
    mov     r10, 0x7FFFFFFFFFFFFFFF
    movq    xmm5, r10
    punpcklqdq xmm5, xmm5
    xor     r8, r8
.stats:
    movsd   xmm6, [rbx + r8*8]         ; x
    movsd   xmm9, [r12 + r8*8]         ; y
    movapd  xmm7, xmm6
    mulsd   xmm7, xmm6                 ; x^2
    mulsd   xmm6, xmm1                 ; b1*x
    addsd   xmm6, xmm0                 ; + b0
    movapd  xmm8, xmm7
    mulsd   xmm8, xmm2                 ; b2*x^2
    addsd   xmm6, xmm8
    subsd   xmm9, xmm6                 ; r
    mulsd   xmm9, xmm9
    NEUMAIER_STEP xmm3, xmm4, xmm9, xmm5, xmm6, xmm7, xmm8
    inc     r8
    cmp     r8, r13
    jb      .stats
    addsd   xmm3, xmm4
    movsd   [r15+24], xmm3             ; sse
    ; sst (centered) = syy - sy^2/n
    movsd   xmm6, [rsp+40]
    movsd   xmm7, [rsp+96]
    movsd   xmm8, [rsp+104]
    movapd  xmm9, xmm6
    mulsd   xmm9, xmm6
    divsd   xmm9, xmm8
    subsd   xmm7, xmm9                 ; sst
    movsd   [r15+32], xmm7
    pxor    xmm8, xmm8
    comisd  xmm7, xmm8
    jbe     .r2_one
    movapd  xmm8, xmm3
    divsd   xmm8, xmm7
    mov     r10d, 1
    cvtsi2sd xmm9, r10d
    subsd   xmm9, xmm8
    movsd   [r15+40], xmm9
    jmp     .nostats
.r2_one:
    mov     r10d, 1
    cvtsi2sd xmm9, r10d
    movsd   [r15+40], xmm9
.nostats:
    xor     eax, eax
    jmp     .epilogue
.earg:
    mov     eax, OLS_E_ARG
    ret
.nonfinite:
    mov     eax, OLS_E_NONFINITE
    pxor    xmm7, xmm7
    %assign ZI 0
    %rep 7
    movsd   [r15 + ZI], xmm7
    %assign ZI ZI + 8
    %endrep
    jmp     .epilogue_keep_rax
.singular:
    mov     eax, OLS_E_SINGULAR
    pxor    xmm7, xmm7
    %assign ZI 0
    %rep 7
    movsd   [r15 + ZI], xmm7
    %assign ZI ZI + 8
    %endrep
    jmp     .epilogue_keep_rax
.epilogue:
    xor     eax, eax
.epilogue_keep_rax:
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
section .note.GNU-stack noalloc noexec nowrite progbits
