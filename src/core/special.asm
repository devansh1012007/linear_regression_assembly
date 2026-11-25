; ------------------------------------------------------------------------------
; special.asm — transcendental kernels for inference (p-values, PLAN COULD).
;   ols_exp_f64_sse2      : e^x via k/r split + Taylor-13; ~1 ulp
;   ols_log_f64_sse2      : ln(x) via s=(m-1)/(m+1), 2sP(s^2); ~1 ulp
;   ols_lgamma_f64_sse2   : ln Gamma(x) — Lanczos g=7 n=9; reflection for
;                           x < 0.5 via sin(pi*x) Taylor; <=2e-13 rel
;   ols_ibeta_f64_sse2    : regularized I_x(a,b) — NR3 betacf (Lentz);
;                           front factor via lgamma/log/exp; ~1e-12
; SSE2 baseline, SysV (xmm0-5 args, xmm0 return). All kernels preserve
; callee-saved GPRs and use xmm0-xmm7/xmm8-15 freely BETWEEN calls only
; where documented; ibeta uses a 48-byte stack frame (a, b, x, 1-x, front,
; m) because its CF loop spans kernel calls that clobber everything.
; Constants: tools/gen_special_consts.py (exact hex; do not hand-edit).
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"

section .rodata align=16
; ---- exp ----
align 16
sc_log2e:       dq 0x3FF71547652B82FE       ; 1.4426950408889634
sc_magic15:     dq 0x4338000000000000       ; 1.5*2^52 round-to-int
sc_ln2hi:       dq 0x3FE62E42FEE00000       ; ln2 hi (round-to-odd split)
sc_ln2lo:       dq 0x3DEA39EF35793C76       ; ln2 - ln2hi (fdlibm pair, exact to 1e-26)
sc_e_ovf:       dq 0x40862E42FEFA39EF       ; 1024*ln2 = 709.782712893384
sc_e_udf:       dq 0xC086232BDD7ABCD2       ; -1022*ln2
sc_e13:         dq 0x3DE6124613A86D09       ; 1/13!
sc_e12:         dq 0x3E21EED8EFF8D898       ; 1/12!
sc_e11:         dq 0x3E5AE64567F544E4       ; 1/11!
sc_e10:         dq 0x3E927E4FB7789F5C       ; 1/10!
sc_e9:          dq 0x3EC71DE3A556C734       ; 1/9!
sc_e8:          dq 0x3EFA01A01A01A01A       ; 1/8!
sc_e7:          dq 0x3F2A01A01A01A01A       ; 1/7!
sc_e6:          dq 0x3F56C16C16C16C17       ; 1/6!
sc_e5:          dq 0x3F81111111111111       ; 1/5!
sc_e4:          dq 0x3FA5555555555555       ; 1/4!
sc_e3:          dq 0x3FC5555555555555       ; 1/3!
sc_e2:          dq 0x3FE0000000000000       ; 1/2!
sc_e1:          dq 0x3FF0000000000000       ; 1
; ---- log ----
align 16
sc_half:        dq 0x3FE0000000000000
sc_sqrt2:       dq 0x3FF6A09E667F3BCD
sc_l10:         dq 0x3FA8618618618618       ; 1/21
sc_l9:          dq 0x3FAAF286BCA1AF28       ; 1/19
sc_l8:          dq 0x3FAE1E1E1E1E1E1E       ; 1/17
sc_l7:          dq 0x3FB1111111111111       ; 1/15
sc_l6:          dq 0x3FB3B13B13B13B14       ; 1/13
sc_l5:          dq 0x3FB745D1745D1746       ; 1/11
sc_l4:          dq 0x3FBC71C71C71C71C       ; 1/9
sc_l3:          dq 0x3FC2492492492492       ; 1/7
sc_l2:          dq 0x3FC999999999999A       ; 1/5
sc_l1:          dq 0x3FD5555555555555       ; 1/3
sc_l0:          dq 0x3FF0000000000000       ; 1
; ---- shared ----
align 16
sc_one:         dq 0x3FF0000000000000
sc_two:         dq 0x4000000000000000
sc_three:       dq 0x4008000000000000
sc_four:        dq 0x4010000000000000
sc_five:        dq 0x4014000000000000
sc_six:         dq 0x4018000000000000
sc_seven:       dq 0x401C000000000000
sc_eight:       dq 0x4020000000000000
sc_zero:        dq 0x0000000000000000
sc_qnan:        dq 0x7FF8000000000001
sc_abs:         dq 0x7FFFFFFFFFFFFFFF
; ---- lgamma (Lanczos g=7 n=9) ----
align 16
sc_lg_c0:       dq 0x3FEFFFFFFFFFF950
sc_lg_c1:       dq 0x40852429B6C30B05
sc_lg_c2:       dq 0xC093AC8E8ED4171B
sc_lg_c3:       dq 0x40881A9661D3B4D8
sc_lg_c4:       dq 0xC06613AE51A32F5D
sc_lg_c5:       dq 0x402903C27F8B9C81
sc_lg_c6:       dq 0xBFC1BCB2992B2855
sc_lg_c7:       dq 0x3EE4F0514E4E324F
sc_lg_c8:       dq 0x3E8435508F3FAEEF
sc_lg_g:        dq 0x401C000000000000       ; g = 7
sc_lg_hln2pi:   dq 0x3FED67F1C864BEB4       ; 0.5*ln(2pi)
sc_lg_lnpi:     dq 0x3FF250D048E7A1BD       ; ln(pi)
sc_pi:          dq 0x400921FB54442D18       ; pi
sc_s5:          dq 0xBE5AE64567F544E4       ; -1/11!
sc_s6:          dq 0x3DE6124613A86D09       ; +1/13!
sc_s7:          dq 0xBD6AE7F3E733B81F       ; -1/15!
sc_s8:          dq 0x3CE952C77030AD4A       ; +1/17!
sc_s9:          dq 0xBC62F49B46814157       ; -1/19!
sc_s10:         dq 0x3BD71B8EF6DCF572       ; +1/21!
sc_s4:          dq 0x3EC71DE3A556C734       ; +1/9!
sc_s3:          dq 0xBF2A01A01A01A01A       ; -1/7!
sc_s2:          dq 0x3F81111111111111       ; +1/5!
sc_s1:          dq 0xBFC5555555555555       ; -1/3!
sc_s0:          dq 0x3FF0000000000000       ; 1
; ---- ibeta ----
align 16
sc_ib_eps:      dq 0x3D719799812DEA11       ; 1e-15
sc_ib_fpm:      dq 0x39B4484BFEEBC2A0       ; 1e-30 (FPMIN guard)
sc_ib_maxit:    dd 300
sc_ib_half:     dq 0x3FE0000000000000

section .text

; ==============================================================================
; double ols_exp_f64_sse2(double x)
; double ols_exp_f64_sse2(double x)
; Args    : xmm0 = x
; Returns : xmm0 = e^x (ovf -> +inf, udf -> 0, NaN -> NaN)
; Requires: SSE2
; Memory  : none (pure)
; Clobbers: rax, xmm0-xmm4
; ==============================================================================
global ols_exp_f64_sse2:function
ols_exp_f64_sse2:
    movsd   xmm1, [rel sc_e_ovf]
    comisd  xmm0, xmm1
    jae     .to_inf
    jp      .to_nan
    movsd   xmm1, [rel sc_e_udf]
    comisd  xmm0, xmm1
    jbe     .to_zero
    ; k = round(x*log2e)
    movsd   xmm1, [rel sc_log2e]
    mulsd   xmm1, xmm0
    movapd  xmm3, xmm1
    addsd   xmm3, [rel sc_magic15]
    subsd   xmm3, [rel sc_magic15]          ; k
    ; fdlibm-style reduction: split X (not k). x_hi = x with the low 27
    ; mantissa bits zeroed (x_hi exact, x_lo = x - x_hi exact by
    ; Sterbenz); r = (x_hi - k*ln2hi) + (x_lo - k*ln2lo): both differences
    ; are exact (k*ln2hi has 21 trailing zero mantissa bits; x_lo is tiny),
    ; so r carries x's full precision. The first drafts (plain hi/lo, then
    ; a k-split) lost 1e-14..2.3e-14 at |x| ~ 300..700.
    movq    rax, xmm0                       ; x bits
    mov     rcx, 0xFFFFFFFFFFF00000        ; zero low 21 mantissa bits
    and     rax, rcx
    movq    xmm4, rax                       ; x_hi
    subsd   xmm0, xmm4                      ; x_lo (exact)
    movapd  xmm5, xmm4
    movq    xmm6, xmm3
    ; xmm3 = k (double). k*ln2hi:
    mulsd   xmm6, [rel sc_ln2hi]
    subsd   xmm5, xmm6                      ; x_hi - k*ln2hi (exact)
    movq    xmm6, xmm3
    mulsd   xmm6, [rel sc_ln2lo]
    subsd   xmm0, xmm6                      ; x_lo - k*ln2lo
    addsd   xmm0, xmm5                      ; r (full precision)
    ; Taylor-13 (Horner): v = c13; v = v*r + c_k for k = 12..0
    ; (c0 = 1 is the constant term — the first draft started v = r*c13,
    ; shifting every coefficient down one power: exp(1) -> 2.34)
    movsd   xmm1, [rel sc_e13]
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_e12]
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_e11]
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_e10]
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_e9]
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_e8]
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_e7]
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_e6]
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_e5]
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_e4]
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_e3]
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_e2]
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_e1]               ; c1 = 1 (r^1 coeff)
    mulsd   xmm1, xmm0
    addsd   xmm1, [rel sc_one]              ; c0 = 1 (constant)
    ; 2^k splice
    cvttsd2si rax, xmm3
    add     rax, 1023
    cmp     rax, 1
    jl      .to_zero
    cmp     rax, 2046
    jg      .to_inf
    shl     rax, 52
    movq    xmm2, rax
    mulsd   xmm1, xmm2
    movapd  xmm0, xmm1
    ret
.to_inf:
    mov     rax, 0x7FF0000000000000
    movq    xmm0, rax
    ret
.to_zero:
    xorps   xmm0, xmm0
    ret
.to_nan:
    mov     rax, 0x7FF8000000000001
    movq    xmm0, rax
    ret

; ==============================================================================
; double ols_log_f64_sse2(double x)     (x > 0; x <= 0 / NaN -> qNaN)
; double ols_log_f64_sse2(double x)
; Args    : xmm0 = x
; Returns : xmm0 = ln(x) (x <= 0 or NaN -> NaN; +inf -> +inf)
; Requires: SSE2
; Memory  : none (pure)
; Clobbers: rax, rdx, rcx, r10, r11, xmm0-xmm7
; ==============================================================================
global ols_log_f64_sse2:function
ols_log_f64_sse2:
    mov     rax, 0x7FF0000000000000
    movq    xmm1, rax
    comisd  xmm0, xmm1
    jae     .inf_or_nan
    pxor    xmm1, xmm1
    comisd  xmm0, xmm1
    jbe     .to_nan
    movq    rax, xmm0
    mov     rdx, rax
    shr     rdx, 52
    and     edx, 0x7FF
    mov     r11, 0x000FFFFFFFFFFFFF
    mov     r10, rax
    and     r10, r11
    test    rdx, rdx
    jnz     .normal
    mov     rcx, 0x4330000000000000
    movq    xmm2, rcx
    mulsd   xmm0, xmm2                      ; subnormal -> normal * 2^52
    movq    rax, xmm0
    mov     rdx, rax
    shr     rdx, 52
    and     edx, 0x7FF
    mov     r10, rax
    and     r10, r11
    sub     rdx, 52
.normal:
    mov     rcx, rdx
    sub     rcx, 1023                       ; e
    mov     rax, 0x3FF0000000000000
    or      rax, r10
    movq    xmm1, rax                       ; m in [1,2)
    comisd  xmm1, [rel sc_sqrt2]
    jb      .m_ok
    mulsd   xmm1, [rel sc_half]
    inc     rcx
.m_ok:
    ; s, u = s^2
    movapd  xmm2, xmm1
    subsd   xmm2, [rel sc_one]
    movapd  xmm3, xmm1
    addsd   xmm3, [rel sc_one]
    divsd   xmm2, xmm3                      ; s
    movapd  xmm4, xmm2
    mulsd   xmm4, xmm4                      ; u
    ; P(u) Horner: v = l10; v = v*u + c for c = l9..l0 (l0 = 1 constant;
    ; the first draft started v = u*l10 — same one-power shift as exp)
    movsd   xmm5, [rel sc_l10]
    mulsd   xmm5, xmm4
    addsd   xmm5, [rel sc_l9]
    mulsd   xmm5, xmm4
    addsd   xmm5, [rel sc_l8]
    mulsd   xmm5, xmm4
    addsd   xmm5, [rel sc_l7]
    mulsd   xmm5, xmm4
    addsd   xmm5, [rel sc_l6]
    mulsd   xmm5, xmm4
    addsd   xmm5, [rel sc_l5]
    mulsd   xmm5, xmm4
    addsd   xmm5, [rel sc_l4]
    mulsd   xmm5, xmm4
    addsd   xmm5, [rel sc_l3]
    mulsd   xmm5, xmm4
    addsd   xmm5, [rel sc_l2]
    mulsd   xmm5, xmm4
    addsd   xmm5, [rel sc_l1]
    mulsd   xmm5, xmm4
    addsd   xmm5, [rel sc_l0]
    mulsd   xmm5, xmm2
    addsd   xmm5, xmm5                      ; ln(m) = 2sP
    cvtsi2sd xmm6, rcx
    movsd   xmm7, [rel sc_ln2hi]
    mulsd   xmm7, xmm6
    addsd   xmm5, xmm7
    movsd   xmm7, [rel sc_ln2lo]
    mulsd   xmm7, xmm6
    addsd   xmm5, xmm7
    movapd  xmm0, xmm5
    ret
.inf_or_nan:
    jp      .to_nan
    ret
.to_nan:
    mov     rax, 0x7FF8000000000001
    movq    xmm0, rax
    ret

; ==============================================================================
; double ols_lgamma_f64_sse2(double x)   (x > 0; else qNaN)
; Direct (x >= 0.5):  lgamma(x) = (x-1/2) ln(x+g-1/2) - (x+g-1/2)
;                       + hln2pi + ln(A),  A = c0 + sum_{k=1..8} ck/(x+k-1)
; Reflection (x < 0.5): lgamma(x) = ln(pi) - ln(sin(pi x)) - lgamma(1-x)
; (the two ln's keep the reflection stable; sin via Taylor on (0, pi/2))
; double ols_lgamma_f64_sse2(double x)
; Args    : xmm0 = x
; Returns : xmm0 = ln Gamma(x) (x <= 0 or NaN -> NaN)
; Requires: SSE2
; Memory  : its own 32-byte frame
; Clobbers: rax, xmm0-xmm7 (log's footprint)
; ==============================================================================
global ols_lgamma_f64_sse2:function
ols_lgamma_f64_sse2:
    ; guards (frame-free)
    mov     rax, 0x7FF0000000000000
    movq    xmm1, rax
    comisd  xmm0, xmm1
    jae     .to_nan
    pxor    xmm1, xmm1
    comisd  xmm0, xmm1
    jbe     .to_nan
    comisd  xmm0, [rel sc_ib_half]
    jae     .lg_core                        ; x >= 0.5: direct
    ; ---- reflection: lgamma(x) = ln(pi) - ln(sin(pi x)) - lgamma(1-x) ----
    push    rbp
    mov     rbp, rsp
    sub     rsp, 16
    movsd   [rbp-16], xmm0                  ; x
    movsd   xmm6, [rel sc_pi]
    mulsd   xmm6, xmm0                      ; w = pi*x in (0, pi/2)
    ; sin(w) = w * P(w^2), P(u) = s0 + s1 u + s2 u^2 + s3 u^3 + s4 u^4
    ;          + s5 u^5  (s_k = (-1)^k/(2k+1)!)  — nested in u = w*w
    movapd  xmm7, xmm6
    mulsd   xmm7, xmm6                      ; u = w*w
    movsd   xmm5, [rel sc_s10]
    mulsd   xmm5, xmm7
    addsd   xmm5, [rel sc_s9]
    mulsd   xmm5, xmm7
    addsd   xmm5, [rel sc_s8]
    mulsd   xmm5, xmm7
    addsd   xmm5, [rel sc_s7]
    mulsd   xmm5, xmm7
    addsd   xmm5, [rel sc_s6]
    mulsd   xmm5, xmm7
    addsd   xmm5, [rel sc_s5]
    mulsd   xmm5, xmm7
    addsd   xmm5, [rel sc_s4]
    mulsd   xmm5, xmm7
    addsd   xmm5, [rel sc_s3]
    mulsd   xmm5, xmm7
    addsd   xmm5, [rel sc_s2]
    mulsd   xmm5, xmm7
    addsd   xmm5, [rel sc_s1]
    mulsd   xmm5, xmm7
    addsd   xmm5, [rel sc_s0]               ; P(u) (11 terms: to u^10)
    mulsd   xmm5, xmm6                      ; sin(w) = w P(u)
    movapd  xmm7, xmm5
    movsd   [rbp-8], xmm7
    movsd   xmm0, [rel sc_one]
    subsd   xmm0, [rbp-16]                  ; t = 1-x >= 0.5
    call    .lg_core                        ; lgamma(t) -> xmm0
    movsd   [rbp-16], xmm0                  ; reuse slot
    movsd   xmm0, [rbp-8]                   ; sin(pi x)
    call    ols_log_f64_sse2                ; ln(sin)
    movsd   xmm1, [rel sc_lg_lnpi]
    subsd   xmm1, xmm0                      ; ln(pi) - ln(sin)
    subsd   xmm1, [rbp-16]                  ; - lgamma(1-x)
    movapd  xmm0, xmm1
    leave
    ret

; ---- Lanczos core (x >= 0.5), self-contained: own frame, [rbp-8]=ln(t) ----
.lg_core:
    push    rbp
    mov     rbp, rsp
    sub     rsp, 32
    movapd  xmm6, xmm0                      ; x
    ; A = c0 + c1/x + c2/(x+1) + ... + c8/(x+7)
    ; (denominators x+k-1 with INTEGER k; the first draft added the
    ;  COEFFICIENT — c1/(x+676.5) — and lost ln(c1/x) ~ 4.9 nats)
    movsd   xmm1, [rel sc_lg_c0]
    movapd  xmm2, xmm6
    movsd   xmm3, [rel sc_lg_c1]
    divsd   xmm3, xmm2                      ; c1/x
    addsd   xmm1, xmm3
    movapd  xmm2, xmm6
    addsd   xmm2, [rel sc_one]
    movsd   xmm3, [rel sc_lg_c2]
    divsd   xmm3, xmm2
    addsd   xmm1, xmm3
    movapd  xmm2, xmm6
    addsd   xmm2, [rel sc_two]
    movsd   xmm3, [rel sc_lg_c3]
    divsd   xmm3, xmm2
    addsd   xmm1, xmm3
    movapd  xmm2, xmm6
    addsd   xmm2, [rel sc_three]
    movsd   xmm3, [rel sc_lg_c4]
    divsd   xmm3, xmm2
    addsd   xmm1, xmm3
    movapd  xmm2, xmm6
    addsd   xmm2, [rel sc_four]
    movsd   xmm3, [rel sc_lg_c5]
    divsd   xmm3, xmm2
    addsd   xmm1, xmm3
    movapd  xmm2, xmm6
    addsd   xmm2, [rel sc_five]
    movsd   xmm3, [rel sc_lg_c6]
    divsd   xmm3, xmm2
    addsd   xmm1, xmm3
    movapd  xmm2, xmm6
    addsd   xmm2, [rel sc_six]
    movsd   xmm3, [rel sc_lg_c7]
    divsd   xmm3, xmm2
    addsd   xmm1, xmm3
    movapd  xmm2, xmm6
    addsd   xmm2, [rel sc_seven]
    movsd   xmm3, [rel sc_lg_c8]
    divsd   xmm3, xmm2
    addsd   xmm1, xmm3                      ; A
    ; t = x + g - 1/2 ; (x - 1/2) — ALL THREE (A, t, x-.5) spilled before
    ; the log calls: ols_log clobbers xmm0-xmm7 wholesale (the first draft
    ; carried A/t/(x-.5) in xmm1/xmm3/xmm4 across the calls — garbage)
    movapd  xmm2, xmm6
    addsd   xmm2, [rel sc_lg_g]
    subsd   xmm2, [rel sc_ib_half]
    movsd   [rbp-8], xmm2                   ; t
    subsd   xmm6, [rel sc_ib_half]
    movsd   [rbp-16], xmm6                  ; x - 1/2
    movsd   [rbp-24], xmm1                  ; A      (frame grows to 32)
    ; ln(t)
    movapd  xmm0, xmm2
    call    ols_log_f64_sse2
    movsd   [rbp-32], xmm0                  ; ln(t)
    ; ln(A)
    movsd   xmm0, [rbp-24]
    call    ols_log_f64_sse2                ; ln(A) -> xmm0
    movapd  xmm5, xmm0
    ; combine: (x-.5)*ln(t) - t + hln2pi + ln(A)
    movsd   xmm0, [rbp-32]
    mulsd   xmm0, [rbp-16]                  ; (x-.5)*ln(t)
    subsd   xmm0, [rbp-8]                   ; - t
    addsd   xmm0, [rel sc_lg_hln2pi]
    addsd   xmm0, xmm5                      ; + ln(A)
    leave
    ret
.to_nan:
    mov     rax, 0x7FF8000000000001
    movq    xmm0, rax
    ret

; ==============================================================================
; double ols_ibeta_f64_sse2(double a, double b, double x)
;   xmm0 = a, xmm1 = b, xmm2 = x.  a,b > 0; x in [0,1].
; NR3 6.4:  bt = exp(lgamma(a+b)-lgamma(a)-lgamma(b) + a ln x + b ln(1-x));
; x > (a+1)/(a+b+2)  ->  I = 1 - I_{1-x}(b,a)  (symmetry swap);
; I = bt * betacf(a,b,x) / a   with betacf the Lentz CF (MAXIT=300,
; EPS=1e-15, FPMIN=1e-30). Non-convergence -> -1.
; Frame (6 slots): [rsp+8]=a [rsp+16]=b [rsp+24]=x [rsp+32]=1-x
;                   [rsp+40]=bt [rsp+48]=m(i) — rbp-based, standard.
; double ols_ibeta_f64_sse2(double a, double b, double x)
; Args    : xmm0 = a, xmm1 = b, xmm2 = x
; Returns : xmm0 = I_x(a,b); non-convergence -> -1; domain err -> NaN
; Requires: SSE2; a,b > 0, x in [0,1]
; Memory  : its own 64-byte frame
; Clobbers: rax, rcx, r14, r15 (saved), xmm0-xmm5
; ==============================================================================
global ols_ibeta_f64_sse2:function
ols_ibeta_f64_sse2:
    ; SysV: save r14/r15 (loop counter + swap flag)
    push    r15
    push    r14
    ; ---- domain guards (frame-free exits pop the two saves) ----
    pxor    xmm3, xmm3
    comisd  xmm2, xmm3
    jbe     .zero_out
    jp      .nan_out
    comisd  xmm2, [rel sc_one]
    jae     .one_out
    movapd  xmm3, xmm0
    comisd  xmm3, xmm3
    jp      .nan_out
    movapd  xmm3, xmm1
    comisd  xmm3, xmm3
    jp      .nan_out
    ; ---- frame: slots a b x 1-x bt ----
    push    rbp
    mov     rbp, rsp
    sub     rsp, 64
    movsd   [rbp-8], xmm0                   ; a
    movsd   [rbp-16], xmm1                  ; b
    movsd   [rbp-24], xmm2                  ; x
    movsd   xmm3, [rel sc_one]
    subsd   xmm3, xmm2
    movsd   [rbp-32], xmm3                  ; 1-x
    ; ---- symmetry swap: x > (a+1)/(a+b+2) -> I = 1 - I_{1-x}(b,a) ----
    xor     r15, r15
    movapd  xmm3, xmm0
    addsd   xmm3, [rel sc_one]
    movapd  xmm4, xmm0
    addsd   xmm4, xmm1
    addsd   xmm4, [rel sc_two]
    divsd   xmm3, xmm4
    comisd  xmm2, xmm3
    jbe     .no_swap
    ; swap a<->b and x<->1-x in the SLOTS
    movupd  xmm0, [rbp-16]
    movupd  xmm1, [rbp-8]
    movsd   [rbp-8], xmm0
    movsd   [rbp-16], xmm1
    movupd  xmm0, [rbp-32]
    movsd   [rbp-24], xmm0
    movsd   xmm3, [rel sc_one]
    subsd   xmm3, xmm0
    movsd   [rbp-32], xmm3
    mov     r15, 1
.no_swap:
    ; ---- bt (accumulator slot [rbp-40]; kernels clobber xmm0-7) ----
    movupd  xmm0, [rbp-8]
    addsd   xmm0, [rbp-16]
    call    ols_lgamma_f64_sse2
    movsd   [rbp-40], xmm0                  ; lgamma(a+b)
    movupd  xmm0, [rbp-8]
    call    ols_lgamma_f64_sse2
    movupd  xmm1, [rbp-40]
    subsd   xmm1, xmm0
    movsd   [rbp-40], xmm1
    movupd  xmm0, [rbp-16]
    call    ols_lgamma_f64_sse2
    movupd  xmm1, [rbp-40]
    subsd   xmm1, xmm0
    movsd   [rbp-40], xmm1
    movupd  xmm0, [rbp-24]
    call    ols_log_f64_sse2
    mulsd   xmm0, [rbp-8]
    movupd  xmm1, [rbp-40]
    addsd   xmm1, xmm0
    movsd   [rbp-40], xmm1
    movupd  xmm0, [rbp-32]
    call    ols_log_f64_sse2
    mulsd   xmm0, [rbp-16]
    movupd  xmm1, [rbp-40]
    addsd   xmm1, xmm0
    movsd   [rbp-40], xmm1                  ; full log-front
    movupd  xmm0, [rbp-40]
    call    ols_exp_f64_sse2
    movsd   [rbp-40], xmm0                  ; bt
    ; ---- Lentz CF (NR3 two-step; state: d [rbp-48], c [rbp-56],
    ; h xmm4; m r14) ----
    ; NR init: c = 1; d = 1 - (a+b)x/(a+1); d = 1/d; h = d.
    ; (The first draft set c = 1/d too, which makes d = 1/c an INVARIANT:
    ;  every del = c*(1/d) = 1 exactly, h never moves, and the loop exits
    ;  on the first convergence test — ibeta(2,3,0.5) -> 2/3.)
    movupd  xmm0, [rbp-8]
    addsd   xmm0, [rbp-16]
    mulsd   xmm0, [rbp-24]
    movupd  xmm1, [rbp-8]
    addsd   xmm1, [rel sc_one]
    divsd   xmm0, xmm1
    movsd   xmm2, [rel sc_one]
    subsd   xmm2, xmm0                      ; 1 - qab x/qap
    movsd   xmm3, [rel sc_one]
    divsd   xmm3, xmm2                      ; d = 1/(...)
    movsd   [rbp-48], xmm3                  ; d slot
    movsd   [rbp-56], xmm2                  ; c slot — WAIT: c must be 1
    ; (fixed below: c = 1.0)
    movsd   xmm2, [rel sc_one]
    movsd   [rbp-56], xmm2                  ; c = 1.0
    movapd  xmm4, xmm3                      ; h = d
    mov     r14, 1
.cf_loop:
    cmp     r14, 300
    jg      .no_converge
    ; EVEN: aa = m(b-m)x/((a+2m-1)(a+2m))  — m from r14 EVERY use:
    ; ols_dbg_lentz clobbers xmm5, the first draft cached m there and the
    ; ODD step ran with a garbage m (the whole "wrong h" mystery)
    cvtsi2sd xmm5, r14
    movupd  xmm0, [rbp-16]
    subsd   xmm0, xmm5
    mulsd   xmm0, xmm5
    mulsd   xmm0, [rbp-24]
    cvtsi2sd xmm5, r14
    movupd  xmm1, [rbp-8]
    addsd   xmm1, xmm5
    addsd   xmm1, xmm5
    subsd   xmm1, [rel sc_one]
    movupd  xmm2, [rbp-8]
    addsd   xmm2, xmm5
    addsd   xmm2, xmm5
    mulsd   xmm1, xmm2
    divsd   xmm0, xmm1
    call    .lentz
    ; ODD: aa = -(a+m)(a+b+m)x/((a+2m)(a+1+2m)) — m reloaded (xmm5 was
    ; clobbered by the lentz call)
    cvtsi2sd xmm5, r14
    movupd  xmm0, [rbp-8]
    addsd   xmm0, xmm5
    movupd  xmm1, [rbp-8]
    addsd   xmm1, [rbp-16]
    addsd   xmm1, xmm5
    mulsd   xmm0, xmm1
    mulsd   xmm0, [rbp-24]
    cvtsi2sd xmm5, r14
    movupd  xmm1, [rbp-8]
    addsd   xmm1, xmm5
    addsd   xmm1, xmm5
    cvtsi2sd xmm5, r14
    movupd  xmm2, [rbp-8]
    addsd   xmm2, [rel sc_one]
    addsd   xmm2, xmm5
    addsd   xmm2, xmm5
    mulsd   xmm1, xmm2
    divsd   xmm0, xmm1
    mov     rax, 0x8000000000000000
    movq    xmm1, rax
    xorpd   xmm0, xmm1
    call    .lentz
    ; convergence on the odd del (xmm2)
    movsd   xmm3, [rel sc_one]
    subsd   xmm3, xmm2
    movq    rax, xmm3
    mov     rcx, 0x7FFFFFFFFFFFFFFF
    and     rax, rcx
    movq    xmm3, rax
    comisd  xmm3, [rel sc_ib_eps]
    jbe     .cf_done
    inc     r14
    jmp     .cf_loop
.cf_done:
    ; ---- result ----
    ; I = bt*h/a ; swapped: 1 - that
    movupd  xmm0, [rbp-40]
    mulsd   xmm0, xmm4                      ; bt*h   (xmm4 = h from loop)
    movupd  xmm1, [rbp-8]
    divsd   xmm0, xmm1
    test    r15, r15
    jz      .result
    movsd   xmm1, [rel sc_one]
    subsd   xmm1, xmm0
    movapd  xmm0, xmm1
.result:
    leave
    pop     r14
    pop     r15
    ret
.zero_out:
    pop     r14
    pop     r15
    xorps   xmm0, xmm0
    ret
.one_out:
    pop     r14
    pop     r15
    movsd   xmm0, [rel sc_one]
    ret
.nan_out:
    pop     r14
    pop     r15
    mov     rax, 0x7FF8000000000001
    movq    xmm0, rax
    ret

; ---- one Lentz half-step (aa in xmm0; h in xmm4; del out xmm2) ----
.lentz:
    movsd   xmm1, [rbp-48]                  ; d
    mulsd   xmm1, xmm0                      ; aa*d
    addsd   xmm1, [rel sc_one]              ; 1+aa*d
    movq    rax, xmm1
    mov     rcx, 0x7FFFFFFFFFFFFFFF
    and     rax, rcx
    movq    xmm3, rax
    comisd  xmm3, [rel sc_ib_fpm]
    jae     .ld1
    movsd   xmm1, [rel sc_ib_fpm]
.ld1:
    movsd   [rbp-48], xmm1                  ; d
    movsd   xmm1, [rbp-56]                  ; c
    divsd   xmm0, xmm1                      ; aa/c
    addsd   xmm0, [rel sc_one]              ; c
    movq    rax, xmm0
    and     rax, rcx
    movq    xmm3, rax
    comisd  xmm3, [rel sc_ib_fpm]
    jae     .ld2
    movsd   xmm0, [rel sc_ib_fpm]
.ld2:
    movsd   [rbp-56], xmm0                  ; c
    movsd   xmm1, [rel sc_one]
    divsd   xmm1, [rbp-48]                  ; 1/d
    movsd   [rbp-48], xmm1
    mulsd   xmm0, xmm1                      ; del = c*(1/d)
    movapd  xmm2, xmm0
    mulsd   xmm4, xmm2                      ; h *= del
    ret

.no_converge:
    mov     rax, 0xBFF0000000000000
    movq    xmm0, rax
    leave
    pop     r14
    pop     r15
    ret

; ==============================================================================
; double ols_t_pvalue_f64_sse2(double t, double df)
;   Two-sided p = P(|T| >= t), T ~ t(df):  p = I_x(df/2, 1/2), x = df/(df+t^2)
;   (identity verified vs mpmath to 1.6e-14 incl. the Cauchy df=1 edge).
;   t = 0 -> 1; NaN -> NaN; df <= 0 -> NaN; |t| = inf -> 0.
; double ols_t_pvalue_f64_sse2(double t, double df)
; Args    : xmm0 = t, xmm1 = df
; Returns : xmm0 = two-sided p; df <= 0 or NaN -> NaN
; Requires: SSE2
; Memory  : its own 32-byte frame (+ ibeta's)
; Clobbers: rax, xmm0-xmm3 (+ ibeta footprint)
; ==============================================================================
global ols_t_pvalue_f64_sse2:function
ols_t_pvalue_f64_sse2:
    ; xmm0 = t, xmm1 = df
    push    rbp
    mov     rbp, rsp
    sub     rsp, 32
    movsd   [rbp-8], xmm0                   ; t
    movsd   [rbp-16], xmm1                  ; df
    ; guards
    pxor    xmm2, xmm2
    comisd  xmm1, xmm2
    jbe     .tp_nan                         ; df <= 0
    jp      .tp_nan
    movapd  xmm3, xmm0
    comisd  xmm3, xmm3
    jp      .tp_nan
    ; |t| via mask
    mov     rax, 0x7FFFFFFFFFFFFFFF
    movq    xmm2, rax
    andpd   xmm0, xmm2                      ; |t|
    movsd   [rbp-8], xmm0
    ; a = df/2 ; x = df/(df + t^2)
    movsd   xmm0, [rbp-16]
    mulsd   xmm0, [rel sc_ib_half]
    movsd   [rbp-24], xmm0                  ; a
    movapd  xmm1, xmm0                      ; save a
    movsd   xmm0, [rbp-8]
    mulsd   xmm0, xmm0                      ; t^2
    addsd   xmm0, [rbp-16]                  ; df + t^2
    movsd   xmm2, [rbp-16]
    divsd   xmm2, xmm0                      ; x
    movapd  xmm0, xmm1                      ; a
    movsd   xmm1, [rel sc_ib_half]          ; b = 1/2
    call    ols_ibeta_f64_sse2
    leave
    ret
.tp_nan:
    mov     rax, 0x7FF8000000000001
    movq    xmm0, rax
    leave
    ret

; ==============================================================================
; double ols_f_pvalue_f64_sse2(double f, double df1, double df2)
;   Upper-tail p = P(F >= f):  p = I_x(df2/2, df1/2), x = df2/(df2 + df1 f).
;   f <= 0 -> 1; NaN -> NaN; df <= 0 -> NaN.
; double ols_f_pvalue_f64_sse2(double f, double df1, double df2)
; Args    : xmm0 = f, xmm1 = df1, xmm2 = df2
; Returns : xmm0 = upper-tail p; f <= 0 -> 1; df <= 0/NaN -> NaN
; Requires: SSE2
; Memory  : its own 32-byte frame (+ ibeta's)
; Clobbers: rax, xmm0-xmm3 (+ ibeta footprint)
; ==============================================================================
global ols_f_pvalue_f64_sse2:function
ols_f_pvalue_f64_sse2:
    ; xmm0 = f, xmm1 = df1, xmm2 = df2
    push    rbp
    mov     rbp, rsp
    sub     rsp, 32
    movsd   [rbp-8], xmm0                   ; f
    movsd   [rbp-16], xmm1                  ; df1
    movsd   [rbp-24], xmm2                  ; df2
    ; guards
    pxor    xmm3, xmm3
    comisd  xmm1, xmm3
    jbe     .fp_nan
    jp      .fp_nan
    comisd  xmm2, xmm3
    jbe     .fp_nan
    jp      .fp_nan
    movapd  xmm3, xmm0
    comisd  xmm3, xmm3
    jp      .fp_nan
    pxor    xmm3, xmm3                      ; re-zero (xmm3 held f for the
    comisd  xmm0, xmm3                      ; NaN check — comparing f vs f
    jbe     .fp_one                         ; made EVERY input hit p=1)
    ; a = df2/2 ; b = df1/2 ; x = df2/(df2 + df1*f)
    movsd   xmm0, [rbp-24]
    mulsd   xmm0, [rel sc_ib_half]
    movsd   [rbp-32], xmm0                  ; a
    movsd   xmm1, [rbp-16]
    mulsd   xmm1, [rel sc_ib_half]          ; b
    movsd   xmm0, [rbp-16]
    mulsd   xmm0, [rbp-8]                   ; df1*f
    addsd   xmm0, [rbp-24]                  ; df2 + df1 f
    movsd   xmm2, [rbp-24]
    divsd   xmm2, xmm0                      ; x
    movsd   xmm0, [rbp-32]                  ; a
    call    ols_ibeta_f64_sse2
    leave
    ret
.fp_one:
    movsd   xmm0, [rel sc_one]
    leave
    ret
.fp_nan:
    mov     rax, 0x7FF8000000000001
    movq    xmm0, rax
    leave
    ret

; ==============================================================================
; double ols_t_qvalue_f64_sse2(double p, double df)
;   t-quantile: the smallest t >= 0 with two-sided P(|T|>=t) <= p.
;   Inverse of ols_t_pvalue_f64_sse2 (which is strictly decreasing in |t|
;   from 1 at t=0 to 0 at inf), so plain bisection on [0, 1e5] converges
;   to a full double in ~60 steps (120 run for margin). Bracket sized so
;   even df=1 is covered to alpha ~ 6e-6 (p(t) ~ 2/(pi*t) there);
;   p-values below P(|T|>=1e5) clamp to 1e5.
; Args    : xmm0 = p (in (0,1]), xmm1 = df (> 0)
; Returns : xmm0 = t critical value; p <= 0 -> +inf; p >= 1 -> 0;
;           NaN/domain -> NaN
; Requires: SSE2
; Memory  : its own 56-byte frame
; Clobbers: rax, xmm0-xmm3 (+ pvalue/ibeta footprint)
; ==============================================================================
global ols_t_qvalue_f64_sse2:function
ols_t_qvalue_f64_sse2:
    push    rbp
    mov     rbp, rsp
    sub     rsp, 56
    movsd   [rbp-8], xmm0                   ; p
    movsd   [rbp-16], xmm1                  ; df
    ; guards: p NaN / df NaN / df <= 0 -> NaN; p <= 0 -> +inf; p >= 1 -> 0
    movapd  xmm2, xmm0
    comisd  xmm2, xmm2
    jp      .tq_nan
    movapd  xmm3, xmm1
    comisd  xmm3, xmm3
    jp      .tq_nan
    pxor    xmm2, xmm2
    comisd  xmm1, xmm2
    jbe     .tq_nan
    comisd  xmm0, xmm2
    jbe     .tq_inf
    movsd   xmm2, [rel sc_one]
    comisd  xmm0, xmm2
    jae     .tq_zero
    ; bisect [lo, hi] = [0, 1e5]. ALL loop state (lo, hi, mid, counter)
    ; lives in THIS frame — never in registers. ols_t_pvalue -> ols_ibeta
    ; clobbers rcx AND xmm4/xmm5 (ibeta keeps h in xmm4 through .lentz),
    ; so the first two drafts (rcx counter; lo/hi in xmm4/xmm5) both
    ; diverged. ADR-0010 bug-class #4, now enforced structurally.
    ; frame: [rbp-8]=p  [rbp-16]=df  [rbp-24]=mid  [rbp-32]=lo
    ;        [rbp-40]=hi  [rbp-48]=counter (frame is 48+8 rounded -> 56)
    pxor    xmm0, xmm0
    movsd   [rbp-32], xmm0                  ; lo = 0
    movsd   xmm0, [rel sc_tq_hi]
    movsd   [rbp-40], xmm0                  ; hi = 1e5
    mov     qword [rbp-48], 120             ; iterations (~60 to a double)
.tq_loop:
    ; mid = 0.5*(lo+hi)
    movsd   xmm0, [rbp-32]
    addsd   xmm0, [rbp-40]
    mulsd   xmm0, [rel sc_ib_half]
    movsd   [rbp-24], xmm0                  ; mid
    ; pmid = t_pvalue(mid, df)
    movapd  xmm1, [rbp-16]
    call    ols_t_pvalue_f64_sse2
    ; p(mid) > target  <=>  mid too small  <=>  lo = mid ; else hi = mid.
    ; (First draft had this inverted — bounds crossed and never converged.)
    comisd  xmm0, [rbp-8]
    ja      .tq_lo_up
    movsd   xmm0, [rbp-24]
    movsd   [rbp-40], xmm0                  ; hi = mid
    jmp     .tq_next
.tq_lo_up:
    movsd   xmm0, [rbp-24]
    movsd   [rbp-32], xmm0                  ; lo = mid
.tq_next:
    sub     qword [rbp-48], 1
    jnz     .tq_loop
    ; return hi: guaranteed P(|T| >= hi) <= target (upper bound)
    movsd   xmm0, [rbp-40]
    leave
    ret
.tq_zero:
    pxor    xmm0, xmm0
    leave
    ret
.tq_inf:
    mov     rax, 0x7FF0000000000000
    movq    xmm0, rax
    leave
    ret
.tq_nan:
    mov     rax, 0x7FF8000000000001
    movq    xmm0, rax
    leave
    ret

; ==============================================================================
; double ols_confint_f64_sse2(double beta, double se, double df, double level)
;   CI = beta ± t_{1-(1-level)/2, df} * se — via the t-quantile kernel:
;   two-sided t-quantile at alpha = 1 - level (the same critical value).
; Args    : xmm0 = beta, xmm1 = se, xmm2 = df, xmm3 = level in (0,1)
; Returns : xmm0 = center; +/- half-width written to *lo/*hi (rdi, rsi)
; Requires: SSE2; se >= 0, df > 0
; Memory  : reads constants; writes 16 bytes at rdi/rsi
; Clobbers: rax, rcx, xmm0-xmm5
; ==============================================================================
global ols_confint_f64_sse2:function
ols_confint_f64_sse2:
    push    rbp
    mov     rbp, rsp
    sub     rsp, 32
    ; save args
    movsd   [rbp-8], xmm0                   ; beta
    movsd   [rbp-16], xmm1                  ; se
    ; alpha = 1 - level ; tcrit = t_qvalue(alpha, df)  [two-sided tail alpha]
    movsd   xmm0, [rel sc_one]
    subsd   xmm0, xmm3                      ; alpha
    movapd  xmm1, xmm2                      ; df
    call    ols_t_qvalue_f64_sse2
    ; half = tcrit * se
    mulsd   xmm0, [rbp-16]
    ; lo = beta - half ; hi = beta + half
    movsd   xmm1, [rbp-8]
    subsd   xmm1, xmm0
    movsd   [rdi], xmm1
    movsd   xmm2, [rbp-8]
    addsd   xmm2, xmm0
    movsd   [rsi], xmm2
    movapd  xmm0, xmm2
    ; (xmm0 = hi is a convenient nonzero return; callers use the pointers)
    leave
    ret

section .rodata
sc_tq_hi:       dq 1.0e5
