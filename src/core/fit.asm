; ------------------------------------------------------------------------------
; fit.asm — ols_fit: the public Tier-1 solver (Phase 2).
; Pipeline: validate -> pivot means (first min(n,256) rows) -> fused shifted
; Gram -> center C = S - s s'/W (intercept only) -> Cholesky -> solve ->
; residual pass (SSE, sigma2) -> se via explicit inverse diag -> cond bound.
; The core never allocates: the caller passes one workspace blob laid out by
; ols_workspace_size (asm-project-delivery §2).
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"
%include "src/macros.inc"

extern ols_gram_scratch_bytes
extern ols_pivot_means_f64_sse2
extern ols_gram_f64_avx2
extern ols_gram_f64_sse2
extern ols_residual_f64_sse2
extern ols_chol_f64_sse2
extern ols_chol_solve_f64_sse2
extern ols_isa
extern ols_qr_build_f64_sse2
extern ols_qr_f64_sse2
extern ols_qr_solve_f64_sse2
extern ols_qr_pivot_f64_sse2

section .text

; ==============================================================================
; size_t ols_workspace_size(size_t n, size_t p, uint32_t method, uint32_t flags)
; Args    : rdi = n, rsi = p, edx = method, ecx = flags
; Returns : rax = bytes (64-aligned). Layout (q = p+1), offsets:
;   C   = ws + 0        (q)
;   G   = ws + q*8      (q*q, FULL matrix mirrored for the inverse)
;   S   = ws + q*8+q*q*8 (q)
;   W   = S + q*8       (1)
;   RSH = W + 8         (q: rhs / solve buffer)
;   XTR = RSH + q*8     (p, CSNE X'r; XTR[p] = sse slack, XTR[p+1] = rsum slack
;                        / QR-path SST temp)
;   classic total = 8*(q*q + 3q + p + 2), rounded to 64.
;   method QR additionally: AUG (n*(q+ic) doubles, 64-aligned), norms (q),
;   work (q+ic) — placed between the classic regions and the gram scratch.
; Requires: nothing (permissive; ols_fit validates).
; Memory  : none.
; Clobbers: rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11.
; v2: total includes ols_gram_scratch_bytes(q) — the packed-row buffers and
; column-major G scratch consumed by the AVX2 Gram kernel (ADR-0005).
; v3: + QR regions (ADR-0007).
; ==============================================================================
global ols_workspace_size:function
ols_workspace_size:
    jmp     ols__ws_size_calc
ols__ws_size_calc:                  ; local body: PC32 link-safe for -shared
    ; layout: C(q) + G(q*q) + S(q) + W(1) + RHS(q) + XTR(p) + sse(1)
    ;       = q*q + 3q + p + 2 doubles  (the old formula undercounted q by
    ;       one slot — XTR/sse ran past the caller's buffer by q*8 bytes,
    ;       corrupting the heap; found by the padded-buffer tail check)
    lea     rax, [rsi+1]            ; q
    mov     r8, rax
    imul    r8, rax                 ; q*q
    lea     r9, [rax*2]
    add     r8, r9                  ; + 2q
    add     r8, rax                 ; + q  (3q total)
    add     r8, rsi                 ; + p
    add     r8, 2                   ; + W + sse
    ; ---- QR extras: AUG n*(q+ic) + norms q + work (q+ic) [+ perm] ----
    cmp     edx, OLS_M_QR
    je      .qr_extras
    cmp     edx, OLS_M_QR_PIVOT
    jne     .no_qr
.qr_extras:
    mov     r10, rdi                ; n (public call passes n directly)
    mov     r9, rcx
    and     r9, OLS_FIT_INTERCEPT   ; ic
    mov     r11, rax
    add     r11, r9                 ; m = q + ic
    imul    r11, r10                ; n * m (AUG doubles)
    add     r8, r11                 ; + AUG
    add     r8, rax                 ; + q  (norms)
    mov     r11, rax
    add     r11, r9                 ; m
    add     r8, r11                 ; + m  (work)
    cmp     edx, OLS_M_QR_PIVOT
    jne     .no_perm
    dec     r11                     ; m-1 perm slots (size_t)
    add     r8, r11
.no_perm:
.no_qr:
    lea     rax, [r8*8]
    add     rax, 63
    and     rax, -64
    push    rbx                     ; save the CALLER's rbx (the old
    mov     rbx, rax                ; mov-then-push code left rbx = classic
                                    ; bytes (0x140 for n=500,p=3) in the
                                    ; caller — the CLI heap crash). rbx is
                                    ; used ONLY between push and pop; the
                                    ; classic-bytes value crosses the call
                                    ; in rbx (callee-saved) and the CALLER's
                                    ; value crosses on the stack.
    lea     rdi, [rsi+1]            ; q (rsi = p; consumed before the call)
    call    ols_gram_scratch_bytes wrt ..plt   ; (caller-saved regs die here)
    ; rax = scratch bytes; rbx = classic bytes (survived the call, being
    ; callee-saved); [rsp] = the caller's rbx
    ; +64: the gram kernel aligns its PKG base up to 64 INSIDE the scratch;
    ; ols_fit hands it an aligned tail pointer, and the slack covers the
    ; worst-case align-up when the caller's ws is only 16-aligned (numpy/
    ; malloc) — without it GSCR wrote past ws and corrupted the heap.
    add     rax, 64
    add     rax, rbx                ; + classic bytes
    pop     rbx                     ; restore the CALLER's rbx
    and     rax, -64
    ret

; ==============================================================================
; int32_t ols_fit(const ols_problem_t *prob, ols_result_t *res,
;                 void *workspace, size_t ws_bytes)
; Args    : rdi = prob, rsi = res, rdx = workspace, rcx = ws_bytes
; Returns : eax = OLS_OK | OLS_E_*; on error res scalars are zeroed,
;           rank = -1, method_used/isa_used filled, pointers untouched.
; Requires: SSE2 min; AVX2 gram when ols_isa() >= 1.
; Memory  : reads prob/res pointers + X/y/w; writes res scalars, beta[0..p),
;           se[0..p+1) (when stats on and se != NULL); workspace per layout.
; Clobbers: caller-saved + rbx,rbp,r12-r15 (pushed/popped).
; Numerics: pivot-shifted Gram; C = S - s s'/W; chol eps_rel = 8q*u;
;           sigma2 = sse / max(1, n - q); se_j = sqrt(sigma2 * inv_jj);
;           cond_est = (maxR_ii/minR_ii)^2 (lower bound).
; Cost    : PREDICTED (perf-log #17): gram dominates; q<=32 solve+stats
;           O(q^3/3 + q^3) scalar ~ 100-300 kcy — negligible vs the sweep.
; Frame (rsp, 256 B):
;   [0] prob [8] res [16] ws [24] wsbytes [32] q [40] p [48] n [56] flags
;   [64] method [72] intercept [80] W [88] sse [96] xtr [104] nb
;   [112] GA args (80 B) -> [112..192)   [192] scratch [200] isa [208] eps
; ==============================================================================
global ols_fit:function
ols_fit:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 352
    and     rsp, -16
    mov     [rsp+0], rdi
    mov     [rsp+8], rsi
    mov     [rsp+16], rdx
    mov     [rsp+24], rcx
    ; ---------- validation ----------
    test    rdi, rdi
    jz      .earg
    test    rsi, rsi
    jz      .earg
    test    rdx, rdx
    jz      .earg
    cmp     dword [rdi+PRB_STRUCT_SIZE], PRB_SIZEOF   ; u32 fields: dword!
    jb      .earg
    cmp     dword [rdi+PRB_VERSION], 1
    jne     .earg
    mov     r12, [rdi+PRB_N]        ; n
    mov     r13, [rdi+PRB_P]        ; p
    test    r12, r12
    jz      .earg
    test    r13, r13
    jz      .earg
    mov     rax, [rdi+PRB_X]
    test    rax, rax
    jz      .earg
    mov     rax, [rdi+PRB_Y]
    test    rax, rax
    jz      .earg
    mov     rax, [rdi+PRB_LDX]
    cmp     rax, r13
    jb      .earg
    mov     r14d, [rdi+PRB_FLAGS]
    mov     [rsp+56], r14
    mov     eax, r14d
    and     eax, OLS_FIT_INTERCEPT
    mov     [rsp+72], rax           ; intercept flag (0/1)
    mov     eax, [rdi+PRB_METHOD]
    mov     [rsp+64], rax
    ; methods: AUTO/GRAM_CHOL/GRAM_CSNE/QR/QR_PIVOT supported; CG reserved
    cmp     eax, OLS_M_QR_PIVOT
    ja      .earg                   ; > QR_PIVOT rejected (CG reserved)
    ; ridge penalty must be >= 0 (negative lambda would break PD-ness and
    ; fail deep inside chol with a misleading SINGULAR; NaN also rejected)
    movsd   xmm0, [rdi+PRB_LAMBDA]
    pxor    xmm1, xmm1
    ucomisd xmm0, xmm1              ; lambda vs 0
    jb      .earg                   ; lambda < 0 or NaN -> OLS_E_ARG
    ; underdetermined: n <= p + intercept
    mov     rcx, r13
    add     rcx, [rsp+72]
    cmp     r12, rcx
    jbe     .eunder
    ; workspace size (rdi = n: the QR extras scale with n — the old call
    ; passed the prob pointer here, which the body never read until the
    ; QR block started reading it as n; r12 = n from validation — [rsp+48]
    ; is not initialized until AFTER this check)
    mov     rdi, r12
    mov     rsi, r13
    xor     edx, edx
    mov     edx, [rsp+64]
    mov     ecx, [rsp+56]
    call    ols_workspace_size
    cmp     [rsp+24], rax
    jb      .ews
    ; ---------- pointers into the workspace ----------
    lea     r15, [r13+1]            ; q
    mov     [rsp+32], r15
    mov     [rsp+40], r13
    mov     [rsp+48], r12
    ; ws layout offsets kept in callee-saved regs for the whole body:
    ;   rbx = G   r12' = ... (r12 is n; recompute after use) — use frame slots
    mov     rbx, [rsp+16]           ; ws
    ; C at ws+0
    lea     rax, [r15*8]
    ; G at ws + q*8
    mov     rcx, rax                ; q*8
    lea     rdx, [rbx + rcx]        ; G
    mov     [rsp+112+0], rdx        ; reuse GA area later; G ptr -> frame [192]
    mov     [rsp+232], rdx
    ; S at G + q*q*8
    mov     rax, r15
    imul    rax, r15
    lea     rax, [rax*8]
    add     rax, rdx
    mov     [rsp+240], rax          ; S
    ; W at S + q*8
    lea     rax, [rax + r15*8]
    mov     [rsp+248], rax          ; W
    ; RHS at W + 8
    add     rax, 8
    mov     [rsp+256], rax          ; RHS
    ; XTR at RHS + q*8
    lea     rax, [rax + r15*8]
    mov     [rsp+264], rax          ; XTR
    ; ---------- pivots ----------
    ; intercept: c = weighted means of the first min(n,256) rows (bounds the
    ; centering cancellation). NO intercept: c = 0 — the raw Gram is required
    ; (a shifted solve without an intercept column returns slopes for
    ; (x - c), not x; that was the no-intercept failure).
    cmp     qword [rsp+72], 0
    je      .pivots_zero
    mov     rax, 256
    cmp     r12, rax
    jae     .nb_set
    mov     rax, r12
.nb_set:
    mov     [rsp+104], rax          ; nb
    mov     rax, [rsp+0]
    mov     rdi, [rax+PRB_X]
    mov     rsi, [rax+PRB_LDX]
    lea     rsi, [rsi*8]
    mov     rdx, [rax+PRB_Y]
    mov     rcx, [rax+PRB_W]
    mov     r8, [rsp+104]
    mov     r9, r13
    sub     rsp, 16
    mov     [rsp], rbx              ; 7th arg: c out (ws+0)
    call    ols_pivot_means_f64_sse2 wrt ..plt
    add     rsp, 16
    test    eax, eax
    jnz     .err_code
    jmp     .pivots_done
.pivots_zero:
    ; c = 0 (q doubles at ws+0)
    mov     rcx, r15
    pxor    xmm0, xmm0
    xor     rax, rax
.pz:
    cmp     rax, rcx
    jae     .pivots_done
    movsd   [rbx + rax*8], xmm0
    inc     rax
    jmp     .pz
.pivots_done:
    ; QR path for BOTH QR (3) and QR_PIVOT (4) — the old `cmp OLS_M_QR;
    ; jne .gram_path` sent method 4 down the GRAM path, whose chol
    ; refused the rank-deficient G (.fsing): the mystery rc=-7/rank=-1
    cmp     dword [rsp+64], OLS_M_QR
    jb      .gram_path              ; methods 0-2: gram
    cmp     dword [rsp+64], OLS_M_QR_PIVOT
    ja      .gram_path              ; > 4: never reached (validated earlier)
    ; ---------- QR path (ADR-0007; pivot variant since phase 4b) ----------
    ; QA args reuse the GA area [rsp+112..224) (the gram is skipped here).
    ; AUG = align64(ws + round64(classic bytes)); norms/work packed after.
    mov     rax, [rsp+16]           ; ws
    mov     rcx, [rsp+32]           ; q
    mov     r8, rcx
    imul    r8, rcx
    lea     r9, [rcx*2]
    add     r8, r9
    add     r8, rcx
    add     r8, [rsp+40]            ; + p
    add     r8, 2                   ; classic doubles
    lea     rdx, [r8*8]
    add     rdx, 63
    and     rdx, -64
    lea     rdi, [rax + rdx]        ; AUG (64-aligned)
    mov     [rsp+168], rdi          ; QA_AUG
    mov     rax, [rsp+48]           ; n
    mov     rcx, [rsp+32]           ; q
    add     rcx, [rsp+72]           ; m = q + ic
    imul    rax, rcx
    lea     rdx, [rdi + rax*8]      ; norms = AUG + n*m
    mov     [rsp+200], rdx          ; QA_NORMS
    lea     rdx, [rdx + rcx*8]      ; work = norms + m? work needs m... q+ic
    ; work = norms + q doubles is enough? The qr kernel zeroes/uses work[k]
    ; for k in j+1..m-1 -> m doubles. norms region is q doubles; place work
    ; at norms + q*8 and size the region m in ws_size (m <= q+1 fits).
    mov     rcx, [rsp+32]
    lea     rdx, [rdx + rcx*8]
    mov     [rsp+208], rdx          ; QA_WORK
    ; ---- fill the QA block ----
    lea     rdi, [rsp+112]
    mov     rax, [rsp+0]
    mov     rdx, [rax+PRB_X]
    mov     [rdi+QA_X], rdx
    mov     rdx, [rax+PRB_LDX]
    lea     rdx, [rdx*8]
    mov     [rdi+QA_LDX8], rdx
    mov     rdx, [rax+PRB_Y]
    mov     [rdi+QA_Y], rdx
    mov     rdx, [rax+PRB_W]
    mov     [rdi+QA_W], rdx
    mov     rdx, [rsp+48]
    mov     [rdi+QA_N], rdx
    mov     rdx, [rsp+40]
    mov     [rdi+QA_P], rdx
    mov     [rdi+QA_C], rbx         ; pivots (C buffer)
    mov     rdx, [rsp+240]
    mov     [rdi+QA_S], rdx
    mov     rdx, [rsp+248]
    mov     [rdi+QA_WOUT], rdx
    ; SST -> XTR[p+1] (the rsum slot; CSNE never runs on the QR path)
    mov     rdx, [rsp+264]
    mov     rcx, [rsp+40]
    lea     rcx, [rcx*8]
    lea     rdx, [rdx + rcx + 8]
    mov     [rdi+QA_SST], rdx
    mov     rcx, [rsp+72]
    mov     [rdi+QA_INTERCEPT], rcx
    ; ---- build ----
    lea     rdi, [rsp+112]
    call    ols_qr_build_f64_sse2 wrt ..plt
    test    eax, eax
    jnz     .err_code
    ; ---- factor (rcond: caller value or default 8*dc*u) ----
    mov     rax, [rsp+0]
    movsd   xmm0, [rax+PRB_RCOND]
    mov     rcx, [rsp+40]           ; p
    mov     rdx, [rsp+72]           ; ic
    lea     rcx, [rcx + rdx]        ; dc = p + ic
    pxor    xmm1, xmm1
    ucomisd xmm0, xmm1
    jp      .rc_default
    ja      .rc_have
.rc_default:
    cvtsi2sd xmm0, rcx
    mov     rax, 8
    cvtsi2sd xmm1, rax
    mulsd   xmm0, xmm1              ; 8*dc
    mov     rax, 0x3CA0000000000000 ; 2^-52
    movq    xmm1, rax
    mulsd   xmm0, xmm1
.rc_have:
    mov     rdi, [rsp+168]          ; AUG
    mov     rsi, [rsp+48]           ; n
    mov     rdx, [rsp+32]           ; q
    add     rdx, [rsp+72]           ; m
    mov     rcx, rdx                ; stride = m
    mov     r8, [rsp+208]           ; work
    mov     qword [rsp+216], 0       ; perm stash = 0 (identity) unless the
    ; pivot path sets it — .braw dereferences a garbage non-zero stash on
    ; the PLAIN QR path and segfaulted (test_fit regression)
    cmp     dword [rsp+64], OLS_M_QR_PIVOT
    jne     .qr_nopivot
    ; perm = work + m doubles (ws_size reserves m-1 more slots for it)
    mov     rax, [rsp+208]          ; work
    mov     rcx, [rsp+32]
    add     rcx, [rsp+72]           ; m
    lea     r9, [rax + rcx*8]       ; perm
    mov     [rsp+216], r9           ; stash for the unpermute loops
    ; perm = identity (the kernel swaps entries; it does not build them)
    xor     r11, r11
.qr_perm_init:
    cmp     r11, rcx
    jae     .qr_perm_init_done
    mov     [r9 + r11*8], r11
    inc     r11
    jmp     .qr_perm_init
.qr_perm_init_done:
    ; rcond sentinel: negative pins column 0 (ones column) — the caller's
    ; negative rcond passes through; a non-negative one becomes -1 (pin
    ; with the default threshold |rcond| = 1 in the kernel's rank check
    ; would refuse everything, so the kernel uses 8*dc*u when rcond == -1
    ; exactly; simpler: pass -rcond when rcond>0, i.e. -|rcond|)
    cmp     dword [rsp+64], OLS_M_QR_PIVOT
    jne     .qr_nopivot             ; plain QR: original args, no sentinel
    pxor    xmm5, xmm5
    comisd  xmm5, xmm0
    ja      .qr_call_pivot          ; rcond < 0: caller pin request (as-is)
    ; rcond >= 0: pin with max(rcond, 8*dc*u) as the threshold. (The first
    ; draft sent -rcond — for rcond = 0 that is -0.0, which is NOT < 0 to
    ; comisd, so pinning never engaged AND the threshold was -0.0: nothing
    ; was ever rank-deficient. -0.0 == 0.0 in compares.)
    movsd   xmm4, xmm0              ; rcond
    ; default = 8*dc*u
    mov     rcx, [rsp+40]
    add     rcx, [rsp+72]           ; dc
    cvtsi2sd xmm1, rcx
    mov     rax, 8
    cvtsi2sd xmm2, rax
    mulsd   xmm1, xmm2
    mov     rax, 0x3CA0000000000000
    movq    xmm2, rax               ; 2^-52
    mulsd   xmm1, xmm2              ; 8*dc*u
    comisd  xmm4, xmm1
    jae     .qr_pin_have            ; rcond >= default: use rcond
    movapd  xmm4, xmm1              ; else default
.qr_pin_have:
    ; NEGATE via the SIGN MASK, not by xoring with the value: xorpd(x, x)
    ; is +0.0, not -x — the first draft xor'ed xmm0 with xmm4 (the
    ; threshold) and, when rcond <= default, xmm4 == xmm0 bitwise, so
    ; the kernel received +0.0 (no pin, no rank detection) — the whole
    ; "sentinel produces 0" mystery
    mov     rax, 0x8000000000000000
    movq    xmm3, rax                ; sign mask (both lanes)
    movapd  xmm0, xmm4               ; xmm0 = max(rcond, default) ...
    xorpd   xmm0, xmm3               ; ... negated: pin + |threshold|
    ; RE-SET the integer args: the sentinel math above clobbers rcx (it
    ; was the stride) and possibly others — the first draft called with a
    ; garbage stride and the factorization silently misindexed Aug
    mov     rdi, [rsp+168]          ; AUG
    mov     rsi, [rsp+48]           ; n
    mov     rdx, [rsp+32]           ; q
    add     rdx, [rsp+72]           ; m
    mov     rcx, rdx                ; stride = m
    mov     r8, [rsp+208]           ; work
    mov     r9, [rsp+216]           ; perm
.qr_call_pivot:
    call    ols_qr_pivot_f64_sse2 wrt ..plt
    jmp     .qr_factored
.qr_nopivot:
    call    ols_qr_f64_sse2 wrt ..plt
.qr_factored:
    mov     rcx, [rsp+40]
    add     rcx, [rsp+72]           ; dc = p + ic (recomputed: the call
    ; setup loaded ecx with the stride m = dc+1, and the first draft
    ; compared rank against that — every full-rank fit reported SINGULAR)
    cmp     eax, ecx
    jl      .qr_singular
    ; ---- solve -> beta_s in RHS ----
    mov     rdi, [rsp+168]
    mov     rsi, [rsp+32]
    add     rsi, [rsp+72]           ; m
    mov     rdx, rsi                ; stride
    mov     rcx, [rsp+256]          ; RHS
    call    ols_qr_solve_f64_sse2 wrt ..plt
    ; ---- beta_raw: position pos = j+ic holds ORIGINAL column perm[pos];
    ;      beta[perm[pos]-ic] = beta_s[pos] / norms[perm[pos]] ----
    ; (unpivoted path: perm == identity, same formulas)
    mov     rax, [rsp+8]
    mov     rdi, [rax+RES_BETA]
    mov     rcx, [rsp+168]          ; AUG (base for the sse tail later)
    mov     r8, [rsp+200]           ; norms (ORIGINAL column order)
    mov     r9, [rsp+72]            ; ic
    mov     r11, [rsp+256]          ; RHS
    xor     r10, r10                ; j
.braw:
    cmp     r10, r13
    jae     .braw_done
    lea     rdx, [r10 + r9]         ; pos = j + ic
    ; src = perm[pos] (identity when no pivoting: perm ptr [rsp+216] = 0)
    mov     rax, rdx
    mov     rsi, [rsp+216]          ; perm (0 = identity)
    test    rsi, rsi
    jz      .braw_ident
    mov     rax, [rsi + rdx*8]      ; perm[pos]
.braw_ident:
    movsd   xmm0, [r11 + rdx*8]     ; beta_s[pos]
    divsd   xmm0, [r8 + rax*8]      ; / norms[orig col]
    ; dest index: orig col - ic (design columns start at ic)
    sub     rax, r9
    movsd   [rdi + rax*8], xmm0     ; res->beta[orig-ic]
    movsd   [r11 + r10*8], xmm0     ; RHS[j] (intercept block reads it)
    inc     r10
    jmp     .braw
.braw_done:
    ; ---- sse byproduct: tail sum over i = dc..n-1 of Aug[i][m-1]^2
    ; (Neumaier; the residual TAIL, not R_dd^2 — see mp_qr_ols) ----
    mov     r11, [rsp+32]           ; q
    add     r11, [rsp+72]           ; m
    dec     r11                     ; col = m-1
    pxor    xmm0, xmm0              ; s
    pxor    xmm1, xmm1              ; c
    mov     rax, 0x7FFFFFFFFFFFFFFF
    movq    xmm7, rax
    punpcklqdq xmm7, xmm7           ; abs mask
    mov     r9, [rsp+72]
    lea     r10, [r13 + r9]         ; i = dc = p + ic
.sse_tail:
    cmp     r10, [rsp+48]           ; i vs n
    jae     .sse_tail_done
    mov     rax, r10
    mov     rdx, [rsp+32]           ; q
    add     rdx, [rsp+72]           ; m
    imul    rax, rdx                ; i * m
    add     rax, r11                ; + (m-1) = &Aug[i][m-1]
    movsd   xmm2, [rcx + rax*8]     ; Aug[i][m-1]
    mulsd   xmm2, xmm2
    movapd  xmm3, xmm0
    addsd   xmm3, xmm2              ; t
    movapd  xmm4, xmm0
    andpd   xmm4, xmm7
    movapd  xmm5, xmm2
    andpd   xmm5, xmm7
    cmpnltpd xmm4, xmm5             ; mask
    movapd  xmm5, xmm0
    subsd   xmm5, xmm3
    addsd   xmm5, xmm2              ; e
    andpd   xmm5, xmm4
    addsd   xmm1, xmm5
    movapd  xmm0, xmm3              ; s = t
    inc     r10
    jmp     .sse_tail
.sse_tail_done:
    addsd   xmm0, xmm1
    mov     rax, [rsp+264]          ; XTR
    mov     rdx, [rsp+40]
    movsd   [rax + rdx*8], xmm0     ; XTR[p] = sse slack (the chol path's
    ; residual pass uses the same slot; .stats_qr reloads both from here —
    ; writing res->sse directly here was wiped by the stats-zero block)
    jmp     .beta_done
.qr_singular:
    ; QR_PIVOT reports rank + refuses (the caller learns WHICH columns
    ; are independent via perm; the unpivoted QR keeps E_SINGULAR).
    ; rank = eax (first deficient position); res->rank filled, error out.
    mov     rdx, [rsp+8]
    mov     dword [rdx+RES_RANK], eax
    mov     eax, OLS_E_SINGULAR
    jmp     .err_code
.gram_path:
    ; ---------- gram ----------
    ; GA args at [rsp+112..192)
    mov     rax, [rsp+0]
    lea     rdi, [rsp+112]
    mov     rdx, [rax+PRB_X]
    mov     [rdi+GA_X], rdx
    mov     rdx, [rax+PRB_LDX]
    lea     rdx, [rdx*8]
    mov     [rdi+GA_LDX8], rdx
    mov     rdx, [rax+PRB_Y]
    mov     [rdi+GA_Y], rdx
    mov     rdx, [rax+PRB_W]
    mov     [rdi+GA_W], rdx
    mov     [rdi+GA_N], r12
    mov     [rdi+GA_P], r13
    mov     [rdi+GA_C], rbx         ; C (ws+0)
    mov     rdx, [rsp+232]
    mov     [rdi+GA_G], rdx
    mov     rdx, [rsp+240]
    mov     [rdi+GA_S], rdx
    mov     rdx, [rsp+248]
    mov     [rdi+GA_WOUT], rdx
    ; scratch = 64-ALIGNED ws tail: [aligned, aligned + scratch_bytes) with
    ; aligned = align64(ws + total - 64 - scratch_bytes). The workspace_size
    ; formula reserves classic + scratch + 64 (rounded), so the aligned
    ; region always fits. (The old code passed ws + total - scratch_bytes
    ; UNALIGNED: the kernel's internal align-up then wrote past ws — the
    ; multi-fit heap corruption.)
    mov     rdi, r15                ; q
    call    ols_gram_scratch_bytes wrt ..plt   ; cross-object: PLT32
    mov     [rsp+280], rax          ; scratch_bytes temp on the frame (not
    ; [rsp+192] = GA_SCRATCH; not rbx = ws/C)
    mov     rdi, [rsp+48]           ; n — the ws body reads n from rdi for
    mov     rsi, r13                ; the QR AUG sizing (the old call passed
    xor     edx, edx                ; the prob pointer here; the body never
    mov     edx, [rsp+64]           ; read it, so nothing else changes)
    mov     ecx, [rsp+56]
    call    ols__ws_size_calc
    mov     rdx, [rsp+16]           ; ws
    sub     rax, 64                 ; total - align slack
    sub     rax, [rsp+280]          ; - scratch_bytes
    add     rdx, rax
    add     rdx, 63
    and     rdx, -64                ; align up to 64
    lea     rdi, [rsp+112]
    mov     [rdi+GA_SCRATCH], rdx
    mov     rax, [rsp+280]
    mov     [rdi+GA_SCRATCH_BYTES], rax
    ; choose ISA (isa value is NOT stored — only used for the branch; the
    ; result's isa_used is filled later from a fresh ols_isa call. Slot 232
    ; holds the G pointer and must survive.)
    call    ols_isa wrt ..plt
    cmp     eax, OLS_ISA_AVX2
    jb      .gram_sse2
    lea     rdi, [rsp+112]
    call    ols_gram_f64_avx2 wrt ..plt
    jmp     .gram_done
.gram_sse2:
    lea     rdi, [rsp+112]
    call    ols_gram_f64_sse2 wrt ..plt
.gram_done:
    test    eax, eax
    jnz     .err_code
    ; ---------- centering (intercept only): G = S - s s'/W ----------
    cmp     qword [rsp+72], 0
    je      .nocenter
    ; W = *Wptr
    mov     rax, [rsp+248]
    movsd   xmm0, [rax]             ; W
    ; for j in 0..q-1: for k in j..q-1: G[j][k] -= s[j]*s[k]/W
    xor     r12, r12                ; j (n no longer needed; saved at [rsp+48])
.ct_j:
    cmp     r12, r15
    jae     .ct_done
    mov     rax, [rsp+240]          ; S
    movsd   xmm4, [rax + r12*8]     ; s[j]
    xor     r13, r13                ; k
.ct_k:
    cmp     r13, r15
    jae     .ct_j_next
    mov     rax, [rsp+240]
    movsd   xmm5, [rax + r13*8]     ; s[k]
    mulsd   xmm5, xmm4
    divsd   xmm5, xmm0
    ; G[j][k] -= that
    mov     rax, [rsp+232]          ; G
    mov     rcx, r12
    imul    rcx, r15
    add     rcx, r13
    movsd   xmm6, [rax + rcx*8]
    subsd   xmm6, xmm5
    movsd   [rax + rcx*8], xmm6
    inc     r13
    jmp     .ct_k
.ct_j_next:
    inc     r12
    jmp     .ct_j
.ct_done:
.nocenter:
    ; ---------- ridge (lambda > 0): G_jj += lambda, slopes only ----------
    ; Contract: minimize sum w(y-b0-Xb)^2 + lambda*||b||^2 (intercept never
    ; penalized). The factored block is the W-scaled centered gram C, so
    ; this adds lambda*I to (Xc'WXc + lambda*I) exactly. With lambda > 0
    ; the se/t statistics come from the REGULARIZED inverse (ridge
    ; covariance, not OLS inference) while sse/sst/r2/f_stat stay
    ; unpenalized (residual pass) — documented in include/ols.h; the
    ; mpmath ridge differential pins the solver.
    mov     rax, [rsp+0]
    movsd   xmm0, [rax+PRB_LAMBDA]
    pxor    xmm1, xmm1
    ucomisd xmm0, xmm1
    jbe     .no_ridge               ; lambda <= 0: plain OLS path
    mov     r13, [rsp+40]           ; p (clobbered by the centering loops)
    xor     rcx, rcx
.ridge_j:
    cmp     rcx, r13
    jae     .no_ridge
    mov     rax, rcx
    imul    rax, r15
    add     rax, rcx                ; G[j][j] (stride q)
    mov     rdx, [rsp+232]          ; G
    movsd   xmm2, [rdx + rax*8]
    addsd   xmm2, xmm0
    movsd   [rdx + rax*8], xmm2
    inc     rcx
    jmp     .ridge_j
.no_ridge:
    ; ---------- RHS[j] = G[j][p]; chol+solve the leading p x p block ----
    mov     r13, [rsp+40]           ; p
    mov     r12, [rsp+232]          ; G
    xor     rcx, rcx
.rhs:
    cmp     rcx, r13
    jae     .rhs_done
    mov     rax, rcx
    imul    rax, r15                ; j*q
    add     rax, r13                ; + p (y column)
    movsd   xmm0, [r12 + rax*8]
    mov     rdx, [rsp+256]          ; RHS
    movsd   [rdx + rcx*8], xmm0
    inc     rcx
    jmp     .rhs
.rhs_done:
    ; chol on G's leading p x p block with stride q
    mov     rdi, r12                ; G
    mov     rsi, r13                ; order p
    ; eps = 8*p*u with u = 2^-52: load the DOUBLE 2^-52 by its bit pattern
    ; 0x3CA0000000000000 (the old `movq` of the integer 2^52 passed 4.5e15
    ; and made every factorization "singular")
    mov     rax, 0x3CA0000000000000
    movq    xmm0, rax               ; xmm0 = 2^-52
    cvtsi2sd xmm1, r13
    mulsd   xmm0, xmm1
    mov     rdx, r15                ; stride = q
    call    ols_chol_f64_sse2 wrt ..plt
    test    eax, eax
    jnz     .err_code
    ; solve: R R' beta = RHS (in place over RHS)
    mov     rdi, r12                ; G (R in its upper block)
    mov     rsi, r13
    mov     rdx, [rsp+256]          ; RHS
    mov     rcx, r15                ; stride
    call    ols_chol_solve_f64_sse2 wrt ..plt
    test    eax, eax
    jnz     .err_code
    ; ---------- beta out + intercept ----------
    mov     rax, [rsp+8]            ; res
    mov     rdi, [rax+RES_BETA]
    mov     rdx, [rsp+256]
    xor     rcx, rcx
.beta_out:
    cmp     rcx, r13
    jae     .beta_done
    movsd   xmm0, [rdx + rcx*8]
    movsd   [rdi + rcx*8], xmm0
    inc     rcx
    jmp     .beta_out
.beta_done:
    ; intercept: b0 = ybar - xbar'beta  (ybar/xbar from pivots + shifted sums)
    cmp     qword [rsp+72], 0
    je      .no_intercept
    ; means: mean_k = c[k] + S[k]/W
    mov     rax, [rsp+248]          ; W ptr
    movsd   xmm4, [rax]             ; W
    ; xbar'beta
    pxor    xmm0, xmm0
    xor     rcx, rcx
.xb:
    cmp     rcx, r13
    jae     .xb_done
    mov     rax, [rsp+16]           ; ws -> C at +0
    movsd   xmm1, [rax + rcx*8]     ; c[k]
    mov     rax, [rsp+240]          ; S
    movsd   xmm2, [rax + rcx*8]
    divsd   xmm2, xmm4
    addsd   xmm1, xmm2              ; mean_k
    mov     rdx, [rsp+256]
    mulsd   xmm1, [rdx + rcx*8]     ; * beta[k]
    addsd   xmm0, xmm1
    inc     rcx
    jmp     .xb
.xb_done:
    ; ybar = c[p] + S[p]/W
    mov     rax, [rsp+16]
    movsd   xmm1, [rax + r13*8]     ; c[p]
    mov     rax, [rsp+240]
    movsd   xmm2, [rax + r13*8]
    divsd   xmm2, xmm4
    addsd   xmm1, xmm2              ; ybar
    subsd   xmm1, xmm0              ; b0
    mov     rax, [rsp+8]
    movsd   [rax+RES_INTERCEPT], xmm1
    jmp     .have_intercept
.no_intercept:
    mov     rax, [rsp+8]
    pxor    xmm0, xmm0
    movsd   [rax+RES_INTERCEPT], xmm0
.have_intercept:
    ; ---------- CSNE refinement (method 2): two fixed-point steps ----------
    ; Corrected semi-normal equations: with the weighted residual
    ; r_i = w_i(y_i - b0 - x_i'b) from the residual pass, the correction is
    ;   intercept model: C d = X'Wr - m*(sum w r)  (C = factored centered
    ;                    gram, m = weighted column means),  b0 += (sum w r)/W
    ;   no-intercept:    C d = X'Wr               (raw gram, c = 0)
    ; computed with the SAME factor (classic CSNE; the residual comes from
    ; the ORIGINAL data via the rsum/xtr kernel, not from b - Ax, so
    ; cancellation does not eat the correction). Two steps; the stats pass
    ; below then measures the REFINED fit. Previously method 2 silently
    ; aliased plain chol — an honesty bug found by the session-3 audit.
    cmp     dword [rsp+64], OLS_M_GRAM_CSNE
    jne     .no_csne
    ; NOTE (P3.7 follow-up): AUTO ran CSNE briefly this phase, but the L8
    ; offset properties regressed — the refinement residual loses
    ; u*|offset| precision at 1e9 offsets (kappa conditioning is not the
    ; only failure axis). AUTO stays plain chol until a data-derived rule
    ; accounts for BOTH kappa (kappa*=1e6, results/accuracy/kappa-sweep.json)
    ; and offset magnitude.
    ; stash m[j] = c[j] + S[j]/W into C (ws+0): the correction needs the
    ; true weighted column means (S is still intact; the se block's later
    ; stash recomputes the same values — idempotent).
    mov     rax, [rsp+248]
    movsd   xmm4, [rax]             ; W
    mov     rax, [rsp+240]          ; S
    mov     rdx, [rsp+16]           ; ws -> C
    xor     rcx, rcx
.csne_m:
    cmp     rcx, r13
    jae     .csne_no_m
    movsd   xmm0, [rax + rcx*8]     ; S[j]
    divsd   xmm0, xmm4
    mov     r8, [rsp+16]
    movsd   xmm1, [r8 + rcx*8]      ; c[j]
    addsd   xmm0, xmm1              ; m[j]
    movsd   [rdx + rcx*8], xmm0
    inc     rcx
    jmp     .csne_m
.csne_no_m:
    mov     r12d, 2                 ; two refinement steps (r12 = n is dead:
                                    ; later n reads go through [rsp+48])
.csne_iter:
    ; ---- residual pass with the CURRENT b0/beta ----
    mov     rax, [rsp+0]
    lea     rdi, [rsp+112]
    mov     rdx, [rax+PRB_X]
    mov     [rdi+RA_X], rdx
    mov     rdx, [rax+PRB_LDX]
    lea     rdx, [rdx*8]
    mov     [rdi+RA_LDX8], rdx
    mov     rdx, [rax+PRB_Y]
    mov     [rdi+RA_Y], rdx
    mov     rdx, [rax+PRB_W]
    mov     [rdi+RA_W], rdx
    mov     rdx, [rsp+48]
    mov     [rdi+RA_N], rdx
    mov     rdx, [rsp+40]
    mov     [rdi+RA_P], rdx
    mov     rax, [rsp+8]            ; res
    movsd   xmm0, [rax+RES_INTERCEPT]
    movsd   [rdi+RA_B0], xmm0
    mov     rdx, [rax+RES_BETA]
    mov     [rdi+RA_BETA], rdx
    ; sse out -> XTR[p] (slack 1); rsum out -> XTR[p+1] (slack 2)
    mov     rdx, [rsp+264]          ; XTR
    mov     rcx, [rsp+40]           ; p
    lea     rcx, [rcx*8]
    lea     rsi, [rdx + rcx]
    mov     [rdi+RA_SSE], rsi
    add     rsi, 8
    mov     [rdi+RA_RSUM], rsi
    mov     [rdi+RA_XTR], rdx
    call    ols_residual_f64_sse2 wrt ..plt
    test    eax, eax
    jnz     .err_code
    ; ---- RHS: intercept model subtracts m*rsum (centered correction);
    ;      no-intercept model uses xtr directly ----
    mov     rdx, [rsp+264]          ; XTR (xtr)
    mov     rdi, [rsp+256]          ; RHS
    cmp     qword [rsp+72], 0
    je      .csne_rhs_raw
    mov     rax, [rsp+16]           ; C holds m
    mov     rcx, [rsp+40]
    lea     rcx, [rcx*8]
    movsd   xmm0, [rdx + rcx + 8]   ; rsum = XTR[p+1]
    xor     r8, r8
.csne_rhs:
    cmp     r8, r13
    jae     .csne_rhs_done
    movsd   xmm1, [rdx + r8*8]      ; xtr[j] = (X'Wr)_j
    movsd   xmm2, [rax + r8*8]      ; m[j]
    mulsd   xmm2, xmm0
    subsd   xmm1, xmm2
    movsd   [rdi + r8*8], xmm1
    inc     r8
    jmp     .csne_rhs
.csne_rhs_raw:
    xor     r8, r8
.csne_rhs_raw_loop:
    cmp     r8, r13
    jae     .csne_rhs_done
    movsd   xmm1, [rdx + r8*8]
    movsd   [rdi + r8*8], xmm1
    inc     r8
    jmp     .csne_rhs_raw_loop
.csne_rhs_done:
    mov     rdi, [rsp+232]          ; G (factored upper block)
    mov     rsi, r13                ; p
    mov     rdx, [rsp+256]          ; RHS
    mov     rcx, r15                ; stride q
    call    ols_chol_solve_f64_sse2 wrt ..plt
    test    eax, eax
    jnz     .err_code
    ; ---- beta += d ; (intercept model) b0 += rsum/W - m'd ----
    mov     rax, [rsp+8]            ; res
    mov     rdi, [rax+RES_BETA]
    mov     rdx, [rsp+256]          ; RHS = d
    mov     rax, [rsp+16]           ; C = m (zeros when no intercept: m'd = 0)
    pxor    xmm0, xmm0              ; m'd
    xor     rcx, rcx
.csne_add:
    cmp     rcx, r13
    jae     .csne_add_done
    movsd   xmm1, [rdx + rcx*8]     ; d[j]
    movsd   xmm2, [rdi + rcx*8]     ; beta[j]
    addsd   xmm2, xmm1
    movsd   [rdi + rcx*8], xmm2
    movsd   xmm3, [rax + rcx*8]     ; m[j]
    mulsd   xmm3, xmm1
    addsd   xmm0, xmm3
    inc     rcx
    jmp     .csne_add
.csne_add_done:
    cmp     qword [rsp+72], 0
    je      .csne_no_b0
    mov     rdx, [rsp+264]
    mov     rcx, [rsp+40]
    lea     rcx, [rcx*8]
    movsd   xmm1, [rdx + rcx + 8]   ; rsum (reload: chol_solve clobbered xmm*)
    mov     rax, [rsp+248]
    movsd   xmm2, [rax]             ; W
    divsd   xmm1, xmm2              ; rsum/W
    subsd   xmm1, xmm0              ; - m'd
    mov     rax, [rsp+8]
    movsd   xmm2, [rax+RES_INTERCEPT]
    addsd   xmm2, xmm1
    movsd   [rax+RES_INTERCEPT], xmm2
.csne_no_b0:
    dec     r12d
    jnz     .csne_iter
.no_csne:
    ; ---------- stats (unless SKIP_STATS) ----------
    mov     rax, [rsp+8]
    ; zero stats first
    pxor    xmm0, xmm0
    movsd   [rax+RES_SSE], xmm0
    movsd   [rax+RES_SST], xmm0
    movsd   [rax+RES_R2], xmm0
    movsd   [rax+RES_R2ADJ], xmm0
    movsd   [rax+RES_SIGMA2], xmm0
    movsd   [rax+RES_COND_EST], xmm0
    movsd   [rax+RES_F_STAT], xmm0
    mov     dword [rax+RES_RANK], -1
    mov     rcx, [rsp+56]
    test    rcx, OLS_FIT_SKIP_STATS
    jnz     .fill_meta
    cmp     dword [rsp+64], OLS_M_QR
    je      .stats_qr
    ; RA args at [rsp+112]
    mov     rax, [rsp+0]
    lea     rdi, [rsp+112]
    mov     rdx, [rax+PRB_X]
    mov     [rdi+RA_X], rdx
    mov     rdx, [rax+PRB_LDX]
    lea     rdx, [rdx*8]
    mov     [rdi+RA_LDX8], rdx
    mov     rdx, [rax+PRB_Y]
    mov     [rdi+RA_Y], rdx
    mov     rdx, [rax+PRB_W]
    mov     [rdi+RA_W], rdx
    mov     rdx, [rsp+48]
    mov     [rdi+RA_N], rdx
    mov     rdx, [rsp+40]
    mov     [rdi+RA_P], rdx
    ; RA_B0 = res->intercept (scalar, loaded via the res struct)
    mov     rdx, [rsp+8]
    movsd   xmm0, [rdx+RES_INTERCEPT]
    movsd   [rdi+RA_B0], xmm0
    ; RA_BETA = res->beta
    mov     rdx, [rdx+RES_BETA]
    mov     [rdi+RA_BETA], rdx
    ; RA_SSE -> ws XTR? no: sse out must be a writable double; use a dedicated
    ; slot: the workspace tail XTR is p doubles; SSE scalar goes to XTR[p]
    ; and the CSNE rsum to XTR[p+1] — the +2 doubles of slack in ws_size
    ; cover exactly these two.
    mov     rdx, [rsp+264]          ; XTR
    mov     rax, [rsp+40]
    lea     rax, [rax*8]
    add     rdx, rax                ; XTR + p*8 = sse slot
    mov     [rdi+RA_SSE], rdx
    mov     rdx, [rsp+264]
    mov     [rdi+RA_XTR], rdx
    ; RA_RSUM = NULL on the stats pass (rsum is only needed for CSNE; the
    ; v3 resid-args append left this slot UNSORED — the kernel dereferenced
    ; the stale [rsp+192] word and wrote 8 bytes through garbage: the
    ; n=500 C-caller heap corruption the CLI tripped over)
    xor     edx, edx
    mov     [rdi+RA_RSUM], rdx
    call    ols_residual_f64_sse2 wrt ..plt
    test    eax, eax
    jnz     .err_code
    ; ---- fill stats: sse, s2, r2, r2_adj, sigma2 ----
    mov     rax, [rsp+8]            ; res
    mov     rdx, [rsp+264]
    mov     rcx, [rsp+40]
    lea     rcx, [rcx*8]
    add     rdx, rcx                ; sse slot
    movsd   xmm0, [rdx]             ; sse
    movsd   [rax+RES_SSE], xmm0
    ; centered SST = G[p][p] AFTER the centering pass (C_yy), i.e. exactly
    ; sum w (y - ybar)^2. The old code read s[p] (a FIRST moment) as S_yy —
    ; the absurd-r2 bug; use the centered G instead.
    mov     r8, [rsp+232]           ; G
    mov     rcx, [rsp+40]           ; p
    mov     rax, r15                ; q (stride)
    imul    rax, rcx
    add     rax, rcx                ; p*q + p = index of G[p][p]
    movsd   xmm1, [r8 + rax*8]      ; C_yy = centered SST (weighted)
    mov     rcx, [rsp+8]
    movsd   [rcx+RES_SST], xmm1
    jmp     .r2_have
.stats_qr:
    ; sse/sst byproducts live in the XTR slack: XTR[p] = sse (tail),
    ; XTR[p+1] = sst (build) — same slots the chol path uses for sse/rsum
    mov     rdx, [rsp+264]          ; XTR
    mov     rcx, [rsp+40]
    lea     rcx, [rcx*8]
    movsd   xmm1, [rdx + rcx + 8]   ; sst
    movsd   xmm0, [rdx + rcx]       ; sse
    mov     rax, [rsp+8]
    movsd   [rax+RES_SST], xmm1
    movsd   [rax+RES_SSE], xmm0
.r2_have:
    ; r2 (reload res into rax; NEVER `mov eax,1` — it truncates the pointer,
    ; the exact segfault fixed here)
    mov     rax, [rsp+8]
    pxor    xmm4, xmm4
    ucomisd xmm1, xmm4
    jbe     .r2_one
    movapd  xmm4, xmm0
    divsd   xmm4, xmm1
    mov     ecx, 1
    cvtsi2sd xmm5, ecx
    subsd   xmm5, xmm4
    movsd   [rax+RES_R2], xmm5
    jmp     .r2_done
.r2_one:
    mov     ecx, 1
    cvtsi2sd xmm5, ecx
    movsd   [rax+RES_R2], xmm5
.r2_done:
    ; sigma2 = sse / max(1, n - q_effective): dof = n - p - intercept
    mov     rax, [rsp+8]
    mov     rcx, [rsp+48]           ; n
    sub     rcx, [rsp+40]           ; - p
    sub     rcx, [rsp+72]           ; - intercept
    cmp     rcx, 1
    jae     .dof_ok
    mov     rcx, 1
.dof_ok:
    cvtsi2sd xmm1, rcx
    movsd   xmm0, [rax+RES_SSE]
    divsd   xmm0, xmm1
    movsd   [rax+RES_SIGMA2], xmm0
    ; r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - intercept)
    ; guards: sst <= 0 or dof <= 1 -> r2_adj = r2 (degenerate; documented in
    ; docs/numerical_stability.md). Re-applied from docs/dev/r2a_t_wip.asm
    ; after the heap-corruption fix landed.
    mov     rax, [rsp+8]
    movsd   xmm2, [rax+RES_R2]
    mov     rcx, [rsp+48]           ; n
    dec     rcx                     ; n - 1
    cvtsi2sd xmm3, rcx
    mov     rcx, [rsp+48]
    sub     rcx, [rsp+40]           ; - p
    sub     rcx, [rsp+72]           ; - intercept
    cmp     rcx, 1
    jbe     .r2a_deg
    movsd   xmm4, [rax+RES_SST]
    pxor    xmm5, xmm5
    ucomisd xmm4, xmm5
    jbe     .r2a_deg
    cvtsi2sd xmm4, rcx
    mov     ecx, 1
    cvtsi2sd xmm5, ecx
    subsd   xmm5, xmm2              ; 1 - r2
    mulsd   xmm5, xmm3              ; * (n-1)
    divsd   xmm5, xmm4              ; / (n - p - intercept)
    mov     ecx, 1
    cvtsi2sd xmm6, ecx
    subsd   xmm6, xmm5
    movsd   [rax+RES_R2ADJ], xmm6
    jmp     .r2a_done
.r2a_deg:
    movsd   [rax+RES_R2ADJ], xmm2
.r2a_done:
    ; ---- f_stat (v3): F = ((sst-sse)/p) / (sse/dof) --------------------------
    ; guards: p=0 impossible here (validated), sse<=0 or dof<2 -> 0.
    ; sst convention matches r2: centered with intercept, raw without;
    ; dof = n - p - intercept (same convention as sigma2/r2_adj above — the
    ; first draft used n - q, wrong for no-intercept where q = p+1 but the
    ; model has p parameters).
    movsd   xmm0, [rax+RES_SST]
    movsd   xmm1, [rax+RES_SSE]
    mov     rax, [rsp+8]
    pxor    xmm3, xmm3
    ucomisd xmm1, xmm3
    jbe     .f_zero
    mov     rcx, [rsp+48]           ; n
    sub     rcx, r13                ; - p
    sub     rcx, [rsp+72]           ; - intercept
    cmp     rcx, 2
    jb      .f_zero
    subsd   xmm0, xmm1              ; sst - sse
    pxor    xmm2, xmm2
    ucomisd xmm0, xmm2
    jbe     .f_zero                 ; sst<=sse: degenerate model, F=0
    cvtsi2sd xmm2, r13              ; p
    divsd   xmm0, xmm2              ; (sst-sse)/p
    cvtsi2sd xmm2, rcx              ; n - q
    divsd   xmm1, xmm2              ; sse/(n-q)
    divsd   xmm0, xmm1              ; F
    movsd   [rax+RES_F_STAT], xmm0
    jmp     .f_done
.f_zero:
    movsd   [rax+RES_F_STAT], xmm3
.f_done:
    cmp     dword [rsp+64], OLS_M_QR
    jne     .se_enter
    ; ---- QR stats: the formed-gram pipeline (S/C rescale, G from R,
    ; chol, se) runs in POSITION order under pivoting while S/C live in
    ; ORIGINAL column order — a consistent position-ordered pipeline
    ; would double the code for a path (rank-deficient designs) where
    ; inference (se/t) is unreliable anyway. CONTRACT: OLS_QR_PIVOT
    ; returns beta/rank/sse/sst/r2/r2_adj/f_stat and ZEROES se/t/
    ; cond_est; full inference stays on the unpivoted QR path.
    cmp     dword [rsp+64], OLS_M_QR_PIVOT
    je      .se_enter               ; method 4: no gram/se chain (contract
                                    ; above); plain QR (3) falls through
    ; S[j] /= norms[j], j < p
    mov     rax, [rsp+200]          ; norms (QA_NORMS)
    mov     rdx, [rsp+240]          ; S
    xor     rcx, rcx
.sresc:
    cmp     rcx, r13
    jae     .sresc_done
    mov     r8, [rsp+72]            ; ic
    lea     r10, [rcx + r8]
    movsd   xmm0, [rdx + rcx*8]
    divsd   xmm0, [rax + r10*8]     ; norms_pos[j+ic] (position-ordered)
    movsd   [rdx + rcx*8], xmm0
    inc     rcx
    jmp     .sresc
.sresc_done:
    ; C[j] /= norms[j+ic] as well: the se stash computes m = c + S/W, and
    ; the scaled intercept functional is m_s = (c + S_raw/W)/norms — with
    ; BOTH C and S scaled, the existing stash formula yields m_s exactly.
    ; (C is dead after the intercept block, which has already run.)
    mov     rax, [rsp+200]          ; norms
    mov     rdx, rbx                ; C (pivots, ws+0)
    xor     rcx, rcx
.cresc:
    cmp     rcx, r13
    jae     .cresc_done
    mov     r8, [rsp+72]            ; ic
    lea     r10, [rcx + r8]
    movsd   xmm0, [rdx + rcx*8]
    divsd   xmm0, [rax + r10*8]     ; norms[j+ic]
    movsd   [rdx + rcx*8], xmm0
    inc     rcx
    jmp     .cresc
.cresc_done:
    ; G[a][b] = sum_{i=0..off+a} Aug[i][off+a] * Aug[i][off+b], a <= b
    ; (rows 0..off+a: row 0 is the ones-column row when ic=1, off+a the
    ; diagonal — the mp_qr_ols validation pinned both bounds)
    xor     r12, r12                ; a
.ga:
    cmp     r12, r13                ; a vs p
    jae     .ga_done
    mov     r14, r12                ; b = a
.gb:
    cmp     r14, r13
    jae     .gb_done
    pxor    xmm0, xmm0
    mov     rcx, [rsp+72]           ; ic (off)
    lea     r10, [r12 + rcx]        ; off+a
    lea     r11, [r14 + rcx]        ; off+b
    xor     r8, r8                  ; i
.gi:
    cmp     r8, r10
    ja      .gi_done                ; i <= off+a
    mov     rdx, [rsp+168]          ; AUG
    mov     rcx, [rsp+32]           ; q
    add     rcx, [rsp+72]           ; m
    mov     rax, r8                 ; i
    imul    rax, rcx
    add     rax, r10                ; i*m + (off+a)
    movsd   xmm1, [rdx + rax*8]
    mov     rax, r8
    imul    rax, rcx
    add     rax, r11                ; i*m + (off+b)
    movsd   xmm2, [rdx + rax*8]
    mulsd   xmm1, xmm2
    addsd   xmm0, xmm1
    inc     r8
    jmp     .gi
.gi_done:
    ; center (intercept only): G[a][b] -= S[a]*S[b]/W (S rescaled)
    cmp     qword [rsp+72], 0
    je      .gi_store
    mov     rdx, [rsp+240]
    movsd   xmm1, [rdx + r12*8]
    movsd   xmm2, [rdx + r14*8]
    mulsd   xmm1, xmm2
    mov     rcx, [rsp+248]
    movsd   xmm2, [rcx]
    divsd   xmm1, xmm2
    subsd   xmm0, xmm1
.gi_store:
    mov     rcx, [rsp+232]          ; G
    mov     rdx, r12
    imul    rdx, r15                ; a*q (r15 = q stride)
    add     rdx, r14
    movsd   [rcx + rdx*8], xmm0
    inc     r14
    jmp     .gb
.gb_done:
    inc     r12
    jmp     .ga
.ga_done:
    ; chol of the formed centered scaled gram (p-block, stride q)
    mov     rdi, [rsp+232]
    mov     rsi, r13                ; p
    mov     rax, 0x3CA0000000000000 ; 2^-52
    movq    xmm0, rax
    cvtsi2sd xmm1, rsi
    mulsd   xmm0, xmm1              ; eps = 8*p*u (same as the chol path)
    mov     rdx, r15
    call    ols_chol_f64_sse2 wrt ..plt
    test    eax, eax
    jnz     .err_code
.se_enter:
    ; ---------- se via explicit inverse (if se != NULL) ----------
    ; QR_PIVOT contract: se/t/cond_est zeroed (the gram pipeline is
    ; original-order; see the .f_done comment)
    cmp     dword [rsp+64], OLS_M_QR_PIVOT
    jne     .se_normal
    mov     rax, [rsp+8]
    mov     rdi, [rax+RES_SE]
    test    rdi, rdi
    jz      .no_se
    pxor    xmm0, xmm0
    mov     rcx, [rsp+40]           ; p
    lea     rcx, [rcx + 1]          ; p+1 slots
    xor     rdx, rdx
.se_zero_p:
    cmp     rdx, rcx
    jae     .se_zero_p_done
    movsd   [rdi + rdx*8], xmm0
    inc     rdx
    jmp     .se_zero_p
.se_zero_p_done:
    mov     rax, [rsp+8]
    mov     rdi, [rax+RES_T]
    test    rdi, rdi
    jz      .se_pivot_done
    pxor    xmm0, xmm0
    mov     rcx, [rsp+40]
    lea     rcx, [rcx + 1]
    xor     rdx, rdx
.t_zero_p:
    cmp     rdx, rcx
    jae     .t_zero_p_done
    movsd   [rdi + rdx*8], xmm0
    inc     rdx
    jmp     .t_zero_p
.t_zero_p_done:
    jmp     .se_pivot_done          ; around the whole se/cond_est region
.se_pivot_done:
    jmp     .fill_meta              ; (rank was set by the factor/refusal)
.se_normal:
    mov     rax, [rsp+8]
    mov     rdi, [rax+RES_SE]
    test    rdi, rdi
    jz      .no_se
    ; inv of the p x p block -> needs a CONTIGUOUS p*p output. The workspace
    ; has no spare p*p region... but G's LOWER triangle is free! chol_inv
    ; writes B contiguously; we cannot overlay G. ALTERNATIVE: compute only
    ; the diagonal of the inverse via the standard recursion:
    ;   diag(R^-1): d_i = 1/R_ii; d_i = (1 - sum_{k>i} R_ik^2 * d? ...}
    ; For R upper: (R'R)^-1 diag_j = sum over the solve; simplest correct:
    ; inv_diag[j] = ||R^{-T} e_j||^2 = solve R' z = e_j then sum z^2.
    ; That is q small solves — reuse chol_inv? It needs the full B. Do the
    ; small solves here against the G block with e_j RHS.
    ; se_j^2 = sigma2 * ||R^{-T} e_j||^2 (R from G's block).
    ; FIRST stash the column means m[j] = S[j]/W into the C buffer (ws+0):
    ; the solves below overwrite S, and the intercept-se formula needs m.
    mov     rax, [rsp+248]
    movsd   xmm4, [rax]             ; W
    mov     rax, [rsp+240]          ; S (still intact here)
    mov     rdx, [rsp+16]           ; ws -> C at +0
    xor     rcx, rcx
.stash_m:
    cmp     rcx, r13
    jae     .stash_m_done
    ; m[k] = c[k] + S[k]/W — the RAW column means (same as the .xb intercept
    ; loop). The old stash stored S[k]/W alone (the shifted part, ~0 when the
    ; pivots are exact means), which zeroed the intercept-se correction.
    ; c is read via a fresh [rsp+16] (ws) load — never via a long-lived reg.
    movsd   xmm0, [rax + rcx*8]
    divsd   xmm0, xmm4
    mov     r8, [rsp+16]           ; ws -> C at +0
    movsd   xmm1, [r8 + rcx*8]     ; c[k]
    addsd   xmm0, xmm1
    movsd   [rdx + rcx*8], xmm0
    inc     rcx
    jmp     .stash_m
.stash_m_done:
    xor     r14, r14                ; j
.se_j:
    cmp     r14, r13
    jae     .se_done
    ; build e_j in the RHS buffer (q doubles, dead after beta extraction)
    mov     rax, [rsp+256]          ; RHS
    xor     rcx, rcx
.sej_zero:
    cmp     rcx, r13
    jae     .sej_built
    xor     edx, edx
    cmp     rcx, r14
    jne     .sej_z
    mov     edx, 1
.sej_z:
    cvtsi2sd xmm0, edx
    movsd   [rax + rcx*8], xmm0
    inc     rcx
    jmp     .sej_zero
.sej_built:
    ; solve R R' z = e_j  (chol_solve overwrites S with z)
    mov     rdi, [rsp+232]          ; G (R in block)
    mov     rsi, r13
    mov     rdx, rax                ; S buffer
    mov     rcx, r15                ; stride
    call    ols_chol_solve_f64_sse2 wrt ..plt
    test    eax, eax
    jnz     .err_code
    ; (A^{-1})_jj = z[j]  (z = C^{-1} e_j; the j-th COMPONENT is the inverse
    ; diagonal. The old code used ||z||^2 = (A^{-2})_jj — a wrong se that
    ; UNDERESTIMATED by the factor sqrt((A^-2)_jj/(A^-1)_jj), found by the
    ; SciPy differential; the contract header always said inv_jj.)
    mov     rax, [rsp+256]
    movsd   xmm0, [rax + r14*8]     ; z[j]
    ; se_j = sqrt(sigma2 * z[j])  (z[j] > 0: C is PD)
    mov     rax, [rsp+8]
    movsd   xmm2, [rax+RES_SIGMA2]
    mulsd   xmm0, xmm2
    sqrtsd  xmm0, xmm0
    mov     rax, [rsp+8]
    mov     rdi, [rax+RES_SE]
    movsd   [rdi + r14*8], xmm0
    inc     r14
    jmp     .se_j
.se_done:
    cmp     dword [rsp+64], OLS_M_QR
    jne     .se_noscale
    ; se_j /= norms[j]: the inverse came from the equilibrated gram
    mov     rax, [rsp+8]
    mov     rdi, [rax+RES_SE]
    mov     rax, [rsp+200]          ; norms
    xor     rcx, rcx
.seun:
    cmp     rcx, r13
    jae     .se_noscale
    mov     r8, [rsp+72]            ; ic
    lea     r10, [rcx + r8]
    movsd   xmm0, [rdi + rcx*8]
    divsd   xmm0, [rax + r10*8]     ; norms[j+ic]
    movsd   [rdi + rcx*8], xmm0
    inc     rcx
    jmp     .seun
.se_noscale:
    ; se[p] (intercept): ||R^{-T} e||^2 for the FULL q system is not directly
    ; available (we factored only the p-block). Intercept se via the augmented
    ; 1-column formula: se_b0^2 = sigma2 * (1/W + m' C^{-1} m), m = column
    ; means of X (weighted), C = centered X'X/W-ish... our C is W-scaled:
    ; se_b0^2 = sigma2*(1/W + m' (C_xx)^{-1} m).
    ; Compute t = C^{-1} m by solving the p-block with RHS = centered means.
    mov     rax, [rsp+248]
    movsd   xmm4, [rax]             ; W
    ; m[j] from the C stash (S was destroyed by the se solves)
    ; solve C_xx t = m into RHS
    mov     rax, [rsp+16]           ; ws -> C at +0 holds m
    mov     rdx, [rsp+256]          ; RHS buffer
    xor     rcx, rcx
.b0rhs:
    cmp     rcx, r13
    jae     .b0rhs_done
    movsd   xmm0, [rax + rcx*8]
    movsd   [rdx + rcx*8], xmm0     ; t RHS = m[j]
    inc     rcx
    jmp     .b0rhs
.b0rhs_done:
    mov     rdi, [rsp+232]
    mov     rsi, r13
    mov     rdx, [rsp+256]
    mov     rcx, r15
    call    ols_chol_solve_f64_sse2 wrt ..plt
    test    eax, eax
    jnz     .err_code
    ; RELOAD W: chol_solve clobbers xmm0-xmm7 (caller-saved), so the xmm4
    ; holding W from above is garbage here (this made se[p] = inf via 1/0)
    mov     rax, [rsp+248]
    movsd   xmm4, [rax]
    ; m' t = sum m[j]*t[j] (m from the C stash; RHS holds t)
    mov     rax, [rsp+16]           ; ws -> C holds m
    mov     rdx, [rsp+256]          ; RHS holds t
    pxor    xmm0, xmm0
    xor     rcx, rcx
.b0mt:
    cmp     rcx, r13
    jae     .b0mt_done
    movsd   xmm1, [rax + rcx*8]     ; m[j]
    mulsd   xmm1, [rdx + rcx*8]     ; t[j]
    addsd   xmm0, xmm1
    inc     rcx
    jmp     .b0mt
.b0mt_done:
    ; se_b0^2 = sigma2*(1/W + m't)
    mov     rax, [rsp+8]
    movsd   xmm2, [rax+RES_SIGMA2]
    mov     eax, 1
    cvtsi2sd xmm3, eax
    divsd   xmm3, xmm4
    addsd   xmm3, xmm0
    mulsd   xmm2, xmm3
    sqrtsd  xmm2, xmm2
    mov     rax, [rsp+8]
    mov     rdi, [rax+RES_SE]
    movsd   [rdi + r13*8], xmm2
    jmp     .cond_est
.no_se:
    mov     rax, [rsp+8]
    mov     dword [rax+RES_RANK], -1
.cond_est:
    ; cond_est = (max R_ii / min R_ii)^2 over the p x p block
    mov     rax, [rsp+232]          ; G
    pxor    xmm0, xmm0              ; max
    ; min: start from +inf
    mov     rcx, 0x7FF0000000000000
    movq    xmm1, rcx
    xor     rcx, rcx
.cnd:
    cmp     rcx, r13
    jae     .cnd_done
    mov     rdx, rcx
    imul    rdx, r15
    add     rdx, rcx
    movsd   xmm2, [rax + rdx*8]     ; R_jj
    ucomisd xmm2, xmm0
    jbe     .cnd_nmax
    movapd  xmm0, xmm2
.cnd_nmax:
    ucomisd xmm2, xmm1
    jae     .cnd_nmin
    movapd  xmm1, xmm2
.cnd_nmin:
    inc     rcx
    jmp     .cnd
.cnd_done:
    ; (max/min)^2 — guard min <= 0 (singular path already returned)
    pxor    xmm3, xmm3
    ucomisd xmm1, xmm3
    jbe     .cnd_zero
    divsd   xmm0, xmm1
    mulsd   xmm0, xmm0
    mov     rax, [rsp+8]
    movsd   [rax+RES_COND_EST], xmm0
    jmp     .fill_meta
.cnd_zero:
    mov     rax, [rsp+8]
    mov     rcx, 0x7FF0000000000000
    movq    xmm0, rcx
    movsd   [rax+RES_COND_EST], xmm0
.fill_meta:
    ; rank = p (Tier 1 non-singular path), method_used, isa_used
    mov     rax, [rsp+8]
    mov     rcx, [rsp+40]
    mov     [rax+RES_RANK], ecx
    mov     edx, [rsp+64]
    test    edx, edx
    jnz     .mu_store
    mov     edx, OLS_M_GRAM_CHOL    ; AUTO ran the gram+chol path: report
.mu_store:                          ; the CONCRETE method, not the request (0)
    mov     [rax+RES_METHOD_USED], edx
    push    rax
    call    ols_isa wrt ..plt
    mov     edx, eax
    pop     rax
    mov     [rax+RES_ISA_USED], edx
    ; ---- t-stats (v2): t[j] = beta_j / se_j, t[p] = b0 / se[p]; computed
    ; only when res->t and res->se are both non-NULL (se==NULL implies the
    ; se loop was skipped). se<=0 guards write t=0. Re-applied from
    ; docs/dev/r2a_t_wip.asm. ----
    mov     rax, [rsp+8]
    mov     rdi, [rax+RES_T]
    test    rdi, rdi
    jz      .t_skip
    mov     rsi, [rax+RES_SE]
    test    rsi, rsi
    jz      .t_skip                          ; se==NULL => loop skipped
    mov     rdx, [rax+RES_BETA]
    xor     rcx, rcx
.t_j:
    cmp     rcx, r13
    jae     .t_p
    movsd   xmm0, [rdx + rcx*8]
    movsd   xmm1, [rsi + rcx*8]
    pxor    xmm2, xmm2
    ucomisd xmm1, xmm2
    jbe     .t_j_zero
    divsd   xmm0, xmm1
    movsd   [rdi + rcx*8], xmm0
    jmp     .t_j_next
.t_j_zero:
    pxor    xmm0, xmm0
    movsd   [rdi + rcx*8], xmm0
.t_j_next:
    inc     rcx
    jmp     .t_j
.t_p:
    movsd   xmm0, [rax+RES_INTERCEPT]
    movsd   xmm1, [rsi + r13*8]
    pxor    xmm2, xmm2
    ucomisd xmm1, xmm2
    jbe     .t_p_zero
    divsd   xmm0, xmm1
    movsd   [rdi + r13*8], xmm0
    jmp     .t_skip
.t_p_zero:
    pxor    xmm0, xmm0
    movsd   [rdi + r13*8], xmm0
.t_skip:
    xor     eax, eax
    jmp     .out
.earg:
    mov     eax, OLS_E_ARG
    jmp     .err_zero
.eunder:
    mov     eax, OLS_E_UNDERDETERMINED
    jmp     .err_zero
.ews:
    mov     eax, OLS_E_WORKSPACE
    jmp     .err_zero
.err_code:
    ; eax already holds the error; zero res scalars. res comes from the
    ; FRAME slot ([rsp+8]) — rsi/rsi-like registers are clobbered by the
    ; solver stages (the old `test rsi,rsi` path segfaulted via rsi=p=13)
.err_zero:
    mov     rdx, [rsp+8]
    test    rdx, rdx
    jz      .out
    push    rax
    pxor    xmm0, xmm0
    movsd   [rdx+RES_INTERCEPT], xmm0
    movsd   [rdx+RES_SSE], xmm0
    movsd   [rdx+RES_SST], xmm0
    movsd   [rdx+RES_R2], xmm0
    movsd   [rdx+RES_R2ADJ], xmm0
    movsd   [rdx+RES_SIGMA2], xmm0
    movsd   [rdx+RES_COND_EST], xmm0
    movsd   [rdx+RES_F_STAT], xmm0
    ; rank: -1 EXCEPT the QR_PIVOT refusal, which pre-filled the detected
    ; rank before jumping here (do not clobber it: read [rdx+RES_RANK]
    ; only on non-pivot paths — simplest: a sentinel register is messy;
    ; instead the pivot path sets rank AFTER this block by NOT jumping
    ; through .err_zero... it does. So: preserve rank when it is already
    ; non-negative (only the pivot path pre-fills it)
    mov     ecx, [rdx+RES_RANK]
    test    ecx, ecx
    jns     .rank_keep
    mov     dword [rdx+RES_RANK], -1
.rank_keep:
    pop     rax
    jmp     .out
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
