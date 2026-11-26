; ------------------------------------------------------------------------------
; gram.asm — fused pivot-shifted Gram pass (Phase 2 core kernel).
;   ols_gram_f64_avx2(A)      : v3 group-pipelined column-panel kernel
;   ols_gram_f64_sse2(A)      : scalar reference, same contract (ignores scratch)
;   ols_gram_scratch_bytes(q) : scratch the AVX2 kernel needs for q = p+1
;
; v3 design (perf-log #20 prediction, written BEFORE the v3 benchmark):
;   Machine truth (bench/machine/load_bench.c): 1 FMA/cyc (latency 8),
;   ~1.15 ymm-load-uops/cyc, ~1 store/cyc — the per-row FMA COUNT and the
;   load count bind, not tile shape. Therefore:
;   PACK   u_r[k] = sqrt(w_r)*(z_r[k]-c[k]) into a double-buffered PKG
;          (2 groups x 4 rows x QN doubles, QN = (q+7)&-8, rows 64B-aligned,
;          zero-padded to QN); sqrt(w) ONCE per row (v1 did q sqrtsd per row
;          in the s-loop); ymm chunks while k+8 <= p, scalar to p (incl. y at
;          k == p) — NO xmm mid-path (bring-up stride bug; pack is DRAM-bound,
;          simpler is safer). Plain vector sums for s and W per group, folded
;          into the Neumaier (s,s_c)/(W_s,W_c) pairs once per group pair.
;   GSCR   column-major G scratch, stride QSC = QN+8 doubles, QN ROWS per
;          column (QN, not q: kernel A writes whole 8-row chunks; the last
;          chunk must stay in-bounds when q is not 1 mod 8). Chunks of
;          8 G-rows x 1 column are contiguous AND 32B-aligned.
;   PIPELINE: group g+1 is packed while group g is consumed, so the pack's
;          DRAM stream overlaps the L1-resident kernel FMAs.
;   KERNEL C (q <= 8): row fits one ymm; 8 column accumulators ymm0..7
;          persist across groups (flushed once in the epilogue); per row:
;          1 u-load + q bcast-loads + q FMA. One straight-line body per
;          q = 1..8 (macro-generated), dispatch via a cmov chain — no
;          guards in the hot path (skills/asm-x86-64-engineering §7).
;   KERNEL A (q >= 9): per group of 4 rows; per (i0, j>=i0) chunk: 1 GSCR
;          load + 4 FMA + 1 store; u-chunks hoisted at the i0 level;
;          broadcasts in mem form. FMA/row = number of (i0,j) chunks = the
;          aligned-chunk triangle minimum (27 at q=17, 80 at q=32).
;   COMPACT: upper triangle of GSCR -> G (q*q, strictly-lower zeroed).
; Contract: G upper triangle written, strictly-lower zeroed; s = sum w(z-c)
; OVERWRITTEN; *Wout = sum w. NaN in X/y/w, or w < 0 (sqrt -> NaN) ->
; OLS_E_NONFINITE, outputs undefined; Inf propagates (unchanged from v1).
; Deterministic: fixed order, fixed group size — bitwise stable per call.
;
; Bring-up bug log (all fixed here; see src/dev/README.md for the story):
;   fold do-while overrunning caller s for q mod 8 != 0 (guard + scalar
;   tail); vmovaps pad stores at unaligned offsets (vmovups); `mov eax,1`
;   and vmovmskpd-eax clobbering the PKG row base (edx); xmm chunk stride
;   4 vs 2-wide loads (path deleted); kernA chunk count (q*8-i0)/8 and
;   cursor starting at i0; GSCR row count QN (chunk overflow); compact
;   double-advance; Wb accumulation across the two packs per fold; second
;   sqrt(w) factor for the s contract; pack/fold inlined as macros (the
;   call-shifted frame view was unmaintainable). v3: the scalar y load
;   re-derives its address from the global row (no stale y pointer);
;   FOLD_SUMS re-materializes the abs mask after every consumption; Wb is
;   zeroed inside FOLD_SUMS after folding.
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"
%include "src/macros.inc"

; ---- v2/v3 frame slots (AVX2 kernel only; UC_0..127 ctrl reused) -------------
%define G2_QN       128     ; size_t QN = (q+7)&-8
%define G2_QSC      136     ; size_t QSC = QN+8 (doubles; GSCR column stride)
%define G2_SMEM     144     ; void* s_mem (plain per-group sums, q doubles)
%define G2_WB       152     ; double W group plain sum
%define G2_CUR      160     ; void* PKG group being consumed
%define G2_NXT      168     ; void* PKG group packed ahead
%define G2_I        176     ; size_t global row of current group
%define G2_CJMP     184     ; void* kernC body entry
%define G2_Q8       192     ; size_t q*8 (G row stride, bytes)
%define G2_SC       200     ; s_c (q doubles); s_mem lives right after it
; frame = G2_SC + 2*q*8 (s_c + s_mem), 32-aligned.
; ---- v4 (fused q<=8) frame slots: base V4 = frame_start = G2_SC + 2*q*8,
; rounded up 32 in the prologue calc — offsets are FROM THAT ROUNDED BASE,
; kept in a callee-saved-free way: recomputed in .v4 via rbp? No: the v3
; code addresses everything off rsp, so v4 slots are rsp-relative TOO, but
; their offset depends on q. Solution: .v4 stores the rounded base in the
; (v3-unused-for-q<=8) G2_SMEM slot? It IS used (zeroing loop writes s_mem).
; Cleanest: v4 uses G2_CUR (free for q<=8: no PKG) to hold its slot base.
%define V4_ULO      0       ; u lo half staging (32B): also the bcast slots
%define V4_UHI      32      ; u hi half staging (32B)
%define V4_SLO      64      ; s batch plain sums, lo (32B)
%define V4_SHI      96      ; s batch plain sums, hi (32B)
%define V4_MASKL    128     ; lane mask lo: lane k set iff k < q (32B)
%define V4_MASKH    160     ; lane mask hi: lane k set iff 4+k < q (32B)
%define V4_CLO      192     ; c[0..3] staging, zero-padded (32B)
%define V4_CHI      224     ; c[4..7] staging, zero-padded (32B)
%define V4_SW       256     ; sqrt(w) per row — MUST live in memory: xmm8..15
                            ; alias the acc/temp ymms (the 122-vs-121 bug:
                            ; vbroadcastsd xmm8 broadcast the ACC after the
                            ; j>=4 hi-FMA clobbered it; same class as
                            ; ADR-0010 #4). Slot 256: OUTSIDE the 0..256
                            ; staging block (240 collided with CHI[2] = c[6]
                            ; -> u[6] = 16 for 17, the q=8 off-by-one).

; ==============================================================================
; PACK_ROW n — pack row (r8+n) of the group. Inputs: rsi = PKG group base,
; r8 = first global row of the group, r13 = QN8, r14 = q, r9 = s_mem, r11 = c.
; Row base -> rax; X row -> rdi; p -> rcx; k -> r10; W group sum -> xmm15.
; ymm8=z, ymm9=c-chunk, ymm10=u, ymm11=NaN mask, ymm12=sqrt(w) bcast,
; ymm13=s_mem chunk, ymm14=zero vector (MUST stay zero), edx=mask/const
; scratch (NEVER eax: rax is the row base; never r10/rcx/rdi/rsi/r9/r11).
; The global row is RE-DERIVED (lea rdx, [r8+%1]) at every use site — no
; long-lived row register exists to clobber.
; ==============================================================================
%macro PACK_ROW 1
    lea     rdx, [r8 + %1]
    cmp     rdx, [rsp+UC_N]
    jae     %%pad_row
    ; ---- PKG row base -> rax ----
%if %1 == 0
    mov     rax, rsi
%elif %1 == 1
    lea     rax, [rsi + r13]
%elif %1 == 2
    lea     rax, [rsi + r13*2]
%else
    lea     rax, [rsi + r13*2]
    lea     rax, [rax + r13]
%endif
    ; ---- w[row] -> sqrt(w) -> ymm12 broadcast ----
    mov     rdi, [rsp+UC_W]
    test    rdi, rdi
    jz      %%w_one
    vmovsd  xmm8, [rdi + rdx*8]
    jmp     %%w_have
%%w_one:
    mov     edx, 1
    vcvtsi2sd xmm8, xmm8, rdx
%%w_have:
    vucomisd xmm8, xmm8
    jp      .nonfinite
    vsqrtsd xmm8, xmm8, xmm8
    vbroadcastsd ymm12, xmm8
    ; ---- X row base -> rdi ----
    lea     rdx, [r8 + %1]
    mov     rdi, [rsp+UC_X]
    imul    rdx, [rsp+UC_LDX8]
    add     rdi, rdx
    mov     rcx, r14
    dec     rcx                      ; p
    xor     r10, r10                 ; k
    vxorpd  xmm11, xmm11, xmm11     ; row NaN sentinel (NaN propagates)
%%pw:                                ; ymm chunks while k+4 <= p
    lea     rdx, [r10 + 4]
    cmp     rdx, rcx
    jbe     %%pw_go
    jmp     %%pt
%%pw_go:
    vmovupd ymm8, [rdi + r10*8]
    vmovupd ymm9, [r11 + r10*8]
    vsubpd  ymm10, ymm8, ymm9
    vcmppd  ymm11, ymm10, ymm10, 3   ; UNORD(q)
    vmovmskpd edx, ymm11
    test    edx, edx
    jnz     .nonfinite
    vmulpd  ymm10, ymm10, ymm12      ; u = sw*(z-c)
    vmovupd [rax + r10*8], ymm10
    vmulpd  ymm10, ymm10, ymm12      ; sw*u = w(z-c) for the s contract
    vmovupd ymm13, [r9 + r10*8]
    vaddpd  ymm13, ymm13, ymm10
    vmovupd [r9 + r10*8], ymm13
    add     r10, 4
    jmp     %%pw
%%pt:                                ; scalar k..p inclusive (y at k==p)
    cmp     r10, rcx
    ja      %%pt_done
    cmp     r10, rcx
    jae     %%from_y
    vmovsd  xmm8, [rdi + r10*8]
    jmp     %%z_have
%%from_y:
    lea     rdx, [r8 + %1]
    mov     rdi, [rsp+UC_Y]
    vmovsd  xmm8, [rdi + rdx*8]
%%z_have:
    vmovsd  xmm9, [r11 + r10*8]
    vsubsd  xmm8, xmm8, xmm9
    vmulsd  xmm8, xmm8, xmm12              ; low lane of the sw broadcast
    vmovsd [rax + r10*8], xmm8
    vmulsd  xmm8, xmm8, xmm12              ; second sw factor: s = sum w(z-c)
    vmovsd  xmm13, [r9 + r10*8]
    vaddsd  xmm13, xmm13, xmm8
    vmovsd [r9 + r10*8], xmm13
    vaddsd  xmm11, xmm11, xmm8             ; row NaN sentinel (NaN sticks)
    inc     r10
    jmp     %%pt
%%pt_done:
    vucomisd xmm11, xmm11
    jp      .nonfinite
    ; ---- zero pad PKG[r][q..QN) ----
    mov     rdx, [rsp+G2_QN]
    mov     rcx, r14                 ; k = q
%%pz:
    lea     rdi, [rcx + 4]
    cmp     rdi, rdx
    ja      %%pz1
    vmovups [rax + rcx*8], ymm14
    add     rcx, 4
    jmp     %%pz
%%pz1:
    cmp     rcx, rdx
    jae     %%pz_done
    vmovsd [rax + rcx*8], xmm14
    inc     rcx
    jmp     %%pz1
%%pz_done:
    ; ---- W group plain sum += w[row] ----
    mov     rdi, [rsp+UC_W]
    test    rdi, rdi
    jz      %%w2_one
    lea     rdx, [r8 + %1]
    vmovsd  xmm8, [rdi + rdx*8]
    jmp     %%w2_have
%%w2_one:
    mov     edx, 1
    vcvtsi2sd xmm8, xmm8, rdx
%%w2_have:
    vucomisd xmm8, xmm8
    jp      .nonfinite
    vaddsd  xmm15, xmm15, xmm8
    jmp     %%next
%%pad_row:
    ; ---- rows beyond n: zero the whole PKG row ----
    mov     rdx, [rsp+G2_QN]
%if %1 == 0
    mov     rax, rsi
%elif %1 == 1
    lea     rax, [rsi + r13]
%elif %1 == 2
    lea     rax, [rsi + r13*2]
%else
    lea     rax, [rsi + r13*2]
    lea     rax, [rax + r13]
%endif
    xor     rcx, rcx
%%zr:
    lea     rdi, [rcx + 4]
    cmp     rdi, rdx
    ja      %%zr1
    vmovups [rax + rcx*8], ymm14
    add     rcx, 4
    jmp     %%zr
%%zr1:
    cmp     rcx, rdx
    jae     %%next
    vmovsd [rax + rcx*8], xmm14
    inc     rcx
    jmp     %%zr1
%%next:
%endmacro

; ==============================================================================
; FOLD_SUMS — fold the plain group sums (s_mem, Wb) into the Neumaier pairs
; (s, s_c) / (W_s, W_c). Vector fold over full 8-lane chunks below q, scalar
; tail for lanes floor8(q)..q-1 (s/s_c/s_mem are q-double buffers: never
; touch lanes >= q). Uses ONLY ymm8-15: kernel-C accs live in ymm0-7.
; The abs mask is re-materialized after every consumption (no long-lived
; constant register). Wb is folded and re-zeroed here.
; ==============================================================================
%macro FOLD_SUMS 0
    mov     rsi, [rsp+UC_S]
    lea     rdi, [rsp + G2_SC]       ; s_c
    mov     r9, [rsp+G2_SMEM]
    mov     rcx, r14
    and     rcx, -8
    shl     rcx, 3                   ; floor8(q)*8 bytes; ZERO when q < 8
    xor     rax, rax                 ; byte offset
    test    rcx, rcx
    jz      %%fs_tail                ; no full chunks: scalar only
%%fs_chunk:
    mov     rdx, 0x7FFFFFFFFFFFFFFF
    vmovq   xmm15, rdx
    vbroadcastsd ymm15, xmm15        ; abs mask (fresh each iteration)
    vmovupd ymm8,  [rsi + rax]       ; s
    vmovupd ymm9,  [r9 + rax]        ; sb (plain group sums)
    vaddpd  ymm10, ymm8, ymm9        ; t
    vandpd  ymm11, ymm8, ymm15       ; |s|
    vandpd  ymm12, ymm9, ymm15       ; |sb|
    vcmppd  ymm13, ymm11, ymm12, 5   ; |s| >= |sb| (imm 5 = _NLT)
    vsubpd  ymm14, ymm8, ymm10
    vaddpd  ymm14, ymm14, ymm9       ; e1 = (s-t)+sb
    vsubpd  ymm15, ymm9, ymm10
    vaddpd  ymm15, ymm15, ymm8       ; e2 = (sb-t)+s
    vandpd  ymm14, ymm14, ymm13
    vandnpd ymm13, ymm13, ymm15
    vorpd   ymm14, ymm14, ymm13      ; e
    vmovupd ymm15, [rdi + rax]
    vaddpd  ymm15, ymm15, ymm14
    vmovupd [rdi + rax], ymm15       ; s_c += e
    vmovupd [rsi + rax], ymm10       ; s = t
    vxorpd  ymm8, ymm8, ymm8
    vmovupd [r9 + rax], ymm8         ; s_mem = 0
    add     rax, 32
    cmp     rax, rcx
    jb      %%fs_chunk
%%fs_tail:
    mov     rcx, r14
    and     rcx, -8                  ; first tail lane
%%fs_t:
    cmp     rcx, r14
    jae     %%fs_w
    vmovsd xmm8, [r9 + rcx*8]      ; x = s_mem[k]
    vmovsd xmm9, [rsi + rcx*8]     ; s
    vaddsd  xmm10, xmm9, xmm8        ; t
    mov     rdx, 0x7FFFFFFFFFFFFFFF
    vmovq   xmm11, rdx
    vandpd  xmm11, xmm11, xmm9       ; |s|
    vmovq   xmm12, rdx
    vandpd  xmm12, xmm12, xmm8       ; |x|
    vcmpnltpd xmm13, xmm11, xmm12    ; |s| >= |x|
    vsubsd  xmm14, xmm9, xmm10
    vaddsd  xmm14, xmm14, xmm8       ; e1
    vsubsd  xmm15, xmm8, xmm10
    vaddsd  xmm15, xmm15, xmm9       ; e2
    vandpd  xmm14, xmm14, xmm13
    vandnpd xmm13, xmm13, xmm15
    vorpd   xmm14, xmm14, xmm13      ; e
    vmovsd xmm15, [rdi + rcx*8]
    vaddsd  xmm15, xmm15, xmm14
    vmovsd [rdi + rcx*8], xmm15     ; s_c += e
    vmovsd [rsi + rcx*8], xmm10     ; s = t
    vxorpd  xmm8, xmm8, xmm8
    vmovsd [r9 + rcx*8], xmm8       ; s_mem = 0
    inc     rcx
    jmp     %%fs_t
%%fs_w:
    ; ---- W fold (scalar Neumaier) ----
    vmovsd xmm8, [rsp+UC_WS]
    vmovsd xmm9, [rsp+G2_WB]
    vaddsd  xmm10, xmm8, xmm9        ; t
    mov     rdx, 0x7FFFFFFFFFFFFFFF
    vmovq   xmm11, rdx
    vandpd  xmm11, xmm11, xmm8       ; |W_s|
    vmovq   xmm12, rdx
    vandpd  xmm12, xmm12, xmm9       ; |Wb|
    vcmpnltpd xmm13, xmm11, xmm12    ; |W_s| >= |Wb|
    vsubsd  xmm14, xmm8, xmm10
    vaddsd  xmm14, xmm14, xmm9       ; e1
    vsubsd  xmm15, xmm8, xmm10
    vaddsd  xmm15, xmm15, xmm9       ; e2  (xmm15 = (t? no): (0-t)+W_s)
    vandpd  xmm14, xmm14, xmm13
    vandnpd xmm13, xmm13, xmm15
    vorpd   xmm14, xmm14, xmm13      ; e
    vmovsd xmm15, [rsp+UC_WC]
    vaddsd  xmm15, xmm15, xmm14
    vmovsd [rsp+UC_WC], xmm15       ; W_c += e
    vmovsd [rsp+UC_WS], xmm10       ; W_s = t
    vxorpd  xmm9, xmm9, xmm9
    vmovsd [rsp+G2_WB], xmm9        ; Wb = 0
%endmacro

; ------------------------------------------------------------------------------
; PACK_GROUP — inline (macro) pack of one 4-row group. Inputs/outputs:
; rsi = PKG group base (input), r8 = first global row (input); xmm15 = W
; group sum (output, accumulated INTO the G2_WB slot); ymm14 zeroed here.
; ------------------------------------------------------------------------------
%macro PACK_GROUP 0
    vxorpd  xmm15, xmm15, xmm15    ; Wb for this group
    vxorpd  ymm14, ymm14, ymm14    ; zero vec for pads (stays zero)
    mov     r9, [rsp+G2_SMEM]
    mov     r11, [rsp+UC_C]
    PACK_ROW 0
    PACK_ROW 1
    PACK_ROW 2
    PACK_ROW 3
    vmovsd xmm8, [rsp+G2_WB]
    vaddsd  xmm8, xmm8, xmm15       ; Wb accumulates across the two packs
    vmovsd [rsp+G2_WB], xmm8       ; that run before one fold
%endmacro

; ==============================================================================
; size_t ols_gram_scratch_bytes(size_t q)
; Args    : rdi = q (= p+1, 1..512)
; Returns : rax = scratch bytes for the AVX2 Gram kernel (64-aligned):
;           PKG 8*QN doubles + GSCR QN*(QN+8) doubles (QN rows per column:
;           kernel A writes whole 8-row chunks, so rows must reach QN)
; Requires: nothing
; Memory  : none
; Clobbers: rax, rcx, rdx
; Numerics: n/a
; Cost    : trivial; called once per ols_fit / per test arg build.
; ==============================================================================
global ols_gram_scratch_bytes:function
ols_gram_scratch_bytes:
    jmp     ols__scratch_calc
ols__scratch_calc:                  ; local body: PC32 to a local label is
    mov     rax, rdi                ; link-safe for -shared (not preemptible)
    add     rax, 7
    and     rax, -8                 ; QN
    lea     rcx, [rax + 8]          ; QSC
    imul    rcx, rax                ; QN*QSC doubles (GSCR)
    imul    rax, 8                  ; 8*QN doubles (PKG: 2 groups x 4 rows)
    add     rax, rcx
    shl     rax, 3                  ; bytes
    add     rax, 63
    and     rax, -64
    ret

; ==============================================================================
; int32_t ols_gram_f64_avx2(const ols_gram_args_t *A)
; Args    : rdi = A
; Returns : eax = OLS_OK | OLS_E_NONFINITE | OLS_E_WORKSPACE (scratch small)
; Requires: AVX2+FMA3 (dispatcher-checked); scratch 64-byte aligned;
;           scratch_bytes >= ols_gram_scratch_bytes(q)
; Memory  : reads A (96 B), X rows [0..n) cols [0..p) at byte stride A->ldx8,
;           y[0..n), w[0..n) if non-NULL, c[0..q); writes scratch (PKG+GSCR),
;           G[0..q*q) upper + zeroed lower, s[0..q), *A->Wout.
; Clobbers: rax,rcx,rdx,rsi,rdi,r8-r11 + rbx,rbp,r12-r15 (pushed/popped,
;           probe-verified), ymm0-ymm15. vzeroupper at exit. rsp/DF/MXCSR kept.
; Numerics: G via vfmadd with fixed group reassociation (cross-ISA
;           differential bounds cover the order change); s and W: plain
;           per-group sums (<= 8 rows) folded into Neumaier pairs.
; Cost    : PREDICTED (perf-log #20, before benchmark): q=8 ~15-18 cy/row
;           out-of-cache; q=17 ~28-32 pipelined (80-92% of DRAM); q=32
;           in-cache ~82-95 (69-80% of flops bound).
; GPR map (callee-saved): rbp=frame rbx=PKG r12=GSCR r13=QN8 r14=q r15=QSC8
; ==============================================================================
global ols_gram_f64_avx2:function
ols_gram_f64_avx2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8                  ; 7 slots -> rsp 16-aligned here
    mov     rbx, [rdi+GA_SCRATCH]
    mov     r14, [rdi+GA_P]
    inc     r14                     ; q
    ; ---- scratch sufficiency (exact, alignment-aware) ----
    ; The kernel aligns the PKG base UP to 64 inside the caller's scratch.
    ; Bytes consumed from A->scratch = alignpad + calc(q), where alignpad =
    ; (-A->scratch) & 63. The old check compared calc(q) BEFORE the align-up:
    ; a 16-aligned scratch passed and GSCR then wrote up to 63 bytes past the
    ; buffer — the multi-fit heap corruption. Exact check fixes both bugs.
    push    rdi                     ; save A (rbx/r14 already extracted)
    mov     rdi, r14
    call    ols__scratch_calc       ; local body (same file): PC32-safe
    mov     rdi, [rsp]              ; A back (saved by the push above)
    mov     rcx, [rdi+GA_SCRATCH]
    neg     rcx
    and     rcx, 63                 ; alignpad
    add     rax, rcx                ; bytes consumed from scratch start
    cmp     [rdi+GA_SCRATCH_BYTES], rax
    jb      .ework_pop
    pop     rdi
    ; ---- frame: align32(G2_SC + 2*q*8) + 256 for the fused q<=8 path -----
    ; (ULO/UHI/SLO/SHI/MASKL/MASKH/CLO/CHI = 8 x 32B slots). The 256 is added
    ; AFTER the 32-alignment of the v3 part: the .v4 slot base is the
    ; 32-aligned end of the v3 area, and the +256 must lie fully inside the
    ; frame (q=4 bug: pre-align +256 overflowed the frame by the pad bytes
    ; and stomped saved registers / Wout).
    lea     rax, [r14*8]
    lea     rdx, [rax*2 + G2_SC]
    add     rdx, 31
    and     rdx, -32
    add     rdx, 288
    sub     rsp, rdx
    ; ---- ctrl block ----
    mov     [rsp+UC_Q], r14
    mov     rax, [rdi+GA_X]
    mov     [rsp+UC_X], rax
    mov     rax, [rdi+GA_LDX8]
    mov     [rsp+UC_LDX8], rax
    mov     rax, [rdi+GA_Y]
    mov     [rsp+UC_Y], rax
    mov     rax, [rdi+GA_W]
    mov     [rsp+UC_W], rax
    mov     rax, [rdi+GA_N]
    mov     [rsp+UC_N], rax
    mov     rax, [rdi+GA_C]
    mov     [rsp+UC_C], rax
    mov     rax, [rdi+GA_G]
    mov     [rsp+UC_G], rax
    mov     rax, [rdi+GA_S]
    mov     [rsp+UC_S], rax
    mov     rax, [rdi+GA_WOUT]
    mov     [rsp+UC_WOUT], rax
    ; ---- derived constants ----
    mov     rax, r14
    add     rax, 7
    and     rax, -8
    mov     [rsp+G2_QN], rax        ; QN
    lea     rcx, [rax + 8]
    mov     [rsp+G2_QSC], rcx       ; QSC (doubles)
    lea     r13, [rax*8]            ; r13 = QN8 (PKG row stride, bytes)
    lea     r15, [rcx*8]            ; r15 = QSC8 (GSCR column stride, bytes)
    lea     rcx, [r14*8]
    mov     [rsp+G2_Q8], rcx        ; q*8 (G row stride, bytes)
    ; ---- PKG/GSCR split of the scratch blob (both 64B-aligned) ----
    add     rbx, 63
    and     rbx, -64                ; PKG base
    mov     rax, 8
    imul    rax, [rsp+G2_QN]
    shl     rax, 3                  ; PKG bytes (8*QN doubles = 64B multiple)
    lea     r12, [rbx + rax]        ; r12 = GSCR base
    mov     [rsp+G2_CUR], rbx
    lea     rcx, [rbx + r13*4]      ; second group buffer: PKG + 4*QN8
    mov     [rsp+G2_NXT], rcx
    ; ---- zero GSCR (QN*QSC doubles) — v3 RESTORED as the default q<=8 path
    ; (ADR-0012: v4 measured a tie at q=8, 17.36 vs 17.17 ns/row, both 20%
    ; of the DRAM bound, plus a P2 gate-edge s-reassociation regression;
    ; the v4 body below stays as documented iteration-3 evidence) --------
    vxorpd  ymm0, ymm0, ymm0
    mov     rcx, [rsp+G2_QN]
    imul    rcx, [rsp+G2_QSC]       ; QN*QSC doubles
    mov     rdi, r12
    xor     rax, rax
.zg:
    vmovapd [rdi + rax*8], ymm0     ; GSCR is 64B-aligned by construction
    add     rax, 4
    cmp     rax, rcx
    jb      .zg
    ; ---- zero s, s_c, s_mem, W ----
    mov     rdx, [rsp+UC_S]
    lea     r9, [rsp + G2_SC]       ; s_c
    lea     r11, [r9 + r14*8]       ; s_mem
    mov     [rsp+G2_SMEM], r11
    xor     rax, rax
.zs:
    vmovsd  [rdx + rax*8], xmm0
    vmovsd  [r9 + rax*8], xmm0
    vmovsd  [r11 + rax*8], xmm0
    inc     rax
    cmp     rax, r14
    jb      .zs
    vmovsd  [rsp+UC_WS], xmm0
    vmovsd  [rsp+UC_WC], xmm0
    vmovsd  [rsp+G2_WB], xmm0
    cmp     r14, 8
    ; ---- kernC body dispatch (q fixed for the whole call) ---- [v3 path]
    lea     rax, [rel .kC_q1]
    cmp     r14, 2
    lea     rdx, [rel .kC_q2]
    cmovae  rax, rdx
    cmp     r14, 3
    lea     rdx, [rel .kC_q3]
    cmovae  rax, rdx
    cmp     r14, 4
    lea     rdx, [rel .kC_q4]
    cmovae  rax, rdx
    cmp     r14, 5
    lea     rdx, [rel .kC_q5]
    cmovae  rax, rdx
    cmp     r14, 6
    lea     rdx, [rel .kC_q6]
    cmovae  rax, rdx
    cmp     r14, 7
    lea     rdx, [rel .kC_q7]
    cmovae  rax, rdx
    cmp     r14, 8
    lea     rdx, [rel .kC_q8]
    cmovae  rax, rdx
    mov     [rsp+G2_CJMP], rax
    ; ---- zero kernel-C accumulators: accs ymm0..7 (G[j][0..3]) and
    ; haccs ymm10..13 (G[j][4..7], j=4..7). ymm8/9 row data, 14/15 temps. ----
    vxorpd  ymm1, ymm1, ymm1
    vxorpd  ymm2, ymm2, ymm2
    vxorpd  ymm3, ymm3, ymm3
    vxorpd  ymm4, ymm4, ymm4
    vxorpd  ymm5, ymm5, ymm5
    vxorpd  ymm6, ymm6, ymm6
    vxorpd  ymm7, ymm7, ymm7
    vxorpd  ymm10, ymm10, ymm10
    vxorpd  ymm11, ymm11, ymm11
    vxorpd  ymm12, ymm12, ymm12
    vxorpd  ymm13, ymm13, ymm13
    ; ---- pack group 0 into CUR, then run the group pipeline ----
    mov     qword [rsp+G2_I], 0
    mov     rsi, [rsp+G2_CUR]
    xor     r8, r8                  ; first global row = 0
    PACK_GROUP
.group:
    mov     rcx, [rsp+G2_I]
    cmp     rcx, [rsp+UC_N]
    jae     .epilogue
    ; ---- pack NEXT group into NXT (software pipeline: its DRAM stream
    ; overlaps the current group's kernel FMAs) ----
    mov     rsi, [rsp+G2_NXT]
    lea     r8, [rcx + 4]
    PACK_GROUP
    ; ---- fold the plain sums (covers BOTH groups packed since last fold) --
    FOLD_SUMS
    ; ---- consume the CURRENT group ----
    mov     rsi, [rsp+G2_CUR]
    cmp     r14, 8
    ja      .kernA
    jmp     [rsp+G2_CJMP]
.group_tail:
    ; swap CUR/NXT; i += 4
    mov     rax, [rsp+G2_CUR]
    mov     rdx, [rsp+G2_NXT]
    mov     [rsp+G2_CUR], rdx
    mov     [rsp+G2_NXT], rax
    add     qword [rsp+G2_I], 4
    jmp     .group

; ==============================================================================
; KERNEL A (q >= 9) — per group of 4 rows. rsi = PKG group base.
; Per (i0, j>=i0) chunk: 1 GSCR load + 4 FMA + 1 store; u-chunks hoisted at
; the i0 level; broadcasts in mem form. Clobbers rax,rcx,rdx,rdi,r8-r11,ymm0-8.
; ==============================================================================
.kernA:
    mov     rdi, rsi                ; p0
    lea     r9,  [rsi + r13]        ; p1
    lea     r10, [rsi + r13*2]      ; p2
    lea     r8,  [rsi + r13*2]
    lea     r8,  [r8 + r13]         ; p3
    xor     r11, r11                ; i0off (bytes)
.kA_i0:
    ; u row chunks at columns i0..i0+7: TWO 4-double loads per row (lo/hi).
    vmovupd ymm0, [rdi + r11]       ; row0 u[i0..i3]
    vmovupd ymm1, [r9 + r11]        ; row1
    vmovupd ymm2, [r10 + r11]       ; row2
    vmovupd ymm3, [r8 + r11]        ; row3
    vmovupd ymm4, [rdi + r11 + 32]  ; row0 u[i0+4..i0+7]
    vmovupd ymm5, [r9 + r11 + 32]
    vmovupd ymm6, [r10 + r11 + 32]
    vmovupd ymm7, [r8 + r11 + 32]
    mov     rdx, r11
    shr     rdx, 3                  ; i0 in lanes
    imul    rdx, [rsp+G2_QSC]       ; i0_lanes * QSC (doubles)
    lea     rax, [r12 + rdx*8]      ; GSCR addr of chunk (col i0, rows i0..)
    add     rax, r11                ; + i0 bytes (i0 is 64-aligned)
    mov     rcx, r14
    shl     rcx, 3
    sub     rcx, r11
    shr     rcx, 3                  ; chunk count = q - i0lanes
    mov     rdx, r11                ; j8 cursor starts at i0 (j >= i0)
    push    rbp                     ; save frame ptr (need 2 more regs? no —
    ; use ymm9/10 as bcast temps instead of more GPRs)
.kA_j:
    vbroadcastsd ymm9, [rdi + rdx]
    vbroadcastsd ymm10, [r9 + rdx]
    vbroadcastsd ymm11, [r10 + rdx]
    vbroadcastsd ymm12, [r8 + rdx]
    vmovupd  ymm13, [rax]
    vfmadd231pd ymm13, ymm0, ymm9
    vfmadd231pd ymm13, ymm1, ymm10
    vfmadd231pd ymm13, ymm2, ymm11
    vfmadd231pd ymm13, ymm3, ymm12
    vmovupd  [rax], ymm13
    vmovupd  ymm13, [rax + 32]      ; rows i0+4..i0+7 of column j
    vfmadd231pd ymm13, ymm4, ymm9
    vfmadd231pd ymm13, ymm5, ymm10
    vfmadd231pd ymm13, ymm6, ymm11
    vfmadd231pd ymm13, ymm7, ymm12
    vmovupd  [rax + 32], ymm13
    add     rax, r15                ; next column chunk (QSC8 bytes)
    add     rdx, 8
    dec     rcx
    jnz     .kA_j
    pop     rbp
    add     r11, 64
    lea     rcx, [r14*8]
    cmp     r11, rcx
    jb      .kA_i0
    jmp     .group_tail

; ------------------------------------------------------------------------------
; KERNEL C bodies (q = 1..8), macro-generated, cmov-chain dispatched.
; accs ymm0..q-1 persist across groups (flushed once in the epilogue);
; ymm8 = u row, ymm9 = bcast. rsi = PKG group base; r13 = QN8.
; ------------------------------------------------------------------------------
%macro KERN_C_BODY 1
.kC_q%1:
    ; haccs ymm10..13 were clobbered by the intervening PACK/FOLD (both own
    ; ymm8-15); accs ymm0..7 survive them. Zero haccs here every entry (the
    ; per-group flush already re-zeroed accs; this is idempotent for them).
%if %1 > 4
    %assign ZR 10
    %rep %1 - 4
        vxorpd  ymm %+ ZR, ymm %+ ZR, ymm %+ ZR
        %assign ZR ZR+1
    %endrep
%endif
    ; q = %1 columns. Per row: load lo (u[0..3]) and hi (u[4..7], zero-padded)
    ; as one unaligned ymm each; FMA acc_j += lo*bcast(u[j]) for j=0..q-1
    ; (G[j][0..3]); FMA hacc_{j} += hi*bcast(u[j]) for j=4..q-1 (G[j][4..7]),
    ; hacc lives in ymm10..13. PACK (next group) clobbers ymm8-15, so the
    ; accumulators are flushed into GSCR and re-zeroed at the END of every
    ; group, BEFORE .group_tail swaps in the next PACK.
    %assign RR 0
    %rep 4
        vmovupd ymm8, [rsi]             ; u[0..3]
        vmovupd ymm9, [rsi + 32]        ; u[4..7] (zero-padded)
        %assign JJ 0
        %rep %1
            vbroadcastsd ymm15, [rsi + JJ*8]
            vfmadd231pd ymm %+ JJ, ymm8, ymm15
            %if JJ >= 4
            %assign HJ JJ+6
            vfmadd231pd ymm %+ HJ, ymm9, ymm15
            %endif
            %assign JJ JJ+1
        %endrep
        add     rsi, r13
        %assign RR RR+1
    %endrep
    ; ---- per-group flush into GSCR (r12 base, r15 = QSC8) ----
    %assign JJ 0
    %rep %1
        ; rdx = j*QSC8 ; GSCR[j] rows 0..3 += acc_j
        mov     rdx, JJ
        imul    rdx, [rsp+G2_QSC]
        lea     rax, [r12 + rdx*8]
        vmovupd ymm14, [rax]
        vaddpd  ymm14, ymm14, ymm %+ JJ
        vmovupd [rax], ymm14
        vxorpd  ymm %+ JJ, ymm %+ JJ, ymm %+ JJ
        %if JJ >= 4
        %assign HREG JJ+6
        vmovupd ymm14, [rax + 32]
        vaddpd  ymm14, ymm14, ymm %+ HREG
        vmovupd [rax + 32], ymm14
        vxorpd  ymm %+ HREG, ymm %+ HREG, ymm %+ HREG
        %endif
        %assign JJ JJ+1
    %endrep
    jmp     .group_tail
%endmacro

%assign CQ 1
%rep 8
    KERN_C_BODY CQ
    %assign CQ CQ+1
%endrep

; ------------------------------------------------------------------------------
; FLUSH_SLOT n — if j (rcx) == n, add ymm{n} into GSCR[j*QSC8] (rax points
; there). Macro so %%-labels work; straight-line, no fallthrough issue: each
; slot either jumps to .flushC_next or falls through to the next slot.
; ------------------------------------------------------------------------------
%macro FLUSH_SLOT 1
    cmp     rcx, %1
    jne     %%skip
    vmovupd ymm14, [rax]
    vaddpd  ymm14, ymm14, ymm %+ %1
    vmovupd [rax], ymm14
%if %1 >= 4
    vmovupd ymm14, [rax + 32]
%assign HSLOT %1 + 6
    vaddpd  ymm14, ymm14, ymm %+ HSLOT
    vmovupd [rax + 32], ymm14
%endif
    jmp     .flushC_next
%%skip:
%endmacro


; ------------------------------------------------------------------------------
; epilogue — flush kernel-C accs (q <= 8), compact GSCR upper triangle into G
; (strictly-lower zeroed), fold s_c into s, write *Wout.
; ------------------------------------------------------------------------------
.epilogue:
    cmp     r14, 8
    ja      .compact
    ; flush C accs: GSCR[j*QSC8] += acc_j for j < q (lanes >= q stay zero)
    xor     ecx, ecx                ; j
    xor     edx, edx                ; j*QSC8
.flushC:
    cmp     rcx, r14
    jae     .compact
    lea     rax, [r12 + rdx]
    FLUSH_SLOT 0
    FLUSH_SLOT 1
    FLUSH_SLOT 2
    FLUSH_SLOT 3
    FLUSH_SLOT 4
    FLUSH_SLOT 5
    FLUSH_SLOT 6
    FLUSH_SLOT 7
.flushC_next:
    add     rdx, r15
    inc     rcx
    jmp     .flushC

.compact:
    mov     rdi, [rsp+UC_G]         ; G out
    xor     rsi, rsi                ; j
.comp_j:
    cmp     rsi, r14
    jae     .comp_done
    mov     rdx, rsi
    imul    rdx, [rsp+G2_QSC]
    shl     rdx, 3                  ; j*QSC8 bytes
    lea     rcx, [rdi + rsi*8]      ; G + j*8
    xor     r8, r8                  ; i
.comp_copy:
    cmp     r8, rsi
    ja      .comp_zero
    lea     rax, [r12 + rdx]        ; column base (fixed per j)
    vmovsd xmm0, [rax + r8*8]      ; row offset from r8*8 alone
    vmovsd [rcx], xmm0
    add     rcx, [rsp+G2_Q8]
    inc     r8
    jmp     .comp_copy
.comp_zero:
    cmp     r8, r14
    jae     .comp_j_next
    vxorpd  xmm0, xmm0, xmm0
    vmovsd [rcx], xmm0
    add     rcx, [rsp+G2_Q8]
    inc     r8
    jmp     .comp_zero
.comp_j_next:
    inc     rsi
    jmp     .comp_j
.comp_done:
    ; s[k] += s_c[k]; *Wout = W_s + W_c
    mov     rdx, [rsp+UC_S]
    lea     r9, [rsp + G2_SC]
    xor     rcx, rcx
.sfold:
    cmp     rcx, r14
    jae     .sfold_done
    vmovsd xmm0, [rdx + rcx*8]
    vaddsd xmm0, xmm0, [r9 + rcx*8]
    vmovsd [rdx + rcx*8], xmm0
    inc     rcx
    jmp     .sfold
.sfold_done:
    mov     rax, [rsp+UC_WOUT]
    vmovsd xmm0, [rsp+UC_WS]
    vaddsd xmm0, xmm0, [rsp+UC_WC]
    vmovsd [rax], xmm0
    xor     eax, eax
    jmp     .out
.nonfinite:
    mov     eax, OLS_E_NONFINITE
    jmp     .out
.ework_pop:
    pop     rdi
.ework:
    mov     eax, OLS_E_WORKSPACE
.out:
    vzeroupper
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ==============================================================================
; V4 — fused single-pass kernel for q <= 8 (ADR-0012; perf-log #28).
; No PKG, no GSCR, no group pipeline: one row loop, u computed in registers,
; column accumulators persist across the WHOLE call.
;   regs : accs lo ymm0..7 (acc_j lanes = G[j][0..3]), accs hi ymm8..11
;          (G[j][4..7], j=4..7), u_lo ymm12, u_hi ymm13, bcast ymm14,
;          sw bcast / temp ymm15.
;   frame (rsp + V4BASE + off, V4BASE stored in G2_CUR): ULO/UHI u staging,
;          SLO/SHI s batch plain sums, MASKL/MASKH lane masks, SCLO/SCHI
;          s Neumaier corrections, plus s_c scalars handled vectorially.
;   s contract: s = sum w(z-c); per-row vector plain accumulation into
;          SLO/SHI, folded into Neumaier pairs (S, SCLO/SHI) every 32 rows
;          (V4_BATCH) and at the end. W: per-row plain xmm fold into the
;          same 2-level structure v3 uses (WS/WC via G2_WB batch slot).
;   NaN semantics identical: any non-finite z (incl. y) or w, or w < 0
;          (sqrt -> NaN) -> OLS_E_NONFINITE, checked per row BEFORE the
;          accumulators see the row. Lanes >= q are force-zeroed via the
;          lane masks BEFORE the NaN check (garbage X-tail never judged).
; ==============================================================================
%define V4_BATCH 4
; 4: v3's plain-sum group was 4 rows (one PACK group) folded into the
; Neumaier pair per fold. Matching that cut keeps the documented error
; model (P1 bound) AND the P2 permutation slack measured for v3 —
; 32/8-row batches regressed both gates (1.156e-10 on P2 n100p3).

; ==============================================================================
; V4 — fused single-pass kernel for q <= 8 (ADR-0012; perf-log #28).
; No PKG, no GSCR, no group pipeline: one row loop; u computed in registers;
; column accumulators persist across the WHOLE call.
;   regs : lo accs ymm0..7 (acc_j, lanes 0..3 = G[j][0..3], j < q),
;          hi accs ymm8..11 (G[j][4..7], j = 4..q-1), u_lo ymm12, u_hi ymm13,
;          bcast temp ymm14, sw bcast temp ymm15.
;   frame (base = [rsp+G2_CUR] = V4BASE, an rsp-relative offset):
;          ULO/UHI  u staging (bcast source), SLO/SHI s batch plain sums,
;          MASKL/MASKH lane masks (lane k set iff lane-index < q).
;   s contract: s = sum w(z-c). Per row: s_batch += sw*u (vector, exact in
;          the masked lanes). Every V4_BATCH rows and at the end: per-column
;          scalar Neumaier fold into (s, s_c) — same two-level structure and
;          the same algorithm shape as v3's FOLD_SUMS.
;   W contract: per-row plain batch sum in G2_WB; Neumaier fold into
;          (WS, WC) at each batch boundary and at the end (v3 structure).
;   NaN semantics: w NaN, w < 0 (sqrt -> NaN), z or y NaN -> NONFINITE,
;          checked per row BEFORE the accumulators see the row. Garbage
;          lanes (>= q) are masked to 0 BEFORE the NaN check (masked X-tail
;          loads can never trigger a false NONFINITE).
;   G contract: upper triangle written from accs, strictly-lower zeroed.
; ==============================================================================
%define V4_BATCH 4
; 4: v3's plain-sum group was 4 rows (one PACK group) folded into the
; Neumaier pair per fold. Matching that cut keeps the documented error
; model (P1 bound) AND the P2 permutation slack measured for v3 —
; 32/8-row batches regressed both gates (1.156e-10 on P2 n100p3).

; --- v4 fused kernel (q <= 8): NOT DISPATCHED (ADR-0012; measured tie
; at q=8 + P2 gate-edge regression; kept as iteration-3 evidence —
; re-enable for future hosts by jumping here after the .zs zeroing,
; q <= 8 only). ---
.v4:
    ; ---- v4 slot base: end of the v3 frame area (s_c + s_mem), 32-aligned,
    ; plus the 256 the frame calc reserved. Park the rsp-relative offset in
    ; G2_CUR (no PKG exists for q <= 8). ----
    lea     rax, [r14*8]
    lea     rax, [rax*2 + G2_SC]
    add     rax, 31
    and     rax, -32
    lea     rsi, [rsp + rax]          ; rsi = V4 slot base ADDRESS
    mov     [rsp+G2_CUR], rsi         ; parked for the epilogue
    ; ---- zero the v4 frame areas (SLO/SHI; ULO/UHI/MASK written before use)
    vxorpd  ymm0, ymm0, ymm0
    vmovupd [rsi+V4_SLO], ymm0
    vmovupd [rsi+V4_SHI], ymm0
    ; ---- c staging: CLO/CHI = c[0..q) zero-padded to 8 lanes (the row
    ; bodies load c in 4-double chunks; padding keeps every read IN BOUNDS
    ; — c is exactly q doubles at the caller, garbage there is masked but
    ; the READ itself would cross the buffer (guard-test violation).) ----
    mov     rdx, [rsp+UC_C]
    xor     rcx, rcx
.v4_cstage:
    cmp     rcx, r14
    jae     .v4_czero
    mov     rax, [rdx + rcx*8]
    jmp     .v4_cput
.v4_czero:
    xor     eax, eax
.v4_cput:
    cmp     rcx, 4
    jb      .v4_clo
    mov     [rsi+V4_CHI + rcx*8 - 32], rax
    jmp     .v4_cnext
.v4_clo:
    mov     [rsi+V4_CLO + rcx*8], rax
.v4_cnext:
    inc     rcx
    cmp     rcx, 8
    jb      .v4_cstage
    ; ---- lane masks (once per call): MASKL lane k = k < q, MASKH = 4+k < q
    xor     rcx, rcx                  ; k = 0..3
.v4_mask:
    mov     rdx, -1
    cmp     rcx, r14
    jb      .v4_ml_set
    xor     edx, edx
.v4_ml_set:
    mov     [rsi+V4_MASKL + rcx*8], rdx
    mov     rdx, -1
    lea     r8, [rcx + 4]
    cmp     r8, r14
    jb      .v4_mh_set
    xor     edx, edx
.v4_mh_set:
    mov     [rsi+V4_MASKH + rcx*8], rdx
    inc     rcx
    cmp     rcx, 4
    jb      .v4_mask
    ; ---- zero the accumulators (lo ymm0..7, hi ymm8..11) -----------------
    vxorpd  ymm0, ymm0, ymm0
    vxorpd  ymm1, ymm1, ymm1
    vxorpd  ymm2, ymm2, ymm2
    vxorpd  ymm3, ymm3, ymm3
    vxorpd  ymm4, ymm4, ymm4
    vxorpd  ymm5, ymm5, ymm5
    vxorpd  ymm6, ymm6, ymm6
    vxorpd  ymm7, ymm7, ymm7
    vxorpd  ymm8, ymm8, ymm8
    vxorpd  ymm9, ymm9, ymm9
    vxorpd  ymm10, ymm10, ymm10
    vxorpd  ymm11, ymm11, ymm11
    ; ---- register plan for the row loop:
    ;   rbx = X base, r13 = ldx8, r11 = y base, r10 = w base (or 0),
    ;   rdi = n, rsi = V4 slot base, r12 = i (row), r15 = batch left
    ;   (c is consumed via the CLO/CHI staging slots; r9 is free)
    mov     rbx, [rsp+UC_X]
    mov     r13, [rsp+UC_LDX8]
    mov     r11, [rsp+UC_Y]
    mov     r10, [rsp+UC_W]
    mov     rdi, [rsp+UC_N]
    xor     r12, r12                  ; i = 0
    mov     r15, V4_BATCH
    ; ---- per-q row-body dispatch (q is call-fixed; blend imm is q-literal)
    lea     rax, [rel .v4q1]
    cmp     r14, 2
    lea     rdx, [rel .v4q2]
    cmovae  rax, rdx
    cmp     r14, 3
    lea     rdx, [rel .v4q3]
    cmovae  rax, rdx
    cmp     r14, 4
    lea     rdx, [rel .v4q4]
    cmovae  rax, rdx
    cmp     r14, 5
    lea     rdx, [rel .v4q5]
    cmovae  rax, rdx
    cmp     r14, 6
    lea     rdx, [rel .v4q6]
    cmovae  rax, rdx
    cmp     r14, 7
    lea     rdx, [rel .v4q7]
    cmovae  rax, rdx
    cmp     r14, 8
    lea     rdx, [rel .v4q8]
    cmovae  rax, rdx
    mov     [rsp+G2_NXT], rax         ; row-body entry (G2_NXT free: no PKG)
.v4_row:
    cmp     r12, rdi
    jae     .v4_fold_final
    ; ---- w[i] -> sw = sqrt(w) -> [rsi+V4_SW]; NaN and w<0 checks ---------
    ; The momentary load register is xmm12 (u_lo temp, dead between rows):
    ; xmm8..11 alias the PERSISTENT hi accs — any scalar write there (the
    ; diag[4] += sw-per-row bug) corrupts G[4][4].
    test    r10, r10
    jz      .v4_w1
    vmovsd  xmm12, [r10 + r12*8]
    jmp     .v4_wh
.v4_w1:
    mov     eax, 1
    vcvtsi2sd xmm12, xmm12, eax       ; sw = 1
    jmp     .v4_ws
.v4_wh:
    vucomisd xmm12, xmm12             ; w NaN?
    jp      .nonfinite
    vsqrtsd xmm12, xmm12, xmm12       ; sw = sqrt(w); w < 0 -> NaN
.v4_ws:
    vucomisd xmm12, xmm12             ; sw NaN (w < 0 or sqrt NaN)?
    jp      .nonfinite
    vmovsd  [rsi+V4_SW], xmm12        ; park sw
    ; ---- X row base -> r8 --------------------------------------------------
    mov     rax, r12
    imul    rax, r13
    lea     r8, [rbx + rax]
    jmp     [rsp+G2_NXT]              ; per-q row body

; ==============================================================================
; V4_ROW_BODY q — one row for fixed q (1..8). Entry: r8 = X row, r11 = y,
; r12 = i, sw at [rsi+V4_SW], rsi = V4BASE, r14 = q. Builds masked u_lo/u_hi
; (u = sw*(z - c), y at lane q-1, lanes >= q zeroed), NaN-checks, FMAs into
; the persistent accs, accumulates the s batch vectors, jumps .v4_tail.
; Touches ONLY ymm12..15 + GPR scratch outside the FMA accumulators.
; ==============================================================================
%macro V4_ROW_BODY 1
%if %1 <= 4
.v4q%1:
    ; Build u_lo = sw*(z - c), k = 0..q-1, with SCALAR in-bounds loads:
    ; z[k] (k < q-1) from the X row; z[q-1] = y[i]. Exactly q-1 doubles at
    ; [r8 .. r8+(q-2)*8] plus y[i] are read — never past the row, unlike a
    ; 32B chunk load (the last X row would overrun the buffer: guard risk).
    ; Staging slots persist across rows: mask kills lanes >= q after reload.
    %assign KK 0
    %rep %1
      %if KK < %1 - 1
        vmovsd  xmm14, [r8 + KK*8]
      %else
        vmovsd  xmm14, [r11 + r12*8]  ; y at k = q-1
      %endif
        vsubsd  xmm14, xmm14, [rsi+V4_CLO + KK*8]
        vmulsd  xmm14, xmm14, [rsi+V4_SW]  ; u_k = sw*(z_k - c_k)
        vmovsd  [rsi+V4_ULO + KK*8], xmm14
        %assign KK KK+1
    %endrep
    vmovupd ymm12, [rsi+V4_ULO]
    vandpd  ymm12, ymm12, [rsi+V4_MASKL]   ; kill stale lanes >= q
    vxorpd  ymm13, ymm13, ymm13       ; u_hi = 0 for q <= 4
    vxorpd  ymm14, ymm14, ymm14
    vcmppd  ymm14, ymm12, ymm12, 3    ; UNORD(u_lo)?
    vmovmskpd eax, ymm14
    test    eax, eax
    jnz     .nonfinite
%else
.v4q%1:
    ; Scalar in-bounds build for both halves (same scheme as q <= 4;
    ; hi-half lanes >= q-4 masked after reload). c[k] for k >= 4 lives in
    ; the CHI staging slot (the q>=5 dG~1.0 bug read CLO+32..56 — past the
    ; 32B CLO slot into MASKL — so u[4..] subtracted mask bits).
    %assign KK 0
    %rep %1
      %if KK < %1 - 1
        vmovsd  xmm14, [r8 + KK*8]
      %else
        vmovsd  xmm14, [r11 + r12*8]  ; y at k = q-1
      %endif
      %if KK < 4
        vsubsd  xmm14, xmm14, [rsi+V4_CLO + KK*8]
      %else
        vsubsd  xmm14, xmm14, [rsi+V4_CHI + (KK-4)*8]
      %endif
        vmulsd  xmm14, xmm14, [rsi+V4_SW]
      %if KK < 4
        vmovsd  [rsi+V4_ULO + KK*8], xmm14
      %else
        vmovsd  [rsi+V4_UHI + (KK-4)*8], xmm14
      %endif
        %assign KK KK+1
    %endrep
    vmovupd ymm12, [rsi+V4_ULO]
    vmovupd ymm13, [rsi+V4_UHI]
    vandpd  ymm12, ymm12, [rsi+V4_MASKL]
    vandpd  ymm13, ymm13, [rsi+V4_MASKH]
    vxorpd  ymm14, ymm14, ymm14
    vcmppd  ymm14, ymm12, ymm12, 3
    vmovmskpd eax, ymm14
    test    eax, eax
    jnz     .nonfinite
    vcmppd  ymm14, ymm13, ymm13, 3
    vmovmskpd eax, ymm14
    test    eax, eax
    jnz     .nonfinite
%endif
    ; ---- (u halves staged by the scalar build above) -----------------------
    ; ---- acc_j += u_lo * bcast(u[j])   (j = 0..q-1, lo acc ymm_j)
    ;      acc_j += u_hi * bcast(u[j])   (j = 4..q-1, hi acc ymm_{j+4})
    ;      NOTE: at j >= 4, bcast(u[j]) is a UHI slot; u[j] for j < 4 is ULO.
    %assign JJ 0
    %rep %1
      %assign LOR JJ
      %assign HIR JJ+4
      %if JJ < 4
        vbroadcastsd ymm14, [rsi+V4_ULO + LOR*8]
        vfmadd231pd ymm %+ LOR, ymm12, ymm14
      %else
        vbroadcastsd ymm14, [rsi+V4_UHI + (LOR-4)*8]
        vfmadd231pd ymm %+ LOR, ymm12, ymm14     ; G[j][0..3] += u_lo*u[j]
        vfmadd231pd ymm %+ HIR, ymm13, ymm14     ; G[j][4..7] += u_hi*u[j]
      %endif
        %assign JJ JJ+1
    %endrep
    ; ---- s batch: s_batch += sw * u = w(z-c) — lo, and hi when q > 4 ------
    ; vmulpd (NOT vfmadd on ymm14: it still holds the last bcast u[j] — the
    ; q=1 2x-s bug: FMA into a dirty destination added bcast(u[0]) + sw*u[0])
    vbroadcastsd ymm15, [rsi+V4_SW]
    vmulpd  ymm14, ymm12, ymm15       ; ymm14 = sw*u_lo (fresh dest)
    vmovupd ymm15, [rsi+V4_SLO]
    vaddpd  ymm15, ymm15, ymm14
    vmovupd [rsi+V4_SLO], ymm15
%if %1 > 4
    vbroadcastsd ymm15, [rsi+V4_SW]
    vmulpd  ymm14, ymm13, ymm15       ; ymm14 = sw*u_hi (fresh dest)
    vmovupd ymm15, [rsi+V4_SHI]
    vaddpd  ymm15, ymm15, ymm14
    vmovupd [rsi+V4_SHI], ymm15
%endif
    jmp     .v4_tail
%endmacro

; ---- instantiate the 8 per-q row bodies -------------------------------------
%assign VQ 1
%rep 8
    V4_ROW_BODY VQ
    %assign VQ VQ+1
%endrep

; ---- v4 row tail: W batch accumulation + batch bookkeeping + loop ----------
; xmm12..15 are FREE here (row temps); accs ymm0..11 are NOT touched.
; The W batch plain sum lives in G2_WB (zeroed by the v3 prologue .zs path
; — note .zs runs for BOTH paths since it zeroes G2_WB explicitly).
.v4_tail:
    test    r10, r10
    jz      .v4_t_w1
    vmovsd  xmm12, [r10 + r12*8]      ; w[i]
    jmp     .v4_t_wh
.v4_t_w1:
    mov     eax, 1
    vcvtsi2sd xmm12, xmm12, eax
.v4_t_wh:
    vmovsd  xmm13, [rsp+G2_WB]
    vaddsd  xmm13, xmm13, xmm12
    vmovsd  [rsp+G2_WB], xmm13
    inc     r12
    dec     r15
    jnz     .v4_row
    ; ---- batch boundary: fold s batch and W batch, reset ------------------
    ; NO call/ret: a call pushes a return address BELOW rsp, shifting every
    ; rsp-relative slot by 8 inside the callee (the q=4 garbage bug: s[2],
    ; Wout read never-written memory). Straight-line jump chain instead.
    jmp     .v4_fold_batch
.v4_fold_batch:
    ; per-column scalar Neumaier fold of the s batch sums into (s, s_c);
    ; xmm12..15 + UC_RB scratch only (accs untouched). rsi = slot base.
    xor     rcx, rcx                  ; k
.v4_fb_k:
    cmp     rcx, r14
    jae     .v4_fb_w
    lea     rdx, [rsi+V4_SLO]         ; k < 4 -> SLO slots
    cmp     rcx, 4
    jb      .v4_fb_lo
    lea     rdx, [rsi+V4_SHI - 32]    ; k >= 4 -> SHI slot (k-4)*8 = k*8-32
.v4_fb_lo:
    vmovsd  xmm12, [rdx + rcx*8]      ; x = batch s sum, column k
    mov     rax, [rsp+UC_S]
    vmovsd  xmm13, [rax + rcx*8]      ; s
    vaddsd  xmm14, xmm13, xmm12       ; t = s + x
.v4_fb_impl:
    mov     rax, 0x7FFFFFFFFFFFFFFF
    vmovq   xmm15, rax
    vandpd  xmm15, xmm15, xmm13       ; |s|
    vmovq   rdx, xmm15                ; save |s| bits
    vmovq   xmm15, rax
    vandpd  xmm15, xmm15, xmm12       ; |x|
    vmovq   [rsp+UC_RB], xmm15        ; |x| -> UC_RB (free slot in AVX2 path)
    vmovq   xmm15, rdx                ; |s|
    vucomisd xmm15, [rsp+UC_RB]       ; |s| vs |x|
    jae     .v4_fb_e1
    ; e2 = (x - t) + s
    vsubsd  xmm15, xmm12, xmm14
    vaddsd  xmm15, xmm15, xmm13
    jmp     .v4_fb_have_e
.v4_fb_e1:
    ; e1 = (s - t) + x
    vsubsd  xmm15, xmm13, xmm14
    vaddsd  xmm15, xmm15, xmm12
.v4_fb_have_e:
    ; s_c[k] += e ; s[k] = t
    mov     rax, [rsp+UC_S]
    lea     rdx, [rsp + G2_SC]        ; s_c base
    vmovsd  xmm12, [rdx + rcx*8]
    vaddsd  xmm12, xmm12, xmm15
    vmovsd  [rdx + rcx*8], xmm12
    vmovsd  [rax + rcx*8], xmm14      ; s = t
    inc     rcx
    jmp     .v4_fb_k
.v4_fb_w:
    ; ---- W batch Neumaier fold: WS/WC += G2_WB (xmm12..15 only) ----------
    vmovsd  xmm12, [rsp+G2_WB]        ; x = batch W
    vmovsd  xmm13, [rsp+UC_WS]        ; s = WS
    vaddsd  xmm14, xmm13, xmm12       ; t
    mov     rax, 0x7FFFFFFFFFFFFFFF
    vmovq   xmm15, rax
    vandpd  xmm15, xmm15, xmm13       ; |s|
    vmovq   rdx, xmm15                ; save |s|
    vmovq   xmm15, rax
    vandpd  xmm15, xmm15, xmm12       ; |x|
    vmovq   [rsp+UC_RB], xmm15
    vmovq   xmm15, rdx                ; |s|
    vucomisd xmm15, [rsp+UC_RB]
    jae     .v4_fb_we1
    vsubsd  xmm15, xmm12, xmm14
    vaddsd  xmm15, xmm15, xmm13
    jmp     .v4_fb_whave
.v4_fb_we1:
    vsubsd  xmm15, xmm13, xmm14
    vaddsd  xmm15, xmm15, xmm12
.v4_fb_whave:
    vmovsd  xmm12, [rsp+UC_WC]
    vaddsd  xmm12, xmm12, xmm15
    vmovsd  [rsp+UC_WC], xmm12
    vmovsd  [rsp+UC_WS], xmm14        ; WS = t
    vxorpd  xmm12, xmm12, xmm12
    vmovsd  [rsp+G2_WB], xmm12        ; batch W = 0
    jmp     .v4_folded                ; (was: ret — the call is gone)

.v4_folded:
    ; ---- reset the s batch sums; restart the batch counter ----------------
    vxorpd  ymm12, ymm12, ymm12
    vmovupd [rsi+V4_SLO], ymm12
    vmovupd [rsi+V4_SHI], ymm12
    mov     r15, V4_BATCH
    jmp     .v4_row

; ---- v4 final: fold the partial batch, extract G, final W, exit -----------
.v4_fold_final:
    ; The last batch is partial (r15 rows may be < V4_BATCH; r15 can be 0
    ; only when n is a multiple of V4_BATCH — then SLO/SHI/G2_WB are zero
    ; and every fold below is an exact no-op). Straight-line duplicate of
    ; .v4_fold_batch (jump-target, no ret). rsi = V4 slot base.
    xor     rcx, rcx
    jmp     .v4_ff_k
.v4_ff_k:
    cmp     rcx, r14
    jae     .v4_ff_w
    lea     rdx, [rsi+V4_SLO]
    cmp     rcx, 4
    jb      .v4_ff_lo
    lea     rdx, [rsi+V4_SHI - 32]
.v4_ff_lo:
    vmovsd  xmm12, [rdx + rcx*8]      ; x
    mov     rax, [rsp+UC_S]
    vmovsd  xmm13, [rax + rcx*8]      ; s
    vaddsd  xmm14, xmm13, xmm12       ; t
    mov     rax, 0x7FFFFFFFFFFFFFFF
    vmovq   xmm15, rax
    vandpd  xmm15, xmm15, xmm13       ; |s|
    vmovq   rdx, xmm15
    vmovq   xmm15, rax
    vandpd  xmm15, xmm15, xmm12       ; |x|
    vmovq   [rsp+UC_RB], xmm15
    vmovq   xmm15, rdx                ; |s|
    vucomisd xmm15, [rsp+UC_RB]
    jae     .v4_ff_e1
    vsubsd  xmm15, xmm12, xmm14
    vaddsd  xmm15, xmm15, xmm13
    jmp     .v4_ff_have
.v4_ff_e1:
    vsubsd  xmm15, xmm13, xmm14
    vaddsd  xmm15, xmm15, xmm12
.v4_ff_have:
    mov     rax, [rsp+UC_S]
    lea     rdx, [rsp + G2_SC]
    vmovsd  xmm12, [rdx + rcx*8]
    vaddsd  xmm12, xmm12, xmm15
    vmovsd  [rdx + rcx*8], xmm12
    vmovsd  [rax + rcx*8], xmm14
    inc     rcx
    jmp     .v4_ff_k
.v4_ff_w:
    ; final W fold (batch may be empty: G2_WB = 0, folding 0 is a no-op)
    vmovsd  xmm12, [rsp+G2_WB]
    vmovsd  xmm13, [rsp+UC_WS]
    vaddsd  xmm14, xmm13, xmm12       ; t
    mov     rax, 0x7FFFFFFFFFFFFFFF
    vmovq   xmm15, rax
    vandpd  xmm15, xmm15, xmm13       ; |s|
    vmovq   rdx, xmm15
    vmovq   xmm15, rax
    vandpd  xmm15, xmm15, xmm12       ; |x|
    vmovq   [rsp+UC_RB], xmm15
    vmovq   xmm15, rdx
    vucomisd xmm15, [rsp+UC_RB]
    jae     .v4_ff_we1
    vsubsd  xmm15, xmm12, xmm14
    vaddsd  xmm15, xmm15, xmm13
    jmp     .v4_ff_whave
.v4_ff_we1:
    vsubsd  xmm15, xmm13, xmm14
    vaddsd  xmm15, xmm15, xmm12
.v4_ff_whave:
    vmovsd  xmm12, [rsp+UC_WC]
    vaddsd  xmm12, xmm12, xmm15
    vmovsd  [rsp+UC_WC], xmm12
    vmovsd  [rsp+UC_WS], xmm14
    ; ---- Wout = WS + WC ----------------------------------------------------
    mov     rax, [rsp+UC_WOUT]
    vmovsd  xmm12, [rsp+UC_WS]
    vaddsd  xmm12, xmm12, [rsp+UC_WC]
    vmovsd  [rax], xmm12
    ; ---- s final: s[k] += s_c[k] ------------------------------------------
    mov     rdx, [rsp+UC_S]
    lea     rax, [rsp + G2_SC]
    xor     rcx, rcx
.v4_sf:
    cmp     rcx, r14
    jae     .v4_gext
    vmovsd  xmm12, [rdx + rcx*8]
    vaddsd  xmm12, xmm12, [rax + rcx*8]
    vmovsd  [rdx + rcx*8], xmm12
    inc     rcx
    jmp     .v4_sf
    ; ---- G extraction: G[j][k] = acc_j lane k, k <= j; lower zeroed -------
    ; acc_j in ymm_j (j < 4 or j < q with lanes 0..3) and ymm_{j+4} (j >= 4,
    ; lanes 4..7). Store each acc to ULO staging once, then scalar-pick.
.v4_gext:
    mov     rdi, [rsp+UC_G]
    xor     rsi, rsi                  ; j  (rsi free: V4BASE reloaded per j)
.v4_gj:
    cmp     rsi, r14
    jae     .v4_done
    mov     rax, [rsp+G2_CUR]         ; V4 slot base ADDRESS
    ; store acc_j (register chosen at ASSEMBLY time — q-loop unrolled by
    ; macro? No: q is runtime. Register choice needs a j-switch. q <= 8:
    ; an 8-way branch is fine for the epilogue.)
    cmp     rsi, 0
    jne     .v4_gj_n0
    vmovupd [rax+V4_ULO], ymm0
    jmp     .v4_gk
.v4_gj_n0:
    cmp     rsi, 1
    jne     .v4_gj_n1
    vmovupd [rax+V4_ULO], ymm1
    jmp     .v4_gk
.v4_gj_n1:
    cmp     rsi, 2
    jne     .v4_gj_n2
    vmovupd [rax+V4_ULO], ymm2
    jmp     .v4_gk
.v4_gj_n2:
    cmp     rsi, 3
    jne     .v4_gj_n3
    vmovupd [rax+V4_ULO], ymm3
    jmp     .v4_gk
.v4_gj_n3:
    cmp     rsi, 4
    jne     .v4_gj_n4
    vmovupd [rax+V4_ULO], ymm4
    jmp     .v4_gk
.v4_gj_n4:
    cmp     rsi, 5
    jne     .v4_gj_n5
    vmovupd [rax+V4_ULO], ymm5
    jmp     .v4_gk
.v4_gj_n5:
    cmp     rsi, 6
    jne     .v4_gj_n6
    vmovupd [rax+V4_ULO], ymm6
    jmp     .v4_gk
.v4_gj_n6:
    vmovupd [rax+V4_ULO], ymm7    ; j = 7 (q <= 8)
.v4_gk:
    ; k = 0..j: G[j][k] = ULO[k]  (lo lanes 0..3) or UHI[k-4] (k >= 4) —
    ; for j >= 4 the hi acc holds lanes 4..7: it was NOT stored. Store BOTH
    ; halves: hi acc of column j is ymm_{j+4} for 4 <= j < q.
    cmp     rsi, 4
    jb      .v4_gk_lo
    ; store hi acc: register switch again (j+4 in ymm8..11)
    cmp     rsi, 4
    jne     .v4_gk_h5
    vmovupd [rax+V4_UHI], ymm8
    jmp     .v4_gk_go
.v4_gk_h5:
    cmp     rsi, 5
    jne     .v4_gk_h6
    vmovupd [rax+V4_UHI], ymm9
    jmp     .v4_gk_go
.v4_gk_h6:
    cmp     rsi, 6
    jne     .v4_gk_h7
    vmovupd [rax+V4_UHI], ymm10
    jmp     .v4_gk_go
.v4_gk_h7:
    vmovupd [rax+V4_UHI], ymm11
.v4_gk_go:
    ; fall through to the k loop with both halves staged
.v4_gk_lo:
    xor     rcx, rcx                  ; k
    lea     rdx, [rdi + rsi*8]        ; &G[j][0]
.v4_gkk:
    cmp     rcx, rsi
    ja      .v4_gz
    cmp     rcx, 4
    jae     .v4_gk_hi_lane
    vmovsd  xmm12, [rax+V4_ULO + rcx*8]
    vmovsd  [rdx], xmm12
    jmp     .v4_gk_next
.v4_gk_hi_lane:
    vmovsd  xmm12, [rax+V4_UHI + rcx*8 - 32]
    vmovsd  [rdx], xmm12
    jmp     .v4_gk_next
.v4_gz:
    ; k > j: strictly-lower zero
    vxorpd  xmm12, xmm12, xmm12
    vmovsd  [rdx], xmm12
.v4_gk_next:
    add     rdx, [rsp+G2_Q8]          ; next row of column j (G row stride q*8)
    inc     rcx
    cmp     rcx, r14
    jb      .v4_gkk
    inc     rsi
    jmp     .v4_gj
.v4_done:
    xor     eax, eax
    jmp     .out

; ==============================================================================
; int32_t ols_gram_f64_sse2(const ols_gram_args_t *A)
; Scalar reference: identical contract, strictly sequential accumulation
; (G[j][k] += u_j*u_k one row at a time, s per column Neumaier).
; Cost    : PREDICTED (perf-log #12): q=8 ~70-90 cy/row; correctness vehicle.
; ==============================================================================
; Args    : rdi = A (ols_gram_args_t*)
; Returns : eax = OLS_OK | OLS_E_NONFINITE
; Requires: SSE2; 1 <= q = A->p+1 <= 512; G/s OVERWRITTEN (zeroed first).
; Memory  : reads A once; X rows at stride ldx8, y, w?, c; writes G/s/*Wout.
; Clobbers: caller-saved + rbx,rbp,r12-r15 (pushed/popped), xmm0-xmm15.
; Numerics: sequential accumulation; s per column Neumaier (sqrt-w staged).
; Cost    : perf-log #12: q=8 ~70-90 cy/row; correctness vehicle.
global ols_gram_f64_sse2:function
ols_gram_f64_sse2:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8
    mov     rbx, [rdi+GA_X]         ; X
    mov     r12, [rdi+GA_LDX8]
    mov     r13, [rdi+GA_Y]
    mov     r14, [rdi+GA_N]         ; n
    mov     r15, [rdi+GA_P]
    inc     r15                     ; q
    ; frame: ctrl(UC_SZ) + s_c(q*8) + u row(q*8); u at UC_SC + q*8
    ; (the original layout put u at UC_U=112 INSIDE the ctrl block, which
    ; overlapped s_c for q > 2 and contaminated the sums — found by the
    ; unique-value differential test; fixed here and guarded by it.)
    lea     rax, [r15*8]
    lea     rdx, [rax*2 + UC_SZ]
    add     rdx, 15
    and     rdx, -16
    sub     rsp, rdx
    mov     [rsp+UC_Q], r15
    mov     rax, [rdi+GA_G]
    mov     [rsp+UC_G], rax
    mov     rax, [rdi+GA_S]
    mov     [rsp+UC_S], rax
    mov     rax, [rdi+GA_WOUT]
    mov     [rsp+UC_WOUT], rax
    mov     rax, [rdi+GA_C]
    mov     [rsp+UC_C], rax
    mov     rax, [rdi+GA_W]
    mov     [rsp+UC_W], rax
    mov     [rsp+UC_N], r14
    mov     [rsp+UC_X], rbx
    mov     [rsp+UC_Y], r13
    mov     [rsp+UC_LDX8], r12
    ; zero G, s, s_c, W
    mov     r11, [rsp+UC_G]
    mov     rcx, r15
    imul    rcx, r15
    pxor    xmm0, xmm0
    xor     rax, rax
.zG2:
    cmp     rax, rcx
    jae     .zG2_done
    movsd   [r11 + rax*8], xmm0
    inc     rax
    jmp     .zG2
.zG2_done:
    mov     rdx, [rsp+UC_S]
    lea     r10, [rsp + UC_SC]
    xor     rax, rax
.zs2:
    cmp     rax, r15
    jae     .zs2_done
    movsd   [rdx + rax*8], xmm0
    movsd   [r10 + rax*8], xmm0
    inc     rax
    jmp     .zs2
.zs2_done:
    movsd   [rsp+UC_WS], xmm0
    movsd   [rsp+UC_WC], xmm0
    ; u row at UC_U (q doubles)
    mov     qword [rsp+UC_I], 0
.row:
    mov     rcx, [rsp+UC_I]
    cmp     rcx, [rsp+UC_N]
    jae     .row_done
    ; w_i -> sqrt(w) (same sqrt-w staging as the AVX2 kernel)
    mov     rax, [rsp+UC_W]
    test    rax, rax
    jz      .rw1
    movsd   xmm4, [rax + rcx*8]
    jmp     .rw2
.rw1:
    mov     eax, 1
    cvtsi2sd xmm4, eax
.rw2:
    movapd  xmm5, xmm4
    subsd   xmm5, xmm4
    ucomisd xmm5, xmm5
    jp      .nonfinite2
    sqrtsd  xmm4, xmm4
    ; build u[k] = sqrt(w)*(z[k]-c[k]); u base = rsp + UC_SC + q*8 (in r10)
    mov     rax, r15
    lea     rax, [rax*8]
    lea     r10, [rsp + UC_SC]
    add     r10, rax                ; u base (AFTER s_c)
    mov     [rsp+UC_UB], r10
    mov     rax, [rsp+UC_I]
    imul    rax, [rsp+UC_LDX8]
    mov     r9, [rsp+UC_X]
    add     r9, rax                 ; X row
    mov     r11, r15
    dec     r11                     ; p
    xor     rcx, rcx
.rk:
    cmp     rcx, r15
    jae     .rk_done
    cmp     rcx, r11
    jb      .rx
    mov     rax, [rsp+UC_I]
    mov     rdx, [rsp+UC_Y]
    movsd   xmm6, [rdx + rax*8]
    jmp     .rz
.rx:
    movsd   xmm6, [r9 + rcx*8]
.rz:
    movapd  xmm7, xmm6
    subsd   xmm7, xmm6
    ucomisd xmm7, xmm7
    jp      .nonfinite2
    mov     rax, [rsp+UC_C]
    subsd   xmm6, [rax + rcx*8]
    mulsd   xmm6, xmm4
    movsd   [r10 + rcx*8], xmm6
    inc     rcx
    jmp     .rk
.rk_done:
    ; G[j][k] += u[j]*u[k] for k >= j
    mov     r10, [rsp+UC_UB]
    xor     rsi, rsi                ; j
.rj:
    cmp     rsi, r15
    jae     .rj_done
    mov     rbx, [rsp+UC_G]
    mov     rax, r15
    lea     rax, [rax*8]
    imul    rax, rsi
    add     rbx, rax                ; Gj
    movsd   xmm4, [r10 + rsi*8]     ; u[j]
    mov     rcx, rsi                ; k = j
.rk2:
    cmp     rcx, r15
    jae     .rk2_done
    movsd   xmm5, [rbx + rcx*8]
    movsd   xmm6, [r10 + rcx*8]
    mulsd   xmm6, xmm4
    addsd   xmm5, xmm6
    movsd   [rbx + rcx*8], xmm5
    inc     rcx
    jmp     .rk2
.rk2_done:
    inc     rsi
    jmp     .rj
.rj_done:
    ; s[k] Neumaier with u[k]
    mov     rax, 0x7FFFFFFFFFFFFFFF
    movq    xmm7, rax
    punpcklqdq xmm7, xmm7
    xor     rcx, rcx
.rsk:
    cmp     rcx, r15
    jae     .rsk_done
    mov     r10, [rsp+UC_UB]
    movsd   xmm6, [r10 + rcx*8]     ; x = sqrt(w)*(z-c)[k]
    ; scale by sqrt(w) of THIS row so the column sum is w*(z-c)
    mov     rax, [rsp+UC_W]
    test    rax, rax
    jz      .rsk_w1
    mov     rdx, [rsp+UC_I]
    movsd   xmm12, [rax + rdx*8]
    sqrtsd  xmm12, xmm12
    mulsd   xmm6, xmm12
    jmp     .rsk_w2
.rsk_w1:
    mov     eax, 1
    cvtsi2sd xmm12, eax
    mulsd   xmm6, xmm12
.rsk_w2:
    mov     rax, [rsp+UC_S]
    movsd   xmm0, [rax + rcx*8]
    movapd  xmm1, xmm0
    addsd   xmm1, xmm6              ; t
    movapd  xmm2, xmm0
    andpd   xmm2, xmm7
    movapd  xmm3, xmm6
    andpd   xmm3, xmm7
    cmpnltpd xmm2, xmm3
    movapd  xmm4, xmm0
    subsd   xmm4, xmm1
    addsd   xmm4, xmm6
    subsd   xmm6, xmm1
    addsd   xmm6, xmm0
    andpd   xmm4, xmm2
    andnpd  xmm2, xmm6
    orpd    xmm4, xmm2
    lea     r10, [rsp + UC_SC]
    movsd   xmm5, [r10 + rcx*8]
    addsd   xmm5, xmm4
    movsd   [r10 + rcx*8], xmm5
    mov     rax, [rsp+UC_S]
    movsd   [rax + rcx*8], xmm1
    inc     rcx
    jmp     .rsk
.rsk_done:
    ; W Neumaier (reload w first; registers were clobbered by the G/s loops)
    movsd   xmm0, [rsp+UC_WS]
    mov     rax, [rsp+UC_W]
    test    rax, rax
    jz      .rw3
    mov     rcx, [rsp+UC_I]
    movsd   xmm6, [rax + rcx*8]
    jmp     .rw4
.rw3:
    mov     eax, 1
    cvtsi2sd xmm6, eax
.rw4:
    movapd  xmm1, xmm0
    addsd   xmm1, xmm6
    movapd  xmm2, xmm0
    andpd   xmm2, xmm7
    movapd  xmm3, xmm6
    andpd   xmm3, xmm7
    cmpnltpd xmm2, xmm3
    movapd  xmm4, xmm0
    subsd   xmm4, xmm1
    addsd   xmm4, xmm6
    subsd   xmm6, xmm1
    addsd   xmm6, xmm0
    andpd   xmm4, xmm2
    andnpd  xmm2, xmm6
    orpd    xmm4, xmm2
    movsd   xmm5, [rsp+UC_WC]
    addsd   xmm5, xmm4
    movsd   [rsp+UC_WC], xmm5
    movsd   [rsp+UC_WS], xmm1
    ; i++
    mov     rax, [rsp+UC_I]
    inc     rax
    mov     [rsp+UC_I], rax
    jmp     .row
.row_done:
    ; fold s_c into s; write W
    mov     rax, [rsp+UC_S]
    lea     r10, [rsp + UC_SC]
    xor     rcx, rcx
.sf2:
    cmp     rcx, r15
    jae     .sf2_done
    movsd   xmm0, [rax + rcx*8]
    addsd   xmm0, [r10 + rcx*8]
    movsd   [rax + rcx*8], xmm0
    inc     rcx
    jmp     .sf2
.sf2_done:
    mov     rax, [rsp+UC_WOUT]
    movsd   xmm0, [rsp+UC_WS]
    addsd   xmm0, [rsp+UC_WC]
    movsd   [rax], xmm0
    xor     eax, eax
    jmp     .out2
.nonfinite2:
    mov     eax, OLS_E_NONFINITE
.out2:
    lea     rsp, [rbp - 40]
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
section .note.GNU-stack noalloc noexec nowrite progbits
