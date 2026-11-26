; ------------------------------------------------------------------------------
; api.asm — ABI introspection: ols_abi_offsets.
; Hashes the frozen offset table (the SAME abi.inc constants the kernels
; assemble against) so the struct-drift test can compare asm truth vs the
; Python mirror; any drift on either side changes the hash (ADR-0006).
; The 8513e04 design finally landed (lost in the e437fe2 re-apply).
; Table = u32 pairs (group, offset): 0 problem, 1 result, 2 gram_args,
; 3 resid_args, 4 (4, OLS_ABI_VERSION). Offsets come from abi.inc, so a
; drifted abi.inc changes the hash; the Python mirror uses literals.
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"

section .rodata align=4
abi_table:
    ; problem (14)
    dd 0, PRB_STRUCT_SIZE
    dd 0, PRB_VERSION
    dd 0, PRB_X
    dd 0, PRB_Y
    dd 0, PRB_W
    dd 0, PRB_N
    dd 0, PRB_P
    dd 0, PRB_LDX
    dd 0, PRB_METHOD
    dd 0, PRB_FLAGS
    dd 0, PRB_LAMBDA
    dd 0, PRB_RCOND
    dd 0, PRB_THREADS
    dd 0, PRB_SIZEOF
    ; result v3 (15)
    dd 1, RES_BETA
    dd 1, RES_INTERCEPT
    dd 1, RES_SSE
    dd 1, RES_SST
    dd 1, RES_R2
    dd 1, RES_R2ADJ
    dd 1, RES_SIGMA2
    dd 1, RES_COND_EST
    dd 1, RES_SE
    dd 1, RES_RANK
    dd 1, RES_METHOD_USED
    dd 1, RES_ISA_USED
    dd 1, RES_T
    dd 1, RES_F_STAT
    dd 1, RES_SIZEOF
    ; gram args v2 (13)
    dd 2, GA_X
    dd 2, GA_LDX8
    dd 2, GA_Y
    dd 2, GA_W
    dd 2, GA_N
    dd 2, GA_P
    dd 2, GA_C
    dd 2, GA_G
    dd 2, GA_S
    dd 2, GA_WOUT
    dd 2, GA_SCRATCH
    dd 2, GA_SCRATCH_BYTES
    dd 2, GA_SIZEOF
    ; resid args (12)
    dd 3, RA_X
    dd 3, RA_LDX8
    dd 3, RA_Y
    dd 3, RA_W
    dd 3, RA_N
    dd 3, RA_P
    dd 3, RA_B0
    dd 3, RA_BETA
    dd 3, RA_SSE
    dd 3, RA_XTR
    dd 3, RA_RSUM
    dd 3, RA_SIZEOF
    ; pred args (8)
    dd 4, PA_X
    dd 4, PA_LDX8
    dd 4, PA_BETA
    dd 4, PA_B0
    dd 4, PA_N
    dd 4, PA_P
    dd 4, PA_OUT
    dd 4, PA_SIZEOF
    ; qr args (15)
    dd 5, QA_X
    dd 5, QA_LDX8
    dd 5, QA_Y
    dd 5, QA_W
    dd 5, QA_N
    dd 5, QA_P
    dd 5, QA_C
    dd 5, QA_AUG
    dd 5, QA_S
    dd 5, QA_WOUT
    dd 5, QA_SST
    dd 5, QA_NORMS
    dd 5, QA_WORK
    dd 5, QA_INTERCEPT
    dd 5, QA_SIZEOF
    ; version marker (1)
    dd 6, OLS_ABI_VERSION
abi_table_end:

%define ABI_WORDS ((abi_table_end - abi_table) / 4)

section .text

; ==============================================================================
; uint64_t ols_abi_offsets(void)
; Args    : none
; Returns : rax = FNV-1a 64 hash over the u32 words of the frozen offset
;           table, finalized with the word count (table-length changes fail).
; Requires: nothing
; Memory  : reads only the local .rodata table (ABI_WORDS u32 words).
; Clobbers: rax, rcx, rdx, rsi, r9
; Notes   : the Python mirror computes the identical hash over its literal
;           table; equal hashes == no drift between asm and ctypes layouts.
; ==============================================================================
global ols_abi_offsets:function
ols_abi_offsets:
    lea     rsi, [abi_table]
    mov     rcx, ABI_WORDS
    mov     rax, 0xcbf29ce484222325       ; FNV-1a offset basis
    mov     r9, 0x100000001b3             ; FNV-1a prime (no imm64 imul)
.loop:
    test    rcx, rcx
    jz      .fini
    mov     edx, [rsi]                    ; zero-extends to rdx
    xor     rax, rdx
    imul    rax, r9
    add     rsi, 4
    dec     rcx
    jmp     .loop
.fini:
    xor     rax, ABI_WORDS                ; length finalization
    imul    rax, r9
    ret
section .note.GNU-stack noalloc noexec nowrite progbits
