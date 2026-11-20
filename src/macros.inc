; ------------------------------------------------------------------------------
; macros.inc — code-generation macros for asm-ols kernels.
; Rule (skills/asm-x86-64-engineering §7): unrolled variants are GENERATED with
; %macro/%rep/%assign, never hand-copied, so tile/accumulator counts are
; one-line edits and cannot desynchronize.
; ------------------------------------------------------------------------------
%include "src/abi.inc"

; VZERO_RET — single disciplined exit for any function that touched ymm.
; Why: SSE/AVX transition penalties (skills/asm-x86-64-engineering §5.2).
%macro VZERO_RET 0
    vzeroupper
    ret
%endmacro

; REDUCE_ACCUMULATORS nacc — horizontally sum ymm0..ymm{NACC-1} into xmm0
; (scalar). Accumulators must be named ymm0..ymm{NACC-1} by convention.
; Used OUTSIDE hot loops only (skills/asm-x86-64-engineering §5.8).
; Numerics: pairwise-ish tree, fixed order for a given NACC.
%macro REDUCE_ACCUMULATORS 1
    %assign _ra_i 1
    %rep (%1 - 1)
        vaddpd  ymm0, ymm0, ymm%[_ra_i]
        %assign _ra_i _ra_i + 1
    %endrep
    vextractf128 xmm1, ymm0, 1
    vaddpd  xmm0, xmm0, xmm1
    vhaddpd xmm0, xmm0, xmm0
%endmacro

; (The real Neumaier body is written inline in sums.asm; see the file for the
; full branch-free sequence. Register budget per accumulator: acc, corr, and
; two shared temporaries across accumulators.)
