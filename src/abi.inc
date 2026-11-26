; ------------------------------------------------------------------------------
; abi.inc — calling-convention abstraction for asm-ols kernels.
; Kernels reference integer arguments ONLY through ARG1..ARG6 so the same
; source assembles for SysV (elf64) and Win64.
; Extended from skills/asm-x86-64-engineering §2 (verified there for both
; formats); re-verified by tests/abi in this repo.
; ------------------------------------------------------------------------------
%ifidn __OUTPUT_FORMAT__, win64
  %define ARG1 rcx
  %define ARG2 rdx
  %define ARG3 r8
  %define ARG4 r9
  ; 5th/6th integer args sit above the shadow space at entry.
  ; Phase-1 kernels take at most 6 args; none currently needs ARG5/ARG6 on
  ; Win64. Define them for completeness; using them requires reading [rsp+40]
  ; BEFORE any push, or after accounting for the frame size.
  %define ABI_WIN64 1
%else
  %define ARG1 rdi
  %define ARG2 rsi
  %define ARG3 rdx
  %define ARG4 rcx
  %define ARG5 r8
  %define ARG6 r9
%endif

; ---- OLS error codes (must match include/ols.h) ------------------------------
%define OLS_OK                 0
%define OLS_E_ARG             -1
%define OLS_E_ALIGN           -2
%define OLS_E_SIZE_OVERFLOW   -3
%define OLS_E_WORKSPACE       -4
%define OLS_E_UNDERDETERMINED -5
%define OLS_E_NONFINITE       -6
%define OLS_E_SINGULAR        -7
%define OLS_E_ILLCOND         -8

; ---- simple_fit flags (must match include/ols.h) ------------------------------
%define OLS_F_STATS           1       ; run the residual pass, fill sse/sst/r2

; ---- ols_fit flags (separate namespace; must match include/ols.h) -------------
%define OLS_FIT_INTERCEPT     1
%define OLS_FIT_REPRODUCIBLE  2
%define OLS_FIT_DOT2          4
%define OLS_FIT_SKIP_STATS    8

; ---- ols_fit methods (must match include/ols.h) -------------------------------
%define OLS_M_AUTO            0
%define OLS_M_GRAM_CHOL       1
%define OLS_M_GRAM_CSNE       2
%define OLS_M_QR              3
%define OLS_M_QR_PIVOT        4
%define OLS_M_CG              5

; ---- ols_problem_t field offsets (FROZEN; verified by struct_check.py) --------
%define PRB_STRUCT_SIZE       0       ; u32
%define PRB_VERSION           4       ; u32
%define PRB_X                 8       ; const double*
%define PRB_Y                 16      ; const double*
%define PRB_W                 24      ; const double* (NULL = unweighted)
%define PRB_N                 32      ; size_t
%define PRB_P                 40      ; size_t
%define PRB_LDX               48      ; size_t
%define PRB_METHOD            56      ; u32
%define PRB_FLAGS             60      ; u32
%define PRB_LAMBDA            64      ; double
%define PRB_RCOND             72      ; double
%define PRB_THREADS           80      ; u32
%define PRB_SIZEOF            88

; ---- ols_result_t field offsets (FROZEN; verified by the drift test) ----------
%define RES_BETA              0       ; double*
%define RES_INTERCEPT         8
%define RES_SSE               16
%define RES_SST               24
%define RES_R2                32
%define RES_R2ADJ             40
%define RES_SIGMA2            48
%define RES_COND_EST          56
%define RES_SE                64      ; double*
%define RES_RANK              72      ; i32
%define RES_METHOD_USED       76      ; i32
%define RES_ISA_USED          80      ; i32
%define RES_PAD               84      ; i32
%define RES_T                 88      ; double* (v2)
%define RES_F_STAT            96      ; double  (v3; ADR-0006)
%define RES_SIZEOF            104     ; v3 (v2 was 96)

; ---- ABI version + table marker (api.asm / drift test; ADR-0006) --------------
; 1 = phase-2 problem/result v1, 2 = result v2 (t), 3 = result v3 (f_stat).
%define OLS_ABI_VERSION       3

; ---- ISA ids (must match include/ols.h) ---------------------------------------
%define OLS_ISA_SSE2          0
%define OLS_ISA_AVX2          1
%define OLS_ISA_AVX512        2

; ---- ols_gram_args_t / ols_resid_args_t offsets (include/ols.h) --------------
%define GA_X                  0
%define GA_LDX8               8
%define GA_Y                  16
%define GA_W                  24
%define GA_N                  32
%define GA_P                  40
%define GA_C                  48
%define GA_G                  56
%define GA_S                  64
%define GA_WOUT               72
%define GA_SCRATCH            80      ; void* caller workspace (PKG + GSCR)
%define GA_SCRATCH_BYTES      88      ; size_t; >= ols_gram_scratch_bytes(q)
%define GA_SIZEOF             96

; ---- gram kernel frame offsets (rsp-relative, fixed; see gram.asm) -----------
%define UC_Q                  0
%define UC_G                  8
%define UC_S                  16
%define UC_WOUT               24
%define UC_C                  32
%define UC_W                  40
%define UC_N                  48
%define UC_I                  56
%define UC_WS                 64
%define UC_WC                 72
%define UC_RB                 80
%define UC_X                  88
%define UC_Y                  96
%define UC_LDX8               104
%define UC_U                  112     ; (unused in fixed layout; u lives after s_c)
%define UC_UB                 120     ; u-buffer base pointer
%define UC_SZ                 128     ; control size; s_c at UC_SC=UC_SZ
%define UC_SC                 128

; ---- ols_resid_args_t offsets (include/ols.h) ---------------------------------
%define RA_X                   0
%define RA_LDX8                8
%define RA_Y                   16
%define RA_W                   24
%define RA_N                   32
%define RA_P                   40
%define RA_B0                  48
%define RA_BETA                56
%define RA_SSE                 64
%define RA_XTR                 72      ; p doubles of X'r (CSNE refinement)
%define RA_RSUM                80      ; double* sum w_i r_i (NULL = skip; v3)
%define RA_SIZEOF              88      ; v3 (v2 was 80)

; ---- ols_pred_args_t offsets (include/ols.h) ----------------------------------
%define PA_X                  0
%define PA_LDX8               8
%define PA_BETA               16
%define PA_B0                 24
%define PA_N                  32
%define PA_P                  40
%define PA_OUT                48
%define PA_SIZEOF             56

; ---- ols_qr_args_t offsets (include/ols.h) ------------------------------------
%define QA_X                  0
%define QA_LDX8               8
%define QA_Y                  16
%define QA_W                  24
%define QA_N                  32
%define QA_P                  40
%define QA_C                  48
%define QA_AUG                56
%define QA_S                  64
%define QA_WOUT               72
%define QA_SST                80
%define QA_NORMS              88
%define QA_WORK               96
%define QA_INTERCEPT          104
%define QA_SIZEOF             112
