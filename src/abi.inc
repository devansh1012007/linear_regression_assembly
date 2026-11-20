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

; ---- ISA ids (must match include/ols.h) ---------------------------------------
%define OLS_ISA_SSE2          0
%define OLS_ISA_AVX2          1
%define OLS_ISA_AVX512        2
