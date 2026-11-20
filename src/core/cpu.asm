; ------------------------------------------------------------------------------
; cpu.asm — CPU feature detection (skills/asm-x86-64-engineering §6).
; ols_cpu_has_avx2_fma : eax = 1/0  (AVX2 + FMA3 + OSXSAVE + OS YMM state)
; ols_cpu_has_avx512   : eax = 1/0  (F + DQ + BW + VL + OS opmask/ZMM state)
; Vendor string is also returned through the out-pointer API for machine.json.
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"

section .text

; int32_t ols_cpu_has_avx2_fma(void)
; Args    : none
; Returns : eax = 1 if AVX2+FMA3 usable (CPU bits AND OS XCR0 enabled), else 0
; Requires: n/a (must run on any x86-64 CPU; cpuid leaf 1 always exists)
; Memory  : none (writes rbx via push/pop only)
; Clobbers: rax, rbx(saved), rcx, rdx
global ols_cpu_has_avx2_fma:function
ols_cpu_has_avx2_fma:
    push    rbx                        ; cpuid clobbers rbx (callee-saved)
    mov     eax, 1
    xor     ecx, ecx
    cpuid
    and     ecx, (1<<12)|(1<<27)|(1<<28)   ; FMA | OSXSAVE | AVX
    cmp     ecx, (1<<12)|(1<<27)|(1<<28)
    jne     .no
    xor     ecx, ecx
    xgetbv                              ; edx:eax = XCR0
    and     eax, 6                      ; XMM + YMM state enabled by OS
    cmp     eax, 6
    jne     .no
    mov     eax, 7
    xor     ecx, ecx
    cpuid
    bt      ebx, 5                      ; CPUID.7:EBX.AVX2
    jnc     .no
    mov     eax, 1
    pop     rbx
    ret
.no:
    xor     eax, eax
    pop     rbx
    ret

; int32_t ols_cpu_has_avx512(void)
; Args    : none
; Returns : eax = 1 if AVX-512F/DQ/BW/VL usable (CPU bits AND OS state), else 0
; Requires: n/a
; Memory  : none
; Clobbers: rax, rbx(saved), rcx, rdx
global ols_cpu_has_avx512:function
ols_cpu_has_avx512:
    push    rbx
    ; OSXSAVE first (leaf 1)
    mov     eax, 1
    xor     ecx, ecx
    cpuid
    test    ecx, 1<<27                  ; OSXSAVE
    jz      .no
    xor     ecx, ecx
    xgetbv
    and     eax, 0xE6                   ; XMM|YMM|opmask|ZMM_Hi256|Hi16_ZMM (bits 1,2,5,6,7)
    cmp     eax, 0xE6
    jne     .no
    mov     eax, 7
    xor     ecx, ecx
    cpuid
    and     ebx, (1<<16)|(1<<17)|(1<<30)|(1<<31)   ; F | DQ | BW | VL
    cmp     ebx, (1<<16)|(1<<17)|(1<<30)|(1<<31)
    jne     .no
    mov     eax, 1
    pop     rbx
    ret
.no:
    xor     eax, eax
    pop     rbx
    ret

; void ols_cpu_ident(char vendor_out[13], uint32_t *family_model_stepping)
; Args    : rdi = vendor_out (>=13 bytes), rsi = fms_out (may be NULL)
; Returns : nothing; vendor_out gets "GenuineIntel"/"AuthenticAMD"/... + NUL
; Requires: n/a
; Memory  : writes 13 bytes at vendor_out, 4 bytes at fms_out
; Clobbers: rax, rbx(saved), rcx, rdx
global ols_cpu_ident:function
ols_cpu_ident:
    push    rbx
    xor     eax, eax
    cpuid                              ; leaf 0: max leaf + vendor
    mov     [rdi], ebx
    mov     [rdi+4], edx
    mov     [rdi+8], ecx
    mov     byte [rdi+12], 0
    test    rsi, rsi
    jz      .done
    mov     eax, 1
    xor     ecx, ecx
    cpuid
    mov     [rsi], eax                 ; family/model/stepping bits
.done:
    pop     rbx
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
