; ------------------------------------------------------------------------------
; abi_probe.asm — SysV ABI conformance probe (asm-verification §2, verified
; template, extended here with MXCSR + DF + vzeroupper checks driven from C).
;
; uint64_t abi_probe_call(void *fn, u64 a1, u64 a2, u64 a3, u64 a4,
;                         uint32_t *corrupt_mask)
;   Calls fn(a1,a2,a3,a4) with canaries in rbx, rbp, r12-r15; on return the
;   bitmask of clobbered callee-saved GPRs is stored at *corrupt_mask
;   (bit: rbx=1, rbp=2, r12=4, r13=8, r14=16, r15=32) and the function's
;   rax is returned. MXCSR/DF/ymm-upper checks are done by the C caller.
; ------------------------------------------------------------------------------
default rel
%include "src/abi.inc"

%macro CHK 3                          ; reg, canary, bit
    mov     rcx, %2
    cmp     %1, rcx
    je      %%ok
    or      r8d, %3
%%ok:
%endmacro

section .text
; Args    : SysV rdi=fn, rsi=a1, rdx=a2, rcx=a3, r8=a4, r9=mask_out | Win64: rcx,rdx,r8,r9
; Returns : rax = fn's rax; *mask_out = clobber bitmask
; Requires: n/a (test-only harness export)
; Memory  : writes 4 bytes at mask_out only
; Clobbers: caller-saved GPRs + xmm0 (returns whatever fn returned in rax)
global abi_probe_call:function
abi_probe_call:
    push    rbp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8                    ; 6 pushes + 8 -> rsp 16-aligned at call
    mov     [rsp], r9
    mov     rax, rdi                  ; fn
    mov     rdi, rsi                  ; a1
    mov     rsi, rdx                  ; a2
    mov     rdx, rcx                  ; a3
    mov     rcx, r8                   ; a4
    mov     rbx, 0x1111111111111111
    mov     rbp, 0x2222222222222222
    mov     r12, 0x3333333333333333
    mov     r13, 0x4444444444444444
    mov     r14, 0x5555555555555555
    mov     r15, 0x6666666666666666
    call    rax
    xor     r8d, r8d
    CHK     rbx, 0x1111111111111111, 1
    CHK     rbp, 0x2222222222222222, 2
    CHK     r12, 0x3333333333333333, 4
    CHK     r13, 0x4444444444444444, 8
    CHK     r14, 0x5555555555555555, 16
    CHK     r15, 0x6666666666666666, 32
    mov     r9, [rsp]
    mov     [r9], r8d
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ------------------------------------------------------------------------------
; Seeded DEFECT functions for the probe self-test (asm-verification §2: "A
; probe you have not seen fail is not a probe"). Each is intentionally bad in
; exactly one way; test_abi.c asserts the probe reports it.
; ------------------------------------------------------------------------------
; u64 bad_clobber_r12(u64 a, u64 b, u64 c, u64 d) — breaks callee-saved r12.
; Args    : rdi, rsi, rdx, rcx
; Returns : rax = 42
; Requires: n/a (deliberately WRONG)
; Memory  : none
; Clobbers: r12 (ILLEGAL under SysV — that is the point)
global test_bad_clobber_r12:function
test_bad_clobber_r12:
    mov     r12, 42
    mov     rax, 42
    ret

; u64 bad_clobber_rbx_r14(...) — breaks rbx and r14 (mask 1|16 = 17).
; Args    : rdi, rsi, rdx, rcx
; Returns : rax = 7
; Requires: n/a (deliberately WRONG)
; Memory  : none
; Clobbers: rbx, r14 (ILLEGAL — that is the point)
global test_bad_clobber_rbx_r14:function
test_bad_clobber_rbx_r14:
    mov     rbx, 1
    mov     r14, 2
    mov     rax, 7
    ret

; u64 good_leaf(u64 a, u64 b, u64 c, u64 d) — returns a+b+c+d; ABI-clean.
; Args    : rdi, rsi, rdx, rcx
; Returns : rax = a+b+c+d
; Requires: n/a
; Memory  : none
; Clobbers: rax, rcx, rdx, rsi, rdi (all caller-saved)
global test_good_leaf:function
test_good_leaf:
    mov     rax, rdi
    add     rax, rsi
    add     rax, rdx
    add     rax, rcx
    ret

; u64 bad_no_vzeroupper(...) — touches ymm then returns without vzeroupper.
; The C harness checks ymm upper halves are zero after any ymm-using callee.
; Args    : rdi, rsi, rdx, rcx
; Returns : rax = 0
; Requires: AVX2 (test-only)
; Memory  : none
; Clobbers: ymm0 upper half left dirty (ILLEGAL per contract rule — that is
;           the point; forces SSE/AVX transition penalty in the caller)
global test_bad_no_vzeroupper:function
test_bad_no_vzeroupper:
    vxorpd  ymm0, ymm0, ymm0
    vpcmpeqd ymm1, ymm1, ymm1
    vaddpd  ymm2, ymm1, ymm1
    vaddpd  ymm3, ymm2, ymm1
    xor     eax, eax
    ret

; u64 bad_overread_end(const double *a, u64 n, ...) — reads a[n] (one element
; past the end) so the guard-page test MUST fault (asm-verification §3).
; Args    : rdi = a, rsi = n
; Returns : rax = bits of a[n] as u64
; Requires: n/a (deliberately WRONG)
; Memory  : reads a[0..n] — the last element is OUT OF CONTRACT
; Clobbers: rax, rcx
global test_bad_overread_end:function
test_bad_overread_end:
    mov     rax, [rdi + rsi*8]        ; a[n]: one past the end
    ret

; u64 good_read_exact(const double *a, u64 n, ...) — reads a[0..n) exactly.
; Args    : rdi = a, rsi = n
; Returns : rax = bits of a[n-1] if n>0 else 0
; Requires: n/a
; Memory  : reads a[0..n) only
; Clobbers: rax, rcx
global test_good_read_exact:function
test_good_read_exact:
    test    rsi, rsi
    jz      .zero
    mov     rax, [rdi + (rsi-1)*8]
    ret
.zero:
    xor     eax, eax
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
