# ADR-0010: Transcendental kernels for inference (exp/log/lgamma/ibeta, t/F p-values)

Date: 2026-11-25 (Phase 6, PLAN COULD "p-values/CIs") · Status: accepted

## Decision
`src/core/special.asm` (13th asm file, SSE2 baseline): exp (k/r split,
fdlibm x-split reduction, Taylor-13), log (atanh-form P(s²) degree 10),
lgamma (Lanczos g=7 n=9 + reflection with sin(w)=w·P(w²) degree 11),
ibeta (NR3 two-step Lentz CF + symmetry swap), plus t/F p-value wrappers:
p_t = I_x(df/2, ½), x = df/(df+t²); p_F = I_x(d2/2, d1/2), x = d2/(d2+d1 f)
(identities verified vs mpmath incl. Cauchy df=1).

## Measured (mpmath 40-50 digit differential, tests/py/test_special.py)
exp 2.1e-16, log 2.2e-16 (both sub-ulp), lgamma 2.3e-15 (abs-floored; the
g=7 set's own ~1e-13 relative error dominates near its zeros — the asm
equals the double-precision formula bit-for-bit), ibeta 3.9e-14, t/F
wrappers 1.3e-14/1.6e-14.

## Bugs found (recorded for the write-up's "what I got wrong")
1. Horner off-by-one: starting v=r·c13 shifts every coefficient one power
   (exp(1)=2.34); same in log and sin.
2. ln2lo constant truncated to 0x3DEA39EF00000000 (pair summed to
   ln2−2.3e-17; ×k=1006 → 2.3e-14 in r). Correct: 0x3DEA39EF35793C76.
3. lgamma A-sum used the COEFFICIENT as the k=1 denominator (c1/(x+676.5)).
4. Cross-call register clobbering: lgamma carried A/t/(x−½) in xmm1/3/4
   across ols_log calls; ibeta's bt accumulated in xmm7 across five calls;
   the CF loop cached m in xmm5 across the lentz call.
5. movapd on 8-aligned frame slots (#GP at rbp−8).
6. r14/r15 clobbered (SysV callee-saved) — the caller crashed in unrelated
   library code; found via core-dump RIP parsing.
7. Lentz init: c=1/d makes d=1/c an invariant → every del = 1 exactly,
   h never moves (NR's c starts at 1, h = d).
8. F-wrapper guard compared f vs xmm3 (which held f after the NaN check)
   — every input returned p=1.
9. Test-side: the suite's relerr abs()'d the reference (negative refs
   reported 2.0 on exact answers — a "kernel bug" that didn't exist).
