# Stretch triage (analysis-only)

Date: 2026-11-26 · branch `phase9-g4` · read-only pass; the only write is
this doc; no benchmarks run (AVX-512 figures are the measurement agent's
artifact, this session). PLAN §3/§6/§10 name the stretch items; this doc
triages each for a decision.

## Precondition (PLAN §6 Phase 6: "only when G0-G5 are green") — NOT MET

Scoreboard at HEAD: H1 RED, H2 GREEN, H3 NOT BUILT, H4 CONFIRMED, H5 RED
(kernel-attributed, results/gram-bench-mt-study.json), H6 RED
(README.md:115-120); G2 CLOSED RED-with-evidence
(ADR-0012); G4 SPLIT (phase9-session9.md:101-106). All stretch work is
formally deferred; this triage ranks what to do first if a green-light
comes. One sandbox fact gates every kernel-side item: `nasm` is not
installed in this environment (avx512-probe.json toolchain: "nasm:
unavailable in this environment") — no asm change can be built here.

## Summary

| Item | verdict-relevant? | effort | recommendation |
|---|---|---|---|
| AVX-512 kernels | H6; G2 re-open cond. (ADR-0012:86-87) | M-L | NO-GO for verdict changes on this host (re-open cond. (b) not met: 1.23x vs ~2x needed); optional time-boxed experiment = adjacent evidence only |
| Dot2 accumulation (H3) | H3 only; no failing gate needs it | M | DEFER |
| Leverage / Cook's distance | none (inference completeness) | S py / M asm+ABI | DEFER |
| libc-free threads (clone/futex) | none (H5 is issue-bound, not pthread-bound) | M | DEFER |
| lasso / CG (OLS_CG=5 reserved) | none | M | NO-GO (CG) / DEFER (lasso) |
| Win64 port | none | L | NO-GO (no toolchain here) |

## 1. Dot2 (double-double) accumulation — DEFER

Reserved, unbuilt: OLS_FIT_DOT2 (include/ols.h:38, src/abi.inc:44) is
consumed by no kernel (src/core/dot.asm:26 calls it "the Phase-6 Dot2 COULD
item"); H3 (PLAN.md:16) is "<15% wall time when memory-bound". Accuracy: no
gate needs it — results/nist-phase2.json passes every row (Wampler1/3
min_lre 7.40 vs gate 7.0; LRE higher = better, so README.md:129's "above
its 7.0 gate" is a PASS; Longley 11.72/11.0, Wampler2 10.10/9.0), and the
conditioning tail (kappa-sweep.json: gram_chol and CSNE 0.0 LRE at kappa
1e8, QR 7.6-8.0; h4_verdict kappa* = 1e6) is already served by requesting
CSNE/QR (nist-phase3.json Wampler1: CSNE 10.91, QR 9.63); G3 is green.
Performance: it would not fix H6 — it would worsen it. The 36-50%-of-peak
shortfall (README.md:120; recomputed from gram-bench-v3/v4.json: q32
50.45/48.88 ns/row = 36.3-37.5%, q65 152.89/149.87 = 48.7-49.7% of the
57.62 GFLOP/s peak, machine.json) is load-port/issue-bound
(ADR-0012:74-85), and Dot2 adds uops to that stream (error-free transform
6-20 flops/element, TwoProd via FMA ~17 — Ogita-Rump-Oishi literature
figure, PLAN §7 reading list; not measured here). H3's premise also fails
here: the DRAM-resident cases run at ~20% of the DRAM bound (17.17-24.45
ns/row), so there is no free-FLOP slack even out-of-cache. DEFER: no gate
is blocked on it, the accuracy tail has two green solvers, and the extra
uops cost wall time in every regime on this host.

## 2. Win64 port — NO-GO (this sandbox)

src/abi.inc:9-19 already abstracts integer args for win64 (ARG1-4 =
rcx/rdx/r8/r9, shadow-space ARG5/6 rule documented) — written, never
assembled: Makefile:10 hardcodes `-f elf64`. The real cost is the register
contract: Microsoft x64 callee-saves xmm6-15, exactly where the hot
accumulators live (gram v4's persistent accumulators are ymm8-11;
ADR-0012:37-43), so every upper-half kernel needs 10-pair save/restore
prologues plus re-audit; unwind metadata changes to SEH (.pdata/.xdata);
ols_mt.c (pthreads) and cli/freestanding.c (Linux raw syscalls, _start
thunk at :240-256) each need a separate port. Blocker, verified here: no
x86_64-w64-mingw32-gcc, no wine, no nasm — nothing could be built, run,
or verified, violating AGENTS.md rule 8 (no claim without an artifact).
Effort L. NO-GO here; a port needs a different environment and its ADR.

## 3. AVX-512 kernels — NO-GO for verdict changes on this host (re-open condition not met)

Reconciled verdict (the probe's own triage in results/avx512-probe.json
says NOT-WORTH; this doc originally said GO-conditional on a looser
threshold — the ADR-0012 re-open condition (b), "32B loads halve the load
count", is the binding standard and it is NOT met).

src/core/dispatch.asm:5-7 clamps AVX-512 requests to AVX2 ("no AVX-512
kernel exists yet"); the host has full AVX-512 (machine.json
host.avx512 = true). The only stretch item aimed at the measured binding
constraint: G2 closed on load-port/issue grounds, and ADR-0012:86-87 names
AVX-512 as re-open condition (b) — "32B loads halve the load count". See
results/avx512-probe.json (this session): zmm FMA 93.59 vs ymm 50.84
GFLOP/s (ratio 1.84 — clears the probe's own ">1.4" materiality
threshold), L1 load 145.96 vs 118.84 GB/s (ratio 1.23), p*_avx512 = 34.01
vs the committed 23.4. Thresholds that would flip this to NO-GO: an L1
zmm/ymm load ratio < ~1.2 (no load-port relief), or zmm downclocking
erasing the 1.84 FMA ratio. Two things will not move regardless: G2's DRAM
clause (no ISA adds bandwidth; probe dram_1t 21.39 vs the committed 18.89
GB/s is host run-to-run variance), and H6's denominator (70% of a ~93.6
GFLOP/s zmm peak is a higher bar than 70% of 57.62, and H6 as
pre-registered (PLAN.md:19) names the AVX2 6x8 tile — a zmm result is
adjacent evidence, not a retroactive flip). Effort M-L (kernel, dispatch
unclamp, differential/guard/mutation, predict-then-measure). GO only as a
time-boxed, prediction-first experiment.

## 4. libc-free threads (raw clone/futex for ols_mt) — DEFER

PLAN.md:24 COULD item. The value case — the freestanding story — is
already complete WITHOUT threads: ADR-0008:44-45 (the freestanding CLI
does not link ols_mt) and asm-ols-fs shipped with bitwise JSON parity
(phase9-session9.md:69-80). It is a C-only shim change, so it builds here
with gcc (no nasm needed; ols_mt.c is the C-outside-the-core precedent).
Risk: bring-up is exactly the bug class ADR-0012 logged 8 times, plus a
latent one seen in read — ols_mt.c:82-93 joins uninitialized th[t] when
pthread_create partially fails — and the payoff is unmeasurable on this host (2 CPUs visible via cpuset,
no CFS quota — results/gram-bench-mt-study.json): H5's 18.0-18.7% of
all-thread BW on the committed 37.6 GB/s basis (gram-bench-v4.json
mt2: 6.76/7.01 GB/s; re-measured 19.5-20.8% of 33.41 GB/s in the mt
study) is per-thread issue-boundness, not pthread overhead; raw clone
adds neither bandwidth nor issue slots. Effort M. DEFER; no verdict moves.

## 5. lasso / CG (OLS_CG = 5 reserved) — NO-GO (CG) / DEFER (lasso)

PLAN.md:24 COULD items; OLS_CG = 5 is reserved and refused OLS_E_ARG
(include/ols.h:52, src/core/fit.asm:181); PLAN.md:131 cuts lasso/CG first.
The historical motivation was Filip (degree-10, rank-11): every solver
refuses it at fit level (rc = -7 across the board, nist-phase3.json),
pre-registered without a target as rank-deficient in double precision
(ADR-0007:54, PLAN.md:63). Iteration does not cure rank deficiency —
regularization does, and ridge already exists (problem.lambda,
include/ols.h:268-272). CG on the Gram is normal-equations iteration: it
inherits the kappa-squared conditioning the sweep measured as chol's
failure mode (0.0 LRE at kappa 1e8) and would trail the shipped QR/CSNE
everywhere those are green. Lasso is a real new capability (L1) with no
gate behind it. Effort M either way. NO-GO for CG (dominated by shipped
solvers); DEFER lasso until a use-case demands it.

## 6. Leverage / Cook's distance — DEFER (Python prototype first)

PLAN.md:24 COULD; inference completeness after p-values/CIs shipped
(README.md:19-27). The ABI exposes no hat values or per-observation
residuals (ols_result_t, include/ols.h:278-298, carries beta/se/t/f_stat/
sse/sst/r2/sigma2/cond_est; ols_resid_args_t, :144-157, returns only
sse/X'r/rsum) — but a pure-Python prototype needs no core change:
ols_gram_f64_* -> G, ols_chol_f64 + ols_chol_inv_f64_sse2 (:189-191) ->
(X'X)^-1, ols_predict_f64 (:165-178) -> fitted values; h_ii = x_i'(X'X)^-1
x_i and Cook's D follow, checked against mpmath/NumPy. Production is M:
an asm kernel writing n doubles plus result v4 — the 104-byte result is
frozen v3, so an ABI bump touches abi.inc, ols.h, the ctypes mirrors and
the drift test. No verdict moves. DEFER; the S-effort script is the first
step if the inference story is ever extended.

## Ordered recommendation

H5 is kernel-attributed (a bare-metal re-run would not flip it — the
per-thread shortfall is kernel-intrinsic, results/gram-bench-mt-study.json;
what could flip it is a kernel with more per-thread load throughput,
ADR-0012's re-open conditions), and G2 is closed, not open — so if
stretch work is ever green-lit:

1. AVX-512 kernel, time-boxed, prediction written before measurement
   (AGENTS rule 2). The only item aimed at the measured binding
   constraint; ADR-0012's own re-open condition; the probe's hardware
   thresholds are cleared. Accept up front: the DRAM clause stays closed
   and H6's bar rises with the peak.
2. Dot2, only after (1) — H3 becomes cheap to test exactly if zmm frees
   issue slots in cache; there is no accuracy motivation meanwhile.
3. Leverage/Cook's Python prototype (S, no ABI risk) — extends the
   inference story the p-values work started.
4. libc-free threads — only if a real freestanding multi-thread use-case
   appears; unmeasurable on this host.
5. lasso — feature work, no evidence value. CG: NO-GO.
6. Win64 — NO-GO in this sandbox (toolchain absent); needs a new
   environment and an ADR, not effort here.
