# Phase 8 report — G2 escalation, iteration 3 (gram v4), and gate closure

Session 8, 2026-11-26. Scope: the one open MUST-gate item from Phase 5's
honest verdicts — G2 (>= 85% of single-thread DRAM bandwidth for the fused
Gram pass). Executed per the escalation rule: third evidence-backed
iteration, then closure with an impossibility argument.

## What was built

1. **gram v4** (`src/core/gram.asm`, documented-unreachable section +
   bench/kernels/gram_bench_v4.py): fully fused single-pass kernel for
   q <= 8 — u built in registers, column accumulators persisting across
   the whole call, no PKG/GSCR round-trip (ADR-0005's named next step).
   Complete verification: gram differential ALL OK, NIST 11.9 LRE, P1
   green at the documented 4-row fold cut.
2. **8 bring-up bugs**, each root-caused and fixed (ADR-0012 has the log;
   the xmm8/ymm8 scalar-vector aliasing struck twice in one kernel —
   ADR-0010 class #4 again: "no register state across calls" now extends
   to "no scalar parking in ymm8..15 while accs live").
3. **mt benchmark rung**: ols_gram_mt measured for the first time (H5).
4. **G2 closed RED-with-evidence** (ADR-0012): v1 2% -> v3 22% -> v4 20%;
   both q regimes have host-specific impossibility arguments; re-open
   conditions named (AVX-512 host / real load throughput).

## Measured results (all from results/gram-bench-v4.json, 30 reps, median)

| case | ns/row | GB/s | % of DRAM bound (18.89 GB/s 1t) |
|---|---|---|---|
| q=2 n=2e5 | 13.02 | 1.23 | 6.5% (16 B/row) |
| q=4 n=2e5 | 14.98 | 2.14 | 11% |
| q=6 n=2e5 | 16.73 | 2.87 | 15% |
| q=8 n=2e5 | 17.36 | 3.69 | 20% |
| q=17 n=2e5 | 24.45 | 5.56 | 29% (136 B/row) |
| q=32 n=4096 L2 | 48.88 | 5.24 | — (L2-resident) |
| q=65 n=4096 L2 | 149.87 | 3.47 | — |
| q=8 mt2 | 9.47 | 6.76 | 18% of 37.6 all-thread (H5 RED) |
| q=17 mt2 | 19.39 | 7.01 | 18.7% |

## What I got wrong (this phase's own list)

1. Prediction #28 was FALSIFIED: I predicted 14±3 ns/row at q=8 from a
   uop model that assumed 32B vector row loads; shipping guard-safe
   scalar loads (correctness first) erased the predicted saving. The
   measured tie (17.36 vs 17.17) is the honest outcome.
2. I initially mislabeled a recurring 1.39e-309 subnormal in s as
   "denormal noise = 0" during probing — it was bug #8 (unzeroed xmm0),
   and it only fired when the true sum cancelled to zero, which my
   random probes rarely produced. The P1 suite caught it; my probes
   didn't. Lesson re-learned: a differential that never exercises exact
   cancellation is not a differential.
3. The first v4 draft was written with a call/ret inside the kernel that
   shifted every rsp-relative slot by 8 in the callee — found by
   inspection before it ever ran, but only after I'd written three
   conflicting versions of the same fold. The clean rewrite was faster
   than the patch chain.

## Gate status after this phase

> **Correction (2026-11-26, session 9):** the line below originally read
> "G0/G1/G3/G4: green". The G4 claim was UNEARNED as written: of G4's four
> clauses, only the E2E work existed, and the thread clause is H5 (RED on
> this host). Session 9 built the missing artifacts (1e7 E2E, fuzz with a
> 4-bug fix, freestanding CLI) and graded G4 as a split verdict — see the
> phase-9 report and README. G0/G1/G3 remain green.

- G2: RED, CLOSED with evidence (ADR-0012) — was the last open MUST-gate
  performance clause.
- H5 (>= 85% of all-thread BW via N threads): RED on this host (18.0% /
  18.7% measured, 2-vCPU quota). The mt machinery itself is verified
  (same-T bitwise reproducibility per ADR-0008 unchanged).
- G0/G1/G3/G4: green (unchanged). Pyramid: ALL GREEN at final HEAD with
  the v3 dispatch restored. Mutation: 28/34 killed = 0.8235
  (results/mutation-phase1.json regenerated this session; 3 SURVIVED
  rows, 2 provably equivalent documented in results/mutation-phase3.json
  survivor_analysis).

## Files

- src/core/gram.asm (v4 body, documented-unreachable; dispatch = v3)
- bench/kernels/gram_bench_v4.py, results/gram-bench-v4.json
- docs/adr/0012-gram-v4-and-g2-closure.md, docs/perf-log.md (#28 + measured)
- results/mutation-phase1.json (regenerated), this report
