# ADR-0009: Baseline ladder verdict — honest placement of the asm engine

Date: 2026-11-25 (Phase 5) · Status: accepted

## Context

PLAN Phase 5 requires the baseline ladder (C -O2/-O3, intrinsics, NumPy/
OpenBLAS) and honest verdicts on H2 (the engine beats the ladder) and G2
(≥ 85% of bandwidth). The ladder: same workload (upper-triangle G = Z'Z,
double), four cases (n ∈ {2e5, 1e6} × q ∈ {8, 17}), best-of-3, sandbox
CPU-quota capped (~2 effective cores, throttled clock).

## Measurement (ns/row)

| rung | 2e5×8 | 2e5×17 | 1e6×8 | 1e6×17 |
|---|---|---|---|---|
| naive-O2 | 18.05 | 68.96 | 19.04 | 70.37 |
| naive-O3 | 17.23 | 42.96 | 17.41 | 44.16 |
| intrin | 16.08 | 51.39 | 16.21 | 59.96 |
| asm-v3 | 16.62 | 23.97 | 16.73 | **24.79** |
| numpy-blas | **7.42** | **24.19** | **7.24** | 27.48 |

## Verdicts

- **H2: RED.** OpenBLAS dsyrk wins 3 of 4 cases (2.2x at q=8 — it uses
  the whole machine, the ladder's asm rung is single-threaded). asm-v3
  wins exactly one: q=17, n=1e6 (DRAM-bound), +11% over numpy — the
  regime the panel kernel was designed for.
- **G2: RED** (standing verdict; the ladder adds context, it does not
  change it).
- Two honest surprises, both recorded in the perf log: gcc -O3 made the
  intrinsics rung SLOWER (the vectorizer broke the broadcast/FMA stripe);
  asm-v3's edge over compiled C grows with q (1.09x at q=8 → 2.9x at
  q=17) but has no headroom at q=8 in this sandbox.

## Consequences

- The claim in the README/write-up is scoped to what the data supports:
  the engine's value is ACCURACY (12-15 LRE, QR/CSNE, kappa study) and
  the q≥16 DRAM-bound single-thread regime, NOT a blanket speed win over
  OpenBLAS.
- A bare-metal `make repro` re-run is the honest path to a stronger H2
  verdict: there, ols_gram_mt (threads) meets OpenBLAS's threading, and
  the clock is not throttled. The ladder artifacts (bench/ladder/,
  results/ladder.json) are committed for exactly that re-run.
- No further kernel optimization in this project phase — the remaining
  gap at q=8 is a threading + sandbox artifact, and optimizing against
  a throttled quota would tune for the wrong machine.
