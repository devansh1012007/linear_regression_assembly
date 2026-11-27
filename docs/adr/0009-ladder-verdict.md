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

> **Correction (2026-11-26, session 8):** this ADR originally wrote "H2"
> for the asm-vs-OpenBLAS verdict. PLAN.md's pre-registered H2 is the
> asm-vs-INTRINSICS clause; the OpenBLAS comparison is **H1**. Corrected
> labels below; the measurements were always correct, only the label
> moved. The previously-unstated H2 and H6 verdicts from this same data
> are added at the end.

- **H1: RED.** OpenBLAS dsyrk wins both q=8 cases decisively (2.24x /
  2.31x — it uses the whole machine, the ladder's asm rung is
  single-threaded); n=2e5, q=17 is a statistical tie (asm 23.97 vs numpy
  24.19 ns/row, -0.9%); asm-v3 wins n=1e6, q=17 (DRAM-bound), +10.9%
  over numpy — the regime the panel kernel was designed for.
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
- A bare-metal `make repro` re-run is the honest path to a stronger H1
  verdict: there, ols_gram_mt (threads) meets OpenBLAS's threading, and
  the clock is not throttled. The ladder artifacts (bench/ladder/,
  results/ladder.json) are committed for exactly that re-run.
- No further kernel optimization in this project phase — the remaining
  gap at q=8 is a threading + sandbox artifact, and optimizing against
  a throttled quota would tune for the wrong machine.

## Added at correction (2026-11-26): the verdicts this data actually yields for PLAN's H2 and H6

- **H2 (asm within ~5% of well-written AVX2 intrinsics): GREEN.**
  asm-v3 vs the intrinsics rung (O2): +3.4% / +3.2% at q=8 (n=2e5/1e6)
  — inside the 5% clause; and 2.1x-2.4x FASTER at q=17 (51.4-60.0 vs
  24.0-24.8 ns/row). The "value of asm is control" reading of PLAN §2
  is supported: the intrinsics rung needed the same hand-tiled stripe to
  get within 3% at q=8, and loses the panel-amortized regime entirely.
- **H6 (6x8 tile >= 70% of measured FMA peak in cache, q+1 > p*): RED.**
  From results/gram-bench-v3.json (n=4096, L2-resident): q=32 = 21
  GFLOP/s = 36% of the measured 57.62 GFLOP/s FMA peak; q=65 = 28
  GFLOP/s = 49% (p* = 23.4, both cases above it). The same cases
  re-measured in results/gram-bench-v4.json: 37% / 50%. Cause per
  ADR-0005: the per-row FMA count and ~1.15 ymm-load-uops/cy limit the
  in-cache tile to ~half of the 2-FMA/cy peak on this host — the same
  load-port ceiling that closes G2 (ADR-0012).
