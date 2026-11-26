# performance.md — machine limits, measured kernels, honest gates

Rule: predictions land in `perf-log.md` *before* the measurement that tests
them; every number below cites its producer script + results file. Claims
derived from these numbers inherit the CV policy of the benchmarking skill
(shared KVM host → run-to-run variance is part of the claim).

## 1. Machine characterization (`bench/machine/characterize.py` → `results/machine.json`)

Host: Intel Xeon KVM guest, 2 vCPU, 4 GiB, AVX2+FMA3+AVX-512 present
(commit `f658f3a` tree, 2026-11-20, second run):

| Limit | Value |
| --- | --- |
| FMA peak (scalar accumulators swept 1..16) | 57.62 GFLOP/s (best at 15 accumulators) |
| Bandwidth ladder (1t) | L1 195.05 · L2 143.13 · L3 29.14 · DRAM 18.89 GB/s (all-thread 37.6) |
| Roofline crossover `p*` (fused Gram, 1t DRAM) | 23.4 — `p* + 1 = 8 · peak_flops / dram_bw_1t` |

Reproducibility note (machine.json `notes`): the first run measured DRAM
16.23 GB/s and p* 27.4; run-to-run DRAM variance is 16.2–18.9 GB/s and
p* 23–27 on this shared host. Any README/doc claim states the run it came
from — the current README cites these values from `results/machine.json`,
not from memory.

## 2. Kernel measurements

**Gram engine** (`results/gram-bench.json` v1, `results/gram-bench-v3.json`
v3, AVX2+FMA3, 1 thread, 30 reps, medians with CV 0.5–3.6%):

| Case | v1 | v3 | Speedup |
| --- | --- | --- | --- |
| q=8, n=200k (DRAM) | 0.33 GB/s | 3.73 GB/s (17.2 ns/row ≈ 61 cy) | 11.3× |
| q=17, n=200k | 0.26 GB/s | 5.57 GB/s (24.4 ns/row ≈ 87 cy) | 21.4× |
| q=32 (in-cache) | — | 50.5 ns/row (≈ 180 cy) | — |
| q=65 (in-cache) | — | 152.9 ns/row (≈ 545 cy) | — |

**Everything else** (sum/dot/axpy, simple fits, chol, residual, pivot means)
is differential-tested; their throughput work is deliberately *not* started
until the Phase-5 baseline ladder exists to measure against.

## 3. Gate G2: CORRECTNESS GREEN, PERFORMANCE RED (honest)

Pre-registered target: fused Gram ≥ 85% of the measured applicable
bandwidth. FINAL STATUS (2026-11-26, ADR-0012): RED, CLOSED with evidence
after the third iteration. v1 2% -> v3 22-54% -> v4 (fully fused q<=8,
verified correct) 20% at q=8 — a measured tie with v3. Both q regimes are
load-port/issue-bound on this host (~1.15 ymm-load-uops/cy, 1-2 FMA/cy,
measured machine.json): q>=9 needs ~125 uops/row >= 31 cy/row vs a 12-41
cy DRAM budget (impossible at 4 uops/cy); q<=8's in-bounds load sequence
alone floors at ~12-14 cy/row. Re-open on an AVX-512 host or measured
better load throughput. Data: results/gram-bench-v4.json (incl. the first
ols_gram_mt rungs: 18-19% of all-thread BW, H5 RED here).

Recorded root causes (perf-log): per-element `sqrt(w)` reloads (~q `sqrtsd`
per row on the weighted path), scalar G tails, strided G access — the kernel
is latency-bound, not bandwidth-bound, so bandwidth-targeted micro-tuning
alone cannot close the gap. v2 (group-pipelined column-panel) was paused
with a written negative result; v3 landed a partial redesign worth 11–21×
but still under gate. The next iteration plan and its prediction are in
`perf-log.md` (#20 and following); ADR-0005 documents the tile/gate design
and the pre-registration discipline.

This red gate is displayed, not hidden: the pyramid prints it, the phase-2
report headlines it, and the README "status" section carries it.

## 4. What the numbers do NOT yet cover

- **No regression-engine benchmark yet.** The measured kernels are the
  building blocks; end-to-end `ols_fit` throughput (ns/row incl. pivots +
  stats + solve) is predicted in perf-log #17 but not yet benchmarked —
  that is Phase-5 scope.
- **No external baselines yet.** C `-O2/-O3`, C AVX2 intrinsics, NumPy,
  SciPy/OpenBLAS comparisons, n-scaling 1e3..1e7, observations/s and
  roofline-utilization curves are the Phase-5 deliverable (PLAN.md §Phase 5).
  H2 ("hand-written asm within ~5% of intrinsics") is explicitly falsifiable
  and untested until then. [Phase-5 measured, ladder.json: H2 GREEN —
  +3.2-3.4% at q=8 (inside the clause), 2.1-2.4x faster at q=17. The
  asm-vs-OpenBLAS result is H1 (RED); labels corrected 2026-11-26.]
- **Threading**: single-threaded only; the roofline all-thread numbers
  (DRAM 37.6 GB/s) bound what N threads could recover (H5), but no threaded
  kernel exists.
