# asm-ols — ordinary least squares in x86-64 assembly

OLS (simple and multiple regression) in hand-written NASM, engineered to be
undeniable: certified NIST StRD accuracy, ABI-probe and guard-page evidence,
mutation-tested kernels, measured-not-assumed performance.

Status: **Phase 0–2 complete** (G0, G1 green; Phase-2 pyramid green — Gram
engine, Cholesky, `ols_fit` Tier-1, statistics). The Phase-2 throughput gate
G2 is honestly RED: v3 kernel reaches 20–54% of the per-case binding bounds
(DRAM-resident cases vs the 18.89 GB/s 1-thread DRAM ceiling, cache cases
vs the 57.6 GFLOP/s FMA peak; gate ≥ 85%), evidence in docs/adr/0005 and
docs/perf-log.md.

## Quick start
    make verify          # lint + build + full test pyramid (== CI)
    make bench           # machine characterization -> results/machine.json

## Status (end of Phase 7 session, 2026-11-25)
- **Inference complete** (ADR-0010 + ADR-0011): transcendental kernels
  (exp/log/lgamma/ibeta, 2.1e-16..3.9e-14 vs mpmath) → t/F p-values →
  **inverse**: `ols_t_qvalue_f64_sse2` (t critical value, bisection on the
  p-value kernel, 2.3e-10 worst vs scipy) and `ols_confint_f64_sse2`
  (β ± t·se endpoints, 2.8e-10 vs scipy, CI99 df=1 exact).
- **CLI**: `--json` now emits `p_values` (per coefficient + intercept),
  `f_pvalue`, `f_df`; `--confint[=LEVEL]` (default 0.95) adds
  `CI95% [lo, hi]` per coefficient in human output and a `confint`
  object in JSON. End-to-end fit→CI vs lstsq+scipy: 1.1e-15.

## Status (end of Phase 4/5 sessions, 2026-11-25)
- **Phase 4 complete**: CLI + `.olsb`/CSV I/O (`cli/asm_ols.c`, coefficients
  exact vs numpy 4e-15, ridge vs mpmath 3.6e-15); OLS_QR_PIVOT (method 4)
  fully wired — pivoted Householder with column-0 pinning, rank-refusal with
  detected rank, perm-aware unscaling (LRE 12-14 vs mpmath on mixed-scale
  designs; se/t zeroed per contract, ADR-0007 successor note in ols.h);
  multithreaded Gram `ols_gram_mt` (row-partitioned pthreads, fixed-index-
  order merge; same-T bitwise reproducible, ADR-0008).
- **Phase 5 ladder measured** (ADR-0009, `results/ladder.json`): honest
  verdicts — H1 RED (numpy wins both q=8 cases decisively, 2.24x/2.31x;
  n=2e5 q=17 is a statistical tie awarded to numpy; asm-v3 wins only
  q=17, n=1e6 DRAM-bound, +10.9%), G2 RED (standing). [The
  phase-5 docs originally labeled this H2; corrected 2026-11-26 — PLAN's
  H2 is the asm-vs-intrinsics clause, see the hypothesis table below.]
  The engine's headline value is ACCURACY (12-15 LRE, kappa study) and
  the q>=16 DRAM-bound single-thread regime.
- Critical fix this session: `ols_workspace_size` clobbered the caller's
  rbx (mov-then-push); every C caller could crash post-free.
- **Phase 8 (2026-11-26): G2 CLOSED RED-with-evidence** (ADR-0012). The
  third evidence-backed iteration — gram v4, a fully fused q<=8 kernel
  with call-persistent register accumulators — verified correct (gram
  differential ALL OK, NIST simple-fit table 11.9) but measured a TIE
  with v3 at q=8 (17.36 vs 17.17 ns/row, both 20% of the DRAM bound;
  prediction #28 FALSIFIED, perf-log). v3 dispatch restored; v4 kept in
  source as documented evidence. Closure argument: both q regimes are
  load-port/issue-bound on this host (measured ~1.15 ymm-load-uops/cy),
  so the 85%-of-DRAM clause is unreachable regardless of kernel tuning
  (v1 2% -> v3 20% -> v4 20%). First ols_gram_mt measurements: 18.0-18.7%
  of all-thread BW (H5 RED on this 2-vCPU host). 8-bug bring-up log in
  ADR-0012 (the xmm8/ymm8 aliasing class struck twice more).
- **Phase 9 (2026-11-26): G4 closure + fuzzer + freestanding CLI.**
  The blanket "G4 green" claim in the phase-8 report was UNEARNED: three
  of its four clauses had no artifacts. All three built this session:
  (1) **1e7 x 16 E2E** (`tests/py/e2e_1e7.py`, `results/e2e-1e7.json`):
  .olsb generated + fit vs a streamed numpy normal-equations reference —
  max rel slope diff 2.1e-13, max abs diff 2.9e-13 (gates: rel slopes
  5e-10, abs intercept 1e-8; all clause_ok true); the honest timing
  (asm 14.4 s vs OpenBLAS-threaded reference 1.4 s) is
  recorded, consistent with the H1 verdict.
  (2) **Parser fuzzing** (`tests/py/fuzz_csv.py`; committed artifacts:
  `results/fuzz-csv.json` seed 42 — 5 s, 5,731 cases, and
  `results/fuzz-csv-seed1337.json` seed 1337 — 20 s, 23,956 cases; both
  clean, 29,687 total) — found **4 real CLI bugs** (unescaped
  name in JSON keys; `-nan`/`inf` stats tokens; `--features` values that
  resolve to no column ran the fit on UNINITIALIZED X; raw invalid-UTF-8
  bytes in .olsb names passed through to stdout). All fixed + locked by
  `tests/py/cli_regression.py` (now in `make verify`). Fuzz runs clean.
  (3) **Freestanding CLI** (`cli/freestanding.c` -> `asm-ols-fs`, 112 KB):
  `_start` + raw syscalls, `-nostdlib -static`, readelf-verified no
  dynamic section + NX stack, links the PRODUCTION core objects; JSON
  output is BITWISE identical to the libc CLI (13/13 fields,
  `tests/py/freestanding_check.py` in `make verify`). Two honest notes:
  numbers print via a hand-rolled %.17g on __float128 + libgcc
  (300k-case sweep vs libc: ALL OK — double/long-double rescaling
  measurably loses the 17th digit), and `ols_init(-1)` (auto) is
  REQUIRED: the first draft passed 0, force-selecting SSE2 and silently
  changing every result by ~1e-15 (the documented sse2-vs-avx2
  accumulation contract).
  Also: `scripts/to_olsb.py` (CSV/.npy -> .olsb converter, PLAN-named,
  was missing), `tools/make_plots.py` + 3 evidence plots (roofline,
  thread-scaling, ladder; `make plots`, wired into `make repro`),
  `results/binary-size.json`, and a repair of `results/nist-phase2.json`
  (committed copy was all-zero LRE — a stale artifact; regenerated green).
- **G4 final grade (honest split)**: E2E clause GREEN; fuzz clause GREEN
  (after the 4 fixes); static-binary clause GREEN; thread-scaling clause
  RED — it is the H5 verdict (18.0-18.7% of all-thread BW on this 2-vCPU
  quota host; machinery verified, re-run on bare metal remains open).

## Status (end of Phase 3, 2026-11-25)
- CSNE refinement + ridge wired; AUTO = Gram+Cholesky (the CSNE study
  measured kappa* = 1e6; AUTO was reverted to plain chol after the L8
  1e9-offset finding — commit de7239c, docs/reports/phase3-accuracy.md).
- OLS_QR: Householder on an equilibrated weighted augmented copy; per-method
  NIST table green (QR 13.1 Longley, 12.2 Norris; Filip refused at fit
  level, raw-kernel LRE 7.11 vs certified). f_stat + ols_predict shipped.
- H4 CONFIRMED: results/accuracy/kappa-sweep.json (+ plot).
- Mutation pipeline repaired — the phase-1 score was inflated; honest
  28/34 = 0.8235 (results/mutation-phase3.json). Details:
  docs/reports/phase3-accuracy.md.

## Results (generated from results/, never hand-typed)
### Hypothesis verdicts (PLAN §2, one line each, measured not assumed)
| # | Hypothesis (pre-registered) | Verdict | Evidence |
|---|---|---|---|
| H1 | Fused single-pass beats NumPy/SciPy pipelines by ~the pass-count ratio in the memory-bound regime | **RED** | `results/ladder.json`: numpy wins the two q=8 cases decisively (2.24x/2.31x faster, threaded vs our single thread); n=2e5 q=17 is a statistical tie (24.19 vs 23.97 ns/row, 0.9%) awarded to numpy; asm-v3 wins exactly one — q=17 n=1e6 DRAM-bound, +10.9%. The pass-count-ratio model is falsified on this host: OpenBLAS's threading + throttled sandbox dominate. |
| H2 | Hand-written asm within ~5% of well-written AVX2 intrinsics | **GREEN** | `results/ladder.json`: asm-v3 vs intrinsics(O2) = +3.4%/+3.2% at q=8 (within the 5% clause) and 2.1x-2.4x FASTER at q=17 (51.4-60.0 vs 24.0-24.8 ns/row). The value-is-control reading of PLAN holds. |
| H3 | Dot2 accumulation costs <15% when memory-bound | **NOT BUILT** | Dot2 is a COULD-tier item, never built — no verdict. Recorded honestly rather than dropped. |
| H4 | CSNE restores QR-like accuracy up to a measured kappa*; plain Gram+Cholesky fails near kappa ~ 1e8 | **CONFIRMED** | `results/accuracy/kappa-sweep.json` (+ plot): kappa* = 1e6 measured. |
| H5 | N threads reach >= 85% of measured all-thread bandwidth | **RED (this host)** | `results/gram-bench-v4.json`: ols_gram_mt 2 threads = 6.76 GB/s (q=8) / 7.01 (q=17) vs 37.6 GB/s all-thread ceiling = 18.0-18.7%. 2-vCPU quota-capped KVM host; the rung is measured and committed for a bare-metal re-run. |
| H6 | 6x8 tile >= 70% of measured FMA peak in cache, q+1 > p* | **RED** | `results/gram-bench-v3.json` (and v4 re-run): q=32 (n=4096, L2) = 21-22 GFLOP/s = 36-37% of the measured 57.62 GFLOP/s FMA peak; q=65 = 28 GFLOP/s = 49-50%. p* = 23.4 (machine.json), so both cases are in the FMA-bound regime. |
| G2 | Fused Gram >= 85% of measured applicable bandwidth | **RED, CLOSED** | ADR-0012: v1 2% -> v3 20-54% -> v4 20% of bound across three evidence-backed iterations; load-port/issue-bound on this host; re-open conditions named. |

- NIST StRD LRE (worst coefficient), simple fits: Norris 12.1, Pontius 11.9,
  NoInt1 14.7, NoInt2 15.0 — pre-registered gate was >= 10.
  Table: results/nist-phase1.json
- NIST through the full `ols_fit` path (Gram + Cholesky + stats): the
  11.9 worst-LRE in results/pyramid-summary.json is the simple-fit table;
  the full-path Gram+Cholesky table (results/nist-phase2.json) lands at
  worst LRE 7.40 (Wampler1, above its 7.0 gate). se/t-stats/r2_adj match
  SciPy; 1e9-offset adversarial data agrees with mpmath. Summary:
  results/pyramid-summary.json, results/nist-phase2.json
- Machine: FMA peak 57.6 GFLOP/s (2 FMA/cy @ 3.57 GHz), DRAM 18.89 GB/s
  1-thread / 37.6 GB/s all-thread (run-to-run 16.2–18.9 on this shared KVM
  host), roofline crossover p* = 23.4 (23–27 across runs). Full data:
  results/machine.json
- Gram kernel v3 (AVX2+FMA, group-pipelined column-panel): 17.2 ns/row at
  q=8 and 24.4 ns/row at q=17, DRAM-resident, 11–21x over v1 — but 20–54%
  of the per-case binding bounds (19.7% of the 1-thread DRAM ceiling at
  q=8, 29.5% at q=17; cache cases vs the FMA peak), so gate G2 stays
  open. Data: results/gram-bench-v3.json, verdict in
  docs/adr/0005-gram-v3-tile-and-gate.md
- Mutation score (Phase-1 kernels): 28/34 killed = 0.8235 (regenerated
  2026-11-26; the old 20/31 survives only in the historical phase-1
  report); survivors analyzed in results/mutation-phase1.json (test gaps
  queued, equivalents documented).

## Layout
src/ = NASM core (abi.inc, macros.inc, core/) plus one C thread shim
(thread/ols_mt.c, pthreads — ships as ols_mt.o in libols.a/libols.so).
Other C: cli/ (asm_ols.c, freestanding.c) and the bench/, tests/ harnesses;
Python in tests/, bench/, tools/, scripts/. include/ols.h is the public
ABI. docs/adr/ holds decisions, docs/perf-log.md the predict-then-measure
log. Build outputs land in build/lib (libols.so + libols.a, production
objects only), build/tests, build/cli — never in the source tree.

## Documentation
docs/README.md is the index: design.md (architecture rationale),
architecture.md (module map, call graph, layouts), numerical_stability.md
(numerics + NIST evidence), performance.md (machine limits + measured
kernels + gate status), perf-log.md (predict-then-measure ledger),
agent-comms.md (cross-session log), adr/ (decision records),
reports/ (phase reports).

## What I got wrong (so far)

Phase 8 additions (ADR-0012): prediction #28's uop model assumed 32B
vector row loads; shipping guard-safe scalar loads erased the predicted
win — the v4 kernel tied v3 and was not dispatched. A recurring subnormal
(1.39e-309) was initially mislabeled as probe noise; it was a real bug
(unzeroed xmm0 preloading the batch sums) that only fired on exact
cancellation — caught by the P1 suite, not by my random probes.
1. Raw second moments instead of pivot-shifted sums: Sxx cancelled to 0 at
   offset 1e9 (intrinsic to the formula, not accumulation). Fixed per
   numerical-ols-algorithms §2.
2. Mean computed as (c + s')/n instead of c + s'/n — Norris b0 LRE 0.4.
3. Quad-fit back-substitution dropped the -a1*cx + a2*cx^2 terms.
4. A "NaN guard" that rejected every negative slope (caught by the mutation
   baseline).
5. Instruction-width confusion in the v3 Gram kernel: vmovupd ymm moves 4
   doubles, not 8 — the packer chunk stride and per-group flushes were wrong
   until a listing-vs-objdump audit caught it (docs/dev history).
6. Legacy SSE scalar ops (movsd/ucomisd) after VEX code dirtied upper YMM
   halves: the SSE/VEX transition penalty silently cost 2x. All scalar ops
   in AVX2 regions are now VEX-encoded.
1–4 are in docs/reports/phase0-1/phase1-simple-regression.md; 5–6 in
docs/adr/0005-gram-v3-tile-and-gate.md and docs/perf-log.md.
