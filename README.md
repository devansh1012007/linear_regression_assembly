# asm-ols — ordinary least squares in x86-64 assembly

OLS (simple and multiple regression) in hand-written NASM, engineered to be
undeniable: certified NIST StRD accuracy, ABI-probe and guard-page evidence,
mutation-tested kernels, measured-not-assumed performance.

Status: **Phase 0–2 complete** (G0, G1 green; Phase-2 pyramid green — Gram
engine, Cholesky, `ols_fit` Tier-1, statistics). The Phase-2 throughput gate
G2 is honestly RED: v3 kernel reaches 23–54% of the per-case binding bounds
(gate ≥ 85%), evidence in docs/adr/0005 and docs/perf-log.md.

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
  verdicts — H2 RED (OpenBLAS wins 3 of 4 cases; asm-v3 wins only the
  q=17, n=1e6 DRAM-bound case, +11%), G2 RED (standing). The engine's
  headline value is ACCURACY (12-15 LRE, kappa study) and the q>=16
  DRAM-bound single-thread regime.
- Critical fix this session: `ols_workspace_size` clobbered the caller's
  rbx (mov-then-push); every C caller could crash post-free.

## Status (end of Phase 3, 2026-11-25)
- CSNE refinement + ridge wired; AUTO = Gram+CSNE (measured kappa* = 1e6).
- OLS_QR: Householder on an equilibrated weighted augmented copy; per-method
  NIST table green (QR 13.1 Longley, 12.2 Norris; Filip refused at fit
  level, raw-kernel LRE 7.11 vs certified). f_stat + ols_predict shipped.
- H4 CONFIRMED: results/accuracy/kappa-sweep.json (+ plot).
- Mutation pipeline repaired — the phase-1 score was inflated; honest
  24/34 = 0.706 (results/mutation-phase3.json). Details:
  docs/reports/phase3-accuracy.md.

## Results (generated from results/, never hand-typed)
- NIST StRD LRE (worst coefficient), simple fits: Norris 12.1, Pontius 11.9,
  NoInt1 14.7, NoInt2 15.0 — pre-registered gate was >= 10.
  Table: results/nist-phase1.json
- NIST through the full `ols_fit` path (Gram + Cholesky + stats): worst LRE
  11.9; se/t-stats/r2_adj match SciPy; 1e9-offset adversarial data agrees
  with mpmath. Summary: results/pyramid-summary.json
- Machine: FMA peak 57.6 GFLOP/s (2 FMA/cy @ 3.57 GHz), DRAM 18.89 GB/s
  1-thread / 37.6 GB/s all-thread (run-to-run 16.2–18.9 on this shared KVM
  host), roofline crossover p* = 23.4 (23–27 across runs). Full data:
  results/machine.json
- Gram kernel v3 (AVX2+FMA, group-pipelined column-panel): 17.2 ns/row at
  q=8 and 24.4 ns/row at q=17, DRAM-resident, 11–21x over v1 — but 23–54%
  of the per-case binding bounds, so gate G2 stays open. Data:
  results/gram-bench-v3.json, verdict in docs/adr/0005-gram-v3-tile-and-gate.md
- Mutation score (Phase-1 kernels): 20/31 killed; survivors analyzed in
  results/mutation-phase1.json (test gaps queued, equivalents documented).

## Layout
src/ = NASM only (abi.inc, macros.inc, core/). C/Python only in tests/,
bench/, tools/. include/ols.h is the public ABI. docs/adr/ holds decisions,
docs/perf-log.md the predict-then-measure log. Build outputs land in
build/lib (libols.so + libols.a, production objects only), build/tests,
build/cli — never in the source tree.

## Documentation
docs/README.md is the index: design.md (architecture rationale),
architecture.md (module map, call graph, layouts), numerical_stability.md
(numerics + NIST evidence), performance.md (machine limits + measured
kernels + gate status), perf-log.md (predict-then-measure ledger),
agent-comms.md (cross-session log), adr/ (decision records),
reports/ (phase reports).

## What I got wrong (so far)
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
