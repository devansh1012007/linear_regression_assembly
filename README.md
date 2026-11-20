# asm-ols — ordinary least squares in x86-64 assembly

OLS (simple and multiple regression) in hand-written NASM, engineered to be
undeniable: certified NIST StRD accuracy, ABI-probe and guard-page evidence,
mutation-tested kernels, measured-not-assumed performance.

Status: **Phase 0 + Phase 1 complete** (G0, G1 green). See PLAN.md for the
roadmap and docs/reports/phase0-1/ for evidence.

## Quick start
    make verify          # lint + build + full test pyramid (== CI)
    make bench           # machine characterization -> results/machine.json

## Phase-1 results (generated from results/, never hand-typed)
- NIST StRD LRE (worst coefficient): Norris 12.1, Pontius 11.9, NoInt1 14.7,
  NoInt2 15.0 — pre-registered gate was >= 10. Table: results/nist-phase1.json
- Machine: FMA peak 57.61 GFLOP/s (2 FMA/cy @ 3.567 GHz), DRAM 16.2 GB/s
  1-thread / 23.3 GB/s all-thread, roofline crossover p* = 27.4.
  Full data: results/machine.json
- Mutation score: 20/31 killed; survivors analyzed in
  results/mutation-phase1.json (test gaps queued, equivalents documented).

## Layout
src/ = NASM only (abi.inc, macros.inc, core/). C/Python only in tests/,
bench/, tools/. include/ols.h is the public ABI. docs/adr/ holds decisions,
docs/perf-log.md the predict-then-measure log.

## What I got wrong (so far)
1. Raw second moments instead of pivot-shifted sums: Sxx cancelled to 0 at
   offset 1e9 (intrinsic to the formula, not accumulation). Fixed per
   numerical-ols-algorithms §2.
2. Mean computed as (c + s')/n instead of c + s'/n — Norris b0 LRE 0.4.
3. Quad-fit back-substitution dropped the -a1*cx + a2*cx^2 terms.
4. A "NaN guard" that rejected every negative slope (caught by the mutation
   baseline).
All four are in docs/reports/phase0-1/phase1-simple-regression.md.
