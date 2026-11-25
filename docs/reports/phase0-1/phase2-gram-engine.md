# Phase 2 report — the Gram engine (PARTIAL: correctness green, perf gate red)

**Date:** 2026-11-20 · **Commits:** 43a29ea, c3e331c, 4a02340

## What was done
1. **ABI layer**: ols_problem_t/ols_result_t (frozen layouts, struct-drift
   test), GA_/RA_ arg structs, ols_workspace_size, full validation paths.
2. **Kernels (all NASM)**: fused pivot-shifted Gram (AVX2 4-row panel with
   sqrt-w staging + SSE2 reference), stride-aware Cholesky factor/solve/inverse
   (compensated dots), fused residual pass (SSE + X'r, Neumaier), weighted
   pivot means, and the ols_fit Tier-1 driver (validation -> pivots -> Gram ->
   centering -> chol/solve -> residual -> se/cond_est; SSE2/AVX2 dispatch).
3. **Tests**: kernel differentials (shapes p<=33 x n<=257 x weights x ISAs —
   ALL OK), ols_fit end-to-end vs NumPy/mpmath (ALL OK incl. adversarial 1e9
   offset to 2e-6 and weighted WLS to 1e-16), NIST via ols_fit (LRE >= 10 on
   all four datasets), guard pages for every Phase-2 kernel (n=1..70, p=1..9),
   struct drift check, cross-ISA bounds. make verify green.
4. **G2 benchmark**: run with pre-registered predictions.

## Gate G2 status: CORRECTNESS GREEN, PERFORMANCE RED
- All tests green on SSE2/AVX2; adversarial-offset pass; guard/ABI green.
- In-cache/DRAM gram throughput measured ~0.24-0.34 GB/s = ~2% of the 85%-of-
  bandwidth target [pre-registered]. Honest negative result, root causes
  logged in perf-log (per-element sqrt(w) reloads ~q sqrtsd/row, scalar G
  tails, strided G): the kernel is latency-bound, not bandwidth-bound.
  First Phase-2b task: stash sqrt(w) per row once, panel-local s sums,
  contiguous G blocking. Correctness harness already in place to re-gate.

## Bugs found and fixed this phase (all caught by the harness)
- gram: u/s_c frame overlap; size_t loop underflow (jb never fires on -1);
  stale-xmm0 zero-fill across blocks; OOB sqrt(w) loads on partial blocks;
  missing loop-back (symbol emitted into .note section).
- fit: u32 field compares; r2/sst stores via truncated pointers (mov eax,1);
  sst read a first moment instead of the centered G; se destroyed the means
  it later needed; W not reloaded after a call clobbers xmm4; err_zero used
  clobbered rsi (segfault via rsi=p); eps loaded as integer 2^52 bits;
  workspace formula undercounted q*8 bytes (heap corruption!); no-intercept
  needs zero pivots (shifted solve returns slopes for (x-c)).
- test refs: weighted design must scale the intercept column too; weighted sst.

## Not finished (Phase 2 remainder + Phases 3-6)
Per PLAN.md: CSNE, QR/TSQR, ridge/rank/auto (P3); threading, .olsb/CSV I/O,
freestanding CLI, fuzzing (P4); sweeps/baselines/plots/README/auditor (P5);
Dot2 (P6). The G2 performance iteration (kernel redesign above) is the
highest-priority incomplete item.
