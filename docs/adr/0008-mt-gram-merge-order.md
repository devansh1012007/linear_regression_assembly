# ADR-0008: Row-partitioned multithreaded Gram — merge order and the honest reproducibility contract

Date: 2026-11-25 (Phase 4c) · Status: accepted

## Context

PLAN Phase 4c requires "multithreading with fixed-order merge + reproducible
mode (bitwise equal for 1..N threads)" for the Gram engine. The kernel
(`ols_gram_f64_avx2`) is a fused pivot-shifted pass over rows; the natural
parallelization is row slices, each producing partial (G_t, s_t, W_t).

## Decision

1. **Orchestration in C, core stays pure asm.** `src/thread/ols_mt.c`
   (linked into libols with `-lpthread`) spawns T−1 workers + the main
   thread; each runs the UNCHANGED kernel on its slice into its own
   partials (same pivots c, per-thread kernel scratch). The asm core
   remains syscall-free and thread-unaware.
2. **Fixed-index-order merge.** Partials fold t = 0, 1, ... T−1 — never
   completion order. Same T ⇒ bitwise identical results across runs and
   schedulings.
3. **Reproducibility contract (honest revision of the PLAN wording).**
   Across DIFFERENT T, bitwise equality is mathematically impossible with
   a partials merge: the sequential kernel adds rows 0..n−1 in one
   compensated chain; T slices add slice-wise then merge — different float
   addition ORDER, different rounding (measured: ~1e-15 relative, well
   inside the compensated-summation bound). The gate therefore reads:
   (a) same T → bitwise, (b) across T → within ~n·u relative. A
   bitwise-1-vs-N scheme would need a merge replaying the kernel's exact
   intra-slice order — a serialization that defeats the parallelism.

## Consequences

- `ols_gram_mt(A, threads, scratch, scratch_bytes)`;
  `ols_gram_mt_scratch_bytes(q, T)` sizes ONE caller block (T·(q²+q+1)
  partial doubles + T kernel-scratch blocks, 64-aligned).
- threads ≤ 1 or n < 2T falls back to the single-thread kernel (tiny-n
  behavior identical).
- Slice arithmetic: X advances by row·ldx8 (bytes); y/w by row (elements).
- Measured in-sandbox (CPU-quota capped): n=2e6, p=7 — 41.8 ms (T=1) →
  20.3 ms (T=2) → 19.7 ms (T=4): T=2 saturates this allocation; the real
  scaling verdict (G4 ≥ 85% of all-thread bandwidth) waits for the
  Phase-5 ladder on uncapped hardware.
- The freestanding CLI does not link ols_mt (no pthreads) — single-thread
  by construction there.
