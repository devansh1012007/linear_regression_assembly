# ADR-0003: Data layout for the kernels — strided arrays now, Z=[X|y] packing in Phase 2

**Status:** Accepted (Phase 0/1) **Date:** 2026-11-20

## Context
PLAN.md §5 fixes the long-term data model: row-major `Z = [X|y]`, zero-padded
packed blocks, pivot-shifted accumulation. Phase 1's kernels (sum/dot/axpy/
simple regression) run directly on caller-owned arrays, so the *interim*
layout must be pinned too, and it must not paint Phase 2 into a corner.

## Decision (Phase-1 interim)
- Kernels operate on **caller-owned 1-D strided-free arrays** (`const double *x`
  with element stride 1, any 8-byte alignment — contracts say "any alignment",
  guard-page + misalignment tests enforce it).
- Simple regression reads x and y as **separate arrays** (not Z-packed): the
  Phase-1 slice is a ctypes/CLI-friendly kernel set, and copying to Z would
  add a pass and hide the raw-kernel evidence the pyramid wants.
- All accumulators live in registers / small stack frames; **no allocation,
  no syscalls** in the core (asm-project-delivery §2 holds from day one).
- Output structs are plain double arrays with fixed layout documented in each
  contract header (`out[0..7)` etc.), so the ABI between asm and ctypes/C is
  byte-stable and testable.

## Decision (Phase-2, unchanged from PLAN)
- `Z = [X|y]` row-major, q = p+1 columns, packed to a multiple of NR=8 with
  zero padding; row stride padded to 4 doubles; blocks of rows sized so one
  block's Z stays in L1/L2 while all tiles consume it; accumulators in ymm;
  upper-triangle tiles only, mirrored once at flush.
- Rationale: row-major Z gives the rank-1-update microkernel contiguous
  per-row features (2 vmovupd + 6 vbroadcastsd + 12 vfmadd231pd per row for
  the 6x8 tile); zero padding removes every column tail and masked op from
  the hot loop (simd-kernel-optimization §3).
- Pivot-shifted accumulation + Chan merge for centering (numerical-ols §2).

## Why not Z-packing already in Phase 1
The Phase-1 kernels are the *evidence generators* for the harness (ABI probe,
guard pages, mutation testing, NIST) at minimum complexity. Introducing the
packing layer before the Gram engine exists would couple the harness to an
unmeasured layout. Phase 1 keeps kernels pure; Phase 2 adds packing as its
own task with its own guard-page and misalignment tests.

## Consequences
- Phase-1 fits (simple/noint/quad) are 2-3-pass over x,y. That is fine: they
  are correctness/accuracy vehicles, not the performance story. The fused
  single pass is Phase 2's H1 test.
- The quad fit derives z = x^2 on the fly (no third array) — 3 sequential
  compensated passes; measured cost goes to perf-log #10.
- Misalignment tests sweep offsets 0..63 bytes in 8-byte steps; contracts
  allow any 8-byte-aligned pointer, so odd byte offsets are OUT of contract
  and untested (documented; a `OLS_E_ALIGN` error exists for the future
  API layer if we ever tighten this).
- ctypes bindings use `C_CONTIGUOUS` ndarrays; strided inputs are copied by
  numpy at the boundary (test-side concern only).

## Revisit trigger
Phase-2 packing task (the layout above is re-litigated there with llvm-mca +
measurement, per PLAN §5 "change if the evidence says so").
