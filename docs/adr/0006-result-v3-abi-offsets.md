# ADR-0006: Result struct v3 (f_stat) and the ols_abi_offsets repair

Date: 2026-11-25 (session 3) · Status: accepted

## Context
The result struct is frozen-but-versioned: fields are append-only and every
change must ship with (a) the new offsets in `src/abi.inc`, (b) the mirrored
ctypes layout in `tests/py/test_fit.py`, (c) the drift check in
`tests/py/test_phase2.py`, and (d) a version marker. Two findings from the
session-3 audit forced this ADR:

1. **`ols_abi_offsets()` is declared in `include/ols.h` but not exported from
   the library.** The commit that designed it (`8513e04`) had its r2/t block
   reverted pending the heap fix; when `e437fe2` re-applied the stats, the
   fingerprint export never came back. The drift-test docstring still claims
   it consumes that export.
2. The statistics block is missing the F-statistic (audit §2C): the
   regression-wide F = ((SST − SSE)/p) / (SSE/(n − q)) has no home in the
   result struct, and callers cannot do model comparison without it.

## Decision
1. **Result v3** = append `f_stat` (double) at offset 96 → size 104. Semantics:
   F = ((sst − sse)/p) / (sse/max(1, n − q)); guarded: p = 0, sse ≤ 0, or
   n − q < 2 → f_stat = 0. `sst` is the centered SST when the intercept is
   on, raw otherwise (same convention as r2, so F is the standard
   regression F-test under both models). Zeroed on error, like the other
   scalars. `intercept` is NOT added to v3; that belongs to problem v2/P3.4.
2. **`ols_abi_offsets()` is implemented and becomes the single source of
   truth the drift test consumes**: it packs the frozen offsets
   (problem fields, result fields, gram/resid arg-blocks, struct sizes)
   into a u64 FNV-1a-style checksum. `test_phase2.py::test_struct_drift`
   fails unless the asm-computed checksum equals the Python-side computed
   checksum over the SAME table. Any drift in either side changes its hash
   and fails the test. (The u32 per-field approach in the old design was
   replaced by a whole-table hash: one comparison, no ordering
   ambiguity, same failure guarantee.)
3. `ols.h` reference fixes shipped with this ADR: `struct_check.py` →
   `test_phase2.py`; ADR-0006 citation now points at this file.
4. Bumping problem v2 (adding `predict` support fields) is deferred to the
   `ols_predict` task (P3.4) so each API change lands with its feature.

## Consequences
- The drift test now exercises the actual exported symbol — a compile-time
  ABI drift between header and asm is impossible to miss.
- `f_stat` adds one load/stores per fit; negligible (stats block is
  O(q), fit is O(n·q)).
- The u64 return avoids one syscall-ish comparison per field; the Python
  side computes the same hash with the same constants (offsets are already
   mirrored in ctypes).
- This ADR backfills documentation for the v2 append (`t` at 88, size 96,
  `e437fe2`) which shipped without its ADR — a process miss noted in the
  phase-3 report.
