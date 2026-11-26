# ADR-0007: QR solver scope for Phase 3 — Householder on an explicit copy

Date: 2026-11-25 (session 3) · Status: accepted

## Context
PLAN.md Phase 3 lists "Householder/TSQR on `[X y]`" and the audit (§2B)
recommends a QR-based solver as the technically important addition: normal
equations square the condition number (κ²·u digits lost), QR loses only
κ·u. The design space for `qr.asm` had three candidate shapes:

- **(a) In-place factorization of a workspace copy of [1|X|y]** — Householder
  with the LAPACK convention (reflected β so R_00 ≤ 0; sign(R_jj) flip
  guards cancellation in the norm).
- **(b) TSQR / communication-avoiding** — the tall-skinnyblocked variant.
- **(c) Column-pivoted QR** — rank-revealing.

## Decision
**v1 ships (a) only**: `ols_qr_householder(A, q, nb, stride)` factoring an
explicit workspace copy of the augmented design matrix A = [1|X|y]
(n × q, q = p+1, nb = number of rows to factor), with `ols_qr_solve`
(R-back-substitution) and se from the standard inverse-of-R diagonal
recursion. SSE comes out as the byproduct ρ² of the last column — this is
the audit's "fused [X y] byproduct" idea in its cheapest form.

Rationale:
- The correctness target (NIST G3: QR ≥ 10 LRE on the average set, ≥ 6 on
  Longley/Wampler) is about *numerics*, not parallel blocking; plain
  Householder on the tall matrix is sufficient and the reference (mpmath
  Householder QR) is straightforward to verify against.
- An explicit copy costs one n·q pass — measured against the Gram pass
  it is the same order (Phase-5 material); the copy also decouples the
  factor from caller memory (caller X/y stay const, no in-place mutation
  surprises at the ABI boundary).
- Column pivoting adds a norm-tracking outer loop and a permutation API
  (rank + column order reported); deferred as `OLS_QR_PIVOT = OLS_E_ARG`
  until the non-pivoted path is green and measured.

Performance: SSE2-scalar asm (register-blocked where it matters at q ≤ 32),
prediction in perf-log **before** measurement per house rules; SIMD
vectorization of the Householder update is Phase-5 baseline-ladder scope,
not Phase-3 scope. This is an accuracy phase; its win condition is LRE,
not ns/row.

## Consequences
- `OLS_QR` gains a real implementation; `method_used` reports truthfully.
- Workspace grows: `ols_workspace_size` must account for the n×q copy for
  QR problems. Formula: classic + n·q·8 + 64 slack, rounded to 64 (the
  n·q copy replaces nothing; CSNE reuses the existing residual pass).
- `rcond` handling: QR reports rank via R_jj ≤ rcond·R_00; rank < p →
  `OLS_E_SINGULAR`; else 0 < smallest ratio < rcond·1e3 (a measured
  soft floor) → `OLS_E_ILLCOND` (executes but flags). Default rcond = 0
  → use `max(p, q)·u` — the same default LAPACK uses for geqp3-ish work
  (not a spec quote; chosen for consistency with our chol eps_rel).
- Filip: expected verdict QR-refused (rank-deficient at double precision)
  or solved with poor LRE — either way reported without a target, per the
  pre-registered G3 note.
- TSQR and pivoting stay SHOULD-tier; starting them requires this ADR's
  successor.
