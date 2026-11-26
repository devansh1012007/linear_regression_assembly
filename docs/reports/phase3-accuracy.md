# Phase 3 — Accuracy engine (session 3, 2026-11-25)

Execution checklist: PLAN.md §Phase 3 · decisions: ADR-0006/0007 · all
numbers script-produced at commit `HEAD` of the phase-3 branch.

## Shipped

| Item | Evidence |
| --- | --- |
| API truth repairs: `ols_abi_offsets` exported for real (FNV-1a table hash, drift test consumes it), `ols_cpu_ident` declared, stale header refs fixed, `.gitignore` root-anchored | `b77d302`, `src/core/api.asm`, drift test |
| Result v3: `f_stat` at 96 (size 104), no-intercept dof fix pinned by closed-form differential | ADR-0006, test_fit stats differential |
| **CSNE actually implemented** (method 2 silently aliased plain chol before) — two fixed-point steps via the residual rsum/xtr | Gate MET: chol 8.3 → CSNE **13.1 LRE** at κ=1e4 vs mpmath |
| Ridge wired (`lambda`, slopes-only, weighted) | mpmath `mp_ridge` differential incl. weighted |
| `ols_predict_f64_sse2` | numpy differential p=0..16, guard case |
| **Householder QR** (`qr.asm`): equilibrated weighted augmented copy, reflected alpha, rank via rcond on the unit-norm diagonal, SSE = residual TAIL norm², wired as `OLS_QR` (rcond default 8·dc·u) | `d26b274`; kernel+end-to-end differentials; guard case; QA block hashed |
| `ols_predict` + QR API in ols.h; QA/PredArgs blocks in the drift table | drift test green |
| Kappa sweep study → **H4 CONFIRMED**: chol loses 2·log10(κ) digits, **CSNE is QR-grade to the measured κ\* = 1e6**, QR loses only log10(κ) (13.5→8.0 at κ=1e8) | `results/accuracy/kappa-sweep.json` + plot |
| AUTO now runs Gram+CSNE (data-derived rule); `method_used` reports truth | `032b69f` **[corrected 2026-11-26: AUTO reverted to plain Gram+Cholesky at `de7239c` after the L8 offset finding — the refinement residual loses u·\|offset\| precision at 1e9 offsets; the κ* = 1e6 data is retained]** |
| Per-method NIST table, 11 datasets × 3 solvers, G3 gates ALL GREEN: QR 13.1 Longley / 12.2 Norris / 12.0 Pontius; CSNE 13.7 Norris; Filip refused at fit level (raw-kernel solve: rank 11, LRE 7.11 vs certified — lstsq truncates and disagrees) | `results/nist-phase3.json` |
| Mutation pipeline repaired + honest score | `993b65f` |

## What I got wrong (this session)

1. **The phase-1 mutation score was fake.** `mutate.py` still linked the
   pre-review object layout, so every mutant failed to link and counted as
   "killed (link fail)" — 28 of 34 in the first rerun; and the mutated
   objects were never actually linked (duplicate symbols resolve to the
   original). The honest pre-fix score was **3/34**. After the lint kill
   and suite additions: **24/34 = 0.706**, 10 survivors analyzed by class
   **[superseded 2026-11-26: final honest score 28/34 = 0.8235 after the
   cancellation battery — see phase-4 report; the regenerated
   results/mutation-phase3.json holds the final run]**
   (`results/mutation-phase3.json`). A score that looks too good is a bug.
2. **Five QR bring-up bugs** were all caught by the harness stack, in
   order: solve's unsigned loop bound (segfault, pinned by a C
   SIGSEGV-handler driver), rank compared against the stride instead of
   dc, `norms[j]` vs `norms[j+ic]` (three sites), the sse tail's stale
   `imul` (garbage addressing), and the intercept functional missing the
   pivot rescale. Every one is now pinned by a test.
3. **The mp_qr_ols reference caught three of my own spec errors** before
   any asm existed: SSE = tail norm² (not R_dd²), the R[:,1:] gram
   includes row 0 AND each column diagonal, and the se solver needs a
   real L·L' factor.
4. The kappa-generator docstring claimed cond2(X) ≈ κ² — wrong (it is
   κ exactly); fixed.

## Open (next session)

- Mutant survivors (10, per-class list in results/mutation-phase3.json)
  **[now superseded: final run has 3 SURVIVED rows, 2 provably equivalent
  — survivor_analysis in results/mutation-phase3.json]**;
  runtime upper-half dirty check as a C helper.
- TSQR / column pivoting / multithreading (Phase 4), CLI + binary I/O
  (Phase 4), the full baseline ladder (Phase 5).
- PR staging refresh: the push kit in `download/` predates Phase 3 —
  regenerate `staging.json` from `docs/push-manifest.json` before pushing.
