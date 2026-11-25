# Phase 6 — session 6 (2026-11-25): transcendental kernels + t/F p-values

Execution: PLAN COULD item "p-values/CIs (asm exp/log/lgamma/incomplete
beta, own accuracy test vs mpmath)". Commits: see git log; `make verify`
ALL GREEN at HEAD (13 asm files + src/thread/).

## Delivered
- `src/core/special.asm`: six kernels — exp, log, lgamma, ibeta, t/F
  p-value wrappers. All SSE2 baseline, contract-linted.
- Differential suite `tests/py/test_special.py` (in `make verify`):
  exp 2.1e-16, log 2.2e-16, lgamma 2.3e-15, ibeta 3.9e-14, t/F identity
  + wrappers ~1.6e-14 vs mpmath — every target met.
- ADR-0010 with the nine-bug post-mortem (three were test-side or
  reference-side; six real, four of them the recurring cross-call
  register-clobber class).

## Not finished
- CLI `--json` p-value fields (wrappers are exported and tested; the CLI
  wiring is a small follow-up).
- Confidence intervals (needs the t-quantile — inverse ibeta — a further
  COULD item).
- PR push (no GitHub credential in the sandbox; kit + staging ready).

## Open next session
1. CLI p-values in --json output (one field per coefficient + model F).
2. Optionally: inverse-ibeta (CI bounds), then the PLAN's stretch list.
3. Push via the resume kit once a PAT exists.
