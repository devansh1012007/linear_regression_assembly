# Phase 7 — session 7 (2026-11-25): inverse inference — CLI p-values, t-quantile, confidence intervals

Execution: the two Phase-6 follow-ups (CLI `--json` p-value fields;
inverse-ibeta CIs) from docs/reports/phase6-session6.md "Open next
session". Commits: see git log; full suite re-run ALL GREEN at HEAD.

## Delivered
- **CLI `--json` p-values**: `p_values` (two-sided t, every coefficient +
  `(intercept)`), `f_pvalue` (model F), `f_df: [p, n−p−intercept]`. df
  conventions mirror fit.asm's t/F exactly. NaN/rank-deficient → JSON
  `null`. Kernel-level check: p vs 40-digit mpmath on the library's OWN
  t-statistics — 4.1e-59 absolute (the apparent 5e-4 gap on x1 was the
  numpy reference's own t differing at t≈88; solver conditioning, not
  the kernel).
- **`ols_t_qvalue_f64_sse2(p, df)`**: t critical value by bisection on
  [0, 1e5] over the p-value kernel (monotone by construction; 120 its,
  ~60 exhaust a double). Guards: p≤0 → +inf, p≥1 → 0, df≤0/NaN → NaN.
  Differential vs scipy 1.14 `stats.t.ppf`: worst 2.3e-10 across 18
  cases (df=1, 1e6, 1e8, fractional 2.5, α=1e-8; typical 1e-13);
  round-trip |p(q(α))−α| ≤ 2.3e-10.
- **`ols_confint_f64_sse2(β, se, df, level, *lo, *hi)`**: β ± t_crit·se.
  200-random differential vs scipy: worst endpoint 2.8e-10; CI99 df=1
  ±63.656741 exact. CLI `--confint[=LEVEL]` (default 0.95): human
  `CI95% [lo, hi]` per coefficient + JSON `confint` object +
  `conf_level`; level outside (0,1) refused with rc 2. End-to-end
  fit→JSON CI vs lstsq+scipy: 1.1e-15.
- **tests/py/test_special.py**: permanent quantile (scipy-gated),
  round-trip (no scipy needed), 8 guard cases, 100-random confint.
- **ADR-0011** with the five-bug post-mortem (three real, one the
  recurring cross-call register class — twice in one loop; one
  diagnostic lesson about stale libolstest.so).

## Debugging narrative (the value is in the loop, not the answer)
The t-quantile hung. Three drafts:
1. rcx loop counter: clobbered by the p-value call (caller-saved),
   wrapped to ~2^64 iterations. → stack slot.
2. Still hung: lo/hi lived in xmm4/xmm5 across the call; ibeta keeps h
   in xmm4 through its lentz helper. Both bounds destroyed per
   iteration. → ALL loop state (lo, hi, mid, counter) in the frame;
   registers hold nothing across calls. Frame 32→56 bytes.
3. Still "hung" — but this was the stale `build/tests/libolstest.so`:
   ols_bind.py prefers it over libols.so, and I had rebuilt only
   libols.so. The loop was already fixed. Diagnosis via: Python
   replication of the exact loop (converged in ms), single-kernel
   probes (fast), bounded-iteration build (returned), then finally
   noticing the two .so timestamps 24 minutes apart. Lesson recorded:
   rebuild BOTH libs (make harness) after every asm edit.
Plus a real logic bug caught by the Python walk: bisection branch
inverted in draft 1 (p(mid) > target means mid too SMALL → lo=mid).

## Not finished
- PR push: PAT was provided late this session; staging must be
  regenerated first (see below) — left as the immediate next step
  rather than rushing a stale-sha push.
- PLAN stretch list (optional items) — not triaged this session.

## Open next session
1. Push via the resume kit: re-run the staging generator at HEAD first
   (the kit in download/pr-resume-kit/ pins Phase-3 shas; the tree has
   moved through Phases 4-7 since), then `GITHUB_TOKEN=<PAT>
   python3 gh_flow.py --allow-existing all`.
2. Stretch list triage (PLAN.md COULD items).
3. Optional: Welch-style fractional-dof CI convenience (kernel already
   accepts any df > 0).
