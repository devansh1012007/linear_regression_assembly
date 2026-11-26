# ADR-0011: Inverse inference — t-quantile, confidence intervals, CLI p-values

Date: 2026-11-25 (Phase 7, PLAN COULD "p-values/CIs" follow-ups) · Status: accepted

## Decision
Three deliverables complete the inference story started in ADR-0010:

1. **CLI `--json` p-values** (`cli/asm_ols.c`): the fit summary now carries
   `p_values` (two-sided t, per coefficient + `(intercept)`), `f_pvalue`
   (model F upper tail) and `f_df: [p, n-p-intercept]`. df conventions
   match `src/core/fit.asm` exactly: t dof = n−p−intercept, F df = (p,
   n−p−intercept). Rank-deficient fits (se/t zeroed) and NaN p-values
   emit JSON `null` (JSON has no NaN; `%.17g` of NaN would print "nan").

2. **`ols_t_qvalue_f64_sse2(p, df)`** (`src/core/special.asm`): the t
   critical value — smallest t ≥ 0 with two-sided P(|T|≥t) ≤ p — by plain
   bisection on [0, 1e5] over `ols_t_pvalue_f64_sse2`, 120 iterations
   (~60 exhaust a double; margin for denormal tails). Bracket sized for
   df=1 (Cauchy): t_{0.001,1}=636.6 covered; alpha below P(|T|≥1e5)
   (~6e-6 at df=1, far smaller at real df) clamps to 1e5. Bisection, not
   Newton: the p-value kernel is only piecewise-smooth (ibeta swap) and
   monotone by construction, so the interval method is robust at 1e5×
   fewer lines. Guard semantics mirror the forward kernels: p ≤ 0 → +inf,
   p ≥ 1 → 0, df ≤ 0/NaN → NaN.

3. **`ols_confint_f64_sse2(beta, se, df, level, *lo, *hi)`**:
   β ± t_crit(1−level, df)·se with t_crit from the quantile kernel
   (level 0.95 → the two-sided 0.05 critical value). Six-arg SysV:
   first four in xmm0-3, pointers in rdi/rsi.

## Measured (scipy 1.14 `stats.t.ppf` differential, tests/py/test_special.py)
- quantile worst rel 2.3e-10 across 18 (alpha, df) cases incl. df=1, 1e6,
  1e8, fractional 2.5, alpha 1e-8 (bisection resolution at extreme
  curvature; typical case 1e-13).
- round-trip |p(q(α))−α| worst 1.5e-12 (test) / 2.3e-10 (α=1e-8 case).
- confint 200-random (β, se, df) endpoints vs scipy: worst 2.8e-10.
- CI99 df=1: ±63.656741 — exact to printed digits.
- CLI end-to-end (fit → JSON confint vs lstsq+scipy): 1.1e-15.
- P7.1 p-values: kernel vs 40-digit mpmath on identical t: 4.1e-59 abs.

## Bugs found (this phase's tax return)
1. **Loop counter in rcx across a call** (draft 1): `dec rcx` after
   `call ols_t_pvalue` — rcx is caller-saved and clobbered; the counter
   wrapped to ~2^64 iterations. ADR-0010 bug #4 class.
2. **lo/hi in xmm4/xmm5 across the call** (draft 2): ibeta keeps h in
   xmm4 through its .lentz helper, so both bounds were destroyed each
   iteration. Same bug class, twice, in the same loop — the fix moved ALL
   loop state (lo, hi, mid, counter) into the frame; registers hold
   nothing across the call. Frame 32 → 56 bytes.
3. **Bisection branch inverted** (draft 1): p(mid) > target means mid is
   too SMALL (need larger t), so lo=mid — I wrote hi=mid. Bounds crossed,
   no convergence.
4. **Stale `build/tests/libolstest.so`** (diagnostic, not a code bug):
   `tests/py/ols_bind.py` prefers libolstest.so over libols.so; rebuilding
   only `libols.so` left every later probe running the pre-fix binary and
   "the fix didn't work" three times. Lesson: `make harness` (or both
   libs) after every asm edit; the harness rule exists for this.
5. Test-side, twice: ctypes `argtypes` needs a flat list — a tuple of
   lists raises "no from_param method"; and calling a kernel without
   setting argtypes first passes ints.

## Bracket note
[0, 1e5] covers every realistic use; the clamp is documented in the
kernel contract. df→0 (e.g. 0.01) has true quantiles ~1e100 — callers
fitting with dof < 1 have degenerate models anyway (n ≤ p); the library's
own fit path returns UNDERDETERMINED there before inference is reached.
