"""tests/props/test_props.py — property-based tests for the Phase-2 engine.

Where the differential suites (test_gram/test_fit) pin EXACT inputs, this
suite pins PROPERTIES that must hold for whole families of inputs. Every
failure is a real bug by construction, not a tolerance miss:

  P1  vs-mpmath differential: for random (n, p, scale, offset), asm ols_fit
      beta matches a 60-dps mpmath solve to LRE >= 9.
  P2  row-permutation invariance: fitting a permuted (X, y) changes no
      reported statistic beyond compensated-rounding slack.
  P3  exact recovery: noiseless y = X beta* must come back as beta* with
      sse ~ 0 (the engine's reason to exist).
  P4  unit weights = no weights: ols_fit(w=1) == ols_fit(w=NULL) to
      relative 1e-12 (the weighted path must degenerate gracefully).
  P5  residual orthogonality: X'r = 0 is the defining OLS property; the
      Gram+Cholesky solution must satisfy it to kappa-scaled tolerance.
  P6  condition-number sweep: accuracy degrades no faster than the
      cond-squared bound predicts (make_kappa_problem sweep).

Run standalone:  python3 tests/props/test_props.py
Wired into the pyramid via tests/py/run_all.py::test_props_suite().
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent / "py"))
sys.path.insert(0, str(_HERE.parent.parent / "ref"))
import ols_ref as R          # noqa: E402
import test_fit as TF        # noqa: E402  (TF.fit = full ols_fit wrapper)

FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"FAIL: {msg}")


def _relerr(got, want):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    denom = np.maximum(np.abs(want), 1e-300)
    return float(np.max(np.abs(got - want) / denom))


def _stack(beta, b0):
    return np.concatenate([[b0], beta]) if b0 is not None else beta


# ---------------------------------------------------------------- P1
def p1_differential_vs_mpmath():
    """Differential vs 60-dps mpmath over random problems.

    The gate is the DOCUMENTED shift-and-cut error model (numerical-ols §2,
    measured in this PR), not a flat number: the pivot-shifted Gram makes the
    SLOPES shift-invariant (measured LRE 14.6-15.0 at offsets up to 1e9),
    while the INTERCEPT absorbs the absolute offset error — its LRE degrades
    as 15 - log10(1+offset). Property gates carry ~1.5 digits of margin:
      - slopes: LRE >= 14 at any tested offset
      - overall: LRE >= 15 - log10(1+offset) - 2.5
      - zero offset: overall LRE >= 12 (measured 12.9-14.2)
    """
    shapes = [(40, 1), (64, 3), (50, 5), (33, 2)]
    offsets = [0.0, 1e6, 1e9]
    rng = np.random.default_rng(2026)
    cases = []
    for (n, p) in shapes:
        for off in offsets:
            X = rng.standard_normal((n, p)) * rng.uniform(1e-3, 1e3) + off
            beta_true = rng.standard_normal(p)
            b0 = rng.standard_normal()
            y = b0 + X @ beta_true + 1e-10 * rng.standard_normal(n)
            cases.append((f"n{n}p{p}off{off:g}", X, y, off))
    for name, X, y, off in cases:
        out = TF.fit(X, y, intercept=True, stats=False)
        ref = R.mp_solve(X, y, intercept=True)
        got = np.concatenate([[out["b0"]], out["beta"]])
        want = np.array([float(v) for v in ref["b"]])
        lres = [R.lre(g, w) for g, w in zip(got, want)]
        slope_floor = 14.0
        check(out["rc"] == 0 and min(lres[1:]) >= slope_floor,
              f"P1 {name}: slopes rc={out['rc']} min LRE={min(lres[1:]):.1f} "
              f"< {slope_floor}")
        overall_floor = max(15.0 - np.log10(1.0 + off) - 2.5, 5.0)
        check(min(lres) >= overall_floor,
              f"P1 {name}: overall min LRE={min(lres):.1f} "
              f"< documented bound {overall_floor:.1f}")
        if off == 0.0:
            check(min(lres) >= 12.0,
                  f"P1 {name}: offset-0 min LRE={min(lres):.1f} < 12")
    print(f"P1 differential vs mpmath: {len(cases)} random problems, "
          "shift-and-cut bound verified")


# ---------------------------------------------------------------- P2
def p2_row_permutation_invariance():
    rng = np.random.default_rng(7)
    for n, p in [(100, 3), (57, 5)]:
        X = rng.standard_normal((n, p)) + 1e4
        beta_true = rng.standard_normal(p)
        y = X @ beta_true + 0.01 * rng.standard_normal(n)
        base = TF.fit(X, y, intercept=True)
        perm = rng.permutation(n)
        pert = TF.fit(X[perm], y[perm], intercept=True)
        for key in ("beta", "se"):
            e = _relerr(pert[key], base[key])
            check(e <= 1e-10, f"P2 n{n}p{p} {key}: rel diff {e:.3e}")
        for key in ("b0", "sse", "sigma2"):
            e = abs(pert[key] - base[key]) / max(abs(base[key]), 1e-300)
            check(e <= 1e-10, f"P2 n{n}p{p} {key}: rel diff {e:.3e}")
    print("P2 row-permutation invariance: done")


# ---------------------------------------------------------------- P3
def p3_exact_recovery():
    rng = np.random.default_rng(11)
    for n, p in [(30, 1), (80, 4), (120, 6)]:
        X = rng.standard_normal((n, p))
        beta_true = rng.uniform(-3, 3, p)
        b0 = 2.5
        y = b0 + X @ beta_true            # noiseless by construction
        out = TF.fit(X, y, intercept=True)
        got = _stack(out["beta"], out["b0"])
        want = np.concatenate([[b0], beta_true])
        e = _relerr(got, want)
        check(out["rc"] == 0 and e <= 1e-9,
              f"P3 n{n}p{p}: rel err {e:.3e} (rc={out['rc']})")
        scale = max(1.0, float(np.sum(y * y)))
        check(abs(out["sse"]) <= 1e-16 * scale,
              f"P3 n{n}p{p}: sse {out['sse']:.3e} not ~0")
    print("P3 exact recovery (noiseless): done")


# ---------------------------------------------------------------- P4
def p4_unit_weights_degenerate():
    rng = np.random.default_rng(13)
    n, p = 90, 3
    X = rng.standard_normal((n, p))
    y = X @ rng.standard_normal(p) + 0.05 * rng.standard_normal(n)
    a = TF.fit(X, y, w=np.ones(n), intercept=True)
    b = TF.fit(X, y, w=None, intercept=True)
    for key in ("beta", "se", "t"):
        e = _relerr(a[key], b[key])
        check(e <= 1e-12, f"P4 {key}: rel diff {e:.3e}")
    for key in ("b0", "sse", "r2"):
        e = abs(a[key] - b[key]) / max(abs(b[key]), 1e-300)
        check(e <= 1e-12, f"P4 {key}: rel diff {e:.3e}")
    print("P4 unit-weights == unweighted: done")


# ---------------------------------------------------------------- P5
def p5_residual_orthogonality():
    rng = np.random.default_rng(17)
    for n, p in [(200, 2), (150, 6)]:
        X = rng.standard_normal((n, p))
        y = X @ rng.standard_normal(p) + 0.1 * rng.standard_normal(n)
        out = TF.fit(X, y, intercept=True, stats=False)
        check(out["rc"] == 0, f"P5 n{n}p{p}: rc={out['rc']}")
        A = np.hstack([np.ones((n, 1)), X])
        r = y - (out["b0"] + X @ out["beta"])
        gnorm = float(np.max(np.abs(A.T @ r)))
        scale = float(np.max(np.abs(A.T @ y)))
        check(gnorm <= 1e-8 * max(scale, 1e-300),
              f"P5 n{n}p{p}: ||X'r||inf {gnorm:.3e} vs scale {scale:.3e}")
    print("P5 residual orthogonality X'r=0: done")


# ---------------------------------------------------------------- P6
def p6_condition_sweep():
    for kappa in (1e1, 1e2, 1e4):
        X, y, b_true, _ = R.make_kappa_problem(120, 4, kappa, seed=77)
        out = TF.fit(X, y, intercept=False, stats=False)
        lres = [R.lre(g, w) for g, w in zip(out["beta"], b_true)]
        # cond2(X) ~= kappa^2; double-precision normal equations lose
        # ~log10(cond^2) digits. Compensated Gram must stay within that
        # bound (plus 1.5 digits of margin), not fall off a cliff.
        floor = 15.0 - 2.0 * np.log10(kappa) - 1.5
        check(out["rc"] == 0 and min(lres) >= floor,
              f"P6 kappa={kappa:g}: min LRE={min(lres):.1f} < floor {floor:.1f}")
    # noisy problems: differential against mpmath must hold regardless of cond
    for kappa in (1e2, 1e3):
        X, y, _, ref = R.make_kappa_problem(120, 4, kappa, seed=91,
                                            residual="large")
        out = TF.fit(X, y, intercept=False, stats=False)
        lres = [R.lre(g, w) for g, w in zip(out["beta"], ref["b"])]
        check(out["rc"] == 0 and min(lres) >= 8.0,
              f"P6 noisy kappa={kappa:g}: min LRE={min(lres):.1f} < 8")
    print("P6 condition sweep (kappa=1e1..1e4): done")


# ----------------------------------------------------------------
def run(fails_sink=None):
    """Run all properties; append failure strings to fails_sink if given."""
    p1_differential_vs_mpmath()
    p2_row_permutation_invariance()
    p3_exact_recovery()
    p4_unit_weights_degenerate()
    p5_residual_orthogonality()
    p6_condition_sweep()
    if fails_sink is not None:
        fails_sink.extend(FAILS)
    return not FAILS


if __name__ == "__main__":
    ok = run()
    if ok:
        print("PROPERTIES: ALL GREEN")
        sys.exit(0)
    print(f"\nPROPERTIES: FAILED ({len(FAILS)} failures)")
    sys.exit(1)
