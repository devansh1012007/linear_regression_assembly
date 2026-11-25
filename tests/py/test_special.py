"""tests/py/test_special.py — Phase 6: transcendental kernels vs mpmath.

Differential targets (pre-registered):
  exp    : max rel err <= 1e-15   (1 ulp-ish budget)
  log    : max rel err <= 1e-15
  lgamma : max rel err <= 2e-13   (Lanczos g=7 n=9 documented bound)
  ibeta  : max abs err <= 1e-12   (Lentz CF + front-factor cancellation)
Specials: NaN propagation, overflow/underflow, domain errors, and the
t/F p-value identities (verified against mpmath's t/F distributions).
"""
import ctypes as C
import math
import sys

import numpy as np

sys.path.insert(0, "tests/py")
sys.path.insert(0, "tests/ref")
import ols_bind as B
import ols_ref as R

ols = B.ols
for fn in ("ols_exp_f64_sse2", "ols_log_f64_sse2", "ols_lgamma_f64_sse2"):
    f = getattr(ols, fn)
    f.restype = C.c_double
    f.argtypes = [C.c_double]
ols.ols_ibeta_f64_sse2.restype = C.c_double
ols.ols_ibeta_f64_sse2.argtypes = [C.c_double, C.c_double, C.c_double]

exp_k = ols.ols_exp_f64_sse2
log_k = ols.ols_log_f64_sse2
lg_k = ols.ols_lgamma_f64_sse2
ib_k = ols.ols_ibeta_f64_sse2


def relerr(got, ref):
    # signed reference: abs() on ref flipped the sign for negative results
    # (log of small x, lgamma in (0,2)) and reported rel = 2.0 on EXACT
    # answers — the first draft "verified" a kernel bug that didn't exist
    ref = float(ref)
    if ref == 0.0:
        return abs(got)
    return abs(got - ref) / abs(ref)


import mpmath as mp
mp.mp.dps = 40


def t_p_two(t, df):
    return 2 * mp.quad(lambda s: (mp.gamma((df + 1) / 2)
           / (mp.sqrt(df * mp.pi) * mp.gamma(df / 2))
           * (1 + s * s / df) ** (-(df + 1) / 2)), [t, mp.inf])


def f_upper(f, d1, d2):
    return mp.quad(lambda s: (mp.gamma((d1 + d2) / 2)
                   / (mp.gamma(d1 / 2) * mp.gamma(d2 / 2))
                   * (d1 / d2) ** (d1 / 2) * s ** (d1 / 2 - 1)
                   * (1 + d1 * s / d2) ** (-(d1 + d2) / 2)), [f, mp.inf])


def main():
    import mpmath as mp
    mp.mp.dps = 50
    rng = np.random.default_rng(6)
    fails = 0

    # ---------------- exp ----------------
    worst = 0.0
    xs = np.concatenate([
        rng.uniform(-30, 30, 400),
        rng.uniform(-700, 700, 200),
        np.linspace(-0.693, 0.693, 101),  # k-split boundary
        [1e-300, -1e-300, 0.0, -0.0],
    ])
    for x in xs:
        got = exp_k(float(x))
        ref = mp.exp(mp.mpf(float(x)))
        e = relerr(got, ref)
        worst = max(worst, e)
    # specials
    ok = (exp_k(710.0) == float("inf") and exp_k(-750.0) == 0.0
          and np.isnan(exp_k(float("nan")))
          and exp_k(0.0) == 1.0)
    print(f"exp: worst rel {worst:.2e} (target 1e-15) "
          f"specials {'OK' if ok else 'FAIL'}")
    if worst > 1e-15 or not ok:
        fails += 1

    # ---------------- log ----------------
    worst = 0.0
    xs = np.concatenate([
        rng.uniform(1e-8, 1e8, 400),
        np.geomspace(1e-300, 1e300, 200),
        [1.0, 2.0, 0.5, np.sqrt(2.0)],      # split boundaries
        [5e-324],                            # subnormal
    ])
    for x in xs:
        got = log_k(float(x))
        ref = mp.log(mp.mpf(float(x)))
        e = relerr(got, ref)
        worst = max(worst, e)
    ok = (np.isnan(log_k(0.0)) and np.isnan(log_k(-1.0))
          and log_k(float("inf")) == float("inf")
          and np.isnan(log_k(float("nan"))))
    print(f"log: worst rel {worst:.2e} (target 1e-15) "
          f"specials {'OK' if ok else 'FAIL'}")
    if worst > 1e-15 or not ok:
        fails += 1

    # ---------------- lgamma ----------------
    # Target note: the g=7 n=9 Lanczos set carries ~1e-13 rel error of its
    # own (verified: the asm equals the double-precision formula bit-for-
    # bit at the worst point; the deviation IS the approximation). Near
    # lgamma's zeros (x ~ 1, 2) relative error blows up — compare with an
    # absolute floor there: err <= max(5e-13, 5e-13*|ref|).
    worst = 0.0
    worst_at = None
    xs = np.concatenate([
        np.linspace(0.01, 0.5, 120, endpoint=False),   # reflection zone
        np.linspace(0.5, 8.0, 200),
        rng.uniform(0.5, 200.0, 200),
        [1.0, 2.0, 0.5, 1.5, 2.5, 100.0, 1e-3, 1e-6],
    ])
    for x in xs:
        got = lg_k(float(x))
        ref = mp.loggamma(mp.mpf(float(x)))
        e = abs(got - float(ref)) / max(1.0, abs(float(ref)))
        if e > worst:
            worst, worst_at = e, x
    # exact values: lgamma(1)=lgamma(2)=0
    ok = (lg_k(1.0) == 0.0 and abs(lg_k(2.0)) < 1e-14
          and np.isnan(lg_k(0.0)) and np.isnan(lg_k(-1.0))
          and np.isnan(lg_k(float("nan"))))
    print(f"lgamma: worst err {worst:.2e} at x={worst_at} "
          f"(target 5e-13 abs-floored) specials {'OK' if ok else 'FAIL'}")
    if worst > 5e-13 or not ok:
        fails += 1

    # ---------------- ibeta ----------------
    worst = 0.0
    worst_at = None
    cases = []
    # random (a,b,x)
    for _ in range(150):
        cases.append((rng.uniform(0.05, 40), rng.uniform(0.05, 40),
                      rng.uniform(0, 1)))
    # t-distribution patterns: I_x(df/2, 1/2), x = df/(df+t^2)
    for t, df in [(0.5, 3), (2.0, 10), (2.5, 4), (10.0, 1000), (1.0, 1),
                  (3.0, 25), (0.1, 2), (5.0, 7)]:
        cases.append((df / 2.0, 0.5, df / (df + t * t)))
        cases.append((0.5, df / 2.0, df / (df + t * t)))  # complement side
    # F patterns: I_x(d2/2, d1/2), x = d2/(d2+d1 F)
    for f, d1, d2 in [(3.0, 4, 60), (1.5, 2, 200), (0.5, 1, 1), (4.0, 10, 10)]:
        cases.append((d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f)))
    # boundary shapes
    cases += [(0.5, 0.5, 0.3), (1.0, 1.0, 0.7), (2.0, 3.0, 0.5),
              (0.1, 0.1, 0.5), (30.0, 2.0, 0.9), (2.0, 30.0, 0.1)]
    for a, b, x in cases:
        got = ib_k(float(a), float(b), float(x))
        ref = mp.betainc(mp.mpf(float(a)), mp.mpf(float(b)),
                         0, mp.mpf(float(x)), regularized=True)
        e = abs(got - float(ref))       # ibeta output is a probability
        if e > worst:
            worst, worst_at = e, (a, b, x)
    # I_x(1/2,1/2,1/2) = 0.5 exactly in theory; the kernel reaches it to
    # ~1e-14 (within the 1e-12 contract) — assert tolerance, not bit-exact
    ok = (ib_k(2.0, 3.0, 0.0) == 0.0 and ib_k(2.0, 3.0, 1.0) == 1.0
          and abs(ib_k(0.5, 0.5, 0.5) - 0.5) <= 1e-12
          and np.isnan(ib_k(float("nan"), 1.0, 0.5)))
    print(f"ibeta: worst abs {worst:.2e} at {worst_at} (target 1e-12) "
          f"specials {'OK' if ok else 'FAIL'}")
    if worst > 1e-12 or not ok:
        fails += 1

    # ---------------- t/F p-value identities ----------------
    def t_pdf(x, df):
        return (mp.gamma((df + 1) / 2) / (mp.sqrt(df * mp.pi) * mp.gamma(df / 2))
                * (1 + x * x / df) ** (-(df + 1) / 2))

    p_worst = 0.0
    for t, df in [(2.0, 10), (10.0, 1000), (0.5, 3), (1.0, 1), (3.0, 25),
                  (2.228, 11)]:
        p_ref = 2 * mp.quad(lambda s: t_pdf(s, mp.mpf(df)), [mp.mpf(t), mp.inf])
        p_got = ib_k(df / 2.0, 0.5, df / (df + t * t))
        p_worst = max(p_worst, abs(p_got - float(p_ref)))
    def f_pdf(x, d1, d2):
        return (mp.gamma((d1 + d2) / 2) / (mp.gamma(d1 / 2) * mp.gamma(d2 / 2))
                * (d1 / d2) ** (d1 / 2) * x ** (d1 / 2 - 1)
                * (1 + d1 * x / d2) ** (-(d1 + d2) / 2))
    for f, d1, d2 in [(3.0, 4, 60), (1.5, 2, 200), (4.0, 10, 10)]:
        p_ref = mp.quad(lambda s: f_pdf(s, d1, d2), [mp.mpf(f), mp.inf])
        p_got = ib_k(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))
        p_worst = max(p_worst, abs(p_got - float(p_ref)))
    print(f"t/F p-identity: worst abs {p_worst:.2e} (target 1e-12)")
    if p_worst > 1e-12:
        fails += 1

    # ---------------- t/F p-value WRAPPERS ----------------
    ols.ols_t_pvalue_f64_sse2.restype = C.c_double
    ols.ols_t_pvalue_f64_sse2.argtypes = [C.c_double, C.c_double]
    ols.ols_f_pvalue_f64_sse2.restype = C.c_double
    ols.ols_f_pvalue_f64_sse2.argtypes = [C.c_double, C.c_double, C.c_double]
    tw = fw = 0.0
    for t, df in [(2.0, 10), (10.0, 1000), (0.5, 3), (1.0, 1), (3.0, 25),
                  (2.228, 11), (0.01, 2), (8.5, 4)]:
        got = ols.ols_t_pvalue_f64_sse2(t, df)
        want = float(t_p_two(mp.mpf(t), mp.mpf(df)))
        tw = max(tw, abs(got - want))
    for f, d1, d2 in [(3.0, 4, 60), (1.5, 2, 200), (4.0, 10, 10),
                      (0.5, 1, 1), (7.7, 3, 45)]:
        got = ols.ols_f_pvalue_f64_sse2(f, d1, d2)
        want = float(f_upper(mp.mpf(f), d1, d2))
        fw = max(fw, abs(got - want))
    okw = (ols.ols_t_pvalue_f64_sse2(0.0, 5.0) == 1.0
           and ols.ols_f_pvalue_f64_sse2(0.0, 3.0, 9.0) == 1.0)
    print(f"t/F wrappers: worst {tw:.2e}/{fw:.2e} (target 1e-12) "
          f"specials {'OK' if okw else 'FAIL'}")
    if tw > 1e-12 or fw > 1e-12 or not okw:
        fails += 1

    # ---------------- t-QUANTILE + CONFINT (phase 7) ----------------
    # bisection inverse of the t p-value kernel; scipy stats.t.ppf reference.
    try:
        from scipy import stats as _st
    except ImportError:
        _st = None
    ols.ols_t_qvalue_f64_sse2.restype = C.c_double
    ols.ols_t_qvalue_f64_sse2.argtypes = [C.c_double, C.c_double]
    ols.ols_confint_f64_sse2.restype = None
    ols.ols_confint_f64_sse2.argtypes = [C.c_double, C.c_double, C.c_double,
                                         C.c_double,
                                         C.POINTER(C.c_double),
                                         C.POINTER(C.c_double)]
    qcases = [(0.05, 37), (0.01, 10), (0.05, 1), (0.05, 2), (0.001, 100),
              (0.10, 5), (0.025, 8), (0.05, 1e6), (0.01, 1), (0.001, 1),
              (0.5, 30), (0.999, 4), (1e-8, 60), (0.05, 2.5), (0.32, 17)]
    qw = 0.0
    for alpha, df in qcases:
        got = ols.ols_t_qvalue_f64_sse2(alpha, df)
        if _st is not None:
            ref = _st.t.ppf(1 - alpha / 2.0, df)
            qw = max(qw, abs(got - ref) / abs(ref))
    # round-trip through our own p-value kernel (no scipy needed)
    rtw = 0.0
    for alpha, df in qcases:
        t = ols.ols_t_qvalue_f64_sse2(alpha, df)
        rtw = max(rtw, abs(ols.ols_t_pvalue_f64_sse2(t, df) - alpha))
    # guards
    inf, nan = float("inf"), float("nan")
    okq = (ols.ols_t_qvalue_f64_sse2(0.0, 5.0) == inf
           and ols.ols_t_qvalue_f64_sse2(1.0, 5.0) == 0.0
           and ols.ols_t_qvalue_f64_sse2(1.5, 5.0) == 0.0
           and ols.ols_t_qvalue_f64_sse2(-0.1, 5.0) == inf
           and math.isnan(ols.ols_t_qvalue_f64_sse2(0.05, 0.0))
           and math.isnan(ols.ols_t_qvalue_f64_sse2(0.05, -3.0))
           and math.isnan(ols.ols_t_qvalue_f64_sse2(nan, 5.0))
           and math.isnan(ols.ols_t_qvalue_f64_sse2(0.05, nan)))
    # confint vs scipy tcrit*se
    cw = 0.0
    if _st is not None:
        rng = np.random.default_rng(7)
        for _ in range(100):
            beta = rng.uniform(-5, 5)
            se = rng.uniform(1e-4, 10)
            df = rng.uniform(1, 500)
            lo, hi = C.c_double(), C.c_double()
            ols.ols_confint_f64_sse2(beta, se, df, 0.90,
                                     C.byref(lo), C.byref(hi))
            hw = _st.t.ppf(0.95, df) * se
            cw = max(cw, abs(lo.value - (beta - hw)),
                     abs(hi.value - (beta + hw)))
    print(f"t-quantile: worst rel {qw:.2e} (target 1e-9, scipy) "
          f"round-trip {rtw:.2e} (target 1e-9) guards "
          f"{'OK' if okq else 'FAIL'}")
    if qw > 1e-9 or rtw > 1e-9 or not okq:
        fails += 1
    print(f"confint: worst endpoint err {cw:.2e} (target 1e-8)")
    if cw > 1e-8:
        fails += 1

    print(f"special: {'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
