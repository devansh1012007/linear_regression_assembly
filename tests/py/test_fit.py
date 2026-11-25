"""tests/py/test_fit.py — ols_fit Tier-1 end-to-end differential vs NumPy."""
import ctypes as C
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
import ols_bind as B
import ols_ref as R

ols = B.ols

# problem/result structs (mirror include/ols.h; struct_check.py guards drift)
class Problem(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("version", C.c_uint32),
                ("X", C.c_void_p), ("y", C.c_void_p), ("w", C.c_void_p),
                ("n", C.c_size_t), ("p", C.c_size_t), ("ldx", C.c_size_t),
                ("method", C.c_uint32), ("flags", C.c_uint32),
                ("lambda_", C.c_double), ("rcond", C.c_double),
                ("threads", C.c_uint32), ("_pad", C.c_uint32)]

class Result(C.Structure):
    _fields_ = [("beta", C.c_void_p), ("intercept", C.c_double),
                ("sse", C.c_double), ("sst", C.c_double), ("r2", C.c_double),
                ("r2_adj", C.c_double), ("sigma2", C.c_double),
                ("cond_est", C.c_double), ("se", C.c_void_p),
                ("rank", C.c_int32), ("method_used", C.c_int32),
                ("isa_used", C.c_int32), ("_pad", C.c_int32),
                ("t", C.c_void_p)]      # v2 append-only: size 96

ols.ols_fit.restype = C.c_int32
ols.ols_fit.argtypes = [C.POINTER(Problem), C.POINTER(Result), C.c_void_p, C.c_size_t]
ols.ols_workspace_size.restype = C.c_size_t
ols.ols_workspace_size.argtypes = [C.c_size_t, C.c_size_t, C.c_uint32, C.c_uint32]

OLS_FIT_INTERCEPT = 1
OLS_FIT_SKIP_STATS = 8
OLS_AUTO = 0
OLS_GRAM_CHOL = 1


def fit(X, y, w=None, intercept=True, method=OLS_AUTO, stats=True):
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n, p = X.shape
    beta = np.zeros(p)
    se = np.zeros(p + 1)
    t = np.zeros(p + 1)
    prob = Problem(C.sizeof(Problem), 1, X.ctypes.data, y.ctypes.data,
                   w.ctypes.data if w is not None else 0, n, p, X.shape[1],
                   method, (OLS_FIT_INTERCEPT if intercept else 0) |
                   (0 if stats else OLS_FIT_SKIP_STATS),
                   0.0, 0.0, 1, 0)
    res = Result(beta.ctypes.data, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 se.ctypes.data, -1, 0, 0, 0, t.ctypes.data)
    ws_bytes = ols.ols_workspace_size(n, p, method, prob.flags)
    ws = np.zeros(max(ws_bytes // 8, 1), dtype=np.float64)
    rc = ols.ols_fit(C.byref(prob), C.byref(res), ws.ctypes.data, ws_bytes)
    out = {
        "rc": rc, "beta": beta.copy(), "b0": res.intercept,
        "sse": res.sse, "sst": res.sst, "r2": res.r2, "r2_adj": res.r2_adj,
        "sigma2": res.sigma2, "cond_est": res.cond_est,
        "se": se.copy(), "t": t.copy(), "rank": res.rank,
        "method_used": res.method_used, "isa_used": res.isa_used,
    }
    return out


def numpy_ref(X, y, w=None, intercept=True):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if w is not None:
        sw = np.sqrt(np.asarray(w, dtype=np.float64))
        # scale the WHOLE design (including the intercept column) by sqrt(w)
        if intercept:
            A = np.hstack([np.ones((len(y), 1)), X]) * sw[:, None]
        else:
            A = X * sw[:, None]
        yw = y * sw
    else:
        A = np.hstack([np.ones((len(y), 1)), X]) if intercept else X
        yw = y
    b, _, rank, _ = np.linalg.lstsq(A, yw, rcond=None)
    r = yw - A @ b
    sse = r @ r
    if intercept:
        if w is None:
            sst = np.sum((y - y.mean()) ** 2)
        else:
            ybar_w = np.sum(w * y) / np.sum(w)
            sst = np.sum(w * (y - ybar_w) ** 2)
    else:
        sst = yw @ yw
    return {"b0": b[0] if intercept else 0.0, "beta": b[1:] if intercept else b,
            "sse": sse, "sst": sst, "r2": 1 - sse / sst if sst else 1.0}


def main():
    rng = np.random.default_rng(99)
    fails = 0
    for p in (1, 2, 3, 5, 8, 13, 16):
        for n in (p + 3, 2 * p + 5, 100):
            X = rng.standard_normal((n, p)) * (10.0 ** rng.integers(-1, 3))
            beta_true = rng.standard_normal(p)
            y = 1.5 + X @ beta_true + 0.05 * rng.standard_normal(n)
            for intercept in (True, False):
                out = fit(X, y, intercept=intercept)
                ref = numpy_ref(X, y, intercept=intercept)
                if out["rc"] != 0:
                    print(f"FAIL rc={out['rc']} p={p} n={n} int={intercept}")
                    fails += 1
                    continue
                bb = np.r_[out["b0"], out["beta"]] if intercept else out["beta"]
                rb = np.r_[ref["b0"], ref["beta"]] if intercept else ref["beta"]
                scale = max(1.0, np.abs(rb).max())
                if not np.allclose(bb, rb, rtol=1e-6, atol=1e-6 * scale):
                    print(f"FAIL beta p={p} n={n} int={intercept}: "
                          f"maxdiff {np.abs(bb - rb).max():.3e}")
                    fails += 1
                if abs(out["sse"] - ref["sse"]) > 1e-6 * max(1, ref["sse"]):
                    print(f"FAIL sse p={p} n={n} int={intercept}: "
                          f"{out['sse']:.6g} vs {ref['sse']:.6g}")
                    fails += 1
                if abs(out["r2"] - ref["r2"]) > 1e-6:
                    print(f"FAIL r2 p={p} n={n} int={intercept}: "
                          f"{out['r2']:.8f} vs {ref['r2']:.8f}")
                    fails += 1
    # weighted
    for p in (2, 5):
        n = 60
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        w = rng.uniform(0.2, 3.0, n)
        out = fit(X, y, w=w)
        ref = numpy_ref(X, y, w=w)
        bb = np.r_[out["b0"], out["beta"]]
        rb = np.r_[ref["b0"], ref["beta"]]
        if not np.allclose(bb, rb, rtol=1e-6, atol=1e-6):
            print(f"FAIL weighted p={p}: maxdiff {np.abs(bb - rb).max():.3e}")
            fails += 1
    # adversarial offset: X = 1e9 + noise
    n, p = 200, 3
    X = 1e9 + rng.standard_normal((n, p))
    y = 2.0 + X @ np.array([1.5, -2.0, 0.5]) + 0.01 * rng.standard_normal(n)
    out = fit(X, y)
    mp = R.mp_solve(X, y, intercept=True)
    rb = np.array([float(v) for v in mp["b"]])
    bb = np.r_[out["b0"], out["beta"]]
    if not np.allclose(bb, rb, rtol=1e-7, atol=1e-7 * np.abs(rb).max()):
        print(f"FAIL offset1e9: maxdiff {np.abs(bb - rb).max():.3e} "
              f"(asm {bb[:2]} vs mp {rb[:2]})")
        fails += 1
    else:
        print(f"adversarial offset 1e9: agrees with mpmath "
              f"(maxdiff {np.abs(bb - rb).max():.2e})")
    # error paths
    out = fit(np.zeros((5, 2)), np.ones(5))
    if out["rc"] != -7:
        print(f"FAIL singular: rc={out['rc']} (want -7)")
        fails += 1
    Xs = np.random.default_rng(1).standard_normal((4, 8))
    out = fit(Xs, np.zeros(4))
    if out["rc"] != -5:
        print(f"FAIL underdetermined: rc={out['rc']} (want -5)")
        fails += 1
    # ---- full statistics differential vs closed form (se/t/r2_adj/F) ----
    try:
        from scipy import stats as sps
        have_scipy = True
    except ImportError:
        have_scipy = False
    rng2 = np.random.default_rng(311)
    for p in (1, 2, 5, 8):
        n = 120 + 7 * p
        X = rng2.standard_normal((n, p))
        y = 0.75 + X @ rng2.standard_normal(p) * 2 + 0.5 * rng2.standard_normal(n)
        out = fit(X, y)
        A = np.hstack([np.ones((n, 1)), X])
        b = np.r_[out["b0"], out["beta"]]
        r = y - A @ b
        dof = n - p - 1
        sigma2 = r @ r / dof
        Ainv = np.linalg.inv(A.T @ A)
        se_true = np.sqrt(sigma2 * np.diag(Ainv))
        t_true = b / se_true
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - (r @ r) / sst
        r2a = 1 - (1 - r2) * (n - 1) / dof
        F_true = (r2 / p) / ((1 - r2) / dof)
        # asm layout: se/t = [slopes..., intercept]; reorder to [b0, slopes]
        se_asm = np.r_[out["se"][p], out["se"][:p]]
        t_asm = np.r_[out["t"][p], out["t"][:p]]
        if not np.allclose(se_asm, se_true, rtol=1e-9, atol=1e-12):
            print(f"FAIL se p={p}: maxdiff {np.abs(se_asm - se_true).max():.3e}")
            fails += 1
        if not np.allclose(t_asm, t_true, rtol=1e-9, atol=1e-12):
            print(f"FAIL t p={p}: maxdiff {np.abs(t_asm - t_true).max():.3e}")
            fails += 1
        if abs(out["r2_adj"] - r2a) > 1e-10:
            print(f"FAIL r2_adj p={p}: {out['r2_adj']:.12f} vs {r2a:.12f}")
            fails += 1
        if have_scipy:
            pvals = 2 * sps.t.sf(np.abs(t_true), dof)
            p_asm = 2 * sps.t.sf(np.abs(t_asm), dof)
            if not np.allclose(p_asm, pvals, rtol=1e-9):
                print(f"FAIL p-value proxy p={p}")
                fails += 1
    print(f"stats differential (se/t/r2_adj{'+p' if have_scipy else ''}): "
          f"{'OK' if fails == 0 else 'see above'}")
    print(f"ols_fit Tier-1: {'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
