"""tests/py/test_qr.py — Phase-3 QR kernel differentials (P3.5).

Kernel-level path (build -> Householder factor -> back-substitution ->
unscale) vs the mpmath reference (mp_qr_ols, same contract) and numpy
lstsq. The ols_fit wiring gets its own end-to-end checks in test_fit.py
once P3.6 lands; this file pins the KERNELS.
"""
import ctypes as C
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
import ols_bind as B
import ols_ref as R

ols = B.ols


class QrArgs(C.Structure):
    _fields_ = [("X", C.c_void_p), ("ldx8", C.c_size_t), ("y", C.c_void_p),
                ("w", C.c_void_p), ("n", C.c_size_t), ("p", C.c_size_t),
                ("c", C.c_void_p), ("Aug", C.c_void_p), ("S", C.c_void_p),
                ("Wout", C.c_void_p), ("sst", C.c_void_p),
                ("norms", C.c_void_p), ("work", C.c_void_p),
                ("intercept", C.c_size_t)]


ols.ols_qr_build_f64_sse2.restype = C.c_int32
ols.ols_qr_build_f64_sse2.argtypes = [C.c_void_p]
ols.ols_qr_f64_sse2.restype = C.c_int32
ols.ols_qr_f64_sse2.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t,
                                C.c_size_t, C.c_double, C.c_void_p]
ols.ols_qr_solve_f64_sse2.restype = None
ols.ols_qr_solve_f64_sse2.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t,
                                      C.c_void_p]


def qr_kernel_fit(X, y, w=None, intercept=True, rcond=0.0):
    """build -> factor -> solve -> unscale, mirroring the planned fit.asm
    orchestration (pivots c = 0 here; sse from the residual tail)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n, p = X.shape
    q = p + 1
    m = q + (1 if intercept else 0)
    c = np.zeros(q)
    Aug = np.zeros((n, m))
    S = np.zeros(q)
    W = np.zeros(1)
    sst = np.zeros(1)
    norms = np.zeros(m - 1)
    work = np.zeros(m)
    qa = QrArgs(X.ctypes.data, X.strides[0], y.ctypes.data,
                w.ctypes.data if w is not None else 0, n, p, c.ctypes.data,
                Aug.ctypes.data, S.ctypes.data, W.ctypes.data,
                sst.ctypes.data, norms.ctypes.data, work.ctypes.data,
                1 if intercept else 0)
    rc = ols.ols_qr_build_f64_sse2(C.byref(qa))
    if rc != 0:
        return {"rc": rc}
    rank = ols.ols_qr_f64_sse2(Aug.ctypes.data, n, m, m, rcond,
                               work.ctypes.data)
    if rank < 0:
        return {"rc": int(rank)}
    beta_s = np.zeros(m - 1)
    ols.ols_qr_solve_f64_sse2(Aug.ctypes.data, m, m, beta_s.ctypes.data)
    dc = m - 1
    sse = float((Aug[dc:, m - 1] ** 2).sum())
    ic = 1 if intercept else 0
    beta = beta_s[ic:] / norms[ic:]
    b0 = 0.0
    if intercept:
        mean = c + S / W[0]
        b0 = float(mean[p] - mean[:p] @ beta)
    return {"rc": 0, "rank": int(rank), "beta": beta, "b0": b0,
            "sse": sse, "sst": float(sst[0]), "norms": norms.copy(),
            "S": S.copy(), "W": float(W[0])}


def lre_min(got, ref):
    return min(R.lre(float(g), float(r)) for g, r in zip(got, ref))


def test_kernel_differential():
    rng = np.random.default_rng(43)
    fails = 0
    for intercept in (True, False):
        for p in (1, 2, 3, 5, 8):
            for n in (p + 2, 40, 200):
                X = rng.standard_normal((n, p)) * 10.0 ** rng.integers(-1, 3, p)
                y = 2.0 + X @ rng.standard_normal(p) + 0.05 * rng.standard_normal(n)
                out = qr_kernel_fit(X, y, intercept=intercept)
                ref = R.mp_qr_ols(X, y, intercept=intercept)
                rb = np.array([float(v) for v in ref["beta"]])
                lre = lre_min(out["beta"], rb)
                if out["rank"] != ref["rank"] or lre < 10.0:
                    print(f"FAIL qr beta p={p} n={n} int={intercept}: "
                          f"rank {out['rank']}/{ref['rank']} lre {lre:.1f}")
                    fails += 1
                A = np.hstack([np.ones((n, 1)), X]) if intercept else X
                sse_np = float(((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) ** 2).sum())
                if abs(out["sse"] - sse_np) > 1e-9 * max(1.0, sse_np):
                    print(f"FAIL qr sse p={p} n={n} int={intercept}: "
                          f"{out['sse']:.6g} vs {sse_np:.6g}")
                    fails += 1
                if intercept and abs(out["b0"] - float(ref["b0"])) > \
                        1e-9 * max(1.0, abs(float(ref["b0"]))):
                    print(f"FAIL qr b0 p={p} n={n}: {out['b0']:.12g} "
                          f"vs {float(ref['b0']):.12g}")
                    fails += 1
    # weighted
    n, p = 60, 3
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    w = rng.uniform(0.2, 3.0, n)
    out = qr_kernel_fit(X, y, w=w, intercept=True)
    ref = R.mp_qr_ols(X, y, w=w, intercept=True)
    lre = lre_min(out["beta"], np.array([float(v) for v in ref["beta"]]))
    if lre < 10.0:
        print(f"FAIL qr weighted: lre {lre:.1f}")
        fails += 1
    # sst byproduct: intercept -> centered, no-intercept -> raw
    n, p = 50, 2
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n) * 3 + 10.0
    out_i = qr_kernel_fit(X, y, intercept=True)
    out_n = qr_kernel_fit(X, y, intercept=False)
    sst_c = float(((y - y.mean()) ** 2).sum())
    sst_r = float((y ** 2).sum())
    if abs(out_i["sst"] - sst_c) > 1e-9 * sst_c:
        print(f"FAIL qr sst centered: {out_i['sst']:.6g} vs {sst_c:.6g}")
        fails += 1
    if abs(out_n["sst"] - sst_r) > 1e-9 * sst_r:
        print(f"FAIL qr sst raw: {out_n['sst']:.6g} vs {sst_r:.6g}")
        fails += 1
    print(f"qr kernel differential: {'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return fails


def test_rank_and_rcond():
    fails = 0
    rng = np.random.default_rng(44)
    # exact rank deficiency: duplicate column -> rank < dc. The kernel's
    # rcond is an ABSOLUTE threshold on the equilibrated diagonal; pass the
    # fit.asm default (8*dc*u) — with rcond=0 even 1e-16 noise is "rank".
    n = 40
    X = rng.standard_normal((n, 3))
    X[:, 2] = X[:, 0]
    y = rng.standard_normal(n)
    dc = 4
    out = qr_kernel_fit(X, y, intercept=True, rcond=8 * dc * 2.0 ** -52)
    if out["rc"] != 0 or out["rank"] != 3:   # dc = 4, rank must be 3
        print(f"FAIL qr rank-deficient: rc={out['rc']} rank={out.get('rank')}")
        fails += 1
    # rcond respected: kappa=1e4 design, rcond = 1e-3 -> refused at some j
    Xk, yk, _, _ = R.make_kappa_problem(60, 4, 1e4, seed=9)
    out = qr_kernel_fit(Xk, yk, intercept=False, rcond=1e-3)
    if out["rc"] != 0 or out["rank"] >= 4:
        print(f"FAIL qr rcond refuse: rc={out['rc']} rank={out.get('rank')}")
        fails += 1
    # full rank with default rcond on the same design
    out = qr_kernel_fit(Xk, yk, intercept=False)
    if out["rc"] != 0 or out["rank"] != 4:
        print(f"FAIL qr default rcond full rank: rc={out['rc']} rank={out.get('rank')}")
        fails += 1
    print(f"qr rank/rcond: {'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return fails


def test_filip_kernel():
    """Filip through the QR kernels — measured, NO gate (G3). Same degree-10
    polynomial design as the NIST phase-2 test (parse_nist returns raw x)."""
    root = Path(__file__).resolve().parent.parent.parent
    d = R.parse_nist(root / "data" / "nist" / "Filip.dat")
    Xd = np.column_stack([d.X[:, 0] ** k for k in range(1, 11)])
    out = qr_kernel_fit(Xd, d.y, intercept=True)
    if out["rc"] != 0:
        print(f"qr Filip: refused rc={out['rc']} rank={out.get('rank')} "
              f"(reported, no target)")
        return 0
    rb = ref_beta_filip(Xd, d.y)
    lre = lre_min(np.r_[out["b0"], out["beta"]], rb)
    # certified-value LREs too (numpy lstsq may truncate small singulars)
    cert_lre = lre_min(np.r_[out["b0"], out["beta"]],
                       np.array(d.certified_b[:12]))
    print(f"qr Filip: solved rank={out['rank']} "
          f"min LRE vs lstsq {lre:.2f}, vs certified {cert_lre:.2f} "
          f"(reported, no target)")
    return 0


def ref_beta_filip(Xd, y):
    # numpy reference for the reported number only (no gate)
    A = np.hstack([np.ones((len(y), 1)), Xd])
    return np.linalg.lstsq(A, y, rcond=None)[0]



def test_pivot_kernel():
    """P4.5 (Phase 4b): column-pivoted QR KERNEL (fit wiring is an open
    item — method 4 returns OLS_E_ARG; this pins the kernel itself):
    mixed-scale designs vs mpmath, rank-deficiency detection, column-0 pin
    semantics (rcond < 0), full-column swaps (the first draft swapped only
    rows j..n-1 and the solve diverged)."""
    ols.ols_qr_pivot_f64_sse2.restype = C.c_int32
    ols.ols_qr_pivot_f64_sse2.argtypes = [
        C.c_void_p, C.c_size_t, C.c_size_t, C.c_size_t, C.c_double,
        C.c_void_p, C.c_void_p]
    rng = np.random.default_rng(95)
    fails = 0
    for trial in range(4):
        n, p = 40, 4
        X = rng.standard_normal((n, p))
        X[:, 0] *= 1e-6
        X[:, 3] *= 100.0
        y = X @ rng.standard_normal(p) + 0.1 * rng.standard_normal(n)
        ref = R.mp_qr_ols(X, y, intercept=True)
        rb = np.array([float(v) for v in ref["beta"]])
        q = p + 1
        m = q + 1
        c = np.zeros(q); Aug = np.zeros((n, m)); S = np.zeros(q)
        W = np.zeros(1); sst = np.zeros(1); norms = np.zeros(m - 1)
        work = np.zeros(m); perm = np.arange(m - 1, dtype=np.int64)
        qa = QrArgs(X.ctypes.data, X.strides[0], y.ctypes.data, 0, n, p,
                    c.ctypes.data, Aug.ctypes.data, S.ctypes.data,
                    W.ctypes.data, sst.ctypes.data, norms.ctypes.data,
                    work.ctypes.data, 1)
        assert ols.ols_qr_build_f64_sse2(C.byref(qa)) == 0
        norms_orig = norms.copy()
        # pin + default-style threshold: -8*dc*u (a literal -1.0 threshold
        # equals the equilibrated column norms and refuses everything)
        rank = ols.ols_qr_pivot_f64_sse2(Aug.ctypes.data, n, m, m,
                                         -8 * (m - 1) * 2.0 ** -52,
                                         work.ctypes.data, perm.ctypes.data)
        if rank != m - 1:
            print(f"FAIL pivot rank t{trial}: {rank}")
            fails += 1
            continue
        beta_s = np.zeros(m - 1)
        ols.ols_qr_solve_f64_sse2(Aug.ctypes.data, m, m, beta_s.ctypes.data)
        beta_orig = np.zeros(p)
        for j in range(1, m - 1):
            beta_orig[perm[j] - 1] = beta_s[j] / norms_orig[perm[j]]
        lre = min(R.lre(float(g), float(r)) for g, r in zip(beta_orig, rb))
        # pin: the ones column (position 0) never moves
        pinned = perm[0] == 0
        if lre < 9.0 or not pinned:
            print(f"FAIL pivot t{trial}: lre {lre:.1f} pinned {pinned}")
            fails += 1
    # rank deficiency: duplicate column detected at the default threshold
    X = rng.standard_normal((40, 3)); X[:, 2] = X[:, 0]
    y = rng.standard_normal(40)
    q, m = 4, 5
    c = np.zeros(q); Aug = np.zeros((40, m)); S = np.zeros(q)
    W = np.zeros(1); sst = np.zeros(1); norms = np.zeros(m - 1)
    work = np.zeros(m); perm = np.arange(m - 1, dtype=np.int64)
    qa = QrArgs(X.ctypes.data, X.strides[0], y.ctypes.data, 0, 40, 3,
                c.ctypes.data, Aug.ctypes.data, S.ctypes.data,
                W.ctypes.data, sst.ctypes.data, norms.ctypes.data,
                work.ctypes.data, 1)
    assert ols.ols_qr_build_f64_sse2(C.byref(qa)) == 0
    rank = ols.ols_qr_pivot_f64_sse2(Aug.ctypes.data, 40, m, m,
                                     -8 * (m - 1) * 2.0 ** -52,
                                     work.ctypes.data, perm.ctypes.data)
    if rank != 3:
        print(f"FAIL pivot rank-deficient: {rank} (want 3)")
        fails += 1
    print(f"pivot kernel: {'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return fails



def test_pivot_fit():
    """P5.1: OLS_QR_PIVOT through ols_fit (method 4) — the sentinel
    xorpd-with-value bug (xorpd(x, x) = +0.0, not -x) is fixed; the ones
    column pins, beta unpermutes, rank-deficient designs refuse with the
    detected rank, se/t zero per contract."""
    import test_fit as TF
    rng = np.random.default_rng(96)
    fails = 0
    for trial in range(3):
        n, p = 60, 4
        X = rng.standard_normal((n, p)); X[:, 0] *= 1e-5; X[:, 3] *= 50.0
        y = 1.5 + X @ rng.standard_normal(p) + 0.1 * rng.standard_normal(n)
        ref = R.mp_qr_ols(X, y, intercept=True)
        rb = np.array([float(v) for v in ref["beta"]])
        out = TF.fit(X, y, intercept=True, method=4)
        if out["rc"] != 0:
            print(f"FAIL pivot-fit t{trial}: rc={out['rc']}")
            fails += 1
            continue
        lre = min(R.lre(float(g), float(r)) for g, r in zip(out["beta"], rb))
        if lre < 10 or out["method_used"] != 4 or np.any(out["se"] != 0):
            print(f"FAIL pivot-fit t{trial}: lre {lre:.1f}")
            fails += 1
    Xd = rng.standard_normal((50, 3)); Xd[:, 2] = Xd[:, 0]
    out = TF.fit(Xd, rng.standard_normal(50), intercept=True, method=4)
    if out["rc"] != -7 or out["rank"] != 3:
        print(f"FAIL pivot-fit rank-deficient: rc={out['rc']} rank={out['rank']}")
        fails += 1
    # regressions: plain QR + chol still green
    X3 = rng.standard_normal((60, 3)); y3 = 1.0 + X3 @ rng.standard_normal(3)
    for m in (1, 2, 3):
        out = TF.fit(X3, y3, intercept=True, method=m)
        if out["rc"] != 0:
            print(f"FAIL regression method {m}")
            fails += 1
    print(f"pivot fit: {'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return fails


if __name__ == "__main__":
    f = (test_kernel_differential() + test_rank_and_rcond()
         + test_filip_kernel() + test_pivot_kernel() + test_pivot_fit())
    sys.exit(1 if f else 0)
