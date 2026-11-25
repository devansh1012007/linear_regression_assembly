"""tests/py/test_gram.py — Phase-2 Gram/chol/fit differential tests vs NumPy."""
import ctypes as C
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
import ols_bind as B

ols = B.ols

# argtypes for the new kernels
ols.ols_gram_f64_avx2.restype = C.c_int32
ols.ols_gram_f64_avx2.argtypes = [C.c_void_p]
ols.ols_gram_f64_sse2.restype = C.c_int32
ols.ols_gram_f64_sse2.argtypes = [C.c_void_p]
ols.ols_residual_f64_sse2.restype = C.c_int32
ols.ols_residual_f64_sse2.argtypes = [C.c_void_p]
ols.ols_pivot_means_f64_sse2.restype = C.c_int32
ols.ols_pivot_means_f64_sse2.argtypes = [np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                                         C.c_size_t, np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                                         C.c_void_p, C.c_size_t, C.c_size_t,
                                         np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")]
ols.ols_chol_f64_sse2.restype = C.c_int32
ols.ols_chol_f64_sse2.argtypes = [np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                                  C.c_size_t, C.c_double, C.c_size_t]
ols.ols_chol_solve_f64_sse2.restype = C.c_int32
ols.ols_chol_solve_f64_sse2.argtypes = [np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                                        C.c_size_t, np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                                        C.c_size_t]
ols.ols_chol_inv_f64_sse2.restype = C.c_int32
ols.ols_chol_inv_f64_sse2.argtypes = [np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                                      C.c_size_t, np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                                      C.c_size_t]


ols.ols_gram_scratch_bytes.restype = C.c_size_t
ols.ols_gram_scratch_bytes.argtypes = [C.c_size_t]


class GramArgs(C.Structure):
    _fields_ = [("X", C.c_void_p), ("ldx8", C.c_size_t), ("y", C.c_void_p),
                ("w", C.c_void_p), ("n", C.c_size_t), ("p", C.c_size_t),
                ("c", C.c_void_p), ("G", C.c_void_p), ("s", C.c_void_p),
                ("Wout", C.c_void_p), ("scratch", C.c_void_p),
                ("scratch_bytes", C.c_size_t)]


class ResidArgs(C.Structure):
    _fields_ = [("X", C.c_void_p), ("ldx8", C.c_size_t), ("y", C.c_void_p),
                ("w", C.c_void_p), ("n", C.c_size_t), ("p", C.c_size_t),
                ("b0", C.c_double), ("beta", C.c_void_p),
                ("sse_out", C.c_void_p), ("xtr_out", C.c_void_p)]


def ptr(a):
    return a.ctypes.data if a is not None else 0


def run_gram(X, y, w, c, isa="avx2"):
    """X: (n, p) C-contiguous; y: (n,); w: (n,) or None; c: (q,) pivots."""
    n, p = X.shape
    q = p + 1
    G = np.zeros(q * q)
    s = np.zeros(q)
    Wout = np.zeros(1)
    sb = ols.ols_gram_scratch_bytes(q)
    buf = np.zeros((sb + 64) // 8, dtype=np.float64)
    base = buf.ctypes.data
    scratch = (base + 63) & ~(63)
    ga = GramArgs(ptr(X), X.strides[0], ptr(y), ptr(w), n, p, ptr(c), ptr(G),
                  ptr(s), ptr(Wout), C.c_void_p(scratch), sb)
    fn = getattr(ols, f"ols_gram_f64_{isa}")
    rc = fn(C.byref(ga))
    return rc, G.reshape(q, q), s, Wout[0]


def test_gram():
    rng = np.random.default_rng(2027)
    fails = 0
    for p in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33):
        for n in (1, 2, 3, 4, 5, 7, 9, 33, 100, 257):
            if n < 2:
                continue
            X = rng.standard_normal((n, p)) * (10.0 ** rng.integers(-2, 3))
            y = rng.standard_normal(n)
            q = p + 1
            # pivots: means of a few rows (any finite c works)
            nb = min(n, 8)
            Z = np.hstack([X[:nb], y[:nb, None]])
            c = Z.mean(axis=0)
            for isa in ("sse2", "avx2"):
                for wcase in (None, rng.uniform(0.5, 2.0, n)):
                    rc, G, s, W = run_gram(X, y, wcase, c, isa)
                    if rc != 0:
                        print(f"FAIL gram rc={rc} p={p} n={n} {isa} w={wcase is not None}")
                        fails += 1
                        continue
                    ww = np.ones(n) if wcase is None else wcase
                    Zs = np.hstack([X, y[:, None]]) - c
                    Gr = (Zs * ww[:, None]).T @ Zs
                    sr = (Zs * ww[:, None]).sum(axis=0)
                    Wr = ww.sum()
                    scale = max(1.0, np.abs(Gr).max())
                    # contract: kernel fills the UPPER TRIANGLE only
                    if not np.allclose(np.triu(G), np.triu(Gr), rtol=1e-10,
                                       atol=1e-10 * scale):
                        print(f"FAIL gram G p={p} n={n} {isa} w={wcase is not None}: "
                              f"maxdiff {np.abs(G - Gr).max():.3e} scale {scale:.1e}")
                        fails += 1
                    if not np.allclose(s, sr, rtol=1e-10, atol=1e-10 * max(1, np.abs(sr).max())):
                        print(f"FAIL gram s p={p} n={n} {isa}")
                        fails += 1
                    if abs(W - Wr) > 1e-10 * max(1, Wr):
                        print(f"FAIL gram W p={p} n={n} {isa}: {W} vs {Wr}")
                        fails += 1
    print(f"gram differential: {'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return fails


def test_chol():
    rng = np.random.default_rng(11)
    fails = 0
    for q in (1, 2, 3, 4, 5, 8, 13, 16, 21, 32):
        A = rng.standard_normal((q, q))
        M = A @ A.T + q * np.eye(q)          # SPD
        # pack upper triangle into a full matrix (kernel reads upper only)
        Au = np.triu(M).copy()
        rc = ols.ols_chol_f64_sse2(Au, q, 1e-14, 0)
        if rc != 0:
            print(f"FAIL chol rc={rc} q={q}")
            fails += 1
            continue
        R = np.triu(Au)
        # verify R'R ~ M
        if not np.allclose(R.T @ R, M, rtol=1e-10, atol=1e-10 * np.abs(M).max()):
            print(f"FAIL chol factor q={q}: maxdiff {np.abs(R.T @ R - M).max():.3e}")
            fails += 1
        # solve
        b = rng.standard_normal(q)
        x_ref = np.linalg.solve(M, b)
        rc = ols.ols_chol_solve_f64_sse2(Au, q, b, 0)
        if rc != 0 or not np.allclose(b, x_ref, rtol=1e-8, atol=1e-8 * np.abs(x_ref).max()):
            print(f"FAIL chol solve q={q}")
            fails += 1
        # inverse
        Binv = np.zeros((q, q))
        rc = ols.ols_chol_inv_f64_sse2(Au, q, Binv, 0)
        if rc != 0 or not np.allclose(Binv, np.linalg.inv(M), rtol=1e-7,
                                      atol=1e-7 * np.abs(np.linalg.inv(M)).max()):
            print(f"FAIL chol inv q={q}")
            fails += 1
    print(f"chol differential: {'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return fails


def test_residual():
    rng = np.random.default_rng(31)
    fails = 0
    for p in (1, 3, 8, 16):
        for n in (10, 100, 1000):
            X = rng.standard_normal((n, p))
            y = rng.standard_normal(n)
            beta = rng.standard_normal(p)
            b0 = 0.7
            sse_buf = np.zeros(1)
            xtr_buf = np.zeros(p)
            ra = ResidArgs(ptr(X), X.strides[0], ptr(y), 0, n, p, b0, ptr(beta),
                           ptr(sse_buf), ptr(xtr_buf))
            rc = ols.ols_residual_f64_sse2(C.byref(ra))
            r = y - b0 - X @ beta
            if rc != 0 or abs(sse_buf[0] - r @ r) > 1e-9 * max(1, r @ r):
                print(f"FAIL resid sse p={p} n={n}: {sse_buf[0]} vs {r @ r}")
                fails += 1
            if not np.allclose(xtr_buf, X.T @ r, rtol=1e-8, atol=1e-8 * max(1, np.abs(X.T @ r).max())):
                print(f"FAIL resid xtr p={p} n={n}")
                fails += 1
    print(f"residual differential: {'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return fails


if __name__ == "__main__":
    f = test_gram() + test_chol() + test_residual()
    sys.exit(1 if f else 0)
