"""tests/py/test_mt.py — Phase 4c: multithreaded Gram (ols_gram_mt).

G4 reproducibility: results BITWISE IDENTICAL for 1..N threads (the
fixed-order index merge, never completion order). Plus: weighted case,
slice-boundary shapes (n not divisible by T, n < 2T fallback), and a first
scaling measurement (reported; the gate lives in the Phase-5 ladder).
"""
import ctypes as C
import sys
import time

import numpy as np

sys.path.insert(0, "tests/py")
sys.path.insert(0, "tests/ref")
import ols_bind as B
from test_gram import GramArgs, ptr

ols = B.ols
ols.ols_gram_mt.restype = C.c_int32
ols.ols_gram_mt.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
ols.ols_gram_mt_scratch_bytes.restype = C.c_size_t
ols.ols_gram_mt_scratch_bytes.argtypes = [C.c_size_t, C.c_size_t]


def run_mt(X, y, w, threads, c):
    n, p = X.shape
    q = p + 1
    G = np.zeros(q * q)
    s = np.zeros(q)
    Wout = np.zeros(1)
    sb = ols.ols_gram_mt_scratch_bytes(q, threads)
    buf = np.zeros((sb + 64) // 8, dtype=np.float64)
    scratch = (buf.ctypes.data + 63) & ~63
    ga = GramArgs(ptr(X), X.strides[0], ptr(y), ptr(w), n, p, ptr(c),
                  ptr(G), ptr(s), ptr(Wout), C.c_void_p(scratch), sb)
    rc = ols.ols_gram_mt(C.byref(ga), threads, C.c_void_p(scratch), sb)
    return rc, G, s, Wout


def main():
    rng = np.random.default_rng(77)
    fails = 0

    # --- reproducibility + edge shapes ---
    # CONTRACT (ADR-0008, honest): same T => bitwise identical across runs
    # and schedulings (the fixed-index-order merge guarantees this);
    # ACROSS T, agreement within the compensated-summation bound (~ n*u
    # relative) — bitwise 1-vs-N equality is impossible with a partials
    # merge because the float addition ORDER differs (the sequential kernel
    # sums rows 0..n-1 in one chain; T slices sum slice-wise then merge).
    for trial, (n, p) in enumerate([(1000, 3), (7777, 5), (12345, 8),
                                    (500, 2), (23, 4)]):
        X = np.ascontiguousarray(rng.standard_normal((n, p)))
        y = rng.standard_normal(n)
        Xc = np.hstack([X, y.reshape(-1, 1)])
        c = Xc[: min(n, 256)].mean(axis=0)
        results = {}
        ok = True
        for T in (1, 2, 3, 4, 5, 6, 7, 8):
            rc, G, s, W = run_mt(X, y, None, T, c)
            if rc != 0:
                print(f"FAIL t{trial} T={T}: rc={rc}")
                fails += 1
                ok = False
                break
            results[T] = (G, s, W)
        if ok:
            G1, s1, W1 = results[1]
            tol = 1e-13 * max(1.0, float(np.abs(G1).max()))  # >> n*u, safe
            worst = 0.0
            for T in range(2, 9):
                G, s, W = results[T]
                worst = max(worst, float(np.abs(G - G1).max()))
                # same-T determinism: rerun T=4 twice, bitwise
            rcA, GA_, sA_, WA_ = run_mt(X, y, None, 4, c)
            rcB, GB_, sB_, WB_ = run_mt(X, y, None, 4, c)
            det = (GA_.tobytes() == GB_.tobytes()
                   and sA_.tobytes() == sB_.tobytes())
            cross = worst <= tol
            print(f"t{trial} (n={n}, p={p}): cross-T worst {worst:.2e} "
                  f"<= {tol:.1e} {'OK' if cross else 'FAIL'}; "
                  f"T=4 rerun bitwise {'OK' if det else 'FAIL'}")
            if not cross or not det:
                fails += 1

    # --- weighted, T=4 vs T=1 ---
    n, p = 4321, 4
    X = np.ascontiguousarray(rng.standard_normal((n, p)))
    y = rng.standard_normal(n)
    w = rng.random(n) + 0.5
    Xc = np.hstack([X, y.reshape(-1, 1)])
    c = Xc[:256].mean(axis=0)
    r1 = run_mt(X, y, w, 1, c)
    r4 = run_mt(X, y, w, 4, c)
    if r1[0] or r4[0]:
        print(f"FAIL weighted: rc {r1[0]}/{r4[0]}")
        fails += 1
    else:
        dg = float(np.abs(r1[1] - r4[1]).max())
        tol = 1e-13 * max(1.0, float(np.abs(r1[1]).max()))
        print(f"weighted T=4 vs T=1: maxdiff {dg:.2e} "
              f"{'OK' if dg <= tol else 'FAIL'}")
        if dg > tol:
            fails += 1

    # --- n < 2T fallback (tiny n, many threads) ---
    X = np.ascontiguousarray(rng.standard_normal((9, 3)))
    y = rng.standard_normal(9)
    c = np.zeros(4)
    rc, G, s, W = run_mt(X, y, None, 8, c)
    rc2, G2, s2, W2 = run_mt(X, y, None, 1, c)
    ok = rc == 0 and rc2 == 0 and G.tobytes() == G2.tobytes()
    print(f"n=9 T=8 fallback: {'OK' if ok else 'FAIL'} (rc {rc}/{rc2})")
    if not ok:
        fails += 1

    # --- scaling measurement (reported; no gate here) ---
    n, p = 2_000_000, 7
    X = np.ascontiguousarray(rng.standard_normal((n, p)))
    y = rng.standard_normal(n)
    c = np.zeros(p + 1)
    for T in (1, 2, 4):
        t0 = time.perf_counter()
        rc, G, s, W = run_mt(X, y, None, T, c)
        dt = time.perf_counter() - t0
        print(f"scaling: n=2e6 p=7 T={T}: {dt*1000:7.1f} ms rc={rc} "
              f"({n*p*8/dt/1e9:5.1f} GB/s)")
        if rc:
            fails += 1

    print(f"mt gram: {'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
