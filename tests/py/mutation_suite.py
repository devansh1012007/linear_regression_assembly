#!/usr/bin/env python3
"""tests/py/mutation_suite.py — fast differential suite used by the mutator.

Runs a representative subset of the pyramid against the MUTATED library
(pointed to by $OLS_MUT_SO; default build/lib/libols.so) and exits nonzero on any
mismatch. Small enough to run ~50x, sharp enough to catch kernel mutations:
  - sums/dots across boundary sizes (n mod 4, mod 8, mod 16) both ISAs
  - axpy write check
  - simple/noint/quad fits vs the Python reference (pivot-shifted)
  - special-value error codes (NONFINITE for NaN, SINGULAR paths)
"""
import ctypes as C
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tests" / "ref"))
import ols_ref as R

LIB = Path(os.environ.get("OLS_MUT_SO", ROOT / "build" / "lib" / "libols.so"))
ols = C.CDLL(str(LIB))
for fn in ("ols_sum_f64_sse2", "ols_sum_f64_avx2"):
    f = getattr(ols, fn); f.restype = C.c_double
    f.argtypes = [np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), C.c_size_t]
for fn in ("ols_dot_f64_sse2", "ols_dot_f64_avx2"):
    f = getattr(ols, fn); f.restype = C.c_double
    f.argtypes = [np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                  np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), C.c_size_t]
for fn in ("ols_axpy_f64_sse2", "ols_axpy_f64_avx2"):
    f = getattr(ols, fn); f.restype = None
    f.argtypes = [C.c_double,
                  np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                  np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), C.c_size_t]
ols.ols_centered_sums_f64_sse2.restype = C.c_int32
ols.ols_centered_sums_f64_sse2.argtypes = [
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    C.c_size_t, np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")]
for fn in ("ols_simple_fit_f64_sse2", "ols_simple_fit_noint_f64_sse2",
           "ols_quad_fit_f64_sse2"):
    f = getattr(ols, fn); f.restype = C.c_int32
    f.argtypes = [
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        C.c_size_t, C.c_uint32,
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")]

FAIL = []


def chk(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("MUT-CAUGHT:", msg)


def cancellation_battery():
    """Exactly-referenced cancellation cases (P3.9 survivor fix): the sum/dot
    survivors degrade Neumaier to plain sequential summation, which loses
    ~16 digits on these inputs; math.fsum is the exactly-rounded oracle.
    numpy's own sum ALSO fails most of these (sequential too) — that is why
    the differentials never caught the mutants."""
    import math
    cases = [
        # big + small + negative-of-big: exact answer = small chain
        [1e16, 1.0, 2.0, -1e16, 3.0],
        [3e15, 1.0, 1.0, -3e15],
        [1e17, 5.0, -1e17, 5.0],
        # alternating cascade (classic Kahan demo)
        [1.0] + [1e-1, -1e-1] * 40,
        # scaled pair sums at various sizes
        [1e12, 1e-4, 1e-4, 1e-4, -1e12],
        [2.2e14, 1.0, 3.0, -2.2e14, 7.0],
    ]
    for a in cases:
        ref = math.fsum(a)
        # BOTH sum kernels are Neumaier by contract -> oracle-strict
        for isa in ("sse2", "avx2"):
            s = getattr(ols, f"ols_sum_f64_{isa}")(np.ascontiguousarray(a), len(a))
            chk(abs(s - ref) <= 1e-14 * max(1.0, abs(ref)),
                f"sum-cancel {isa} ref={ref!r} got={s!r}")
    # dot: ONLY the sse2 path is compensated by contract (dot.asm header:
    # avx2 = 4x4 FMA accumulators, error <= gamma_2n * sum|ab| — the
    # compensated variant is the sse2 reference and the Dot2 COULD item).
    # So: sse2 strict vs fsum; avx2 only within its DOCUMENTED bound.
    A = np.array([1e16, 1.0, 2.0, -1e16, 3.0])
    B = np.ones(5)
    ref = math.fsum(A * B)
    d = ols.ols_dot_f64_sse2(np.ascontiguousarray(A), B, 5)
    chk(abs(d - ref) <= 1e-14 * max(1.0, abs(ref)),
        f"dot-cancel sse2 ref={ref!r} got={d!r}")
    bound = 1e-15 * float(np.abs(A * B).sum()) * 10   # gamma_2n, n=5, loose
    d = ols.ols_dot_f64_avx2(np.ascontiguousarray(A), B, 5)
    chk(abs(d - ref) <= bound,
        f"dot-cancel avx2 beyond documented bound: ref={ref!r} got={d!r} "
        f"bound={bound!r}")


def main():
    rng = np.random.default_rng(4242)
    cancellation_battery()
    sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 20, 31, 32, 33, 63, 64, 65, 100]
    for n in sizes:
        a = rng.standard_normal(n) * (10.0 ** float(rng.integers(-3, 4)))
        b = rng.standard_normal(n) * (10.0 ** float(rng.integers(-3, 4)))
        rs, rd = float(np.sum(a)), float(np.dot(a, b))
        for isa in ("sse2", "avx2"):
            s = getattr(ols, f"ols_sum_f64_{isa}")(a, n)
            chk(abs(s - rs) <= 1e-8 * max(1, abs(rs)), f"sum {isa} n={n}: {s} vs {rs}")
            d = getattr(ols, f"ols_dot_f64_{isa}")(a, b, n)
            chk(abs(d - rd) <= 1e-8 * max(1, abs(rd)), f"dot {isa} n={n}: {d} vs {rd}")
        y = b.copy()
        ols.ols_axpy_f64_avx2(2.5, a, y, n)
        chk(np.max(np.abs(y - (b + 2.5 * a))) <= 1e-12 * max(1, np.max(np.abs(b))),
            f"axpy avx2 n={n}")
        y2 = b.copy()
        ols.ols_axpy_f64_sse2(2.5, a, y2, n)
        chk(np.max(np.abs(y2 - (b + 2.5 * a))) <= 1e-12 * max(1, np.max(np.abs(b))),
            f"axpy sse2 n={n}")
        if n >= 5:
            x = np.ascontiguousarray(a); yy = np.ascontiguousarray(b)
            out = np.zeros(7)
            rc = ols.ols_simple_fit_f64_sse2(x, yy, n, 1, out)
            ref = R.simple_fit(x, yy)
            chk(rc == 0, f"simple rc n={n}: {rc}")
            chk(abs(out[0] - ref["b1"]) <= 1e-8 * max(1, abs(ref["b1"])),
                f"simple b1 n={n}: {out[0]} vs {ref['b1']}")
            chk(abs(out[1] - ref["b0"]) <= 1e-8 * max(1, abs(ref["b0"])),
                f"simple b0 n={n}: {out[1]} vs {ref['b0']}")
            out5 = np.zeros(5)
            rc = ols.ols_simple_fit_noint_f64_sse2(x, yy, n, 1, out5)
            refn = R.simple_fit_noint(x, yy)
            chk(rc == 0, f"noint rc n={n}: {rc}")
            chk(abs(out5[0] - refn["b"]) <= 1e-8 * max(1, abs(refn["b"])),
                f"noint b n={n}: {out5[0]} vs {refn['b']}")
            out7 = np.zeros(7)
            rc = ols.ols_quad_fit_f64_sse2(x, yy, n, 1, out7)
            refq = R.quad_fit(x, yy)
            chk(rc == 0, f"quad rc n={n}: {rc}")
            for k in range(3):
                chk(abs(out7[k] - refq[f"b{k}"]) <= 1e-7 * max(1, abs(refq[f"b{k}"])),
                    f"quad b{k} n={n}: {out7[k]} vs {refq[f'b{k}']}")
    # NIST quick check (the sharpest accuracy probe)
    d = R.parse_nist(ROOT / "data" / "nist" / "Norris.dat")
    out = np.zeros(7)
    rc = ols.ols_simple_fit_f64_sse2(np.ascontiguousarray(d.X[:, 0]),
                                     np.ascontiguousarray(d.y), d.n, 1, out)
    chk(rc == 0 and R.lre(out[1], d.certified_b[0]) >= 8,
        f"Norris b0 LRE {R.lre(out[1], d.certified_b[0]):.1f} < 8")
    chk(R.lre(out[0], d.certified_b[1]) >= 8,
        f"Norris b1 LRE {R.lre(out[0], d.certified_b[1]):.1f} < 8")
    # special values
    bad = np.arange(8.0); bad[3] = math.nan
    out = np.zeros(7)
    rc = ols.ols_simple_fit_f64_sse2(bad, np.arange(8.0), 8, 0, out)
    chk(rc == -6, f"NaN input must give NONFINITE, got {rc}")
    flat = np.ones(8)
    rc = ols.ols_simple_fit_f64_sse2(flat, np.arange(8.0), 8, 0, out)
    chk(rc == -7, f"constant x must be SINGULAR, got {rc}")
    if FAIL:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
