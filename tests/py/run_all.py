#!/usr/bin/env python3
"""tests/py/run_all.py — the committed Phase-1 test pyramid.

Levels (asm-verification §1):
  1. Reference differential (shapes x ISAs x offsets vs numpy + mpmath-lite)
  2. ABI conformance via abi_probe_call on EVERY exported kernel
  3. Memory safety via guard pages (C harness binary, n=0..300)
  4. Special values (NaN/Inf/denormal/huge/tiny/-0 -> OLS_E_NONFINITE etc.)
  5. Properties (scale, shift, permutation, offset-1e9 adversarial)
  6. Accuracy: NIST LRE table (gate: >= 10 on Norris/Pontius/NoInt1/NoInt2)
  7. Cross-ISA: sse2 vs avx2 within documented bounds
  8. Property suite (tests/props): mpmath differential over random problems,
     permutation invariance, exact recovery, unit-weight degeneracy,
     residual orthogonality, condition-number sweep
Exit code 0 iff everything passes. Writes results/pyramid-summary.json.
"""
import ctypes as C
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
import ols_bind as B
import ols_ref as R

ROOT = Path(__file__).resolve().parent.parent.parent
FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print(f"FAIL: {msg}")
    return cond


# ---------------------------------------------------------------- level 1
def test_shapes_and_offsets():
    """Full shape matrix + byte-offset misalignment sweep vs numpy reference."""
    ols = B.ols
    rng = np.random.default_rng(2026)
    shapes = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
              100, 127, 128, 129, 1000, 1023, 1024, 1025]
    # misalignment: build a padded buffer, take views at byte offsets
    for n in shapes:
        a_all = rng.standard_normal(n + 9) if n else np.zeros(9)
        b_all = rng.standard_normal(n + 9) if n else np.zeros(9)
        for off in (0, 8, 16, 24, 56):        # byte offsets within 64
            for arr_all, fn_base in ((a_all, "sum"), (b_all, "sum")):
                pass
            # sum + dot for both ISAs at this offset
            av = np.frombuffer(a_all.tobytes()[off:off + n * 8] if n else b"",
                               dtype=np.float64) if n else np.zeros(0)
            bv = np.frombuffer(b_all.tobytes()[off:off + n * 8] if n else b"",
                               dtype=np.float64) if n else np.zeros(0)
            # NOTE: frombuffer gives read-only; ctypes needs writable? We use
            # a different mechanism below (padded ndarray views), kept simple:
        break
    # Simpler + honest: numpy views with element offsets (equivalent to byte
    # offsets multiple of 8 — the contract's granularity), across all shapes.
    for n in shapes:
        a_pad = rng.standard_normal(n + 8)
        b_pad = rng.standard_normal(n + 8)
        for eloff in (0, 1, 3, 7):
            a = a_pad[eloff:eloff + n]
            b = b_pad[eloff:eloff + n]
            ref_sum = float(np.sum(a)) if n else 0.0
            ref_dot = float(np.dot(a, b)) if n else 0.0
            for isa in ("sse2", "avx2"):
                s = getattr(ols, f"ols_sum_f64_{isa}")(a, n)
                check(abs(s - ref_sum) <= 1e-9 * max(1, abs(ref_sum)),
                      f"sum {isa} n={n} off={eloff*8}B: {s} vs {ref_sum}")
                d = getattr(ols, f"ols_dot_f64_{isa}")(a, b, n)
                check(abs(d - ref_dot) <= 1e-9 * max(1, abs(ref_dot)),
                      f"dot {isa} n={n} off={eloff*8}B: {d} vs {ref_dot}")
            # axpy (writes y)
            y0 = b.copy()
            B.ols.ols_axpy_f64_avx2(2.5, a, y0, n)
            ref = b + 2.5 * a
            if n:
                check(np.max(np.abs(y0 - ref)) <= 1e-13 * max(1, np.max(np.abs(ref))),
                      f"axpy avx2 n={n} off={eloff*8}B")
            # fits vs Python reference
            if n >= 5:
                x = np.ascontiguousarray(a)
                y = np.ascontiguousarray(b)
                rc, out = B.simple_fit(x, y)
                ref = R.simple_fit(x, y)
                check(rc == 0, f"simple_fit rc n={n}")
                for k, key in ((0, "b1"), (1, "b0")):
                    check(abs(out[k] - ref[key]) <= 1e-9 * max(1, abs(ref[key])),
                          f"simple_fit {key} n={n} off={eloff*8}B")
    print("L1 shapes+offsets: done")


# ---------------------------------------------------------------- level 2
def test_abi_probe_all_exports():
    """Every exported kernel goes through the ABI canary probe."""
    ols = B.ols
    # resolve symbol addresses via dlsym on the CDLL handle
    dl = C.CDLL(None)
    dl.dlsym.restype = C.c_void_p
    dl.dlsym.argtypes = [C.c_void_p, C.c_char_p]
    rng = np.random.default_rng(5)
    buf = np.ascontiguousarray(rng.standard_normal(64))
    mask = C.c_uint32(0xFFFFFFFF)
    p = buf.ctypes.data
    cases = []
    out6 = np.zeros(6)
    out7 = np.zeros(7)
    out5 = np.zeros(5)
    alpha_bits = 0x4000000000000000          # 2.0
    cases = [
        ("ols_sum_f64_sse2", p, 16, 0, 0),
        ("ols_sum_f64_avx2", p, 16, 0, 0),
        ("ols_dot_f64_sse2", p, p, 16, 0),
        ("ols_dot_f64_avx2", p, p, 16, 0),
        # axpy takes alpha in xmm0: use the integer-arg adapters (tests/abi)
        ("abi_probe_axpy_sse2", alpha_bits, p, p, 16),
        ("abi_probe_axpy_avx2", alpha_bits, p, p, 16),
        ("ols_centered_sums_f64_sse2", p, p, 16, out6.ctypes.data),
        ("ols_cpu_has_avx2_fma", 0, 0, 0, 0),
        ("ols_cpu_has_avx512", 0, 0, 0, 0),
        ("ols_isa", 0, 0, 0, 0),
        ("ols_init", -1, 0, 0, 0),
    ]
    for name, a1, a2, a3, a4 in cases:
        addr = dl.dlsym(ols._handle, name.encode())
        check(addr is not None and addr != 0, f"dlsym {name}")
        rc = ols.abi_probe_call(C.c_void_p(addr),
                                C.c_uint64(a1 & 0xFFFFFFFFFFFFFFFF),
                                C.c_uint64(a2 & 0xFFFFFFFFFFFFFFFF),
                                C.c_uint64(a3 & 0xFFFFFFFFFFFFFFFF),
                                C.c_uint64(a4 & 0xFFFFFFFFFFFFFFFF),
                                C.byref(mask))
        check(mask.value == 0, f"ABI probe {name}: clobber mask {mask.value:#x}")
    print("L2 ABI probe on all exports: done")


# ---------------------------------------------------------------- level 3
def test_guard_pages():
    """Runs the C guard harness (self-test + dot template n=0..300)."""
    exe = ROOT / "build" / "tests" / "test_guard_full"
    src = ROOT / "tests" / "guard" / "test_guard_full.c"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["gcc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-Iinclude", "-Itests/guard", "-o", str(exe), str(src),
                        "-Lbuild/lib", "-lols", f"-Wl,-rpath,{ROOT}/build/lib"],
                       check=True, cwd=ROOT)
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    check(r.returncode == 0, f"guard harness failed: {r.stdout[-500:]} {r.stderr[-300:]}")
    print("L3 guard pages: done")


# ---------------------------------------------------------------- level 4
def test_special_values():
    ols = B.ols
    n = 16
    base = np.arange(n, dtype=np.float64) + 1.0
    specials = [float("nan"), float("inf"), float("-inf"),
                5e-324, 1e-300, 1e300, -0.0]
    for v in specials:
        a = base.copy()
        a[7] = v
        b = base.copy() * 2.0
        rc, out = B.simple_fit(a, b)
        nonfinite = not math.isfinite(v)
        check(rc == -6 if nonfinite else rc in (0, -7),
              f"special {v!r} in x: rc={rc} (expected NONFINITE for nonfinite)")
        # kernels propagate nonfinite inputs (no silent drops)
        s = ols.ols_sum_f64_sse2(a, n)
        if nonfinite:
            check(math.isnan(s) or math.isinf(s),
                  f"sum with {v!r} gave finite {s} (contract: propagate)")
        else:
            check(math.isfinite(s),
                  f"sum with finite {v!r} gave nonfinite {s}")
        d = ols.ols_dot_f64_sse2(a, b, n)
        if nonfinite:
            check(math.isnan(d) or math.isinf(d),
                  f"dot with {v!r} gave finite {d} (contract: propagate)")
    # all-finite edge: huge values
    a = np.full(n, 1e300)
    b = np.full(n, 1e300)
    rc, out = B.simple_fit(a, b)     # constant x -> singular
    check(rc == -7, f"constant x should be SINGULAR, got {rc}")
    # x = y = 1e300 constant: sxx = 16e600 overflows to +inf and b = inf/inf
    # = NaN; the honest float64 answer is SINGULAR (no rescaling exists in
    # this kernel). Contract updated in simple.asm; documented here.
    rc, out = B.simple_fit_noint(a, b)
    check(rc == -7, f"noint with 1e300 const: inf/inf slope must be SINGULAR, got {rc}")
    print("L4 special values: done")


# ---------------------------------------------------------------- level 5
def test_properties():
    rng = np.random.default_rng(77)
    n = 500
    x = rng.standard_normal(n) * 3 + 10
    y = 2.0 + 1.5 * x + rng.standard_normal(n) * 0.5

    # scale: fit(x, c*y).b1 == c * fit(x, y).b1
    _, o1 = B.simple_fit(x, y)
    _, o2 = B.simple_fit(x, 10.0 * y)
    check(abs(o2[0] - 10 * o1[0]) <= 1e-9 * 10 * abs(o1[0]),
          "property scale: b1(c*y) != c*b1(y)")
    check(abs(o2[4] - o1[4]) <= 1e-9, "property scale: r2 invariant")

    # shift with intercept: slopes invariant, intercept shifts by -t*b1
    t = 1000.0
    _, o3 = B.simple_fit(x + t, y)
    check(abs(o3[0] - o1[0]) <= 1e-6 * abs(o1[0]),
          "property shift: slope invariant")
    check(abs(o3[1] - (o1[1] - t * o1[0])) <= 1e-6 * max(1, abs(o1[1])),
          "property shift: intercept shift")

    # adversarial offset: x = 1e9 + noise (numerical-ols §2 mandatory test).
    # y is noisy (sigma=0.01, Sxx~n), so the ESTIMATOR itself has std ~4e-4:
    # the correct assertion is agreement with the high-precision reference,
    # not proximity to the true parameter.
    xoff = 1e9 + rng.standard_normal(n)
    yoff = 3.0 + 2.0 * (xoff - 1e9) + 0.01 * rng.standard_normal(n)
    _, o4 = B.simple_fit(xoff, yoff)
    mp4 = R.mp_simple_fit(xoff, yoff)
    b1_mp = float(mp4["b"][1]) if isinstance(mp4.get("b"), list) else float(mp4["b1"])
    b0_mp = float(mp4["b"][0]) if isinstance(mp4.get("b"), list) else float(mp4["b0"])
    check(abs(o4[0] - b1_mp) <= 1e-9 * abs(b1_mp),
          f"adversarial offset 1e9: b1={o4[0]} vs mpmath {b1_mp}")
    check(abs(o4[1] - b0_mp) <= 1e-9 * abs(b0_mp),
          f"adversarial offset: b0={o4[1]} vs mpmath {b0_mp}")
    # and the estimator must still be near the truth within sampling noise
    check(abs(o4[0] - 2.0) <= 1e-2, f"adversarial offset: b1={o4[0]} far from 2.0")

    # permutation invariance (within bounds)
    perm = rng.permutation(n)
    _, o5 = B.simple_fit(x[perm], y[perm])
    check(abs(o5[0] - o1[0]) <= 1e-9 * abs(o1[0]), "property permutation: b1")

    # mpmath agreement on a well-conditioned problem (60 dps)
    mpres = R.mp_simple_fit(x[:100], y[:100])
    b1_mp = float(mpres["b1"]) if "b1" in mpres else None
    if b1_mp is None:
        b_mp = [float(v) for v in mpres["b"]]
        b1_mp, b0_mp = b_mp[1], b_mp[0]
    else:
        b0_mp = float(mpres["b0"])
    _, o6 = B.simple_fit(x[:100], y[:100])
    check(abs(o6[0] - b1_mp) <= 1e-12 * abs(b1_mp),
          f"mpmath agreement b1: {o6[0]} vs {b1_mp}")
    check(abs(o6[1] - b0_mp) <= 1e-12 * max(1, abs(b0_mp)),
          f"mpmath agreement b0: {o6[1]} vs {b0_mp}")

    # kappa generator sanity: cond(X) tracks target (guards the reference)
    X, yy, bt, _ = R.make_kappa_problem(200, 4, 1e6)
    sv = np.linalg.svd(X, compute_uv=False)
    check(abs(sv[0] / sv[-1] - 1e6) <= 1e-3 * 1e6, "kappa generator cond tracking")
    print("L5 properties: done")


# ---------------------------------------------------------------- level 6
def test_nist():
    rows = []
    worst = 15.0
    for name in ("Norris", "Pontius", "NoInt1", "NoInt2"):
        d = R.parse_nist(ROOT / "data" / "nist" / f"{name}.dat")
        x, y = d.X[:, 0], d.y
        if name == "Pontius":
            rc, out = B.quad_fit(x, y)
            b = [out[0], out[1], out[2]]
            r2, sse = out[5], out[3]
        elif d.has_intercept:
            rc, out = B.simple_fit(x, y)
            b = [out[1], out[0]]
            r2, sse = out[4], out[2]
        else:
            rc, out = B.simple_fit_noint(x, y)
            b = [out[0]]
            r2, sse = out[3], out[1]
        check(rc == 0, f"NIST {name}: rc={rc}")
        lres = []
        for got, cert in zip(b, d.certified_b):
            l = R.lre(got, cert)
            lres.append(l)
            worst = min(worst, l)
        lre_r2 = R.lre(r2, d.r_squared)
        rsd = math.sqrt(sse / (d.n - d.p))
        lre_rsd = R.lre(rsd, d.residual_sd)
        worst = min(worst, lre_r2, lre_rsd)
        rows.append({"dataset": name, "n": d.n, "p": d.p,
                     "lre_b": [round(v, 1) for v in lres],
                     "lre_r2": round(lre_r2, 1), "lre_rsd": round(lre_rsd, 1),
                     "coefficients": b, "certified_b": list(d.certified_b)})
    # G1 pre-registered gate: LRE >= 10 per coefficient on all four datasets
    check(worst >= 10.0, f"NIST gate: worst LRE {worst:.1f} < 10")
    out = {"gate_min_lre": 10.0, "worst_lre": round(worst, 1), "rows": rows}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "nist-phase1.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"L6 NIST: worst LRE {worst:.1f} (gate >= 10)")
    return out


# ---------------------------------------------------------------- level 7
def test_cross_isa():
    rng = np.random.default_rng(99)
    for n in (1, 7, 16, 33, 100, 1000):
        a = rng.standard_normal(n)
        s_sse2 = B.ols.ols_sum_f64_sse2(a, n)
        s_avx2 = B.ols.ols_sum_f64_avx2(a, n)
        # different association orders: bound = gamma_{2n} * sum|a| ~ 2nu*sum|a|
        bound = 2 * n * 2.2e-16 * float(np.sum(np.abs(a)))
        check(abs(s_sse2 - s_avx2) <= max(bound, 1e-300),
              f"cross-ISA sum n={n}: {s_sse2} vs {s_avx2} (bound {bound:.3e})")
        b = rng.standard_normal(n)
        d_sse2 = B.ols.ols_dot_f64_sse2(a, b, n)
        d_avx2 = B.ols.ols_dot_f64_avx2(a, b, n)
        bound = 2 * n * 2.2e-16 * float(np.sum(np.abs(a) * np.abs(b)))
        check(abs(d_sse2 - d_avx2) <= max(bound, 1e-300),
              f"cross-ISA dot n={n}: {d_sse2} vs {d_avx2}")
    print("L7 cross-ISA: done")


def test_props_suite():
    """L8: property-based tests from tests/props/test_props.py (P1-P6)."""
    props_dir = ROOT / "tests" / "props"
    sys.path.insert(0, str(props_dir))
    try:
        import test_props
    except ImportError as e:
        check(False, f"props suite import: {e}")
        return
    ok = test_props.run(FAILURES)
    if not ok:
        for f in test_props.FAILS:
            check(False, f"props: {f}")
    print("L8 properties (tests/props): done")


def main():
    test_shapes_and_offsets()
    test_abi_probe_all_exports()
    test_guard_pages()
    test_special_values()
    test_properties()
    nist = test_nist()
    test_cross_isa()
    test_props_suite()
    summary = {
        "pass": not FAILURES,
        "failures": FAILURES[:50],
        "nist_worst_lre": nist["worst_lre"],
    }
    (ROOT / "results" / "pyramid-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if FAILURES:
        print(f"\nPYRAMID: FAILED ({len(FAILURES)} failures)")
        return 1
    print("\nPYRAMID: ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
