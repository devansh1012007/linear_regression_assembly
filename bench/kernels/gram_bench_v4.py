#!/usr/bin/env python3
"""gram bench v4 — same methodology as gram_bench_v3.py (asm-benchmarking §5):
warm-up 3, 30 timed reps, median + min + CV, bytes/row = 8*q, taskset-pinned
(where available). Cases: the v3 DRAM pair for continuity (q=8 fused-v4,
q=17 kernel-A-v3 unchanged control), the v3 L2 pair, PLUS:
  - q=2/4/6 narrow DRAM cases (v4's home turf)
  - ols_gram_mt (2 threads) at q=8/17 DRAM — the H5 rung.
Output: results/gram-bench-v4.json.
"""
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/py"))
import ols_bind as B  # noqa: E402
import ctypes as C  # noqa: E402

ols = B.ols
ols.ols_gram_scratch_bytes.restype = C.c_size_t
ols.ols_gram_scratch_bytes.argtypes = [C.c_size_t]
ols.ols_gram_mt.restype = C.c_int32
ols.ols_gram_mt.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
ols.ols_gram_mt_scratch_bytes.restype = C.c_size_t
ols.ols_gram_mt_scratch_bytes.argtypes = [C.c_size_t, C.c_size_t]


class GramArgs(C.Structure):
    _fields_ = [("X", C.c_void_p), ("ldx8", C.c_size_t), ("y", C.c_void_p),
                ("w", C.c_void_p), ("n", C.c_size_t), ("p", C.c_size_t),
                ("c", C.c_void_p), ("G", C.c_void_p), ("s", C.c_void_p),
                ("Wout", C.c_void_p), ("scratch", C.c_void_p),
                ("scratch_bytes", C.c_size_t)]


def make_case(p, n, seed=20261121):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    q = p + 1
    c = np.zeros(q)
    G = np.zeros(q * q)
    s = np.zeros(q)
    Wout = np.zeros(1)
    sb = ols.ols_gram_scratch_bytes(q)
    buf = np.zeros((sb + 64) // 8)
    scratch = (buf.ctypes.data + 63) & ~63
    ga = GramArgs(X.ctypes.data, X.strides[0], y.ctypes.data, 0, n, p,
                  c.ctypes.data, G.ctypes.data, s.ctypes.data,
                  Wout.ctypes.data, C.c_void_p(scratch), sb)
    # KEEP the backing arrays alive: byref alone dangles once the locals go
    # out of scope (the segfault in the first bench run)
    keep = (X, y, c, G, s, Wout, buf, ga)
    return C.byref(ga), q, n, keep


def bench(p, n, fn, reps=30, warm=3, threads=None):
    if threads is None:
        ptr, q, n, keep = make_case(p, n)
        call = lambda: ols.ols_gram_f64_avx2(ptr)
    else:
        ptr, mscratch, msb, q, n, keep = make_case_mt(p, n, threads)
        call = lambda: ols.ols_gram_mt(ptr, threads, C.c_void_p(mscratch), msb)
    for _ in range(warm):
        rc = call()
        assert rc == 0, rc
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        rc = call()
        t1 = time.perf_counter_ns()
        assert rc == 0, rc
        samples.append((t1 - t0) / 1e9)
    med = statistics.median(samples)
    cv = statistics.stdev(samples) / statistics.fmean(samples)
    bytes_row = 8 * q
    return {
        "samples_s": samples,
        "derived": {
            "median_s": med,
            "min_s": min(samples),
            "cv": cv,
            "ns_per_row": med / n * 1e9,
            "gbps": bytes_row * n / med / 1e9,
            "bytes_per_row": bytes_row,
        },
    }


def make_case_mt(p, n, threads=2, seed=20261121):
    """ols_gram_mt(A, T, scratch, bytes): external thread scratch sized by
    ols_gram_mt_scratch_bytes(q, T); A->scratch is unused by the mt path but
    kept valid."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    q = p + 1
    c = np.zeros(q)
    G = np.zeros(q * q)
    s = np.zeros(q)
    Wout = np.zeros(1)
    sb = ols.ols_gram_mt_scratch_bytes(q, threads)
    buf = np.zeros((sb + 64) // 8)
    scratch = (buf.ctypes.data + 63) & ~63
    ga = GramArgs(X.ctypes.data, X.strides[0], y.ctypes.data, 0, n, p,
                  c.ctypes.data, G.ctypes.data, s.ctypes.data,
                  Wout.ctypes.data, C.c_void_p(scratch), sb)
    keep = (X, y, c, G, s, Wout, buf, ga)
    return C.byref(ga), scratch, sb, q, n, keep


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip() != ""
    host = json.loads((ROOT / "results/machine.json").read_text())["host"]
    cases = [
        ("q2_n200000_DRAM", 1, 200000, 1, None),
        ("q4_n200000_DRAM", 3, 200000, 1, None),
        ("q6_n200000_DRAM", 5, 200000, 1, None),
        ("q8_n200000_DRAM", 7, 200000, 1, None),
        ("q17_n200000_DRAM", 16, 200000, 1, None),
        ("q32_n4096_L2", 31, 4096, 1, None),
        ("q65_n4096_L2", 64, 4096, 1, None),
        ("q8_n200000_DRAM_mt2", 7, 200000, 2, 2),
        ("q17_n200000_DRAM_mt2", 16, 200000, 2, 2),
    ]
    out = {
        "schema": 1,
        "commit": commit,
        "dirty": dirty,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": host,
        "toolchain": {"nasm": "2.16.03", "python": "3.12.16"},
        "config": {
            "kernel": "ols_gram_f64_avx2 (v4 fused q<=8 / v3 kernelA q>=9) "
                      "+ ols_gram_mt",
            "isa": "avx2+fma3", "precision": "f64",
            "reps": 30, "warmup": 3, "bytes_per_row_formula": "8*(p+1)"},
        "cases": {},
        "notes": "v4 prediction perf-log #28: q=8 14+-3 ns/row (75-95% of "
                 "DRAM 18.89 GB/s = 3.39 ns/row at 64 B/row... bound: "
                 "64/18.89 = 3.39 ns/row). q17 v3 control ~24.4. mt2: H5 rung "
                 "vs all-thread DRAM 37.6 GB/s.",
    }
    for name, p, n, _, threads in cases:
        r = bench(p, n, None, threads=threads)
        out["cases"][name] = r
        d = r["derived"]
        print(f"{name}: {d['ns_per_row']:.2f} ns/row  {d['gbps']:.2f} GB/s  "
              f"CV {d['cv']*100:.2f}%")
    (ROOT / "results/gram-bench-v4.json").write_text(json.dumps(out, indent=1))
    print("written results/gram-bench-v4.json")


if __name__ == "__main__":
    main()
