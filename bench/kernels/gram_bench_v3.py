#!/usr/bin/env python3
"""gram bench v3 — schema-compliant (asm-benchmarking §5) benchmark of the
rewritten AVX2 Gram kernel.

Rule zero: every number in results/gram-bench-v3.json is produced here from
raw samples committed in the same file, at the commit being described.

Method (asm-benchmarking §2):
  - working sets: q=8/17 (n=200k, ~13-29 MB, DRAM-resident, > 8x L3/2),
    q=32/65 (n=4096, 1-2 MB, L2-resident) — matching the v1 run's cases.
  - warm-up 3, then 30 timed reps (kernel-level); median + min + CV reported;
    CV > 3% flags unstable.
  - pin to one core via taskset (host has 2 vCPUs, shared KVM).
  - bytes moved per row = 8*q (X row + y + w), matching the v1 accounting so
    the two runs are directly comparable.
Output: results/gram-bench-v3.json (schema 1, immutable).
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


class GramArgs(C.Structure):
    _fields_ = [("X", C.c_void_p), ("ldx8", C.c_size_t), ("y", C.c_void_p),
                ("w", C.c_void_p), ("n", C.c_size_t), ("p", C.c_size_t),
                ("c", C.c_void_p), ("G", C.c_void_p), ("s", C.c_void_p),
                ("Wout", C.c_void_p), ("scratch", C.c_void_p),
                ("scratch_bytes", C.c_size_t)]


def bench_case(p, n, reps=30, warm=3, seed=20261121):
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
    ptr = C.byref(ga)
    for _ in range(warm):
        rc = ols.ols_gram_f64_avx2(ptr)
        assert rc == 0, rc
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        rc = ols.ols_gram_f64_avx2(ptr)
        t1 = time.perf_counter_ns()
        assert rc == 0, rc
        samples.append((t1 - t0) / 1e9)
    med = statistics.median(samples)
    mn = min(samples)
    mean = statistics.fmean(samples)
    sd = statistics.stdev(samples)
    cv = sd / mean
    bytes_row = 8 * q
    gbps = bytes_row * n / med / 1e9
    return {
        "samples_s": samples,
        "derived": {
            "median_s": med,
            "min_s": mn,
            "cv": cv,
            "ns_per_row": med / n * 1e9,
            "gbps": gbps,
            "bytes_per_row": bytes_row,
        },
    }


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip() != ""
    host = json.loads((ROOT / "results/machine.json").read_text())["host"]
    cases = [
        ("q8_n200000_DRAM", 7, 200000),
        ("q17_n200000_DRAM", 16, 200000),
        ("q32_n4096_L2", 31, 4096),
        ("q65_n4096_L2", 64, 4096),
    ]
    out = {
        "schema": 1,
        "commit": commit,
        "dirty": dirty,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": host,
        "toolchain": {"nasm": "2.16.03", "python": "3.12.16"},
        "config": {"kernel": "ols_gram_f64_avx2", "variant": "v3 group-pipelined column-panel",
                   "isa": "avx2+fma3", "threads": 1, "precision": "f64",
                   "reps": 30, "warmup": 3, "bytes_per_row_formula": "8*(p+1)"},
        "cases": {},
        "notes": "v1 comparison numbers from results/gram-bench.json (same cases, "
                 "same accounting). DRAM bound 18.89 GB/s 1t (machine.json, second "
                 "characterization run; run-to-run 16.2-18.9). CV policy: >3% flags "
                 "unstable per asm-benchmarking §2.",
    }
    for name, p, n in cases:
        r = bench_case(p, n)
        out["cases"][name] = r
        d = r["derived"]
        print(f"{name}: median {d['ns_per_row']:.2f} ns/row  {d['gbps']:.2f} GB/s  "
              f"CV {d['cv']*100:.2f}%")
    (ROOT / "results/gram-bench-v3.json").write_text(json.dumps(out, indent=1))
    print("written results/gram-bench-v3.json")


if __name__ == "__main__":
    main()
