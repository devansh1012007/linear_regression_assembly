"""bench/ladder/run_ladder.py — Phase 5 baseline ladder driver.

Runs the C ladder (naive-O2, naive-O3, intrinsics, asm-v3) across the
(n, q) matrix and adds the NumPy rung (Z.T @ Z on the same data, BLAS
dsyrk-equivalent). Writes results/ladder.json + prints a table.

Honesty note: this sandbox is CPU-quota capped (~2 effective cores,
throttled clock); absolute numbers are NOT comparable to bare metal.
The ladder's PURPOSE here is the RELATIVE ordering of rungs on identical
hardware + the artifact for re-running on real hardware (make repro).
"""
import json
import subprocess
import sys
import time

import numpy as np

CASES = [(200000, 8), (200000, 17), (1000000, 8), (1000000, 17)]


def numpy_rung(n, q, reps=3):
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((n, q))
    best = 1e30
    for _ in range(reps):
        t0 = time.perf_counter()
        G = Z.T @ Z
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best / n * 1e9, n * q * 8 / best / 1e9, float(np.abs(G).max())


def main():
    results = []
    for n, q in CASES:
        for tag, exe in (("naive-O2", "build/bench/ladder_o2"),
                         ("naive-O3", "build/bench/ladder_o3")):
            out = subprocess.run([f"./{exe}", str(n), str(q), "3"],
                                 capture_output=True, text=True).stdout
            for line in out.splitlines():
                if line.startswith(("naive", "intrin", "asm")):
                    parts = line.split()
                    r = {"rung": parts[0] + ("-O3" if "O3" in tag and
                         parts[0] == "naive-O2" else ""),
                         "n": n, "q": q,
                         "ns_row": float(parts[parts.index("ns/row") - 1]),
                         "gbps": float(parts[parts.index("GB/s") - 1])}
                    # fix rung name for the O3 binary (naive-O2 label inside)
                    if "O3" in tag and parts[0] == "naive-O2":
                        r["rung"] = "naive-O3"
                    if "O3" in tag and parts[0] == "intrin":
                        r["rung"] = "intrin-O3"
                    if "O3" in tag and parts[0] == "asm-v3":
                        continue  # same kernel in both binaries; keep O2's
                    results.append(r)
        # numpy
        ns_row, gbps, gmax = numpy_rung(n, q)
        results.append({"rung": "numpy-blas", "n": n, "q": q,
                        "ns_row": ns_row, "gbps": gbps, "gmax": gmax})

    # table
    print(f"{'rung':12s} {'n':>8s} {'q':>3s} {'ns/row':>8s} {'GB/s':>7s}")
    for r in sorted(results, key=lambda r: (r["n"], r["q"], r["rung"])):
        print(f"{r['rung']:12s} {r['n']:8d} {r['q']:3d} "
              f"{r['ns_row']:8.2f} {r['gbps']:7.2f}")
    with open("results/ladder.json", "w") as f:
        json.dump({"cases": CASES, "runs": results,
                   "note": "CPU-quota-capped sandbox; relative ordering "
                           "only; re-run on bare metal via make repro"},
                  f, indent=1)
    print("written: results/ladder.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
