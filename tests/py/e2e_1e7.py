#!/usr/bin/env python3
"""G4 clause 1: end-to-end 1e7 x 16 fit via the .olsb path vs numpy pipeline.

PLAN Phase-4 gate G4: "end-to-end on a 1e7 x 16 file matches the NumPy
pipeline within bounds; fuzz runs clean ...; static binary has no dynamic
section and a non-executable stack" (thread clause graded separately as H5).

This script:
  1. streams a synthetic 1e7 x 17 (y + x1..x16) .olsb directly to disk
     (no 1.4 GB python array in RAM; coefficients chosen, sigma known),
  2. runs build/cli/asm-ols fit --json on it,
  3. recomputes the reference with a numpy NORMAL-EQUATIONS pipeline
     (X'X, X'y — the same algorithm class; lstsq's QR at 1e7 x 16 would
     need a second full copy of X and does not fit this host's RAM),
  4. checks |beta_asm - beta_np| <= 5e-10 relative on non-intercept
     coefficients and <= 1e-8 absolute on the intercept, plus sigma2/r2,
  5. writes results/e2e-1e7.json (timings included).

Bounds rationale: well-conditioned by construction (columns ~ N(0,1),
design ~ orthogonal); Gram-based solve vs numpy dgemm normal equations
agree to ~1e-12 typically; 5e-10 gives 3+ orders of headroom without
being vacuous. Run: python3 tests/py/e2e_1e7.py  (NOT in `make verify`;
belongs to `make repro` scale runs).
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CLI = ROOT / "build" / "cli" / "asm-ols"
OUT = ROOT / "results" / "e2e-1e7.json"

N = 10_000_000
Q = 16          # features
SEED = 20261126
SIGMA = 0.25


def stream_olsb(path: Path) -> dict:
    """Write the .olsb incrementally: names then 1e7 rows in 100 chunks."""
    rng = np.random.default_rng(SEED)
    beta_true = np.linspace(0.5, 2.0, Q) * np.array(
        [1 if i % 2 == 0 else -1 for i in range(Q)])
    intercept = -1.25
    names = ["y"] + [f"x{j+1}" for j in range(Q)]
    header_len = 48
    name_block = b"".join(n.encode() + b"\0" for n in names)
    name_off = header_len
    data_off = name_off + len(name_block)
    n_cols = Q + 1
    data_bytes = N * n_cols * 8
    trailer_off = data_off + data_bytes

    crc = 0
    t0 = time.perf_counter()
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQQQQ", 0x4F4C5342, 1, N, n_cols,
                            name_off, data_off, trailer_off))
        f.write(name_block)
        for _ in range(100):
            X = rng.standard_normal((N // 100, Q))
            y = intercept + X @ beta_true + SIGMA * rng.standard_normal(N // 100)
            block = np.column_stack([y, X]).astype("<f8").tobytes()
            crc = zlib.crc32(block, crc)
            f.write(block)
        f.write(struct.pack("<I", crc & 0xFFFFFFFF))
    gen_s = time.perf_counter() - t0
    return {"beta_true": beta_true.tolist(), "intercept": intercept,
            "gen_s": gen_s, "file_MB": path.stat().st_size / 1e6}


def numpy_reference(path: Path) -> dict:
    """Normal-equations reference, streamed in chunks (matches algorithm class).

    Accumulates BOTH the Gram G = A'A and the column sums s = 1'A: the
    intercept's normal-equation cross-terms need 1'X = feature column sums,
    which are NOT derivable from G alone (first draft used X'y there —
    wrong by ~beta*n, caught by the lstsq differential at n=2e5).
    """
    with open(path, "rb") as f:
        raw = f.read(48)
        _, _, n, nc, _, data_off, _ = struct.unpack("<IIQQQQQ", raw)
        f.seek(data_off)
        G = np.zeros((nc, nc))
        s = np.zeros(nc)              # column sums (1'A): s[0]=sum(y)
        N_chunk = 200_000
        rows_left = n
        t0 = time.perf_counter()
        while rows_left:
            k = min(N_chunk, rows_left)
            A = np.frombuffer(f.read(k * nc * 8), dtype="<f8").reshape(k, nc)
            G += A.T @ A
            s += A.sum(axis=0)
            rows_left -= k
        ref_s = time.perf_counter() - t0
    # augmented normal equations [[n, 1'X], [X'1, X'X]] [b0, b] = [1'y, X'y]
    Afull = np.ones((nc, nc))
    Afull[0, 0] = float(n)
    Afull[0, 1:] = s[1:]              # 1'X  (column sums!)
    Afull[1:, 0] = s[1:]
    Afull[1:, 1:] = G[1:, 1:]
    b = np.array([s[0]] + list(G[0, 1:]))
    beta = np.linalg.solve(Afull, b)
    # SSE from sufficient stats: SSE = y'y - 2 beta'c + beta' Afull beta,
    # with c = [1'y, X'y]
    c = np.array([s[0]] + list(G[0, 1:]))
    yty = G[0, 0]
    SSE = yty - 2 * beta @ c + beta @ Afull @ beta
    if SSE < 0:                        # roundoff guard
        SSE = max(SSE, 0.0)
    sigma2 = SSE / (n - nc)
    sst = yty - s[0] ** 2 / n
    r2 = 1 - SSE / sst
    return {"beta": beta.tolist(), "sigma2": float(sigma2),
            "r2": float(r2), "ref_s": ref_s}


def main() -> None:
    if not CLI.exists():
        sys.exit("e2e_1e7: build/cli/asm-ols missing (run make harness)")
    path = Path("/tmp/e2e_1e7.olsb")
    print(f"[1/3] streaming {N:,} x {Q+1} .olsb -> {path} ...")
    gen = stream_olsb(path)
    print(f"      done in {gen['gen_s']:.1f}s ({gen['file_MB']:.0f} MB)")

    print("[2/3] asm-ols fit --json ...")
    t0 = time.perf_counter()
    r = subprocess.run([str(CLI), "fit", str(path), "--target", "y",
                        "--json"], capture_output=True, text=True, check=True)
    fit_s = time.perf_counter() - t0
    fit = json.loads(r.stdout)

    print("[3/3] numpy normal-equations reference (streamed) ...")
    ref = numpy_reference(path)

    ours = np.array([fit["intercept_value"]] +
                    [fit["coefficients"][f"x{j+1}"] for j in range(Q)])
    refb = np.array(ref["beta"])
    abs_d = np.abs(ours - refb)
    rel_d = abs_d / np.maximum(np.abs(refb), 1e-12)
    # bounds (see docstring): rel on slopes, abs on intercept
    slope_ok = bool(np.all(rel_d[1:] <= 5e-10))
    icept_ok = bool(abs_d[0] <= 1e-8)
    sig_ok = abs(fit["sigma2"] - ref["sigma2"]) / ref["sigma2"] <= 1e-8
    r2_ok = abs(fit["r2"] - ref["r2"]) <= 1e-10
    ok = slope_ok and icept_ok and sig_ok and r2_ok

    summary = {
        "schema": "e2e-1e7",
        "host": "sandbox 2-vCPU quota KVM (results/machine.json)",
        "n": N, "features": Q, "seed": SEED, "sigma": SIGMA,
        "gen_s": round(gen["gen_s"], 2),
        "asm_fit_s": round(fit_s, 3),
        "numpy_ref_s": round(ref["ref_s"], 2),
        "max_abs_diff": float(abs_d.max()),
        "max_rel_diff_slopes": float(rel_d[1:].max()),
        "intercept_abs_diff": float(abs_d[0]),
        "sigma2_rel_diff": float(abs(fit["sigma2"] - ref["sigma2"]) / ref["sigma2"]),
        "r2_abs_diff": float(abs(fit["r2"] - ref["r2"])),
        "bounds": {"rel_slopes": 5e-10, "abs_intercept": 1e-8,
                   "rel_sigma2": 1e-8, "abs_r2": 1e-10},
        "pass": bool(ok),
        "clause_ok": {"slopes": slope_ok, "intercept": icept_ok,
                      "sigma2": sig_ok, "r2": r2_ok},
        "method_asm": fit.get("method"),
    }
    OUT.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(("PASS" if ok else "FAIL"), "->", OUT)
    path.unlink(missing_ok=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
