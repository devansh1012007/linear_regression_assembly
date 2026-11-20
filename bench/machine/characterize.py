#!/usr/bin/env python3
"""Machine characterization driver -> results/machine.json.

Measures (simd-kernel-optimization §1):
  1. FMA peak vs accumulator count (1..16)
  2. Load throughput from L1 (ymm)
  3. Bandwidth ladder L1/L2/L3/DRAM, single- and all-threads
  4. Clock calibration (dependent-add chain rate vs TSC-derived time)
  5. Roofline crossover p* = peak_flops / (bw * 1/8)
Prediction was written in docs/perf-log.md BEFORE this benchmark ran.
"""
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "bench" / "machine"
RESULTS = ROOT / "results"
NPROC = os.cpu_count() or 2

# cap DRAM working set: PLAN.md says >= 8x L3 but RAM here is 4 GiB and L3
# (cloud, shared) reports 504 MiB -> 8x would exceed RAM. Capped at 1.5 GiB
# and recorded; the ADR-platform notes the deviation.
L3_BYTES = 504 * 1024 * 1024
DRAM_BYTES = min(8 * L3_BYTES, 1536 * 1024 * 1024)


def sh(*args, timeout=600):
    out = subprocess.run([str(a) for a in args], capture_output=True, text=True,
                         timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"{args} -> {out.returncode}: {out.stderr[:400]}")
    return out.stdout.strip()


def git_commit():
    try:
        sha = sh("git", "-C", ROOT, "rev-parse", "HEAD")
        dirty = bool(sh("git", "-C", ROOT, "status", "--porcelain"))
        return sha, dirty
    except Exception:
        return "unknown", True


def host_info():
    info = {"machine": platform.machine(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "nproc": NPROC}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        info["mem_total_gib"] = round(
            int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) / 1048576, 2)
        info["l3_mib_reported"] = 504
        flags = next(l for l in open("/proc/cpuinfo") if l.startswith("flags")).split()
        info["avx2"] = "avx2" in flags
        info["avx512"] = "avx512f" in flags
        info["fma"] = "fma" in flags
    except Exception as e:
        info["warn"] = str(e)
    return info


def main():
    bin_ = BENCH / "machine_bench"
    if not bin_.exists():
        subprocess.run(["gcc", "-O2", "-march=native", "-o", str(bin_),
                        str(BENCH / "machine_bench.c")], check=True)

    commit, dirty = git_commit()

    # 1. FMA peak vs accumulators
    fma = {}
    for nacc in range(1, 17):
        flops_s = float(sh(bin_, "fma", nacc))
        fma[nacc] = round(flops_s / 1e9, 2)          # GFLOP/s
        print(f"fma nacc={nacc:2d}: {fma[nacc]:8.2f} GFLOP/s", flush=True)
    peak_gflops = max(fma.values())
    best_nacc = max(fma, key=fma.get)

    # 2. load throughput (L1)
    load_bs = float(sh(bin_, "load"))
    print(f"load L1: {load_bs/1e9:.2f} GB/s", flush=True)

    # 3. bandwidth ladder (single thread)
    ladder = {}
    sizes = {"L1": 32 * 1024, "L2": 1024 * 1024, "L3": 64 * 1024 * 1024,
             "DRAM": DRAM_BYTES}
    for name, sz in sizes.items():
        reps = 30 if sz <= 64 * 1024 * 1024 else 8
        bs = float(sh(bin_, "stream", sz, reps))
        ladder[name] = {"bytes": sz, "gbps_1t": round(bs / 1e9, 2)}
        print(f"stream {name} ({sz/1048576:.0f} MiB): {bs/1e9:.2f} GB/s 1t", flush=True)

    # all-threads bandwidth (2 vCPU): concurrent processes, each working set
    # sized so NPROC buffers fit in the 4 GiB host
    per_proc = min(DRAM_BYTES, 1024 * 1024 * 1024)
    procs = []
    for i in range(NPROC):
        p = subprocess.Popen([str(bin_), "stream", str(per_proc), "6"],
                             stdout=subprocess.PIPE, text=True)
        procs.append(p)
    tot = 0.0
    for p in procs:
        out, _ = p.communicate(timeout=900)
        out = out.strip()
        if not out:
            raise RuntimeError("all-thread stream child died (OOM?)")
        tot += float(out)
    ladder["DRAM"]["gbps_allt"] = round(tot / 1e9, 2)
    print(f"stream DRAM all-{NPROC}t: {tot/1e9:.2f} GB/s", flush=True)

    # 4. clock calibration: depadd rate (1 op/cy presumed for add latency 1)
    dep_ops_s = float(sh(bin_, "depadd", 200_000_000))
    # TSC frequency via two wall readings around a depadd run is already in the
    # rate; core-clock estimate assumes 1 dependent add per cycle (true on all
    # modern x86: ALU add reg,reg latency 1).
    est_ghz = dep_ops_s / 1e9
    print(f"depadd rate: {dep_ops_s/1e9:.3f} Gops/s -> ~{est_ghz:.3f} GHz core clock", flush=True)

    # 5. roofline crossover p* (double precision, 8 B per element):
    #    flops/s = (p+1)/8 * B/s  ->  p* + 1 = 8 * peak_flops / B
    dram_bps = ladder["DRAM"]["gbps_1t"] * 1e9
    p_star = 8 * (peak_gflops * 1e9) / dram_bps - 1
    print(f"roofline p* (1 thread, DRAM): {p_star:.1f}", flush=True)

    machine = {
        "schema": 1,
        "commit": commit,
        "dirty": dirty,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": host_info(),
        "toolchain": {
            "gcc": sh("gcc", "-dumpversion"),
            "nasm": sh("nasm", "-v").split()[2] if shutil_which("nasm") else "missing",
        },
        "fma_peak_vs_accumulators_gflops": fma,
        "fma_peak_gflops": peak_gflops,
        "fma_best_nacc": best_nacc,
        "load_l1_gbps": round(load_bs / 1e9, 2),
        "bandwidth_ladder": ladder,
        "clock": {
            "depadd_gops_s": round(dep_ops_s / 1e9, 3),
            "core_clock_est_ghz": round(est_ghz, 3),
            "method": "dependent add chain, 1 op/cycle latency assumption",
        },
        "roofline": {
            "p_star_1t_dram": round(p_star, 1),
            "formula": "p* + 1 = 8 * peak_flops / dram_bw_1t",
            "note": "crossover between DRAM-bandwidth-bound and FMA-bound for "
                    "fused Gram pass (flops/byte = (p+1)/8)",
        },
        "constraints": {
            "dram_working_set_bytes": DRAM_BYTES,
            "note": "PLAN.md asks 8x L3; host RAM 4 GiB < 8x504 MiB, capped at 1.5 GiB (see ADR-0002)",
        },
    }

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / "machine.json"
    out_path.write_text(json.dumps(machine, indent=2) + "\n")
    print(f"\nwrote {out_path}")


def shutil_which(name):
    from shutil import which
    return which(name)


if __name__ == "__main__":
    main()
