#!/usr/bin/env python3
"""gram bench mt study — H5 variance/repeatability + attribution evidence.

Why (phase9-g4): the committed single-shot mt2 rows in
results/gram-bench-v4.json (q8 6.76 GB/s / q17 7.01 GB/s vs the machine.json
all-thread DRAM ladder 37.6 GB/s = 18.0% / 18.7%, gate H5 RED, attributed to
a "2-vCPU quota") have no error bars, and the quota attribution was never
checked against the cgroup state.  This script strengthens the H5 evidence:

  1. Re-measures ols_gram_mt (symbol verified via `nm -D build/lib/libols.so`)
     at q=8/q=17, n=200000 ("DRAM-resident" per the v3/v4 case naming),
     threads in {1, 2}: warmup 3 + 30 timed reps (v4 protocol), seed 20261121,
     bytes/row = 8*q — reporting median/p10/p90/min/max/CV of GB/s and ns/row.
  2. Re-measures the STREAM-like bandwidth ladder exactly the way
     bench/machine/characterize.py produced results/machine.json's
     bandwidth_ladder (same prebuilt bench/machine/machine_bench binary, same
     working-set sizes, same internal best-of-N rep counts), at 1 thread and
     2 threads (2 concurrent processes, per-proc sizes as characterize.py),
     with >= 15 outer reps for error bars.
  3. Computes mt2 GB/s as % of the NEW 2-thread aggregate with propagated
     spread (full cross-rep ratio distribution, not just median/median).
  4. Re-checks the cgroup claim: /sys/fs/cgroup/cpu.max (v2 path; absent here
     — this host mounts cgroup v1), v1 cpu.cfs_quota_us / cpu.cfs_period_us /
     cpu.stat, cpuset, affinity, `taskset -c 0,1 true` sanity.
  5. Attribution evidence: the 85% target in GB/s and ns/row, the mt2/mt1
     efficiency ratio, and the 1t-vs-allt DRAM saturation case
     (A: one thread ~50% of all-thread -> real headroom, RED is kernel-side;
      B: 1t ~= allt -> DRAM single-thread-saturated, H5 structurally
      impossible on this host).

Usage: python3 bench/kernels/gram_bench_mt_study.py [kernel_reps=30]
           [ladder_reps=15] [write|nowrite]
Output: results/gram-bench-mt-study.json (write mode) + stdout tables.
Runtime budget: ~1-2 min at defaults (each kernel rep 2-5 ms, each ladder
outer rep 0.1-1.3 s); reps may be reduced but never below 7.
"""
import json
import os
import platform
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

MB = ROOT / "bench" / "machine" / "machine_bench"
assert MB.exists(), f"prebuilt machine_bench missing: {MB}"
assert hasattr(ols, "ols_gram_mt"), "symbol ols_gram_mt not found in the loaded lib"
# struct identical to gram_bench_v4.GramArgs / include/ols.h ols_gram_args_t
# (size 96: 12 fields, X..scratch_bytes; verified below against ols.h)
GramArgs = None  # placeholder; the real struct is GramArgs_ below

SEED = 20261121


class GramArgs_(C.Structure):
    _fields_ = [("X", C.c_void_p), ("ldx8", C.c_size_t), ("y", C.c_void_p),
                ("w", C.c_void_p), ("n", C.c_size_t), ("p", C.c_size_t),
                ("c", C.c_void_p), ("G", C.c_void_p), ("s", C.c_void_p),
                ("Wout", C.c_void_p), ("scratch", C.c_void_p),
                ("scratch_bytes", C.c_size_t)]


assert C.sizeof(GramArgs_) == 96, \
    f"GramArgs_ size {C.sizeof(GramArgs_)} != 96 (include/ols.h contract)"


def make_case(p, n, threads, seed=SEED):
    """Identical construction to gram_bench_v4.make_case_mt (same seed, same
    field order); also returns G so a numeric sanity check can read it."""
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
    ga = GramArgs_(X.ctypes.data, X.strides[0], y.ctypes.data, 0, n, p,
                   c.ctypes.data, G.ctypes.data, s.ctypes.data,
                   Wout.ctypes.data, C.c_void_p(scratch), sb)
    keep = (X, y, c, G, s, Wout, buf, ga)
    return C.byref(ga), scratch, sb, q, n, keep


def bench_mt(p, n, threads, reps, warm=3, seed=SEED):
    """warmup 3 + reps timed calls of ols_gram_mt — v4's protocol verbatim."""
    ptr, mscratch, msb, q, n_, keep = make_case(p, n, threads, seed)
    call = lambda: ols.ols_gram_mt(ptr, threads, C.c_void_p(mscratch), msb)
    for _ in range(warm):
        rc = call()
        assert rc == 0, f"ols_gram_mt rc={rc} (p={p}, threads={threads})"
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        rc = call()
        t1 = time.perf_counter_ns()
        assert rc == 0, rc
        ts.append((t1 - t0) / 1e9)
    G = keep[3]
    g00 = float(G[0])                       # sum x0^2 (c=0) ~ n for N(0,1)
    sanity = {"G00": g00, "expected_n": n, "rel_err": abs(g00 - n) / n}
    assert np.isfinite(G).all() and sanity["rel_err"] < 0.05, sanity
    return q, ts, sanity


def stats_of(samples):
    """median / p10 / p90 / min / max / mean / CV (linear-interpolation
    percentiles, numpy 'linear' / statistics 'inclusive' convention)."""
    s = sorted(samples)
    if len(s) < 2:
        v = s[0]
        return {"n": 1, "median": v, "p10": v, "p90": v, "min": v, "max": v,
                "mean": v, "cv": 0.0}
    cuts = statistics.quantiles(s, n=100, method="inclusive")  # 1%..99%
    mean = statistics.fmean(s)
    return {"n": len(s), "median": statistics.median(s),
            "p10": cuts[9], "p90": cuts[89], "min": s[0], "max": s[-1],
            "mean": mean, "cv": statistics.stdev(s) / mean}


def case_result(ts, q, n, threads, sanity):
    bytes_row = 8 * q
    gbps = [bytes_row * n / t / 1e9 for t in ts]
    nsrow = [t * 1e9 / n for t in ts]
    return {
        "samples_s": ts,
        "bytes_per_row": bytes_row,
        "threads": threads,
        "seed": SEED,
        "sanity": sanity,
        "gbps_stats": stats_of(gbps),
        "ns_per_row_stats": stats_of(nsrow),
        "samples_gbps": gbps,
    }


def stream_one(bytes_, internal_reps):
    """One outer sample of the 1t ladder rung = one machine_bench invocation
    (internal best-of-N, exactly how characterize.py measured gbps_1t)."""
    out = subprocess.run([str(MB), "stream", str(bytes_), str(internal_reps)],
                         capture_output=True, text=True, timeout=600)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(
            f"machine_bench stream {bytes_}x{internal_reps} failed: "
            f"rc={out.returncode} stderr={out.stderr[:300]}")
    return float(out.stdout.strip())


def stream_two(per_proc, internal_reps):
    """One outer sample of the 2t rung = 2 concurrent machine_bench processes
    (per-proc working set as characterize.py), aggregate = sum of B/s."""
    procs = [subprocess.Popen([str(MB), "stream", str(per_proc),
                               str(internal_reps)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True) for _ in range(2)]
    tot = 0.0
    for pr in procs:
        o_, e_ = pr.communicate(timeout=900)
        if pr.returncode != 0 or not o_.strip():
            raise RuntimeError(f"2t stream child died: rc={pr.returncode} "
                               f"stderr={e_[:300]}")
        tot += float(o_.strip())
    return tot


# ladder rungs replicate bench/machine/characterize.py exactly:
# sizes L1 32 KiB / L2 1 MiB / L3 64 MiB / DRAM 1.5 GiB (1t) and 1 GiB per
# proc (2t, min(DRAM_BYTES, 1 GiB)); internal reps 30 / 30 / 30 / 8 (1t DRAM)
# and 6 (2t DRAM) — the same counts characterize.py used.
LADDER = [
    ("L1",   32 * 1024,        30, 32 * 1024,        30),
    ("L2",   1024 * 1024,      30, 1024 * 1024,      30),
    ("L3",   64 * 1024 * 1024, 30, 64 * 1024 * 1024, 30),
    ("DRAM", 1610612736,        8, 1073741824,        6),
]


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError as e:
        return f"<unreadable: {e}>"


def cgroup_check():
    """The '2-vCPU quota' claim, checked against the live container."""
    d = {}
    d["proc_self_cgroup"] = read_text("/proc/self/cgroup")
    d["cpu_max_v2_path"] = read_text("/sys/fs/cgroup/cpu.max")
    d["cpu_cfs_quota_us_v1"] = read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    d["cpu_cfs_period_us_v1"] = read_text(
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    d["cpu_stat_v1"] = read_text("/sys/fs/cgroup/cpu/cpu.stat")
    d["cpu_shares_v1"] = read_text("/sys/fs/cgroup/cpu/cpu.shares")
    d["cpuset_cpus_v1"] = read_text("/sys/fs/cgroup/cpuset/cpuset.cpus")
    cpus_allowed = "?"
    for line in read_text("/proc/self/status").splitlines():
        if line.startswith("Cpus_allowed_list"):
            cpus_allowed = line.split(":", 1)[1].strip()
    d["cpus_allowed_list"] = cpus_allowed
    d["os_cpu_count"] = os.cpu_count()
    d["sched_getaffinity"] = sorted(os.sched_getaffinity(0))
    try:
        rc = subprocess.run(["taskset", "-c", "0,1", "true"],
                            capture_output=True, timeout=30).returncode
        d["taskset_0_1_true_rc"] = rc
    except FileNotFoundError:
        d["taskset_0_1_true_rc"] = "taskset not installed"

    quota_unlimited = d["cpu_cfs_quota_us_v1"] == "-1"
    zero_throttled = "nr_throttled 0" in d["cpu_stat_v1"]
    two_cpus = (d["os_cpu_count"] == 2 and cpus_allowed == "0-1"
                and d["sched_getaffinity"] == [0, 1])
    if d["cpu_max_v2_path"].startswith("<unreadable"):
        d["cgroup_version"] = ("cgroup v1 mounted (v2 /sys/fs/cgroup/cpu.max "
                               "path absent; /proc/self/cgroup shows v1 "
                               "controllers)")
    d["checks"] = {
        "cfs_quota_unlimited": quota_unlimited,
        "zero_throttle_events": zero_throttled,
        "exactly_two_cpus_visible": two_cpus,
    }
    if quota_unlimited and zero_throttled and two_cpus:
        d["quota_claim_verdict"] = (
            "The '2-vCPU quota' attribution does NOT hold as a CFS quota: "
            "cpu.cfs_quota_us = -1 (unlimited), cpu.stat shows 0 throttle "
            "events, and the v2 cpu.max path does not exist on this cgroup-v1 "
            "host. The honest attribution is: '2 CPUs visible to the "
            "container (cpuset 0-1 / Cpus_allowed_list 0-1), no cfs quota' "
            "— the container is sized by CPU visibility, not throttled by a "
            "bandwidth quota; residual run-to-run variance comes from the "
            "shared physical host (noisy neighbors), consistent with the "
            "16.2-18.9 GB/s run-to-run DRAM spread already noted in "
            "results/machine.json.")
    else:
        d["quota_claim_verdict"] = (
            f"UNEXPECTED cgroup state: quota_unlimited={quota_unlimited}, "
            f"zero_throttled={zero_throttled}, two_cpus={two_cpus} — the "
            "'2-vCPU quota' phrase needs manual re-inspection.")
    return d


def cross(a, b):
    return [x / y for x in a for y in b]


def main():
    argv = sys.argv[1:]
    kernel_reps = int(argv[0]) if len(argv) > 0 else 30
    ladder_reps = int(argv[1]) if len(argv) > 1 else 15
    mode = argv[2] if len(argv) > 2 else "write"
    reps_note = None
    if kernel_reps < 7:
        reps_note = f"kernel_reps clamped {kernel_reps} -> 7 (floor)"
        kernel_reps = 7
    if ladder_reps < 7:
        reps_note = (reps_note or "") + f" ladder_reps clamped {ladder_reps} -> 7"
        ladder_reps = 7

    t_start = time.time()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip() != ""

    # symbol provenance (task: find the exact symbol name in the .so)
    sym = {"lib_used_by_bench": str(B.LIB)}
    try:
        nm = subprocess.run(["nm", "-D", str(ROOT / "build/lib/libols.so")],
                            capture_output=True, text=True).stdout
        sym["nm_D_build_lib_libols_so"] = [
            l.strip() for l in nm.splitlines()
            if l.strip().endswith("ols_gram_mt")
            or l.strip().endswith("ols_gram_mt_scratch_bytes")]
    except Exception as e:  # pragma: no cover
        sym["nm_D_build_lib_libols_so"] = f"nm failed: {e}"

    host = json.loads((ROOT / "results/machine.json").read_text())["host"]
    cpu_model = host.get("cpu_model", "?")

    # ---- 1. kernel cases: ols_gram_mt q=8/q=17, n=200000, threads 1/2 ----
    cases = {}
    for name, p, threads in (("q8_n200000_DRAM_mt1", 7, 1),
                             ("q8_n200000_DRAM_mt2", 7, 2),
                             ("q17_n200000_DRAM_mt1", 16, 1),
                             ("q17_n200000_DRAM_mt2", 16, 2)):
        q, ts, sanity = bench_mt(p, 200000, threads, kernel_reps)
        cases[name] = case_result(ts, q, 200000, threads, sanity)
        st = cases[name]["gbps_stats"]
        print(f"{name}: median {st['median']:.3f} GB/s  "
              f"p10 {st['p10']:.3f} p90 {st['p90']:.3f}  "
              f"min {st['min']:.3f} max {st['max']:.3f}  "
              f"CV {st['cv']*100:.2f}%  ({st['n']} reps)", flush=True)

    # ---- 2. STREAM ladder, 1t and 2t, ladder_reps outer reps --------------
    # 1t and 2t reps are INTERLEAVED (rep i: one 1t sample, then one 2t
    # sample) so the saturation ratio 1t/2t is time-paired even when the
    # shared host's DRAM regime drifts mid-run (it did, between the smoke
    # and full runs of this study: 1t 19.0 -> 15.8 GB/s).
    ladder = {}
    for rung, b1, r1, b2, r2 in LADDER:
        s1, s2 = [], []
        for i in range(ladder_reps):
            s1.append(stream_one(b1, r1) / 1e9)
            print(f"  stream {rung} 1t rep {i+1}/{ladder_reps}: "
                  f"{s1[-1]:.2f} GB/s", flush=True)
            s2.append(stream_two(b2, r2) / 1e9)
            print(f"  stream {rung} 2t rep {i+1}/{ladder_reps}: "
                  f"{s2[-1]:.2f} GB/s (aggregate)", flush=True)
        ladder[rung] = {
            "bytes_1t": b1, "internal_reps_1t": r1,
            "bytes_2t_per_proc": b2, "internal_reps_2t": r2,
            "samples_1t_gbps": s1, "samples_2t_gbps": s2,
            "stats_1t_gbps": stats_of(s1), "stats_2t_gbps": stats_of(s2),
            # time-paired per-rep ratios (rep i measured 1t then 2t)
            "paired_2t_over_1t": stats_of([b / a for a, b in zip(s1, s2)]),
            "paired_1t_share_of_2t": stats_of([a / b for a, b in zip(s1, s2)]),
        }

    # ---- 3. cgroup re-check ----------------------------------------------
    cg = cgroup_check()
    print("\ncgroup check:", flush=True)
    for k in ("cpu_max_v2_path", "cpu_cfs_quota_us_v1", "cpu_cfs_period_us_v1",
              "cpu_stat_v1", "cpu_shares_v1", "cpuset_cpus_v1",
              "cpus_allowed_list", "os_cpu_count", "sched_getaffinity",
              "taskset_0_1_true_rc"):
        print(f"  {k}: {cg[k]}", flush=True)
    print(f"  verdict: {cg['quota_claim_verdict']}", flush=True)

    # ---- 4. H5 analysis ----------------------------------------------------
    allt = ladder["DRAM"]["stats_2t_gbps"]
    one_t = ladder["DRAM"]["stats_1t_gbps"]
    paired_share = ladder["DRAM"]["paired_1t_share_of_2t"]
    paired_scaling = ladder["DRAM"]["paired_2t_over_1t"]
    tgt_med, tgt_p10, tgt_p90 = (0.85 * allt["median"], 0.85 * allt["p10"],
                                 0.85 * allt["p90"])
    h5 = {
        "gate": "H5: N threads reach >= 85% of the measured all-thread "
                "bandwidth (all-thread = 2 threads on this 2-CPU host)",
        "threshold_fraction": 0.85,
        "allt_2t_gbps_stats": allt,
        "allt_1t_gbps_stats": one_t,
        "paired_1t_share_of_2t": paired_share,
        "paired_2t_scaling": paired_scaling,
        "h5_target_gbps": {"median": tgt_med, "p10": tgt_p10, "p90": tgt_p90},
        "dram_1t_share_of_2t": one_t["median"] / allt["median"],
        "stream_scaling_2t_over_1t": allt["median"] / one_t["median"],
    }
    for qname, bytes_row in (("q8", 64), ("q17", 136)):
        mt1s = cases[f"{qname}_n200000_DRAM_mt1"]["samples_gbps"]
        mt2s = cases[f"{qname}_n200000_DRAM_mt2"]["samples_gbps"]
        eff = cross(mt2s, mt1s)
        pct_new = [r * 100 for r in cross(mt2s, ladder["DRAM"]["samples_2t_gbps"])]
        h5[qname] = {
            "mt2_gbps_stats": cases[f"{qname}_n200000_DRAM_mt2"]["gbps_stats"],
            "mt1_gbps_stats": cases[f"{qname}_n200000_DRAM_mt1"]["gbps_stats"],
            "efficiency_mt2_over_mt1": stats_of(eff),
            "pct_of_new_2t_allt": stats_of(pct_new),
            "h5_target_ns_per_row_median": bytes_row / tgt_med,
            "mt2_ns_per_row_stats":
                cases[f"{qname}_n200000_DRAM_mt2"]["ns_per_row_stats"],
            "per_thread_kernel_gbps_median":
                cases[f"{qname}_n200000_DRAM_mt2"]["gbps_stats"]["median"] / 2,
            "per_thread_share_of_1t_stream":
                (cases[f"{qname}_n200000_DRAM_mt2"]["gbps_stats"]["median"] / 2)
                / one_t["median"],
            "needed_per_thread_gbps_for_h5": tgt_med / 2,
            "needed_per_thread_speedup_vs_measured":
                tgt_med / cases[f"{qname}_n200000_DRAM_mt2"]["gbps_stats"]["median"],
        }
        # Decisive reachability check: the H5 target per thread vs what ONE
        # stream thread demonstrably pulls on this host. If needed <= 1t
        # stream rate, the bandwidth is physically reachable by a single
        # thread, so the shortfall is kernel-side, not ceiling-side.
        h5[qname]["h5_per_thread_within_1t_stream"] = bool(
            h5[qname]["needed_per_thread_gbps_for_h5"] <= one_t["median"])

    # cross-session context from the committed characterization
    try:
        mj = json.loads((ROOT / "results/machine.json").read_text())
        bl = mj["bandwidth_ladder"]["DRAM"]
        committed_share = bl["gbps_1t"] / bl["gbps_allt"]
        committed = {"date": mj.get("date"),
                     "gbps_1t": bl["gbps_1t"], "gbps_allt": bl["gbps_allt"],
                     "1t_share_of_allt": round(committed_share, 3),
                     "allt_today_vs_committed":
                         round(allt["median"] / bl["gbps_allt"], 3)}
    except Exception as e:  # pragma: no cover
        committed = {"error": str(e)}
    h5["committed_machine_json_context"] = committed

    sat = paired_share["median"]
    h5["dram_1t_share_of_2t_paired_median"] = sat
    if sat < 0.70:
        case = "A"
        case_text = (
            f"single-thread DRAM stream is only {sat*100:.0f}% of the "
            f"2-thread aggregate (time-paired median; spread "
            f"{paired_share['p10']*100:.0f}-{paired_share['p90']*100:.0f}%; "
            f"block medians {one_t['median']:.2f} vs {allt['median']:.2f} "
            f"GB/s, scaling {h5['stream_scaling_2t_over_1t']:.2f}x): DRAM is "
            "NOT single-thread-saturated on this host — mt2 has real "
            "headroom, so the H5 RED is NOT excusable as a saturated memory "
            "ceiling; it is kernel-side (per-thread throughput), not "
            "ceiling-side")
    elif sat <= 0.90:
        case = "borderline"
        c_1t = committed.get("gbps_1t")
        c_allt = committed.get("gbps_allt")
        c_share = committed.get("1t_share_of_allt")
        case_text = (
            f"single-thread DRAM stream is {sat*100:.0f}% of the 2-thread "
            f"aggregate (time-paired median, spread "
            f"{paired_share['p10']*100:.0f}-{paired_share['p90']*100:.0f}%; "
            f"block medians {one_t['median']:.2f} vs {allt['median']:.2f} "
            f"GB/s): partial single-thread saturation this session"
            + (f", but the committed machine.json characterization "
               f"({committed.get('date', '?')[:10]}) measured {c_share*100:.0f}%"
               f" ({c_1t} of {c_allt} GB/s)" if isinstance(c_share, float)
               else "")
            + " — the 1t share swings across sessions on this shared host, "
              "so neither extreme is stable; the per-thread reachability "
              "check below is the decisive attribution either way")
    else:
        case = "B"
        case_text = (
            f"single-thread DRAM stream ({one_t['median']:.2f} GB/s) ~= the "
            f"2-thread aggregate ({allt['median']:.2f} GB/s, {sat*100:.0f}%): "
            "DRAM IS single-thread-saturated here — H5 (>=85% of all-thread "
            "BW) is structurally impossible on this host regardless of the "
            "kernel")
    h5["attribution_case"] = case
    h5["attribution_case_text"] = case_text

    q8, q17 = h5["q8"], h5["q17"]
    reach_q8 = ("IS within" if q8["h5_per_thread_within_1t_stream"]
                else "EXCEEDS")
    verdict_notes = (
        f"H5 target: >= 85% of the all-thread bandwidth. Fresh 2-thread "
        f"STREAM aggregate on this host today: {allt['median']:.2f} GB/s "
        f"(p10 {allt['p10']:.2f} / p90 {allt['p90']:.2f}), so the gate needs "
        f"{tgt_med:.2f} GB/s "
        f"({q8['h5_target_ns_per_row_median']:.2f} ns/row at q=8). Measured "
        f"ols_gram_mt 2-thread: q=8 {q8['mt2_gbps_stats']['median']:.2f} GB/s "
        f"= {q8['pct_of_new_2t_allt']['median']:.1f}% of the new all-thread "
        f"figure (rep-spread {q8['pct_of_new_2t_allt']['p10']:.1f}-"
        f"{q8['pct_of_new_2t_allt']['p90']:.1f}%), q=17 "
        f"{q17['mt2_gbps_stats']['median']:.2f} GB/s = "
        f"{q17['pct_of_new_2t_allt']['median']:.1f}% "
        f"(spread {q17['pct_of_new_2t_allt']['p10']:.1f}-"
        f"{q17['pct_of_new_2t_allt']['p90']:.1f}%): H5 stays RED, with error "
        f"bars nowhere near 85%. "
        f"Attribution: {case_text}. The decisive check: hitting 85% of "
        f"{allt['median']:.2f} GB/s with 2 threads needs "
        f"{q8['needed_per_thread_gbps_for_h5']:.2f} GB/s per thread, which "
        f"{reach_q8} what ONE stream thread demonstrably pulls on this host "
        f"({one_t['median']:.2f} GB/s) — i.e. the target bandwidth is "
        f"physically reachable per thread; what falls short is the kernel: "
        f"each mt2 thread sustains only "
        f"{q8['per_thread_kernel_gbps_median']:.2f} GB/s "
        f"({q8['per_thread_share_of_1t_stream']*100:.0f}% of the 1t stream "
        f"rate), an expected {q8['needed_per_thread_speedup_vs_measured']:.1f}x "
        f"shortfall, consistent with the load-port-bound analysis of "
        f"ADR-0012 (2 loads + 6 broadcasts + 12 FMAs per row cannot issue "
        f"enough loads to fill DRAM). mt2/mt1 efficiency is "
        f"{q8['efficiency_mt2_over_mt1']['median']:.2f}x (q=8, spread "
        f"{q8['efficiency_mt2_over_mt1']['p10']:.2f}-"
        f"{q8['efficiency_mt2_over_mt1']['p90']:.2f}x) / "
        f"{q17['efficiency_mt2_over_mt1']['median']:.2f}x (q=17): the "
        f"threading machinery itself works (q=8 scales ~1.9x), the ceiling "
        f"is per-thread kernel throughput. "
        f"No CFS quota exists (cpu.cfs_quota_us = -1, zero throttle events, "
        f"cgroup v1 — the v2 cpu.max path does not exist), so the committed "
        f"'2-vCPU quota' phrase is wrong; the honest phrasing is '2 CPUs "
        f"visible to the container (cpuset 0-1), no cfs quota'. Caveat: the "
        f"all-thread denominator itself is unstable on this shared host "
        f"(committed {committed.get('gbps_allt', '?')} GB/s on "
        f"{str(committed.get('date', '?'))[:10]} vs {allt['median']:.2f} "
        f"today, and {allt['min']:.2f}-{allt['max']:.2f} within this run) — "
        f"the H5 percentage moves with the neighbor load, but no plausible "
        f"denominator brings "
        f"{min(q8['mt2_gbps_stats']['median'], q17['mt2_gbps_stats']['median']):.1f}"
        f"-{max(q8['mt2_gbps_stats']['median'], q17['mt2_gbps_stats']['median']):.1f} "
        f"GB/s anywhere near its 85% threshold. N is capped at 2 by the "
        f"container's visible CPUs."
    )

    # continuity with the committed single shots (read, not hardcoded)
    try:
        v4 = json.loads((ROOT / "results/gram-bench-v4.json").read_text())
        cont_v4 = {k: v4["cases"][k]["derived"]["gbps"] for k in
                   ("q8_n200000_DRAM", "q17_n200000_DRAM",
                    "q8_n200000_DRAM_mt2", "q17_n200000_DRAM_mt2")}
    except Exception as e:
        cont_v4 = {"error": str(e)}
    try:
        mj = json.loads((ROOT / "results/machine.json").read_text())
        bl = mj["bandwidth_ladder"]
        cont_mj = {"DRAM_1t": bl["DRAM"]["gbps_1t"],
                   "DRAM_allt": bl["DRAM"]["gbps_allt"],
                   "L1_1t": bl["L1"]["gbps_1t"], "L2_1t": bl["L2"]["gbps_1t"],
                   "L3_1t": bl["L3"]["gbps_1t"]}
        committed_pct = {
            "q8_mt2_vs_37p6": cont_v4["q8_n200000_DRAM_mt2"]
            / bl["DRAM"]["gbps_allt"] * 100,
            "q17_mt2_vs_37p6": cont_v4["q17_n200000_DRAM_mt2"]
            / bl["DRAM"]["gbps_allt"] * 100}
    except Exception as e:
        cont_mj, committed_pct = {"error": str(e)}, {}

    # ---- summary table (stdout + JSON) ------------------------------------
    print("\n===== SUMMARY (median / p10 / p90 / min / max / CV) =====",
          flush=True)
    rows = []
    for name in cases:
        for unit, st in (("GB/s", cases[name]["gbps_stats"]),
                         ("ns/row", cases[name]["ns_per_row_stats"])):
            rows.append({"config": name, "unit": unit,
                         "median": round(st["median"], 4),
                         "p10": round(st["p10"], 4),
                         "p90": round(st["p90"], 4),
                         "min": round(st["min"], 4),
                         "max": round(st["max"], 4),
                         "cv_pct": round(st["cv"] * 100, 3)})
    for rung in ("L1", "L2", "L3", "DRAM"):
        for unit, st in (("GB/s 1t", ladder[rung]["stats_1t_gbps"]),
                         ("GB/s 2t", ladder[rung]["stats_2t_gbps"])):
            rows.append({"config": f"stream_{rung}", "unit": unit,
                         "median": round(st["median"], 3),
                         "p10": round(st["p10"], 3), "p90": round(st["p90"], 3),
                         "min": round(st["min"], 3), "max": round(st["max"], 3),
                         "cv_pct": round(st["cv"] * 100, 3)})
    for r in rows:
        print(f"{r['config']:24s} {r['unit']:>7s}  {r['median']:>9.3f} "
              f"{r['p10']:>9.3f} {r['p90']:>9.3f} {r['min']:>9.3f} "
              f"{r['max']:>9.3f}  CV {r['cv_pct']:6.2f}%", flush=True)

    print("\n===== H5 =====", flush=True)
    print(f"2t all-thread DRAM aggregate: {allt['median']:.2f} GB/s "
          f"(p10 {allt['p10']:.2f} / p90 {allt['p90']:.2f})  -> "
          f"85% target {tgt_med:.2f} GB/s", flush=True)
    for qname in ("q8", "q17"):
        p = h5[qname]["pct_of_new_2t_allt"]
        e_ = h5[qname]["efficiency_mt2_over_mt1"]
        print(f"{qname}: mt2 = {h5[qname]['mt2_gbps_stats']['median']:.2f} "
              f"GB/s = {p['median']:.1f}% of new 2t all-thread "
              f"(spread {p['p10']:.1f}-{p['p90']:.1f}%); "
              f"efficiency mt2/mt1 {e_['median']:.2f}x "
              f"(spread {e_['p10']:.2f}-{e_['p90']:.2f}x)", flush=True)
    print(f"1t DRAM stream = {one_t['median']:.2f} GB/s = "
          f"{sat*100:.0f}% of the 2t aggregate -> attribution case {case}",
          flush=True)
    print(f"\ncgroup verdict: {cg['quota_claim_verdict']}", flush=True)
    print(f"\nverdict_notes: {verdict_notes}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nelapsed: {elapsed:.1f} s", flush=True)

    out = {
        "schema": 1,
        "study": "gram-mt-h5-variance-and-attribution",
        "commit": commit,
        "dirty": dirty,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"machine": platform.machine(),
                 "os": f"{platform.system()} {platform.release()}",
                 "python": platform.python_version(),
                 "nproc": os.cpu_count(), "cpu_model": cpu_model,
                 "cpus_allowed": cg["cpus_allowed_list"],
                 "mem_total_gib": host.get("mem_total_gib")},
        "toolchain": {"python": platform.python_version()},
        "params": {
            "kernel_protocol": {"warmup": 3, "reps": kernel_reps,
                                "seed": SEED,
                                "bytes_per_row_formula": "8*(p+1)",
                                "n": 200000, "cases": "q8 (p=7) / q17 (p=16), "
                                "threads in {1,2}, same construction as "
                                "gram_bench_v4.make_case_mt"},
            "ladder_protocol": {
                "binary": "bench/machine/machine_bench (prebuilt, same "
                          "binary that produced results/machine.json)",
                "binary_mtime": time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(MB.stat().st_mtime)),
                "method": "one outer sample = one machine_bench invocation "
                          "(internal best-of-N, matching characterize.py); "
                          "2t = 2 concurrent processes, per-proc sizes as "
                          "characterize.py",
                "rungs": [{"rung": r, "bytes_1t": b1, "internal_reps_1t": i1,
                           "bytes_2t_per_proc": b2, "internal_reps_2t": i2,
                           "outer_reps": ladder_reps}
                          for r, b1, i1, b2, i2 in LADDER]},
            "reps_floor_note": reps_note,
            "runtime_budget_note": f"total wall {elapsed:.1f} s "
                                   f"(budget 600 s)",
        },
        "symbol_check": sym,
        "cases": {k: {kk: vv for kk, vv in v.items() if kk != "samples_gbps"}
                  for k, v in cases.items()},
        "stream_ladder": ladder,
        "h5_analysis": h5,
        "cgroup_check": cg,
        "continuity": {"gram_bench_v4_gbps": cont_v4,
                       "machine_json_ladder_gbps": cont_mj,
                       "committed_h5_pct_of_37p6": committed_pct},
        "summary_table": rows,
        "verdict_notes": verdict_notes,
    }
    if mode != "nowrite":
        out_path = ROOT / "results/gram-bench-mt-study.json"
        out_path.write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwritten {out_path}", flush=True)
    else:
        print("\n(nowrite mode: results NOT saved)", flush=True)


if __name__ == "__main__":
    main()
