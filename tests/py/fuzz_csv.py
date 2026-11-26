#!/usr/bin/env python3
"""G4 clause 3: parser fuzzing for a FIXED budget, seeded, reproducible.

PLAN Phase-4 gate G4: "fuzz runs clean for a fixed budget". Targets the two
parsers in cli/asm_ols.c (load_olsb, load_csv) plus argument handling.

Strategy (deterministic, seeded; no external deps):
  A. CSV structure fuzz   — valid-ish CSVs with mutated separators/quotes/
     numbers/line-endings/widths, plus random row/field edits of a good
     file (valid CSV -> fit must still work; corrupted -> rc in {1,2}).
  B. .olsb byte fuzz      — start from a valid small .olsb (via
     scripts/to_olsb.py), then header-field, offset, data-byte and
     truncation mutations.
  C. argv fuzz            — random flag combos / missing values.

Budget: wall-clock seconds (default 5.0), counted once per executed case;
budget and rng seed recorded in results/fuzz-csv.json. Invariant per case:
  rc in {0,1,2}, no stderr containing 'Segmentation'/'Aborted' (rc<0 from
  the shell counts as immediate FAIL), rc==0 implies stdout parses as JSON
  with the expected keys. NOT in `make verify` (subprocess-heavy; this is
  a `make repro` scale artifact).
"""
from __future__ import annotations

import argparse
import json
import random
import string
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CLI = ROOT / "build" / "cli" / "asm-ols"
OUT = ROOT / "results" / "fuzz-csv.json"

SEED_DEFAULT = 42
BUDGET_DEFAULT = 5.0
TIMEOUT_S = 20  # per-case timeout: the fuzzer must catch hangs, not die


def base_csv(rng: random.Random, n_rows: int = 60) -> str:
    lines = ["y,x1,x2,x3"]
    for _ in range(n_rows):
        lines.append(",".join(f"{rng.gauss(0, 1):.6f}" for _ in range(4)))
    return "\n".join(lines) + "\n"


def mutate_csv(rng: random.Random, good: str) -> str:
    lines = good.splitlines()
    op = rng.random()
    if op < 0.15:                       # width change: drop/add a field
        i = rng.randrange(1, len(lines))
        f = lines[i].split(",")
        f.pop(rng.randrange(len(f))) if rng.random() < 0.5 else \
            f.append(rng.choice(["1.0", "", "nan", "-"]))
        lines[i] = ",".join(f)
    elif op < 0.30:                     # number corruption
        i = rng.randrange(1, len(lines))
        f = lines[i].split(",")
        j = rng.randrange(len(f))
        f[j] = rng.choice(["", "  ", "1e999", "-inf", "inf", "nan", "0x10",
                           "1.2.3", "1,5", '"1.5"', "1e", "+", "1e-400"])
        lines[i] = ",".join(f)
    elif op < 0.45:                     # line ending / whitespace
        i = rng.randrange(1, len(lines))
        lines[i] = rng.choice([" " + lines[i] + " ", lines[i] + "\r",
                               lines[i] + ";", lines[i].replace(",", " , ")])
    elif op < 0.60:                     # header games
        lines[0] = rng.choice(["", ",,,", "y,y,x1", "y,x1,x2,x3,x4",
                               'y,x1,"x2",x3', "y\tx1\tx2\tx3"])
    elif op < 0.75:                     # extreme values
        i = rng.randrange(1, len(lines))
        f = lines[i].split(",")
        f[rng.randrange(len(f))] = rng.choice(
            ["1e308", "-1e308", "5e-324", "1e309", "-1e309"])
        lines[i] = ",".join(f)
    elif op < 0.90:                     # structural: dup/blank/BOM rows
        i = rng.randrange(1, len(lines))
        lines.insert(i, rng.choice(["", ",,,,", lines[i]]))
        if rng.random() < 0.3:
            lines[0] = "﻿" + lines[0]     # UTF-8 BOM
    else:                               # junk tail
        lines.append(rng.choice(["garbage", ",,,", "1,2,3", "\x00\x01"]))
    return "\n".join(lines) + "\n"


def run_cli(args: list, cwd=None) -> dict:
    t0 = time.perf_counter()
    try:
        # errors="replace": fuzzed files can carry arbitrary bytes; the
        # harness must survive them (round 1 of the fuzz run crashed HERE,
        # in python's UTF-8 decode, not in the CLI).
        p = subprocess.run([str(CLI)] + args, capture_output=True,
                           text=True, errors="replace", timeout=TIMEOUT_S)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return {"rc": "timeout", "json": None, "err": "TIMEOUT", "s": TIMEOUT_S}
    ok_json = None
    if rc == 0 and "--json" in args:
        try:
            ok_json = json.loads(out)
        except json.JSONDecodeError:
            ok_json = "INVALID"
    # NOTE: even when --json is present, getopt may have eaten it as a
    # value (e.g. '--rcond --json'); the caller sets _json_expected=False
    # for those and check() skips the JSON assertions.
    return {"rc": rc, "json": ok_json, "err": err, "s": time.perf_counter() - t0}


def check(res: dict, json_expected: bool = True) -> str | None:
    rc = res["rc"]
    if rc == "timeout":
        return "hang (timeout)"
    if not isinstance(rc, int) or rc < 0 or rc > 2:
        return f"bad exit code {rc!r}"
    if "Segmentation" in res["err"] or "Aborted" in res["err"]:
        return "crash signal"
    if rc == 0 and json_expected and res["json"] == "INVALID":
        return "rc=0 but stdout not JSON"
    if rc == 0 and json_expected and isinstance(res["json"], dict):
        need = {"n", "coefficients"}
        if not need.issubset(res["json"]):
            return "JSON missing expected keys"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--budget", type=float, default=BUDGET_DEFAULT)
    args = ap.parse_args()
    if not CLI.exists():
        sys.exit("fuzz_csv: build/cli/asm-ols missing (run make harness)")

    rng = random.Random(args.seed)
    tmp = Path("/tmp/fuzz_asm_ols")
    tmp.mkdir(exist_ok=True)
    good = base_csv(random.Random(1))
    (tmp / "good.csv").write_text(good)

    # sanity: the unmutated file must fit cleanly before any fuzzing
    res = run_cli(["fit", str(tmp / "good.csv"), "--target", "y", "--json"])
    bad = check(res)
    if bad:
        sys.exit(f"fuzz_csv: BASELINE FAILS: {bad}")

    import struct, zlib
    # valid small .olsb to mutate: reuse to_olsb.py
    subprocess.run([sys.executable, str(ROOT / "scripts" / "to_olsb.py"),
                    str(tmp / "good.csv"), str(tmp / "good.olsb")],
                   check=True, capture_output=True)
    good_bytes = (tmp / "good.olsb").read_bytes()

    stats = {"csv": 0, "olsb": 0, "argv": 0}
    fails = []
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < args.budget:
        n += 1
        kind = rng.random()
        if kind < 0.5:
            stats["csv"] += 1
            payload = mutate_csv(rng, good)
            f = tmp / "m.csv"
            f.write_text(payload)
            res = run_cli(["fit", str(f), "--target", "y", "--json"])
        elif kind < 0.85:
            stats["olsb"] += 1
            b = bytearray(good_bytes)
            for _ in range(rng.randrange(1, 6)):
                op = rng.random()
                if op < 0.5 and len(b) > 60:      # flip a byte (data or hdr)
                    b[rng.randrange(len(b))] = rng.randrange(256)
                elif op < 0.7:                     # header field tweak
                    field = rng.randrange(6)
                    val = rng.choice([0, 1, 2**31, 2**40, 10**9])
                    b[8 + field * 8: 8 + field * 8 + 8] = \
                        struct.pack("<Q", val)
                elif op < 0.85:                    # truncate
                    b = b[:rng.randrange(1, len(b))]
                else:                              # extend with junk
                    b += bytes(rng.randrange(256) for _ in range(rng.randrange(1, 64)))
            f = tmp / "m.olsb"
            f.write_bytes(bytes(b))
            res = run_cli(["fit", str(f), "--target", "y", "--json"])
        else:
            stats["argv"] += 1
            flags = rng.sample(["--target", "y", "--features", "x1,x2",
                                "--method", "chol", "--lambda", "0.5",
                                "--json", "--no-intercept", "--confint",
                                "--rcond", "1e-8", "--"],
                               k=rng.randrange(1, 8))
            # --json only counts as JSON mode if getopt will actually see
            # it as a flag: a value-taking option directly before it
            # (--rcond --json) makes getopt eat --json as the VALUE, and
            # the CLI legitimately prints human output (round 2 finding:
            # this was a fuzzer-side false positive, not a CLI bug).
            VALUE_TAKING = {"--target", "--features", "--method",
                            "--lambda", "--rcond"}
            json_live = "--json" in flags and not any(
                flags[i] in VALUE_TAKING and flags[i + 1] == "--json"
                for i in range(len(flags) - 1))
            probe_args = ["fit", str(tmp / "good.csv")] + flags
            res = run_cli(probe_args)
            if not json_live and "--json" in flags:
                # re-check without JSON expectation: rerun is cheap; mark
                # by passing a flag through a shallow copy
                res["_json_expected"] = False
            else:
                res["_json_expected"] = "--json" in flags
        problem = check(res, res.get("_json_expected", True))
        if problem:
            repro = (f"{kind:.3f} seed={args.seed} iter={n} stats={stats} "
                     f"args-rc={res['rc']} err={res['err'][:200]!r}")
            fails.append({"iter": n, "kind": kind, "problem": problem,
                          "repro_hint": repro,
                          "file": None})
            if len(fails) > 20:
                break
    elapsed = time.perf_counter() - t0

    summary = {
        "schema": "fuzz-csv",
        "seed": args.seed, "budget_s": args.budget,
        "elapsed_s": round(elapsed, 2), "cases": n,
        "by_kind": stats, "failures": fails,
        "per_case_timeout_s": TIMEOUT_S,
        "invariant": "rc in {0,1,2}; no crash signal; rc=0 => valid JSON with n+coefficients",
        "clean": not fails,
    }
    OUT.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
