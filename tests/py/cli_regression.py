#!/usr/bin/env python3
"""CLI regression tests for the four fuzzer-found bugs (G4 clause 3, 2026-11-26).

Locked-in regressions — each case here failed BEFORE the fix:
  1. quoted CSV header name (y,"x1",x2) -> coefficient key must be
     JSON-escaped, output must parse as JSON.
  2. overflow stats (1e308 values -> inf Gram -> nan sse) -> every numeric
     JSON field must be finite-or-null, never nan/inf tokens.
  3. --features value that is not a column ('--features --lambda' via
     getopt eating) -> must be rejected rc=2, never run on uninit X.
  4. mutated .olsb name with a raw 0x88 byte -> rejected at load rc=1,
     never emitted into JSON output.

Also: duplicate feature entries rejected; empty feature entry rejected.
Run as part of make verify via run_all.py (fast, deterministic).
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CLI = ROOT / "build" / "cli" / "asm-ols"
TMP = Path("/tmp/cli_regression")
FAILS: list[str] = []


def run(args, raw=False):
    p = subprocess.run([str(CLI)] + args, capture_output=True, timeout=60,
                       text=True, errors="replace")
    return p.returncode, p.stdout, p.stderr


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL {name} {detail}")


def main() -> int:
    if not CLI.exists():
        print("cli_regression: build/cli/asm-ols missing"); return 1
    TMP.mkdir(exist_ok=True)

    # --- 1. quoted header name must be escaped, JSON must parse ----------
    f1 = TMP / "quoted.csv"
    # row-varying data (first draft used a column-only formula -> every
    # row identical -> SINGULAR, caught by this suite itself)
    rows = ["y,\"x1\",x2,x3"]
    s = 7
    for r in range(30):
        vals = []
        for i in range(4):
            s = (s * 1103515245 + 12345 + i * 7919 + r) & 0x7FFFFFFF
            vals.append(f"{(s % 2000) / 100 - 10:.3f}")
        rows.append(",".join(vals))
    f1.write_text("\n".join(rows) + "\n")
    rc, out, err = run(["fit", str(f1), "--target", "y", "--json"])
    try:
        d = json.loads(out)
        ok1 = rc == 0 and "\"x1\"" in d["coefficients"]
    except json.JSONDecodeError:
        ok1 = False
    check("quoted-name JSON-escaped + parses", ok1, f"rc={rc} out={out[:80]!r}")

    # --- 2. overflow values (1e308 finite, Gram squares overflow) --------
    # The library refuses the design with OLS_E_NONFINITE/-6 before any
    # stats are computed (fits-in-double guard), so the CLI must exit 1
    # with the fit-failed message and NO stdout. (Earlier draft expected
    # NaN stats to flow through; they never do — the guard fires first.
    # json_num() remains defense-in-depth for any future NaN res field.)
    f2 = TMP / "overflow.csv"
    rows = ["y,x1,x2,x3"] + [
        ",".join(("1e308" if (i + r) % 7 == 0 else f"{r}.{i}")
                 for i in range(4))
        for r in range(40)]
    f2.write_text("\n".join(rows) + "\n")
    rc, out, err = run(["fit", str(f2), "--target", "y", "--json",
                        "--confint"])
    bad_tokens = any(t in out for t in ("nan", "inf", "-inf"))
    check("overflow input refused by library, no stdout",
          rc == 1 and out == "" and not bad_tokens and "failed" in err,
          f"rc={rc} out={out[:80]!r} err={err[:80]!r}")

    # --- 3. --features that resolves to no column -> rc 2 ----------------
    rc, out, err = run(["fit", str(f1), "--target", "y",
                        "--features", "--lambda", "--json"])
    check("unusable --features rejected rc=2",
          rc == 2 and "not a usable column" in err, f"rc={rc} err={err[:80]!r}")
    # duplicate entries rejected
    rc, out, err = run(["fit", str(f1), "--target", "y",
                        "--features", "x2,x2"])
    check("duplicate --features rejected rc=2", rc == 2, f"rc={rc}")
    # empty entry rejected
    rc, out, err = run(["fit", str(f1), "--target", "y",
                        "--features", "x2,,"])
    check("empty --features entry rejected rc=2", rc == 2, f"rc={rc}")

    # --- 4. .olsb with raw 0x88 in a name -> load-time rejection ---------
    good = f1.read_text()
    names = good.splitlines()[0].split(",")
    body = [[float(x) for x in ln.split(",")]
            for ln in good.splitlines()[1:]]
    name_block = b"y\x00x1\x88\x00x2\x00x3\x00"     # 0x88 inside second name
    data = b"".join(struct.pack("<4d", *row) for row in body)
    hdr = struct.pack("<IIQQQQQ", 0x4F4C5342, 1, len(body), 4,
                      48, 48 + len(name_block), 48 + len(name_block) + len(data))
    f4 = TMP / "badname.olsb"
    f4.write_bytes(hdr + name_block + data
                   + struct.pack("<I", zlib.crc32(data) & 0xFFFFFFFF))
    rc, out, err = run(["fit", str(f4), "--target", "y", "--json"])
    check("bad UTF-8 name rejected at load rc=1",
          rc == 1 and "bad column name" in err, f"rc={rc} err={err[:80]!r}")

    # --- sanity: a plain good file still fits (no regression) ------------
    rc, out, err = run(["fit", str(f1), "--target", "y", "--json",
                        "--confint", "--features", "x2,x3"])
    try:
        d = json.loads(out)
        sane = rc == 0 and set(d["coefficients"]) == {"x2", "x3"} \
               and "(intercept)" in d["p_values"]
    except json.JSONDecodeError:
        sane = False
    check("healthy path unaffected (features + confint + p_values)",
          sane, f"rc={rc}")

    print("cli_regression:", "ALL GREEN" if not FAILS else f"{len(FAILS)} FAILURES")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
