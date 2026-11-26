#!/usr/bin/env python3
"""G4 clause 4: freestanding CLI checks (build constraints + bitwise parity).

1. readelf: no dynamic section, GNU_STACK is NX (not RWE), static.
2. fit on a generated .olsb: JSON parses; every shared numeric field is
   BITWISE identical to the libc CLI's (same library objects, auto ISA).
3. binary size recorded to results/binary-size.json (DoD item 3: "binary
   size: generated from committed raw results").

Runs in `make verify` (fast: one small fit).
"""
from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FS = ROOT / "build" / "cli" / "asm-ols-fs"
LIBC_CLI = ROOT / "build" / "cli" / "asm-ols"
OUT = ROOT / "results" / "binary-size.json"
FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def readelf(*args):
    return subprocess.run(["readelf", *args, str(FS)],
                          capture_output=True, text=True, check=True).stdout


def make_dataset(path: Path, n=60, nc=4, seed=7):
    s = seed
    def rnd():
        nonlocal s
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        return (s % 2000) / 100 - 10
    names = b"y\x00x1\x00x2\x00x3\x00"[:0]  # rebuilt below
    colnames = ["y", "x1", "x2", "x3"][:nc]
    nb = b"".join(nm.encode() + b"\0" for nm in colnames)
    rows = []
    for i in range(n):
        x = [rnd() for _ in range(nc - 1)]
        y = 1.5 + 0.8 * x[0] - 0.3 * x[1] + 0.1 * x[2] + rnd() * 0.05
        rows.append([y] + x)
    data = b"".join(struct.pack(f"<{nc}d", *r) for r in rows)
    hdr = struct.pack("<IIQQQQQ", 0x4F4C5342, 1, n, nc, 48,
                      48 + len(nb), 48 + len(nb) + len(data))
    path.write_bytes(hdr + nb + data + struct.pack("<I", zlib.crc32(data)))


def main() -> int:
    if not FS.exists():
        print("freestanding_check: build/cli/asm-ols-fs missing"); return 1

    # 1. ELF constraints
    d = readelf("-d")
    check("no dynamic section", "There is no dynamic section" in d)
    ph = readelf("-W", "-l")
    m = re.search(r"GNU_STACK\s+(\S+)", ph)
    check("GNU_STACK present", m is not None)
    if m:
        check("GNU_STACK non-executable (RW, no E)", "RWE" not in m.group(0), m.group(0))
    check("static (no INTERP)",
          not any("INTERP" in ln for ln in ph.splitlines()))

    # 2. bitwise parity with the libc CLI
    ds = Path("/tmp/fs_check.olsb")
    make_dataset(ds)
    fs = subprocess.run([str(FS), "fit", str(ds), "--target", "y", "--json"],
                        capture_output=True, text=True, timeout=60)
    lc = subprocess.run([str(LIBC_CLI), "fit", str(ds), "--target", "y", "--json"],
                        capture_output=True, text=True, timeout=60)
    check("fs rc=0", fs.returncode == 0, fs.stderr[:100])
    check("libc rc=0", lc.returncode == 0, lc.stderr[:100])
    try:
        fa = json.loads(fs.stdout); fb = json.loads(lc.stdout)
        parses = True
    except json.JSONDecodeError:
        parses = False
        fa = fb = {}
    check("fs JSON parses", parses)

    def flat(d, pre=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, pre + k + "."))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[pre + k] = v
        return out

    fla, flb = flat(fa), flat(fb)
    common = set(fla) & set(flb)
    diffs = [k for k in common if fla[k] != flb[k]]
    check(f"bitwise parity on shared fields ({len(common)})", not diffs,
          f"differs: {diffs[:5]}")

    # 3. size artifact (DoD item 3)
    OUT.write_text(json.dumps({
        "schema": "binary-size",
        "asm-ols-fs_bytes": FS.stat().st_size,
        "asm-ols-fs_libc_free": True,
        "asm-ols_bytes": (LIBC_CLI.stat().st_size if LIBC_CLI.exists() else None),
        "note": "asm-ols-fs: -nostdlib -static, no dynamic section, NX stack, "
                "links the production CORE_OBJS + libgcc (q-float formatting)",
    }, indent=1))
    print(f"  ok   size artifact -> {OUT}")

    print("freestanding_check:", "ALL GREEN" if not FAILS else f"{len(FAILS)} FAILURES")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
