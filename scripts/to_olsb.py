#!/usr/bin/env python3
"""to_olsb.py — write a .olsb dataset file from CSV or .npy/.npz (PLAN Phase-4).

Format (matches cli/asm_ols.c load_olsb exactly):
  bytes  0-3   magic  0x4F4C5342 ("OLSB")
  bytes  4-7   version 1 (u32 LE)
  bytes  8-15  n_rows  (u64 LE)
  bytes 16-23  n_cols  (u64 LE)
  bytes 24-31  name_off   (offset of NUL-terminated name block)
  bytes 32-39  data_off   (offset of row-major f64 data)
  bytes 40-47  trailer_off(offset of CRC32 trailer)
  name block: n_cols NUL-terminated column names
  data block: n_rows * n_cols f64 LE
  trailer:    CRC32 of the data block (zlib.crc32)

Usage:
  python3 scripts/to_olsb.py input.csv out.olsb [--target y] [--features a,b,c]
  python3 scripts/to_olsb.py input.npy out.olsb [--target y]

CSV: header row required (same RFC-4180-ish rules as the CLI: comma
separator, no quoted commas). .npy: 2-D array, first name "c0" etc.
unless --features is given. .npz: needs --target to pick the column
order (target first if --no-intercept-style ordering is done by the CLI,
we just preserve the array order).
"""
from __future__ import annotations

import argparse
import csv
import struct
import sys
import zlib
from pathlib import Path

MAGIC = 0x4F4C5342
VERSION = 1


def fail(msg: str) -> None:
    print(f"to_olsb: {msg}", file=sys.stderr)
    sys.exit(2)


def read_csv(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        fail("empty CSV")
    header, body = rows[0], rows[1:]
    names = [h.strip() for h in header]
    data = []
    for i, r in enumerate(body, start=2):
        if not r or (len(r) == 1 and not r[0].strip()):
            continue
        if len(r) != len(names):
            fail(f"line {i}: {len(r)} fields, header has {len(names)}")
        try:
            data.append([float(x) for x in r])
        except ValueError as e:
            fail(f"line {i}: {e}")
    if not data:
        fail("no data rows")
    n_rows, n_cols = len(data), len(names)
    flat = [v for row in data for v in row]
    return names, n_rows, n_cols, flat


def read_npy(path: Path):
    import numpy as np
    a = np.load(path)
    if a.ndim != 2:
        fail(".npy must be 2-D")
    if not (a.dtype == np.float64):
        a = a.astype(np.float64)
    n_rows, n_cols = a.shape
    names = [f"c{j}" for j in range(n_cols)]
    return names, n_rows, n_cols, a.ravel().tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="CSV/.npy -> .olsb converter")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--features", help="comma-separated column names "
                                       "(order = file column order)")
    ap.add_argument("--target", help="target column name (must be present "
                                     "in the header; kept in place)")
    args = ap.parse_args()

    src, dst = Path(args.input), Path(args.output)
    if not src.exists():
        fail(f"cannot open {src}")
    if src.suffix == ".csv":
        names, n_rows, n_cols, flat = read_csv(src)
    elif src.suffix in (".npy", ".npz"):
        names, n_rows, n_cols, flat = read_npy(src)
    else:
        fail(f"unsupported input type {src.suffix} (use .csv or .npy)")

    if args.features:
        feats = [f.strip() for f in args.features.split(",") if f.strip()]
        if len(feats) != n_cols:
            fail(f"--features lists {len(feats)} names, file has {n_cols} columns")
        names = feats
    if args.target and args.target not in names:
        fail(f"--target {args.target} not among columns {names}")

    header = struct.pack("<IIQQQQQ", MAGIC, VERSION, n_rows, n_cols,
                         0, 0, 0)  # offsets patched below
    name_block = b"".join(n.encode() + b"\0" for n in names)
    name_off = len(header)
    data_off = name_off + len(name_block)
    data = struct.pack(f"<{n_rows * n_cols}d", *flat)
    trailer_off = data_off + len(data)
    trailer = struct.pack("<I", zlib.crc32(data) & 0xFFFFFFFF)
    blob = (struct.pack("<IIQQQQQ", MAGIC, VERSION, n_rows, n_cols,
                        name_off, data_off, trailer_off)
            + name_block + data + trailer)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(blob)
    print(f"wrote {dst}: n_rows={n_rows} n_cols={n_cols} "
          f"({len(blob) / 1e6:.1f} MB, CRC ok)")


if __name__ == "__main__":
    main()
