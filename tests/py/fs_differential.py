#!/usr/bin/env python3
"""Ground-truth differential: fs CLI vs libc CLI vs direct ols_fit via ctypes.

Determines whether the freestanding CLI computes identical doubles to the
libc CLI when both load the same .olsb and call ols_fit with identical
problem structs. Any mismatch means the fs shim (not the library) is wrong.
"""
import ctypes as C
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
LIB = ROOT / "build" / "lib" / "libols.so"
FS = ROOT / "build" / "cli" / "asm-ols-fs"
LIBC_CLI = ROOT / "build" / "cli" / "asm-ols"


class Problem(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("version", C.c_uint32),
                ("X", C.c_void_p), ("y", C.c_void_p), ("w", C.c_void_p),
                ("n", C.c_size_t), ("p", C.c_size_t), ("ldx", C.c_size_t),
                ("method", C.c_uint32), ("flags", C.c_uint32),
                ("lambda", C.c_double), ("rcond", C.c_double),
                ("threads", C.c_uint32), ("_pad", C.c_uint32)]


class Result(C.Structure):
    _fields_ = [("beta", C.c_void_p), ("intercept", C.c_double),
                ("sse", C.c_double), ("sst", C.c_double),
                ("r2", C.c_double), ("r2_adj", C.c_double),
                ("sigma2", C.c_double), ("cond_est", C.c_double),
                ("se", C.c_void_p), ("rank", C.c_int32),
                ("method_used", C.c_int32), ("isa_used", C.c_int32),
                ("_pad", C.c_int32), ("t", C.c_void_p),
                ("f_stat", C.c_double)]


def main() -> int:
    raw = open("/tmp/rt.olsb", "rb").read()
    _, _, n, nc, name_off, data_off, _ = struct.unpack("<IIQQQQQ", raw[:48])
    data = np.frombuffer(raw, dtype="<f8", offset=data_off,
                         count=n * nc).reshape(n, nc).copy()
    names = raw[name_off:data_off].split(b"\0")[:-1]
    ti = names.index(b"y")
    y = np.ascontiguousarray(data[:, ti])
    X = np.ascontiguousarray(np.delete(data, ti, axis=1))
    p = X.shape[1]

    lib = C.CDLL(str(LIB))
    lib.ols_fit.restype = C.c_int32
    lib.ols_fit.argtypes = [C.POINTER(Problem), C.POINTER(Result),
                            C.c_void_p, C.c_size_t]
    lib.ols_workspace_size.restype = C.c_size_t
    lib.ols_workspace_size.argtypes = [C.c_size_t, C.c_size_t,
                                       C.c_uint32, C.c_uint32]

    beta = np.zeros(p + 1)
    se = np.zeros(p + 1)
    t = np.zeros(p + 1)
    prob = Problem(88, 1, X.ctypes.data, y.ctypes.data, None,
                   n, p, p, 0, 1, 0.0, 0.0, 1, 0)
    res = Result(beta.ctypes.data, 0, 0, 0, 0, 0, 0, 0,
                 se.ctypes.data, -1, 0, 0, 0, t.ctypes.data, 0)
    ws_bytes = lib.ols_workspace_size(n, p, 0, 1)
    ws = (C.c_char * ws_bytes)()
    rc = lib.ols_fit(C.byref(prob), C.byref(res), ws, ws_bytes)
    print("direct ols_fit rc:", rc, "rank:", res.rank)

    fs = json.loads(subprocess.run(
        [str(FS), "fit", "/tmp/rt.olsb", "--target", "y", "--json"],
        capture_output=True, text=True, check=True).stdout)
    lc = json.loads(subprocess.run(
        [str(LIBC_CLI), "fit", "/tmp/rt.olsb", "--target", "y", "--json"],
        capture_output=True, text=True, check=True).stdout)

    def cmp(name, direct, a, b):
        same = (direct == a, direct == b)
        print(f"{name:16s} direct={direct!r:24s} fs={a!r:24s} libc={b!r:24s} "
              f"fs==direct:{same[0]} libc==direct:{same[1]}")
        return same

    all_ok = True
    all_ok &= cmp("intercept", res.intercept, fs["intercept_value"], lc["intercept_value"])[0]
    for j in range(p):
        all_ok &= cmp(f"beta[{j}]", beta[j], fs["coefficients"][f"x{j+1}"],
                      lc["coefficients"][f"x{j+1}"])[0]
    all_ok &= cmp("sse", res.sse, fs["sse"], lc["sse"])[0]
    all_ok &= cmp("sst", res.sst, fs["sst"], lc["sst"])[0]
    all_ok &= cmp("r2", res.r2, fs["r2"], lc["r2"])[0]
    all_ok &= cmp("sigma2", res.sigma2, fs["sigma2"], lc["sigma2"])[0]
    print("VERDICT:", "fs == direct ols_fit (bitwise)" if all_ok else "fs DIFFERS from direct ols_fit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
