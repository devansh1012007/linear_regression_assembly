#!/usr/bin/env python3
"""Replay fuzz_csv.py's exact loop (seed 42) and dump failing cases verbatim."""
import sys, random, json, struct, subprocess
from pathlib import Path
sys.path.insert(0, 'tests/py')
from fuzz_csv import base_csv, mutate_csv, run_cli, check, CLI, ROOT

SEED, BUDGET = 42, 5.0
rng = random.Random(SEED)
tmp = Path('/tmp/fuzz_replay'); tmp.mkdir(exist_ok=True)
good = base_csv(random.Random(1))
(tmp/'good.csv').write_text(good)
subprocess.run([sys.executable, str(ROOT/'scripts'/'to_olsb.py'),
                str(tmp/'good.csv'), str(tmp/'good.olsb')],
               check=True, capture_output=True)
good_bytes = (tmp/'good.olsb').read_bytes()

import time
t0 = time.perf_counter()
n = 0
fails = 0
while time.perf_counter() - t0 < BUDGET:
    n += 1
    kind = rng.random()
    if kind < 0.5:
        payload = mutate_csv(rng, good)
        f = tmp/'m.csv'; f.write_text(payload)
        argv = ['fit', str(f), '--target', 'y', '--json']
        res = run_cli(argv)
        extra = payload
    elif kind < 0.85:
        b = bytearray(good_bytes)
        for _ in range(rng.randrange(1, 6)):
            op = rng.random()
            if op < 0.5 and len(b) > 60:
                b[rng.randrange(len(b))] = rng.randrange(256)
            elif op < 0.7:
                field = rng.randrange(6)
                val = rng.choice([0, 1, 2**31, 2**40, 10**9])
                b[8+field*8:8+field*8+8] = struct.pack('<Q', val)
            elif op < 0.85:
                b = b[:rng.randrange(1, len(b))]
            else:
                b += bytes(rng.randrange(256) for _ in range(rng.randrange(1, 64)))
        f = tmp/'m.olsb'; f.write_bytes(bytes(b))
        argv = ['fit', str(f), '--target', 'y', '--json']
        res = run_cli(argv)
        extra = f'<binary {len(b)} bytes>'
    else:
        flags = rng.sample(["--target", "y", "--features", "x1,x2",
                            "--method", "chol", "--lambda", "0.5",
                            "--json", "--no-intercept", "--confint",
                            "--rcond", "1e-8", "--"],
                           k=rng.randrange(1, 8))
        argv = ['fit', str(tmp/'good.csv')] + flags
        res = run_cli(argv)
        extra = None
    problem = check(res)
    if problem:
        fails += 1
        print(f"=== FAIL iter {n} kind {kind:.3f} ===")
        print("argv:", argv)
        print("problem:", problem, "| rc:", res['rc'], "| err:", res['err'][:150])
        if extra and not extra.startswith('<binary'):
            print("input first 4 lines:", extra.splitlines()[:4])
        # rerun to capture stdout
        p = subprocess.run([str(CLI)]+argv, capture_output=True, text=True, timeout=20)
        out = p.stdout
        try:
            json.loads(out)
            print("stdout: (parses NOW?!)", out[:120])
        except json.JSONDecodeError as e:
            print("JSONDecodeError:", e)
            print("stdout around error:", out[max(0,e.pos-60):e.pos+40])
        if fails >= 5:
            break
print(f"replayed {n} iters, {fails} fails")
