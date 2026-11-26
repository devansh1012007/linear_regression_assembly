#!/usr/bin/env python3
"""tools/mutate.py — mutation testing for the asm kernels (asm-verification §5).

Applies one mutation at a time to a copy of each src/core/*.asm, rebuilds the
shared library into a scratch build dir, runs the fast differential suite
(tests/py/mutation_suite.py), and records killed/survived mutants.
Every mutant must be killed or be documented as equivalent in
results/mutation-phase1.json.

Mutation operators (register/instruction-level, asm-specific):
  1. tail bound: `and rcx, -N` -> `and rcx, -(N-4)` (over/under-process)
  2. jump condition: jb -> jbe, ja -> jae (off-by-one in tails)
  3. displacement: +32 -> +24 on vmovupd/vmovsd memory operands
  4. drop vzeroupper (replace with nop-equivalent)
  5. swap a register operand in add/sub/mul (xmm4 -> xmm5 in one instruction)
  6. drop one accumulator from a final reduction (vaddpd -> nop-equivalent)
  7. delete an instruction (replace with nop padding of same length not
     possible generically; instead: `inc rax` -> `add rax, 2` skipping)
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "build" / "mutation"
TARGETS = ["src/core/dot.asm", "src/core/axpy.asm", "src/core/sums.asm",
           "src/core/simple.asm"]


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def build_lib(objects, out):
    """Link the mutated library against the CURRENT build/lib objects (the
    2025-11 review moved production objects there; the old build/cpu.o
    paths made EVERY mutant fail to link — the phase-1 "killed (link fail)"
    kills were fake, found by the P3.9 rerun when the score looked too
    good). The mutated file's fresh object replaces its build/lib one."""
    objs = []
    for src in sorted((ROOT / "build" / "lib").glob("*.o")):
        if src.stem in objects:
            objs.append(str(objects[src.stem][1]))
        else:
            objs.append(str(src))
    if not objs:
        return False
    r = run(["gcc", "-O2", "-std=c11", "-fPIC", "-shared", "-o", str(out),
             *objs, str(ROOT / "build" / "tests" / "abi_probe.o"),
             str(ROOT / "tests" / "abi" / "abi_adapters.c"), "-Iinclude"])
    return r.returncode == 0


def mutate_lines(text):
    """Yield (description, mutated_text) one mutation at a time."""
    lines = text.splitlines(keepends=True)
    n = len(lines)
    # op 1: tail-width bound
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*and\s+rcx,\s*-)(\d+)\s*$", ln)
        if m and int(m.group(2)) >= 8:
            new = m.group(1) + str(int(m.group(2)) - 4) + "\n"
            yield (f"line {i+1}: and rcx,-{m.group(2)} -> -{int(m.group(2))-4}",
                   "".join(lines[:i]) + new + "".join(lines[i+1:]))
    # op 2: jb -> jbe / ja -> jae in tail handling
    for i, ln in enumerate(lines):
        for old, new in (("jb\t", "jbe\t"), ("ja\t", "jae\t")):
            if re.search(rf"\s{old}", ln) and "cmp" not in ln:
                mutated = re.sub(rf"\s{old}", f" {new}", ln, count=1)
                yield (f"line {i+1}: {old.strip()} -> {new.strip()}",
                       "".join(lines[:i]) + mutated + "".join(lines[i+1:]))
                break
    # op 3: displacement +32 -> +24 on vector loads
    for i, ln in enumerate(lines):
        if "vmovupd" in ln and "+32]" in ln:
            yield (f"line {i+1}: +32 -> +24 displacement",
                   "".join(lines[:i]) + ln.replace("+32]", "+24]") + "".join(lines[i+1:]))
    # op 4: drop a vzeroupper
    for i, ln in enumerate(lines):
        if ln.strip() == "vzeroupper":
            yield (f"line {i+1}: vzeroupper dropped",
                   "".join(lines[:i]) + "    ; vzerupper dropped (mutation)\n" + "".join(lines[i+1:]))
    # op 5: swap xmm4 -> xmm5 in one arithmetic instruction (first occurrence per line)
    for i, ln in enumerate(lines):
        if re.search(r"\b(vaddpd|vaddsd|vfmadd231pd|vfmadd231sd|addsd|subsd|mulsd)\b", ln) \
           and "xmm4" in ln and "xmm5" not in ln:
            yield (f"line {i+1}: xmm4 -> xmm5 operand swap",
                   "".join(lines[:i]) + ln.replace("xmm4", "xmm5", 1) + "".join(lines[i+1:]))
    # op 6: vaddpd in reductions -> vxorpd (zero an accumulator)
    for i, ln in enumerate(lines):
        if "vaddpd" in ln and "ymm0" in ln:
            yield (f"line {i+1}: vaddpd ymm0 -> vxorpd (accumulator dropped)",
                   "".join(lines[:i]) + ln.replace("vaddpd", "vxorpd", 1) + "".join(lines[i+1:]))
    # op 7: inc rax -> add rax, 2 (skip an element)
    for i, ln in enumerate(lines):
        if ln.strip() == "inc     rax":
            yield (f"line {i+1}: inc rax -> add rax,2",
                   "".join(lines[:i]) + "    add     rax, 2\n" + "".join(lines[i+1:]))


def main():
    import os
    if "NASM_PATH" in os.environ:
        pass
    SCRATCH.mkdir(parents=True, exist_ok=True)
    results = []
    suite = ROOT / "tests" / "py" / "mutation_suite.py"
    so = SCRATCH / "libols_mut.so"
    n_killed = n_equiv = n_total = 0
    for target in TARGETS:
        text = (ROOT / target).read_text()
        seen = set()
        for desc, mutated in mutate_lines(text):
            if mutated in seen:
                continue
            seen.add(mutated)
            n_total += 1
            mpath = SCRATCH / (Path(target).name)
            mpath.write_text(mutated)
            # contract lint first (cheap): a mutation that breaks a
            # contract-header or the vzeroupper discipline is killed here —
            # closes the 2026-11-21 survivor class "vzeroupper dropped"
            rl = run(["python3", str(ROOT / "tools" / "lint_contracts.py"),
                      str(mpath)])
            if rl.returncode != 0:
                n_killed += 1
                n_total += 1
                results.append({"target": str(target), "mutation": desc,
                                "outcome": "killed (contract lint)"})
                print(f"KILLED  {target}:{desc} [lint]", flush=True)
                continue
            obj = SCRATCH / (Path(target).stem + "mut.o")
            r = run(["nasm", "-f", "elf64", "-I.", "-o", str(obj), str(mpath)])
            if r.returncode != 0:
                # assembly failure = trivially dead mutant (still counts as killed)
                n_killed += 1
                results.append({"target": str(target), "mutation": desc,
                                "outcome": "killed (build fail)"})
                continue
            ok = build_lib({Path(target).stem: (str(mpath), obj)}, so)
            if not ok:
                n_killed += 1
                results.append({"target": str(target), "mutation": desc,
                                "outcome": "killed (link fail)"})
                continue
            env = dict(**__import__("os").environ, OLS_MUT_SO=str(so))
            r = run(["python3", str(suite)], env=env, timeout=240)
            killed = (r.returncode != 0)
            if killed:
                n_killed += 1
                results.append({"target": str(target), "mutation": desc,
                                "outcome": "killed"})
            else:
                n_equiv += 1
                results.append({"target": str(target), "mutation": desc,
                                "outcome": "SURVIVED"})
            print(f"{'KILLED ' if killed else 'SURVIV'} {target}:{desc}", flush=True)
    summary = {
        "schema": 1,
        "targets": TARGETS,
        "total_mutants": n_total,
        "killed": n_killed,
        "survived": n_equiv,
        "mutation_score": round(n_killed / n_total, 4) if n_total else None,
        "results": results,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    out_name = ("mutation-phase3.json" if "PHASE3" in __import__("os").environ
                else "mutation-phase1.json")
    (ROOT / "results" / out_name).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nmutation score: {n_killed}/{n_total} = {summary['mutation_score']}")
    return 0 if n_equiv == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
