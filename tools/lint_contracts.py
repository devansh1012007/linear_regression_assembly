#!/usr/bin/env python3
"""Contract-header linter (asm-x86-64-engineering §3, asm-verification §8).

Fails the build if a `global` symbol in src/**/*.asm lacks a contract header
with the mandatory fields: Args, Returns, Requires, Memory, Clobbers.
Optional but encouraged: Numerics, Cost.

Usage:
  lint_contracts.py                 # lint src/ + tests/abi
  lint_contracts.py --check-src     # also enforce assembly purity of src/
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ("Args", "Returns", "Requires", "Memory", "Clobbers")
GLOBAL_RE = re.compile(r"^\s*global\s+([A-Za-z_.$][\w.$]*)\s*(:function)?", re.M)


def lint_file(path: Path) -> list[str]:
    errors = []
    text = path.read_text()
    # Comment block preceding each `global` directive: walk lines backwards.
    for m in GLOBAL_RE.finditer(text):
        sym = m.group(1)
        if sym.startswith("abi_probe"):  # harness export, header lives in tests/abi
            continue
        if sym.startswith("ols__"):       # internal data symbols (not functions)
            continue
        # find end of previous non-empty, non-global line
        start = text.rfind("\n", 0, m.start())
        block = []
        pos = m.start()
        lines = text[:pos].splitlines()
        for line in reversed(lines):
            s = line.strip()
            if not s:  # blank line terminates the comment block
                break
            if s.startswith(";"):
                block.append(s)
            else:
                break
        header = "\n".join(block)
        missing = [f for f in REQUIRED if not re.search(rf"\b{f}\s*:", header)]
        if missing:
            errors.append(f"{path}:{text[:m.start()].count(chr(10))+1}: "
                          f"global '{sym}' missing contract fields: {', '.join(missing)}")
    return errors


def lint_vzeroupper(path: Path) -> list[str]:
    """Discipline check (asm-verification 2b, mutation survivor fix): any
    function body that touches ymm registers must contain vzeroupper before
    returning — an AVX kernel leaving the upper half dirty stalls subsequent
    SSE code. The mutation runner lints the mutated file, so a dropped
    vzeroupper is a killed mutant, not a silent survivor."""
    errors = []
    text = path.read_text()
    lines = text.splitlines()
    # split into function bodies at global/section boundaries
    starts = [i for i, ln in enumerate(lines)
              if re.match(r"^\s*global\s", ln) or re.match(r"^section\b", ln)]
    starts.append(len(lines))
    for a, b in zip(starts, starts[1:]):
        body = lines[a:b]
        if not any(re.search(r"\bymm\d+", ln) for ln in body):
            continue
        if not any("vzeroupper" in ln for ln in body):
            errors.append(f"{path}:{a+1}: function body uses ymm without "
                          f"vzeroupper (upper-half discipline)")
    return errors


def check_purity() -> list[str]:
    # src/thread/ is the ONE sanctioned exception: the Phase-4c pthread
    # orchestration layer (ADR-0008) — the numerical core stays pure asm.
    errors = []
    for p in (ROOT / "src").rglob("*"):
        if (p.is_file() and p.suffix not in (".asm", ".inc")
                and "src/thread/" not in p.as_posix()):
            errors.append(f"src/ purity: non-asm source {p}")
    return errors


def main() -> int:
    file_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if file_args:
        # lint specific files (the mutation runner lints mutated copies)
        targets = [Path(a) for a in file_args]
    else:
        targets = list((ROOT / "src").rglob("*.asm"))
    errors = []
    for t in targets:
        errors += lint_file(t)
        errors += lint_vzeroupper(t)
    if "--check-src" in sys.argv:
        errors += check_purity()
    if errors:
        print("\n".join(errors))
        print(f"lint_contracts: {len(errors)} error(s)")
        return 1
    print(f"lint_contracts: OK ({len(targets)} asm files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
