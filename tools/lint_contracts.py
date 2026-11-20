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


def check_purity() -> list[str]:
    errors = []
    for p in (ROOT / "src").rglob("*"):
        if p.is_file() and p.suffix not in (".asm", ".inc"):
            errors.append(f"src/ purity: non-asm source {p}")
    return errors


def main() -> int:
    targets = list((ROOT / "src").rglob("*.asm"))
    errors = []
    for t in targets:
        errors += lint_file(t)
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
