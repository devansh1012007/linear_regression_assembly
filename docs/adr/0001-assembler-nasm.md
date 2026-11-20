# ADR-0001: Assembler — NASM (not GAS)

**Status:** Accepted (Phase 0) **Date:** 2026-11-20

## Context
PLAN.md §4 defaults to NASM but requires the trade-off to be recorded. All
product code under `src/` is assembly; the assembler choice affects macro
power, unwind info, and toolchain friction for the whole project.

## Options
- **NASM 2.16**: powerful multi-line `%macro`/`%rep`/`%assign` metaprogramming;
  clean Intel syntax; separate from the C toolchain (assembles without gcc);
  verified templates exist in the skills. No `.cfi_*` unwind directives; no
  integrated `--gc-sections`-aware relaxations; one more build dependency.
- **GAS (`.intel_syntax noprefix`)**: one toolchain with gcc/ld; emits CFI
  unwind info (`.cfi_def_cfa_offset`) for free — better gdb backtraces through
  asm frames and proper profiler unwinding; weaker macro language for
  generated tiles (`.macro`/`.irp` exist but are less ergonomic for
  register-indexed tile generation).
- **Other (asmtool, asmc)**: immature on Linux; rejected without measurement.

## Decision
**NASM.** The decisive factor is rule "generated, not copy-pasted"
(asm-x86-64-engineering §7): Phase-2's 6x8 Gram tile expands 12 FMAs per row
from `%rep`; NASM's macro system does this with the least friction, and tile
changes stay one-line edits. The lack of unwind info is acceptable for this
project because (a) all Phase-1 kernels are leaf functions, (b) profiling will
use sampling with frame pointers from the C harness, and (c) the CLI is a
static binary where backtraces are secondary to correctness evidence.

## Consequences
- `.note.GNU-stack` section required in every file (linker exec-stack warning
  otherwise) — enforced by the verified template, all files carry it.
- No CFI: gdb backtraces through asm frames may be truncated; documented as a
  known limitation; revisit if call-graph profiling becomes essential.
- Win64 port (COULD item) remains possible via `-f win64` with `abi.inc`'s
  ARGn indirection (`%ifidn __OUTPUT_FORMAT__, win64`), verified in the skill.
- Build requires NASM >= 2.15; CI installs it via apt.

## Revisit trigger
If a Phase-2+ kernel needs unwind info (e.g. TSQR recursion that must profile
cleanly through asm frames), or if macro debugging becomes a time sink.
