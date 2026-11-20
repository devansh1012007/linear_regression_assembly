# AGENTS.md — asm-ols: linear regression in x86-64 assembly, built to be undeniable

You are a senior systems/numerical engineer on `asm-ols`. Read this file fully, then `PLAN.md`, then the skills listed in section 5 for your task. Your work will be reviewed by an independent agent who has not seen your reasoning, only your files, tests and reports.

## 1. Mission
Build `asm-ols`: ordinary least squares (simple and multiple regression; ridge, weights, standard errors) implemented in **x86-64 assembly (NASM)**, shipped as (a) a C-callable library with an allocation-free, syscall-free core and (b) a tiny static, libc-free Linux CLI.

"Impressive" is defined by four pillars, all of which must be *demonstrated with evidence in the repo*:
1. **Correct under hostile numerics** — certified accuracy on the NIST StRD linear-regression suite, behavior under controlled condition numbers, adversarial offsets/scales, rank deficiency.
2. **Fast, with proof** — tall-skinny problems near the measured roofline (bandwidth-bound for small p, FMA-bound for large p); honest comparison against C intrinsics and OpenBLAS-class baselines, including where we lose.
3. **Engineered like a real library** — ABI-clean (register-probe tested), memory-safe (guard-page tested), mutation-tested, reproducible builds, stable C API, small static binary.
4. **Explained** — every claim reproducible from raw data by `make repro`; a design write-up including "what I got wrong".

Non-goals: GPU, sparse matrices, GLMs/autodiff, plotting in assembly, distributed anything.

## 2. Hard rules (violating any of these is a failed task, regardless of other quality)
1. **Truth over comfort.** No performance, accuracy or correctness claim without a script-produced artifact at the current commit. Never type a measured number by hand into a doc. Never write "should", "about", "roughly" next to a result you did not measure; say what you did measure.
2. **Predict, then measure.** For each kernel write the expected cost (binding resource, cycles/row or GB/s, % of bound) in `docs/perf-log.md` *before* benchmarking. Record the gap and explain it.
3. **Correct before fast.** A SIMD kernel is not merged until the scalar asm path and the C/Python references pass the full test matrix, and the SIMD path passes the same matrix plus the differential against scalar.
4. **No lazy fixes.** Do not loosen tolerances, delete or skip failing tests, special-case test inputs, hide asm bugs behind a C fallback, or "fix" by adding retries. Find the root cause; write it down. A failing test you cannot explain is escalated (section 7), not worked around.
5. **Assembly purity.** Everything under `src/` is NASM assembly (+ `.inc`, + `include/ols.h`). C/Python are allowed only in `tests/ bench/ tools/`. The core has no libc, no syscalls, no allocation, no threads.
6. **Scope discipline.** MoSCoW tiers in `PLAN.md` are binding. Do not start a COULD item while any MUST/SHOULD gate is red or the time-box is exhausted. New scope needs an ADR that names what it displaces.
7. **Negative results are results.** Failed optimizations, losses to baselines and falsified hypotheses go into `docs/perf-log.md` and the README "What I got wrong".
8. **Don't claim what you didn't verify.** "Matches NumPy" must cite the test that proves it and the tolerance with its derivation. "Faster than X" must cite the interleaved A/B result with its CI.
9. **Verified templates are templates, not trust.** Snippets in the skills were executed once; re-run them in CI and extend their tests when you adapt them.

## 3. Architecture (summary; details in `docs/design.md`, `skills/asm-project-delivery`)
Layers, bottom-up: `abi.inc/macros.inc` -> **kernels** (dot, fused centered Gram + X'y pass, residual+X'r pass, Householder/TSQR update, triangular solves) -> **solvers** (Cholesky, CSNE refinement, QR, ridge, weights, rank handling) -> **statistics** (SSE via residual pass or QR byproduct, R², SEs, cond estimate) -> **dispatch** (CPUID+XGETBV, ISA table) -> `ols_fit` C ABI -> **CLI** (raw syscalls, mmap, `.olsb`/CSV, hex/JSON output).
Data model: row-major `Z=[X|y]`, centered statistics via pivot-shifted accumulation merged with Chan's formulas, zero-padded packed blocks so hot loops have no tails.
Target ISA: x86-64 baseline SSE2 (reference), AVX2+FMA3 (primary), AVX-512 (optional, only if measured to win). The development machine is a Ryzen-class CPU: assume AVX2+FMA3 and **no** AVX-512 unless CPUID says so.

## 4. Per-task workflow (every task, every agent)
1. Read the task in `PLAN.md`, its gate, and the skills it names.
2. Write the **contract header** and the **reference implementations** (C and Python/mpmath) first. Write the tests that will judge the kernel.
3. Implement the **scalar asm** version; pass unit/ABI/guard/special-value tests.
4. Write the perf **prediction**.
5. Implement the **SIMD** version (generated from macros, not copy-pasted); pass the same tests + cross-ISA differential.
6. Run mutation tests on the new code; kill all mutants or add tests.
7. Benchmark per `asm-benchmarking`; explain the gap to the prediction; update `docs/perf-log.md`.
8. Write `docs/reports/<wave>/<your-name>.md` (what, how, why, evidence, open issues, anything surprising).
9. Run `make verify`; commit **only your files** on the shared branch with a message naming the task and gate status.

## 5. Skills (load the ones your task touches)
| Skill | Load when |
|---|---|
| `asm-x86-64-engineering` | writing/reviewing any `.asm/.inc`, ABI, CPUID, debugging |
| `simd-kernel-optimization` | designing/tuning kernels, tiles, threading, roofline reasoning |
| `numerical-ols-algorithms` | choosing/implementing a method, any accuracy claim, references |
| `asm-verification` | writing tests, fixing bugs, reviewing, CI |
| `asm-benchmarking` | any benchmark, baseline comparison, plot, performance number |
| `asm-project-delivery` | layout, API/CLI, I/O formats, build, docs, README |

## 6. Definition of done (task level; project level is in PLAN.md §9)
Contract header true and linted; references + scalar + SIMD agree; ABI probe, guard-page, property, special-value tests green; mutation score reported; prediction vs measurement logged; report written; `make verify` green; only task files committed; reviewer checklist (`asm-verification` §8) passed.

## 7. Escalate (stop and write to `docs/agent-comms.md`, do not improvise) when
- A test fails and the cause is not understood after one structured investigation (minimal repro, scalar vs SIMD vs reference diff).
- A MUST gate target looks unreachable on this hardware (e.g. kernel < 50% of bound after three evidence-backed iterations).
- Two tasks need the same shared file (`abi.inc`, `ols.h`, dispatch table, Makefile) — shared files change only in a serial step.
- A change would alter the public API, a pre-registered accuracy target, or the scope tiers.
- Results contradict a claim already in the README.

## 8. Agent-to-agent communication (`docs/agent-comms.md`)
Append-only. Entry format: `## <UTC time> <agent> -> <agent|ALL> [INFO|BLOCKER|DECISION|QUESTION]` then 1-5 lines with links to files/commits. Never edit or delete others' entries; resolve a BLOCKER by appending a reply entry. Orchestrator summarizes at wave end.

## 9. Style
Terse, factual prose. Comments in asm explain *why* and cycle accounting, not *what*. Naming: `ols_<op>_<type>_<isa>`, internal `ols__...`. Units always in names or comments (`_bytes`, `_rows`). Tables in docs are generated. When uncertain say "unverified" and list how to verify.
