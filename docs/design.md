# design.md — architecture rationale

The "why" behind the module map (`architecture.md` has the layouts). Every
decision here traces to a hard rule in `AGENTS.md` §2 or a measured failure
recorded in `perf-log.md` / `agent-comms.md`.

## 1. Assembly purity as a product property

Everything under `src/` is NASM (`.asm` + `.inc`) plus `include/ols.h` and
one sanctioned C file, `src/thread/ols_mt.c` (ADR-0008) — the C header *is*
the ABI contract, not part of the core (rule 5). The core has
no libc, no syscalls, no allocation, no threads: `ols_fit` and every kernel
run on caller-provided buffers. This is what makes the guard-page harness
(meaningful), the ABI register probe (exhaustive), and a future static
libc-free CLI (cheap) possible. C/Python otherwise live only in `tests/`,
`bench/`, `tools/` and `cli/`, and `tools/lint_contracts.py --check-src`
enforces the purity line in CI.

## 2. One contract surface, hash-verified

`include/ols.h` declares everything; `src/abi.inc` mirrors the same constants
as NASM `%define`s. Two hand-maintained views of one layout can drift, so the
asm side exports `ols_abi_offsets()` (src/core/api.asm): a FNV-1a hash over
the frozen offset table (problem, result, gram/resid arg blocks, ABI version
tag), and `tests/py/test_phase2.py::test_struct_drift` recomputes the same
hash from the ctypes mirrors. Any one-sided edit changes the hash and fails
the pyramid (ADR-0006). Literal offset checks pin absolute values on top.

Struct evolution is append-only and version-tagged: result v1 88 → v2 96
(t-stats) → v3 104 (f_stat). New fields are written only when the caller
allocated the current size (the version check in `ols_fit`); old callers
keep working.

## 3. Memory discipline learned from two real corruptions

Two heap corruptions shipped and were caught by the harness before any
release (phase-2 report):

1. The workspace-size formula undercounted by one slot, so the SSE scalar
   ran past the caller's buffer by `q*8` bytes — found by the padded-buffer
   tail check.
2. The gram scratch pointer was passed unaligned; the kernel's internal
   align-up then wrote past the workspace. The fix reserves explicit slack
   and aligns the tail pointer *before* the call.

Consequences encoded in the design: `ols_workspace_size` is computed by one
function used by both the caller and `ols_fit` itself (the fit re-derives the
same total to locate the aligned tail, so the two can never disagree);
guard-page tests run every kernel against page-boundary buffers at misaligned
offsets (L3); kernels validate nothing they can trust the API layer for
(validation once in `ols_fit`, contracts thereafter — asm-project-delivery §2).

## 4. Numerics before speed

The accuracy stack is pivot-shifted compensated accumulation → algebraic
centering → Cholesky with an explicit singularity threshold → closed-form
statistics. Every inner product is Neumaier-compensated; every shift is
chosen so cancellation is bounded by construction, not by luck
(`numerical_stability.md`). Performance work is gated behind the full
correctness matrix: a SIMD kernel merges only when the scalar path, the C/Python
references and the cross-ISA differential all agree within pre-registered
bounds (rule 3; the cross-ISA bound is 1e-9 relative on the Gram, L4).

## 5. ISA dispatch does not advertise vaporware

`ols_cpu_has_avx512` reports chip truth; `ols_init`/`ols_isa` report
*implemented-kernel* truth and are clamped to AVX2 because no AVX-512 kernel
ships (review finding, fixed 2026-11-25; ADR: dispatch.asm header). The
reviewer-facing rule: `ols_isa()` never returns a level for which the
library has no kernel, so a caller can never dispatch into a stub.

## 6. Honest measurement as a first-class artifact

Predict-then-measure is mechanical: `perf-log.md` entries are written before
benchmarking; gaps get root causes, not tolerance relaxations (rule 2/4).
Gate G2 is the worked example: the pre-registered 85%-of-bandwidth target
was missed by v1 (~2%), the v2 redesign was paused with a written negative
result, v3 recovered 11.3-19.9× and *still* misses the gate — the gate stays
red and the next iteration plan is recorded (`performance.md` §3). The
pyramid (`tests/py/run_all.py`, L1-L8) is the single gate `make verify` runs.

## 7. What this design does not do (yet)

Documented, deliberate gaps — each has an owner in `PLAN.md` §Phase 3/4/5:
Dot2 (double-double) accumulation (H3), AVX-512 kernels, Win64/ARM64 ports,
leverage/Cook's distance and the remaining COULD items (`PLAN.md` §3).
The audit trail for scope decisions is `docs/adr/`.
