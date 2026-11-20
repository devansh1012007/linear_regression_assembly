---
name: asm-project-delivery
description: Repo layout, C-callable API, allocation-free core design, binary/CSV formats, freestanding Linux CLI (raw syscalls), build/CI, ADRs, README and results provenance for asm-ols. Load when creating project structure, designing the public API or CLI, writing I/O or parsing code, setting up the build, or writing docs and the final write-up.
---

# Delivering asm-ols as a real library + CLI

## 1. Repo layout
```
asm-ols/
  AGENTS.md  PLAN.md  README.md  Makefile
  include/ols.h              # the only public header (C ABI)
  src/
    abi.inc  macros.inc
    core/    dot.asm gram_*.asm resid_*.asm chol.asm trsv.asm qr_*.asm stats.asm dispatch.asm cpu.asm
    cli/     start.asm args.asm io_bin.asm io_csv.asm fmt.asm main.asm   # freestanding, Linux syscalls
  tests/     ref/ (python+C references)  abi/ (probe)  guard/  props/  nist/  fuzz/  tolerances.py
  bench/     machine/ sweeps/ baselines/ (c, numpy, eigen)  roofline.py  flops.md
  tools/     mutate.py  lint_contracts.py  gen_results_md.py
  data/      nist/ (certified values)  synth/ (generators, never large files in git)
  results/   <date>-<commit>/*.json      # immutable raw outputs
  docs/      adr/  perf-log.md  reports/  design.md
```
Product code (`src/`) is assembly + `.inc` + `include/ols.h`. C/Python/Make exist only in `tests/ bench/ tools/`. A CI job prints the assembly-vs-other line count of `src/` and fails if `src/` contains a non-asm source.

## 2. Core design rules
- **The core allocates nothing, makes no syscalls, spawns no threads.** The caller passes a workspace (size from `ols_workspace_size`). This makes the core embeddable, testable under guard pages, and callable from Python via ctypes.
- **Stable C ABI** with a problem/result struct (versioned) rather than a long argument list:
```c
typedef struct {
  uint32_t struct_size, version;
  const double *X; const double *y; const double *w;   // w may be NULL
  size_t n, p, ldx;                                   // row-major X, ldx >= p
  uint32_t method;   // OLS_AUTO, OLS_GRAM_CHOL, OLS_GRAM_CSNE, OLS_QR, OLS_QR_PIVOT, OLS_CG
  uint32_t flags;    // INTERCEPT, REPRODUCIBLE, DOT2, SKIP_STATS, ...
  double   lambda;   // ridge, 0 = none
  double   rcond;    // rank tolerance, 0 = default
  uint32_t threads;  // 1 = single (core itself never creates threads: this selects partitioning)
} ols_problem_t;
typedef struct {
  double *beta;      // p
  double intercept, sse, sst, r2, r2_adj, sigma2, cond_est;
  double *se;        // p+1 or NULL
  int32_t rank, method_used, isa_used;
} ols_result_t;
int32_t ols_fit(const ols_problem_t*, ols_result_t*, void *workspace, size_t ws_bytes);
size_t  ols_workspace_size(size_t n, size_t p, uint32_t method, uint32_t flags);
```
- Error codes (negative): `OLS_E_ARG, OLS_E_ALIGN, OLS_E_SIZE_OVERFLOW, OLS_E_WORKSPACE, OLS_E_UNDERDETERMINED (n <= p+1), OLS_E_NONFINITE, OLS_E_SINGULAR, OLS_E_ILLCOND`. Every error path is tested. Validation happens once in the API layer; kernels trust their contracts.
- Dispatch table resolved once (`ols_init`) from CPUID/XGETBV; thread-safe by construction (idempotent write of the same table).

## 3. Data formats
**Binary (`.olsb`)**: little-endian header, 64 bytes: `magic "OLSB\0\1\0\0"`, `u32 version`, `u32 flags (bit0: row-major, bit1: has y as last column)`, `u64 n`, `u64 p`, `u32 dtype (1 = f64)`, reserved zeros, then `n*(p+1)` doubles. Validate `n*(p+1)*8 + 64 == file size` with overflow-checked multiply. `mmap` the file (`MAP_PRIVATE|MAP_POPULATE` when it fits RAM), optional `madvise(MADV_HUGEPAGE)`. Converter `tools/to_olsb.py` for CSV/NumPy.
**CSV**: header optional; `--target=<col>`; delimiters `,` `;` tab. Number parsing: Clinger fast path (<= 15 significant digits and |decimal exponent| <= 22 -> exact double ops, correctly rounded); otherwise a slow exact path or, for the first release, reject with a clear message and document it. Differential-fuzz against Python `float()`. Eisel-Lemire (needs a 128-bit power-of-ten table) is a COULD item.
**Output**: default human table; `--format=hex` prints C99 hex-floats (exact, trivial to implement); `--format=json`. Shortest-roundtrip decimal printing (Ryu/Grisu) is a COULD item; until then print 17 significant digits via integer arithmetic with an error-bounded algorithm and state it.

## 4. Freestanding Linux x86-64 CLI
- Entry `_start`: `rsp` points at `argc`, `[rsp+8]` = argv; `rsp` is 16-byte aligned at entry, so a plain `call` keeps the ABI invariant.
- Syscall ABI: number in `rax`; args `rdi, rsi, rdx, r10, r8, r9`; clobbers `rcx, r11`; result in `rax` (value in [-4095, -1] is `-errno`). Numbers: `read 0, write 1, open 2, close 3, fstat 5, mmap 9, mprotect 10, munmap 11, madvise 28, clone 56, exit 60, futex 202, exit_group 231`.
- Handle short reads/writes and `EINTR` (`-4`) in loops. Check `mmap` failures (`rax` in the errno range). Exit code: 0 ok, 2 usage, 3 I/O, 4 data/validation, 5 numerical (singular/ill-conditioned).
- Static, no libc, no dynamic loader, non-executable stack; report binary size in README (`size`, `stat`).
- Threading in the CLI: first build variant linked with libc+pthreads; libc-free `clone`/futex pool is a COULD item (own ADR, own tests).

## 5. Build and CI
`make all | test | verify | bench | bench-report | repro | clean`. Variants: `BUILD=debug` (DWARF, no stripping), `release` (stripped CLI), `harness` (shared library for ctypes + C tests). `make verify` = lint contracts + unit + ABI probe + guard + property + NIST + fuzz smoke + mutation smoke. `make repro` = fresh clone -> build -> verify -> machine characterization -> sweeps -> README tables/plots, writing everything under `results/`. CI mirrors `make verify` (see `asm-verification` section 7).

## 6. Documentation standards
- **ADR per decision** (`docs/adr/NNNN-title.md`: context, options with measured/argued trade-offs, decision, consequences, revisit trigger). Mandatory ADRs: NASM vs GAS, ABI strategy, data layout and padding, tile shape, Gram vs QR default, `auto` thresholds, threading model, reproducible-mode semantics, CSV parsing strategy.
- `docs/design.md`: the life of a row (file -> block -> pack -> tiles -> Gram -> factor -> solve -> statistics), with a diagram.
- Every sub-agent writes `docs/reports/<wave>/<agent>.md`: what, how, why, evidence, open issues. Reports link raw results, never paste hand-typed numbers.
- README (generated sections marked): one-paragraph pitch; install/run in 3 commands; **results** (roofline plot, accuracy table LRE by method x NIST dataset, kappa-sweep plot, thread scaling, binary size, mutation score); design overview; honest limitations; **"What I got wrong"** (hypotheses that failed, with data); reproduce instructions (`make repro`).

## 7. Definition of done for any task
Contract header written; scalar/reference match; guard + ABI + property tests green; mutation score for the new code reported; prediction vs measurement logged; docs/report written; commit contains only the task's files; reviewer checklist passed.

## 8. Scope control
MoSCoW lives in PLAN.md. A COULD item may start only when all MUST/SHOULD gates are green and the time-box in PLAN.md remains. New scope requires an ADR stating what it displaces. "One more feature" is the standard way this project dies.
