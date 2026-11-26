# PLAN.md — asm-ols

Status legend: targets marked **[pre-registered]** are written before measurement; revising one requires a logged reason (`docs/adr/`), never a silent edit. Estimates are in *focused days* and are guesses: after Phase 1, compare actual vs estimate and rescale the rest.

## 1. The real problem (read first)
Linear regression is cheap. For realistic sizes it runs in milliseconds in NumPy, so "I wrote it in assembly" is not impressive by itself, and hand-written assembly often only *ties* compiler intrinsics. What impresses a skeptical engineer is **assembly + rigor**:
1. it gets the right answer where simple implementations don't (NIST StRD, ill-conditioning, huge offsets);
2. it runs at a *measured* hardware limit, with an honest account of where it loses;
3. its correctness is evidenced at register/memory level (ABI probe, guard pages, mutation score);
4. the write-up shows reasoning and mistakes, not just a speed bar.
The defensible performance story is **pass fusion on tall-skinny data**: for small p, OLS is memory-bound, so wall time ~ number of passes over X. One fused pass (centered Gram + X'y + y'y) versus a typical library pipeline (center -> gemm -> solve, several passes and 2x flops) should win by roughly the pass-count ratio. That is hypothesis H1 below, to be tested, not assumed.

## 2. Hypotheses to test (each gets a verdict in the README)
- **H1** In the memory-bound regime a fused single-pass kernel beats NumPy/SciPy pipelines by about the pass-count ratio.
- **H2** Hand-written asm is within ~5% of well-written AVX2 intrinsics; the value of asm here is control (generated tiles, padding scheme, register allocation, tiny freestanding binary), not raw speed. (May be falsified either way.)
- **H3** Dot2 (double-double) accumulation costs <15% wall time when memory-bound.
- **H4** CSNE restores QR-like accuracy up to some kappa* that we measure; plain Gram+Cholesky fails near kappa ~ 1e8.
- **H5** One thread cannot saturate DRAM; N threads reach >= 85% of the measured all-thread bandwidth.
- **H6** For p+1 above the roofline crossover p*, the 6x8 tile reaches >= 70% of measured FMA peak in cache.

## 3. Scope (MoSCoW)
- **MUST:** simple regression; multiple regression via fused centered Gram + Cholesky (SSE2 reference, AVX2+FMA3 primary); accurate SSE/R^2; standard errors + t-stats; CPUID/XGETBV dispatch; C ABI `ols_fit`; binary I/O + CLI (Linux, libc-free); full test pyramid; NIST table; Phase-0 machine characterization; roofline + baseline comparison; generated README; write-up.
- **SHOULD:** CSNE refinement; Householder/TSQR QR with `[X y]` byproduct SSE; `auto` method; ridge; weights; rank detection; condition estimate; multithreading with fixed-order merge + reproducible mode; CSV input; mutation testing; kappa-sweep study.
- **COULD (value order):** Dot2 accumulation path; p-values/CIs (asm exp/log/lgamma/incomplete beta, own accuracy test vs mpmath); Win64 port; AVX-512 path (only if hardware has it and it measures faster); lasso/elastic net by coordinate descent; CG/LSQR on the Gram; Eisel-Lemire CSV parsing + shortest-roundtrip printing; libc-free thread pool (`clone`+`futex`); ARM64 NEON port; leverage/Cook's distance.
- **WON'T:** GPU, sparse, GLMs, autodiff, distributed, GUI.

**Minimum Impressive Product (MIP):** Phases 0-2, CSNE from Phase 3, NIST table, roofline, README. If time runs out, ship the MIP; it is already a strong artifact.

## 4. Assumptions and decisions to confirm in Phase 0 (record as ADRs)
- Primary target: Linux x86-64 ELF, SysV ABI. If you develop on Windows, use WSL2 for all correctness work and wall-clock benchmarks (PMU counters are usually unavailable there); a native Win64 build is a COULD item.
- CPU: Ryzen-class => AVX2+FMA3 baseline; check CPUID for AVX-512 (Zen 4+). Measure FMA latency/throughput and bandwidth instead of trusting tables.
- Assembler: **NASM** (macro power for generated tiles) vs GAS-intel (CFI/unwind, one toolchain). Default NASM; ADR records the trade-off.
- Memory: benchmark sizes need >= 8x L3; confirm RAM and cap `n` accordingly (e.g. p=16, 8 bytes/double: 2e7 rows = 2.7 GB).
- Python harness via ctypes against a `.so` build of the core; NumPy/SciPy/mpmath/hypothesis available.
- Audience/goal (portfolio piece, learning, or both) decides how much to polish writing vs. extend features; decide now, it changes the time-box.

## 5. Architecture snapshot
```
CLI (_start, raw syscalls, mmap)  -->  ols_fit (C ABI, validation, workspace)
   -->  solvers: Cholesky | CSNE | TSQR | ridge | weights | rank
   -->  kernels: fused centered Gram+X'y+y'y | residual+X'r | Householder update | trsv | dot/axpy
   -->  dispatch table (CPUID+XGETBV): sse2 ref | avx2+fma | avx512(opt)
Data: row-major Z=[X|y], zero-padded packed blocks, pivot-shifted sums, Chan merge, fixed-order thread merge
```
Key kernel hypothesis: 6x8 tile of G (12 ymm accumulators), per row: 2 loads + 6 broadcasts + 12 FMAs; upper triangle only; zero padding removes tails; narrow dedicated kernels for p+1 <= 8. Validate with llvm-mca and measurement; change if the evidence says so.

## 6. Phases, tasks, gates

### Phase 0 — Foundations and machine truth (2 d)
Tasks: repo skeleton, `abi.inc`, Makefile (`verify`, `repro`), CI skeleton, contract linter; ABI probe + guard-page harness (templates exist in skills; extend + self-test); Python references (`numpy`, `mpmath`, kappa generator, NIST parser + certified data); machine characterization (FMA peak vs accumulators, load throughput, cache/DRAM bandwidth per thread and all threads, clock calibration) -> `results/machine.json`; ADRs: assembler, platform, layout.
**Gate G0:** probe catches the seeded bad function; guard test passes on the dot template; measured FMA peak within 10% of the theoretical value for the CPU (else the harness is wrong); `machine.json` committed; roofline crossover p* computed.

### Phase 1 — Simple regression end-to-end on a tiny stack (2 d)
Tasks: compensated centered sums + simple regression (SSE2 scalar asm), AVX2 dot/axpy/sum kernels, ctypes bindings, shape/offset matrix, special values, properties, NIST Norris/Pontius/NoInt1/NoInt2, first mutation run.
**Gate G1:** full pyramid green for these kernels; Norris/Pontius LRE >= 10 **[pre-registered]**; mutants killed; estimate-vs-actual recorded (rescale schedule).

### Phase 2 — The Gram engine (6 d)  *(hard task: dedicated agent, heavy context)*
Tasks: packing/padding; fused pass with pivot shift; scalar SSE2 reference kernel; AVX2 narrow kernels (q = p+1 <= 8); generic 6x8 tiled kernel from macros; Chan merge; Cholesky + triangular solves; `ols_fit` Tier 1; stats (SSE via residual pass, R^2, SE, t, cond bound); dispatch + `OLS_FORCE_ISA`; error codes; predictions in `perf-log.md` *before* benchmarks.
**Gate G2:** all tests green on SSE2/AVX2; in-cache >= 70% of compute bound at q in {8,16,32} and out-of-cache single-thread >= 85% of measured single-thread bandwidth for p <= 16 **[pre-registered; if < 50% after three evidence-backed iterations, escalate]**; adversarial-offset tests (x = 1e9+noise) pass; guard/ABI green for every export.

### Phase 3 — Accuracy engine (5 d)  *(hard task: TSQR gets a dedicated agent)*
Tasks: residual + X'r fused pass; CSNE; Householder/TSQR on `[X y]` (SSE = rho^2); column pivoting + rank; ridge, weights; `auto` selection; kappa-sweep study (error vs kappa, per method, zero- and large-residual); full NIST table per method.
**Gate G3:** NIST targets **[pre-registered]**: QR >= 10 LRE on Norris/Pontius/NoInt1/NoInt2 and >= 6 on Longley and Wampler1-3; Gram+CSNE >= 9 on the lower/average set; Filip and Wampler4-5 reported without a target. kappa* for CSNE measured and `auto` thresholds derived from data. Error-bound tests use recorded constants.

### Phase 3 — execution checklist (session 3; elaborates the block above)

Session-3 audit findings that shape this checklist (verified at HEAD `210f08c`, not assumed):
- **API truth bugs**: `OLS_GRAM_CSNE` accepted but aliased to plain Cholesky (no refinement code exists); `problem.lambda` accepted but ignored; `ols_abi_offsets()` declared in `ols.h` and cited by the drift-test docstring but **not exported** from the library; `ols_cpu_ident` exported but undeclared in the header; `ADR-0006` cited in `ols.h` but never written; `ols.h` cites `tests/py/struct_check.py` (the real check lives in `tests/py/test_phase2.py`).
- **Mutation debt** (session-1 comms, still open): no vzeroupper/upper-half discipline check; SSE2-axpy tail case missing.

Order follows the house workflow (reference first, scalar asm before any vectorization; the QR kernel is SSE2-scalar this phase — its performance story belongs to the Phase-5 baseline ladder, but the perf-log prediction is still written before any measurement). Estimates in focused hours; total ≈ 15.

- [x] **P3.0** This checklist + ADR-0006 (result v3 / `ols_abi_offsets` repair) + ADR-0007 (QR scope: Householder on an explicit copy, no pivoting v1) + comms entry. (0.5)
- [ ] **P3.1 API truth**: export a real `ols_abi_offsets` (FNV-1a over the frozen offset table, the `8513e04` design that never landed); drift test consumes the export and fails on any drift; declare `ols_cpu_ident`; fix the stale header references. (0.5)
- [ ] **P3.2 Result v3** (96 → 104 bytes, append-only: `f_stat` at 96, zeroed on error): offsets in `abi.inc` + `ols.h` + both ctypes mirrors; `f_stat` fill in the fit.asm stats block — F = (sst − sse)/sse · (n − q)/p, using the stored `sst` (centered with intercept, raw without — same convention as r2); SciPy differential on random + NIST problems. (1.5)
- [ ] **P3.3 Solver honesty**: (a) **CSNE** — two fixed-point refinement steps via the existing residual pass: d = C⁻¹·X'r, b += d (all existing kernels; orchestration in fit.asm). Gate: recovers ≥ 12 LRE vs mpmath on κ=1e4 problems where plain chol measures ~7. (b) **ridge** — wire `lambda` into the Gram p-block (G_jj += λ, j < p; intercept never penalized), differential vs an mpmath ridge reference. (c) both reflected truthfully in `method_used`. (2)
- [ ] **P3.4 `ols_predict`**: asm kernel (n×p matvec + b0, compensated), differential vs numpy, guard + probe. (1)
- [ ] **P3.5 `qr.asm`**: mpmath Householder-QR reference in `tests/ref/ols_ref.py` FIRST; then asm: explicit workspace copy of A = [1|X|y] (q = p+1 cols), factor (Householder, reflected β choice for stability), R-solve, se via the inv(R) diagonal recursion, sse byproduct (last ρ²); differentials (beta, sse, se) across the shape matrix; guard pages + ABI probe; perf-log prediction before measurement. (4)
- [ ] **P3.6 Wire `OLS_QR`** into `ols_fit`: dispatch, `rcond` → rank(R) check (R_jj ≤ rcond·R_00 → `OLS_E_ILLCOND` with rank filled; rank < p → `OLS_E_SINGULAR`), se from R, sse byproduct, f_stat. `OLS_QR_PIVOT` stays `OLS_E_ARG` (ADR-0007). (2)
- [ ] **P3.7 Kappa sweep**: κ ∈ {1e1..1e8} × {zero, large} residual × {CHOL, CSNE, QR} vs mpmath → `results/accuracy/kappa-sweep.json` + plot; **H4 verdict**: κ* where CSNE drops > 1 digit below QR; `OLS_AUTO` rule derived from that data (cond_est threshold), implemented + tested. (1.5)
- [ ] **P3.8 NIST per-method table**: 9 datasets (download official Wampler4/5) × {CHOL, CSNE, QR} → `results/nist-phase3.json` + README table regenerated from it. Gates as pre-registered above. (1)
- [ ] **P3.9 Mutation debt**: vzeroupper/upper-half discipline check (small C helper per asm-verification §2b) + SSE2-axpy tail in the mutation suite; rerun; survivors re-analyzed. (1)
- [ ] **P3.10 Docs + gate**: `docs/design.md` (cited by AGENTS.md, missing), `docs/reports/phase3-accuracy.md`, README status/tables; `make verify` green; G3 self-check item by item. (1)

Cut order if time runs short: P3.4 predict (caller can matvec) → P3.3b ridge → P3.9 mutation → P3.7 plot polish. Never cut: the CSNE honesty fix, QR + differentials, the per-method NIST table, G3 evidence.

### Phase 4 — Systems: threads, I/O, CLI (4 d)
Tasks: row-partitioned threading (pthreads build), fixed-order merge, reproducible mode (bitwise equal for 1..N threads); `.olsb` + CSV (Clinger fast path), `to_olsb.py`; freestanding `_start` CLI with raw syscalls; exit codes; parser fuzzing.
**Gate G4:** end-to-end on a 1e7 x 16 file matches the NumPy pipeline within bounds; thread scaling reaches >= 85% of measured all-thread bandwidth **[pre-registered]**; fuzz runs clean for a fixed budget; static binary has no dynamic section and a non-executable stack.

### Phase 5 — Evidence and write-up (3 d)
Tasks: full sweeps; baseline ladder (C -O2, -O3, intrinsics, SciPy dsyrk+dposv single/multi-thread, Eigen, NumPy lstsq as accuracy baseline); roofline figures; statistics for each claim; mutation score; `make repro` on a clean checkout (WSL2) and a native-Linux re-run of headline numbers; README generated; "What I got wrong" section; verdicts on H1-H6.
**Gate G5 (project DONE, section 9):** a stranger can clone, run `make repro`, and regenerate every table/plot in the README.

### Phase 6 — Stretch (time-box <= 3 d each, only when G0-G5 are green)
Suggested order by value per effort: Dot2 path (H3) -> p-values/CIs -> Win64 port -> AVX-512 (if present) or ARM64 NEON -> lasso/CG -> libc-free threads.

## 7. Wave plan for the orchestrator (file ownership prevents conflicts)
| Wave | Parallel agents (owned files) | Serial / shared |
|---|---|---|
| 0 | A0 skeleton: Makefile, `abi.inc`, CI, linter (serial first). Then: A machine characterization (`bench/machine/`), B probe+guard+refs (`tests/`), C NIST data + generators (`data/`, `tests/ref/`) | shared files created here only |
| 1 | A sums + simple regression (`src/core/sums.asm simple.asm`), B dot/axpy (`dot.asm axpy.asm`), C tests for A/B, D bench harness | dispatch skeleton: one agent |
| 2 | A pack + fused narrow kernels, B tiled 6x8 generator (dedicated, heavy context), C `chol.asm trsv.asm`, D `stats.asm` | `ols_fit` glue and `ols.h`: one agent, serial |
| 3 | A residual pass + CSNE, B TSQR/QR (dedicated), C ridge/weights/rank, D accuracy study (`tests/nist`, sweeps) | `auto` selection after A-D |
| 4 | A threading, B `io_bin.asm io_csv.asm`, C CLI `main/args/fmt`, D fuzzing | Makefile changes: serial |
| 5 | A sweeps, B baselines, C docs/README/plots, **D fresh-context auditor** (tries to falsify every README claim) | final edit pass by one agent |
Each wave ends with the independent reviewer sub-agent (correctness, optimization opportunities, test quality and extra edge cases), then an improvement step from that review. Hard/long issues get a dedicated agent with heavy context and a callout in the wave summary. Reading list for context packs: Goto & van de Geijn "Anatomy of High-Performance Matrix Multiplication"; BLIS Haswell dgemm kernel; Higham "Accuracy and Stability of Numerical Algorithms"; Golub & Van Loan "Matrix Computations"; Bjorck "Numerical Methods for Least Squares Problems"; Demmel et al. on communication-avoiding/TSQR; Ogita-Rump-Oishi "Accurate Sum and Dot Product"; Williams et al. "Roofline"; Agner Fog's manuals; uops.info; AMD/Intel optimization guides.

## 8. Risks and mitigations
| Risk | Why it matters | Mitigation |
|---|---|---|
| Agent invents or misreports numbers | credibility is the product | rule 1; generated docs; auditor agent in Wave 5 |
| Silent ABI/clobber bugs | crashes far from cause | probe on every export; mutation tests |
| Overreads in tails | works until memory layout changes | guard pages; zero-padded packed blocks |
| Benchmark noise on laptop/WSL2 | false conclusions | pin, interleave, CV gate, native re-run of headlines |
| Unfair baselines | invalidates comparisons | baseline ladder; label algorithm differences |
| Scope creep (p-values, parsers, ports) | project dies at 70% | MoSCoW, MIP, time-boxes, ADR for new scope |
| Kernel stalls < 50% of bound | gate G2 fails | llvm-mca + controlled experiments; escalate with evidence; ship narrower kernel set |
| Accuracy targets unrealistic in double | embarrassing misses | targets pre-registered but honest reporting if missed; Filip excluded from targets |
| Windows/WSL2 tooling friction | time sink | Linux ELF first; Win64 only as stretch |

## 9. Project definition of done
1. Every MUST item shipped; SHOULD items shipped or explicitly cut with reasons in the README.
2. `make verify` green in CI on SSE2 and AVX2 paths; mutation score reported (all killed or documented as equivalent).
3. NIST table, kappa-sweep plot, roofline plots, thread scaling, binary size: all generated from committed raw results.
4. H1-H6 each have a verdict with evidence (including losses).
5. README + `docs/design.md` + ADRs complete; "What I got wrong" is non-empty.
6. A fresh-context auditor agent could not falsify a README claim (or the claim was fixed).

## 10. Cut order if time is short
Stretch items -> libc-free threads -> CSV slow path -> AVX-512 -> column pivoting -> lasso/CG -> weights/ridge polish. Never cut: tests, the roofline honesty, NIST table, the write-up.

## 11. Highest-leverage first move
Do Phase 0's machine characterization and the Phase 1 simple-regression slice end-to-end (asm kernel + ctypes test + NIST Norris) *before* writing more planning or agent infrastructure. That slice exercises the toolchain, harness, references and reporting pipeline, and its actual-vs-estimated time calibrates everything after it.
