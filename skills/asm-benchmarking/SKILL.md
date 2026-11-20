---
name: asm-benchmarking
description: Benchmarking, profiling and honest-reporting rules for asm-ols (machine characterization, roofline, baseline ladder, noise control on laptops/Ryzen/WSL2, statistics, result file schema, plots generated from raw data). Load before writing any benchmark, comparing against NumPy/OpenBLAS/C, producing a performance number for docs, or plotting.
---

# Benchmarking and honest performance claims

The failure mode to prevent: a flattering number nobody can reproduce. Rule zero: **no performance number appears in any document unless a script in this repo produced it from raw samples committed under `results/`, at the commit being described.** README tables and plots are generated; nobody types numbers by hand.

## 1. What to measure (and in which units)
- Primary metrics: **time**, **GB/s** (bytes of X actually streamed / time), **GFLOP/s** (flop count defined per algorithm in `bench/flops.md`), **% of attainable bound** where bound = `min(peak_flops, bandwidth x AI)` from `results/machine.json` for the *same* working-set level.
- Cycles/row or cycles/element only with a calibrated clock (see `simd-kernel-optimization` section 1.4).
- Sweeps (each a separate result file): `p in {1,2,3,4,6,8,12,16,24,32,48,64,128}`; `n` sized so the working set lands in L1, L2, L3, and >= 8x L3; threads 1..N; methods {Tier 1, 1b, 2}; precision {double, Dot2}.
- End-to-end CLI timing (file -> coefficients) is reported separately from kernel timing and states cache state (cold file vs page-cached).

## 2. Noise control (laptop/desktop Ryzen, WSL2)
- Pin to one physical core (`taskset -c`; avoid SMT siblings and core 0). Set performance governor where possible; on laptops use AC power, a fixed power plan, and cool-down between sweeps. Boost clocks and thermal throttling are the dominant noise source: randomize run order across configurations, interleave A/B variants (ABABAB), never run all of A then all of B.
- Warm-up runs, then >= 30 timed repetitions (kernels) or >= 10 (end-to-end). Report **median and min**, the coefficient of variation, and the sample count. If CV > 3% the result is flagged unstable and not used for claims.
- Timing: `clock_gettime(CLOCK_MONOTONIC_RAW)` around a repeated call; `rdtscp` + `lfence` only for sub-microsecond kernels, remembering TSC != core clock. Prevent dead-code elimination in C baselines (consume results via `volatile` sink).
- WSL2: wall-clock timing is usable; hardware PMU counters are usually unavailable; the host scheduler adds jitter. State the environment in every result file. Final headline numbers should be re-measured on native Linux (live USB is enough) if any claim depends on counters or sub-2% differences.
- Check that the OS did not migrate the process; check `lscpu` model, microcode, governor; record all in the result file.

## 3. Baseline ladder (same algorithm, same precision, same threads)
Speed claims compare like with like; accuracy claims are separate.
1. Scalar C `-O2` (floor).
2. C + auto-vectorization `-O3 -march=native` (what the compiler gives for free).
3. C + AVX2 intrinsics, written to the same algorithm (the honest "is assembly buying anything" baseline).
4. Library: NumPy `X.T @ X` + `scipy.linalg.solve(assume_a='pos')` and SciPy `dsyrk` + `dposv` (OpenBLAS/MKL) — single-threaded via `OPENBLAS_NUM_THREADS=1` for the single-thread comparison and fully threaded separately. Record `np.show_config()`.
5. Eigen (normal equations and `ColPivHouseholderQR`) as a C++ reference.
6. NumPy `lstsq` (SVD-based, `gelsd`) is **not** a fair speed baseline for Tier 1 (different algorithm, more accurate and slower); show it in a labeled "different algorithm" row, and use it as the accuracy baseline.
`-ffast-math` baselines change numerics: label them or exclude them. If the asm does **not** beat intrinsics or the library in some regime, say so and explain why (that analysis is valuable; hiding it is fatal for credibility).

## 4. Roofline deliverable
`bench/roofline.py` reads `results/machine.json` + sweep results and draws: compute roof, L1/L2/L3/DRAM bandwidth roofs (per-thread and all-thread), measured points for every kernel/p/size, annotated with % of bound. Include the memory-bound -> compute-bound crossover p*. One figure per thread count.

## 5. Result file schema (JSON, one per run, immutable)
`{ "schema": 1, "commit": "<sha>", "dirty": false, "date": "...", "host": {cpu_model, cores, smt, governor, os, wsl, microcode}, "toolchain": {nasm, gcc, python, numpy, openblas}, "config": {kernel, variant, n, p, ldx, threads, precision, isa}, "samples_s": [...], "derived": {median_s, min_s, cv, gbps, gflops, pct_bound}, "notes": "" }` — refuse to plot results with `dirty: true` or from a different commit than the one named in the figure caption.

## 6. Statistics for claims
A claim "A is X% faster than B" needs: interleaved runs, >= 30 samples each, medians, and a non-parametric test (Mann-Whitney U) or bootstrap CI on the ratio of medians. Report the CI. Differences within noise are reported as "no measurable difference".

## 7. Profiling workflow when a kernel misses its bound
1. Re-check the prediction (flops/bytes counted correctly? padding bytes included?).
2. `llvm-mca -mcpu=<znver*> -iterations=100` on the inner loop: static bottleneck (ports, dependencies, frontend).
3. Controlled experiments: remove loads (compute-only), remove FMAs (load-only), vary accumulator count, vary block size B, vary alignment, vary prefetch. Each isolates a resource.
4. Counters if available (cycles, instructions, L1/L2/L3 misses, stalls) via `perf stat` (native Linux) or AMD uProf; otherwise controlled experiments only.
5. Write the explanation in `docs/perf-log.md`, including when the answer is "hardware limit; here is the evidence".

## 9. Reporting pitfalls checklist
Cold vs warm cache stated? Working-set level stated? Threads stated? Same precision and algorithm? Frequency/thermal state stated? Raw data committed? Figure regenerates from `make bench-report`? No cherry-picked sizes (all swept sizes shown)? Negative results included?
