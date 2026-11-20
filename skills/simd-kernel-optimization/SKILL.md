---
name: simd-kernel-optimization
description: Method and design rules for the performance-critical kernels of asm-ols (centered Gram/X'y accumulation, residual pass, dot/axpy, triangular solves) on AVX2+FMA3 CPUs such as Ryzen. Load before designing, writing, tuning or benchmarking any SIMD kernel, choosing tile sizes, deciding between memory-bound and compute-bound tactics, or adding multithreading.
---

# SIMD kernel optimization for OLS

## 0. Method (non-negotiable)
`model -> predict -> implement -> measure -> explain the gap -> log`. Every kernel gets, *before* the first benchmark, a written prediction in `docs/perf-log.md`: the binding resource (FMA pipes, load ports, L1/L2/L3/DRAM bandwidth), expected cycles per unit of work, expected % of attainable peak. After measuring, the gap must be explained with evidence (llvm-mca output, counters if available, a controlled experiment). "It's faster now" is not an entry. Negative results are logged too.

## 1. Know the machine first (Phase 0 deliverable: `results/machine.json`)
Measure, don't assume. Microbenchmarks, each in asm or intrinsics, pinned to one core:
1. **FMA peak:** register-only `vfmadd231pd` loop with N independent accumulators, N swept 1..16. Find the knee: accumulators needed = (FMA latency) x (FMA/cycle). On Zen 2/3 expect roughly 8-10; verify on uops.info and by measurement. Output: peak GFLOP/s (double) and cycles per FMA.
2. **Load throughput:** `vmovupd ymm, [mem]` over an L1-resident buffer: bytes/cycle. A dot product needs 2 loads per FMA, so this often binds before FMA does.
3. **Bandwidth ladder:** AVX2 read-only streaming over working sets sized to L1, L2, L3, 8x L3 (DRAM): GB/s each, **per thread and for all threads**. A single core usually cannot saturate DRAM (latency x concurrency limit); multithreading is a *bandwidth* tool for tall-skinny problems.
4. **Effective core clock:** run a 1-cycle dependent chain (`add rax, 1` x N) alongside timing to convert seconds to core cycles; TSC ticks at a fixed rate while boost clocks vary. Report time-based metrics (GB/s, GFLOP/s) as primary; cycles only with calibrated clock.
5. Peak ISA support: AVX-512 present? (Zen 4 executes it as 2x256-bit; do not assume a win.)

llvm-mca (`-mcpu=znver2|znver3|znver4`) and uops.info give static throughput/latency; **uiCA models Intel cores only**, so on Ryzen use llvm-mca plus measurement. `perf` PMU counters are usually unavailable under WSL2; on Windows use AMD uProf, on native Linux use `perf stat`.

## 2. Work and traffic for the target problem (n >> p)
Augment the data as `Z = [X | y]` (p+1 columns). One pass computes the Gram `G = Z'Z` (upper triangle only), which contains `X'X`, `X'y`, `y'y`.
- FMAs per row: (p+1)(p+2)/2; flops = 2x that. Bytes per row: 8(p+1) (plus padding).
- Arithmetic intensity ~ (p+1)/8 flop/byte for large p. With DRAM at B GB/s the attainable rate is min(peak_flops, B x AI).
- **Memory-bound region (small p):** the goal is to saturate bandwidth with a *single fused pass* (Gram + column sums/pivot shift + y terms together) and the fewest total passes. Spend spare flops on accuracy (compensated or double-double accumulation, section 6); they are free here.
- **Compute-bound region (large p, or data in L2/L3):** register-blocked microkernel quality decides everything.
- The crossover p* = peak_flops / (B x 1/8) is computed from `machine.json` and plotted; don't guess it.

## 3. Candidate microkernel design (hypothesis to validate, not gospel)
Row-major `Z` already provides the layout a rank-1-update kernel wants: for each row r, the feature values `z[r, j0..j0+7]` are contiguous.
- Tile of G: MR x NR = 6 x 8 doubles -> 12 ymm accumulators (6 rows x 2 vectors). Per row r of the block: 2 `vmovupd` loads (features j0..j0+7), 6 `vbroadcastsd` (features i0..i0+5), 12 `vfmadd231pd`. Registers: 12 acc + 2 loaded + 1-2 broadcast temp = 15-16 of 16 ymm. Per row: 8 load-port uops vs 12 FMAs -> FMA-bound only if the machine has enough load throughput; check with llvm-mca.
- Only tiles with `j-block >= i-block` (upper triangle) are computed; mirror once at the end.
- **Zero-pad the feature dimension to a multiple of NR (and the row stride to a multiple of 4 doubles) in the packed block copy.** Padding removes every column tail and every masked op from the hot loop. Row tails (n mod block) cost nothing special because the k-loop is over rows.
- Block over rows so one block of Z (B x (p+1) doubles) stays in L1/L2 while all tiles consume it; tile accumulators stay in registers across the block, flush (add into the running G) once per block.
- Small p (<= 4..8) does not fill a 6x8 tile: write dedicated narrow kernels (p+1 = 2, 3, 4, 5..8) via the same macro with different `%assign`s, selected by a dispatch table.
- Prefetch: first rely on the hardware prefetcher (pure sequential streams). Add `prefetcht0/nta` only if an A/B measurement shows a gain; record the result either way.
- Alignment: keep packed blocks 64-byte aligned; for the unpadded source matrix use unaligned loads; never let a vector straddle into unmapped memory (the contract and guard-page tests enforce this; last block is copied/padded when needed).

## 4. Other kernels
- **Residual pass** `r = y - Xb`, fused with `X'r` for CSNE refinement and with SSE: one pass, `b` held in registers/L1; it is a memory-bound GEMV-like pass.
- **Cholesky / triangular solves (p x p):** tiny relative to n p^2; correctness and numerical care first. Vectorize inner dot products; no blocking until p > ~64.
- **Householder/TSQR update:** vectorize across columns of the row-major block (updates of trailing columns are contiguous).
- **Reductions:** reduce lanes once per kernel call, never per row; fixed reduction tree if reproducible mode is on.

## 5. Multithreading (after single-thread kernels are at their bound)
- Partition rows into contiguous chunks; each thread owns a private, cache-line-padded (64-128 B) accumulator set; merge with a **fixed-order tree** (so results do not depend on thread timing). Centered statistics merge with Chan's pairwise formulas (see `numerical-ols-algorithms`).
- Reproducible mode: block boundaries independent of thread count -> bitwise-identical results for 1..N threads. Default mode may re-balance.
- Thread creation: first use pthreads in the harness/CLI build; a libc-free `clone` + `futex` pool is a COULD item.
- Scaling plot (1..N threads vs the measured DRAM ceiling) is a required result.
- Huge pages for large buffers: `madvise(MADV_HUGEPAGE)`; measure TLB effect, don't assume.

## 6. Spending free flops on accuracy
- Error-free transformations with FMA: `TwoProd: p = a*b; e = fma(a, b, -p)` (exact). `TwoSum: s = a+b; bb = s-a; e = (a-(s-bb)) + (b-bb)`.
- Dot2 accumulation (Ogita-Rump-Oishi): result is as accurate as computing in twice the working precision. In the memory-bound region the extra ~10x flops should cost little wall time; **measure that claim** per p.

## 7. Anti-patterns (reject in review)
`loop`/`enter`; horizontal ops in hot loops; `div`/`sqrt` per row; mixing legacy SSE and VEX; unaligned 32-byte stores that cross cache lines in hot loops; 4K-aliasing between streams (offset buffers by 64-256 B if a profile shows it); denormals in hot paths (check inputs); copying a working kernel instead of generating it from a macro; optimizing without a written prediction; benchmarking a debug build or a cold cache and calling it a result.

## 8. Experiment log format (`docs/perf-log.md`)
`### YYYY-MM-DD kernel/variant | hypothesis | change | predicted | measured (median, CV) | verdict | evidence link`. Each entry cites the raw result file and the commit.
