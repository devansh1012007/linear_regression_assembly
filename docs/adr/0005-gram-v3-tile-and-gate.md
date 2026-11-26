# ADR-0005: Gram v3 kernel architecture and the G2 gate decision

Date: 2026-11-21 | Status: accepted (supersedes the implicit v1 design)

## Context
The v1 Gram kernel failed the G2 perf gate at ~2% of the bandwidth target
(results/gram-bench.json). simd-kernel-optimization §3 prescribes packed
zero-padded blocks, a 6x8 %rep-generated tile, register-resident
accumulators flushed once per block, dedicated narrow kernels. PLAN §5
allows deviating when host evidence says so.

## Decision
1. The skill's 6x8 register tile assumes 2 FMA/cycle Zen-class hardware
   with ample load bandwidth. This host (Xeon, AVX-512 capable but
   quota-limited vCPU) measured: 1-2 FMA/cyc (knee 15 accumulators),
   ~1.15 ymm-load-uops/cyc, ~1 store/cyc, DRAM 16.2-18.9 GB/s 1t. The
   binding resource for tall-skinny Gram is the per-row FMA count and the
   load count, so v3 uses:
   - kernel C (q<=8): two-half-row loads, q column accumulators + hi-half
     haccs, per-group flush into a column-major G scratch (PACK clobbers
     ymm8-15, so accs must not persist across groups);
   - kernel A (q>=9): per (i0,j) 8-row chunks: 2 loads + 4 bcast + 8 FMA
     + 2 stores, u-chunks hoisted;
   - PACK: pivot-shift, sqrt(w) once per row, width-limited 4-double
     chunks, scalar tail, NaN row-sentinel.
2. All scalar ops in the AVX2 kernel are VEX-encoded (the SSE/VEX
   transition penalty cost ~2x; see perf-log 2026-11-21).
3. G2 gate: NOT met (23-54% of binding bounds vs >=85% required), after
   2 of 3 evidence-backed iterations. The escalation rule (<50% after
   three) is not triggered. Phase-2 MUST-completeness items proceed ahead
   of further kernel tuning; the >50% cases (q=65: 54%) show the kernel A
   core is within 2x of its FMA bound and the remaining gap concentrates
   in the pack/PKG round-trip.

## Consequences
- ols_gram_args_t grew scratch/scratch_bytes fields (size 96, versioned).
- ols_gram_scratch_bytes(q) is the single source of scratch sizing.
- v1's 4-row panel kernel is retired; SSE2 reference unchanged.
- Further tuning (pack-into-kernel fusion) is the documented next step.
