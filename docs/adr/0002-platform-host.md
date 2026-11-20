# ADR-0002: Platform — Linux x86-64 ELF, and the ACTUAL dev machine

**Status:** Accepted (Phase 0) **Date:** 2026-11-20

## Context
PLAN.md §4 assumed a Ryzen-class AVX2+FMA3 machine WITHOUT AVX-512, >= 8x L3
RAM for DRAM-resident benchmarks. Phase 0 measured the actual host. This ADR
records the deviation and its consequences.

## Measured host (results/machine.json, commit pending first commit)
- Intel Xeon (Sapphire Rapids family, model 173) under KVM, **2 vCPUs**, no
  SMT exposure, 4.06 GiB RAM.
- CPUID: AVX2+FMA3 **and AVX-512F/DQ/BW/VL + AMX** present; OS XCR0 enables
  ZMM state (ols_cpu_has_avx512() == 1, verified).
- Effective core clock 3.567 GHz (depadd chain), FMA peak 57.61 GFLOP/s
  (2 FMA/cy, knee at 8 accumulators — matches Skylake-SP-class 4-cycle FMA
  latency; Zen 2/3 also 4-5).
- Bandwidth: L1 195 GB/s, L2 144 GB/s, L3 22.6 GB/s (shared, cloud-noisy),
  DRAM 16.2 GB/s single-thread, 23.3 GB/s with both vCPUs.
- cgroup v1; no CPU quota visible; PMU (`perf stat`) unavailable in this
  sandbox (not needed for G0; asm-benchmarking §2 anticipated this).

## Deviations from PLAN.md assumptions
1. **AVX-512 exists on the dev machine.** PLAN.md said "assume no AVX-512
   unless CPUID says so" — CPUID says so. Decision: keep AVX2+FMA3 as the
   *primary* target (PLAN scope unchanged; AVX-512 remains a COULD item with
   its own measurement, exactly as the plan requires). Do NOT silently make
   AVX-512 primary; that needs an ADR + Phase-6 time-box.
2. **8x L3 exceeds RAM** (8 x 504 MiB = 4.03 GiB vs 4.06 GiB total). The DRAM
   working set is capped at 1.5 GiB (single) / 1 GiB per process (all-thread)
   so the benchmark host itself stays out of swap. "DRAM-resident" claims on
   this host therefore mean ">= 3x the *measured effective* L3 slice" — weaker
   than PLAN's 8x; headline DRAM numbers must be re-measured on a bigger
   machine before being published as claims (recorded as open issue in the
   Phase-0 report).
3. **2 vCPUs** limit thread-scaling evidence (H5): 1 -> 2 threads only. H5's
   "N threads reach >= 85% of measured all-thread bandwidth" is testable but
   the N range is tiny; document when G4 is reached.

## Decision
Linux x86-64 ELF, SysV ABI, gcc + NASM toolchain (in-sandbox NASM 2.16.03
built from source at /home/z/opt/nasm because apt is unavailable without
sudo; CI uses apt's nasm). Primary ISA: AVX2+FMA3. Baseline reference: SSE2
scalar. AVX-512: COULD item, hardware present, decision deferred to Phase 6
with measurement.

## Consequences
- All Phase-0 numbers are cloud-vCPU numbers: CV > 3% results are flagged per
  asm-benchmarking §2 and not used for claims.
- Bandwidth-bound gate targets (G2's >= 85% of single-thread DRAM) refer to
  the *measured* 16.2 GB/s figure from machine.json at the same commit.
- The QEMU dispatch tests (Haswell/SandyBridge -cpu) from asm-verification §7
  are a CI-only item; qemu is not installed in this sandbox (open issue).

## Revisit trigger
Migration to a bigger host (more RAM/cores) -> re-run `make bench` and
regenerate machine.json before any Phase-2 kernel is tuned against it.
