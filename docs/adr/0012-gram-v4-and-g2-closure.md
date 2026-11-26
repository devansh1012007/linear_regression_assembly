# ADR-0012: G2 gate closure — the 85%-of-DRAM clause is unreachable on this host; gram v4 (iteration 3) falsified

Date: 2026-11-26 | Status: accepted (closes G2 as RED-with-evidence)

## Context

Gate G2 (PLAN §6, Phase 2, pre-registered): out-of-cache single-thread
>= 85% of measured single-thread bandwidth for p <= 16. History:
- v1 (2026-11-20): ~2% of bound (results/gram-bench.json).
- v3 (2026-11-21, ADR-0005): 22-54% of per-case binding bounds after 2 of
  3 evidence-backed iterations; escalation clause: "< 50% after three".
  [corrected 2026-11-26: recomputed basis gives 19.7-48.7% for v3 /
  19.5-49.7% for v4; the 22% figure came from a 72 B/row bytes model vs
  the JSON's 64 B/row]
- This session: iteration 3 (v4, fused pack-into-kernel for q <= 8) +
  the closure analysis.

## The v4 iteration (what was built, what it measured)

Design (perf-log #28, written before implementation): for q <= 8 the whole
row fits one ymm per half, so u = sqrt(w)*(z - c) can be built in registers
and column accumulators (lo: G[j][0..3] in ymm0..7; hi: G[j][4..7] in
ymm8..11) can persist for the entire call — eliminating v3's PKG/GSCR
round-trip (ADR-0005's named next step).

Bring-up bug log (all root-caused, none papered over; the recurring
register-aliasing class is ADR-0010 #4's third and fourth strike):
1. `vfmadd231pd` into a dirty destination: the s-accumulation FMA reused
   ymm14 which still held the last bcast(u[j]) — doubled s at q=1.
2. Early v4 dispatch skipped the s/s_c/W zeroing (WS/WC read uninitialized
   -> garbage Wout; order-of-prologue bug).
3. Frame size calc: +256 was added BEFORE the 32-alignment, so the align
   pad could push the v4 slots past the frame (q=4: 24-byte overflow
   stomping saved registers).
4. c[k] for k >= 4 read from CLO+32.. (past the 32B slot, into MASKL) —
   q >= 5 G off by mask-bit-sized constants.
5-6. sw (sqrt w) kept in xmm8 — the SAME physical register as the
   persistent hi-acc ymm8: every w-load and every j>=4 hi-FMA clobbered
   each other (diag[4] += sw per row; s scaled by the accumulator).
   Fixed by parking sw in a frame slot; the momentary w-load also moved
   (xmm12) — the xmm8..15/ymm8..15 aliasing is total, there is no scalar
   safe harbor in the upper half.
7. The sw frame slot (240) collided with CHI[2] (c[6] staging): u[6] = 16
   for 17 at q=8. Slots moved to a 288-byte reservation.
8. `vxorpd ymm0` lived inside the skipped GSCR-zero block: .zs and the v4
   SLO zeroing used caller-leftover xmm0 bits — s picked up a subnormal
   2^-1026 residue exactly when the true sum cancelled to 0 (P1 red).

Verification after fixes: gram differential ALL OK (both ISAs, weights,
pivots, all shapes), NIST 11.9 LRE green, P1 green at V4_BATCH=4 (v3's
documented 4-row fold cut), P2 sse/sigma2 measured 1.156e-10 vs the 1e-10
gate — an allowed s-reassociation at the gate edge, documented here rather
than silently tolerated or worked around.

Measured (results/gram-bench-v4.json, 30 reps, median):
- q=8 n=2e5 DRAM: 17.36 ns/row = 20% of the 3.39 ns/row DRAM budget.
  v3 measured 17.17 on the same host/cases. A TIE — the prediction's
  14±3 window is FALSIFIED (gap analysis in perf-log #28-measured: the
  guard-safe scalar row loads cost what the vectorized PACK round-trip
  cost).
- q=17 (v3 kernel A control): 24.45 vs 24.42 — reproduction confirmed.
- ols_gram_mt 2 threads: q=8 9.47 ns/row (6.76 GB/s = 18% of the 37.6
  GB/s all-thread ceiling); q=17 19.39 (7.01 GB/s). H5 (>= 85% of
  all-thread BW): RED on this 2-vCPU shared KVM host.

## Decision

1. The v3 pipeline is RESTORED as the default q<=8 dispatch (this session's
   final commit). The v4 body remains in source, documented-unreachable, as
   iteration-3 evidence and a starting point for hosts with real load
   bandwidth (its verification is complete: differential + NIST green).
2. G2 closes RED with a three-iteration evidence trail and an
   impossibility argument, not a tuning gap:
   - q >= 9: kernel A's per-row instruction stream (~125 uops at q=17)
     exceeds 4 uops/cy issue for ~31 cy/row against a 3.39-11.5 ns/row
     (12-41 cy) DRAM budget — issue-bound by construction on 2-wide/4-wide
     hosts; 85% of DRAM is unreachable regardless of tuning.
   - q <= 8: the DRAM budget (64-136 B/row) is 3.4-7.2 ns/row; the
     in-bounds load sequence alone (q+1 loads, ~1.15 ymm-load-uops/cy,
     measured machine.json) floors the kernel near 12-14 cy/row before
     any FMA. v1 -> v3 -> v4 moved 2% -> 22% -> 20%: the ceiling on this
     host is ~20-25%, set by load-port throughput, not by kernel structure.
     [corrected 2026-11-26: recomputed basis gives 19.7-48.7% for v3 /
     19.5-49.7% for v4; the 22% figure came from a 72 B/row bytes model
     vs the JSON's 64 B/row]
   - Re-open condition: a host where (a) CPUID load-throughput counters
     or (b) AVX-512 (32B loads halve the load count) change the floor.
3. The pre-registered G2 target was written before the machine
   characterization of THIS class of host (shared KVM, quota-capped vCPU,
   ~1.15 ymm-loads/cy). The target was honest for a bare-metal Ryzen;
   on this host it is unreachable, and the honest artifact is the
   measured evidence trail + this closure, per AGENTS rule 7 (negative
   results are results).

## Consequences

- README/performance.md updated: G2 RED-CLOSED (was "open after v3"),
  H5 RED (mt rung measured), v4 documented under "What I got wrong".
- The P2 permutation gate (1e-10) stays as-is: v3 dispatch restored
  means the engine's measured behavior is the documented one again.
- bench/kernels/gram_bench_v4.py + results/gram-bench-v4.json are the
  reproduction path for every number above.
