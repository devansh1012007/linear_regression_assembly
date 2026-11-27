# architecture.md — module map, call graph, layouts

Reference companion to `design.md` (rationale). Offsets and sizes here are
illustrative — the frozen truth lives in `src/abi.inc` + `include/ols.h`,
hash-checked by the drift test (ADR-0006).

## 1. Module map (src/core/)

| File | Exports (contract in include/ols.h) |
| --- | --- |
| `cpu.asm` | `ols_cpu_ident` (CPUID leaf 0), `ols_cpu_has_avx2_fma`, `ols_cpu_has_avx512` — chip truth |
| `dispatch.asm` | `ols_init`, `ols_isa` — OS-state-checked feature detection (XGETBV); clamped to implemented kernels (SSE2, AVX2) |
| `api.asm` | `ols_abi_offsets` — FNV-1a over the frozen offset table; consumed by the struct-drift test (ADR-0006) |
| `sums.asm` | `ols_sum_f64_{sse2,avx2}`, `ols_centered_sums_f64_sse2`, `ols_simple_fit*` — Phase-1 primitives + fits |
| `simple.asm` | `ols_quad_fit_f64_sse2` — quadratic basis fit |
| `dot.asm`, `axpy.asm` | `ols_dot_f64_{sse2,avx2}`, `ols_axpy_f64_{sse2,avx2}` |
| `gram.asm` | `ols_gram_f64_{sse2,avx2}` (v3 group-pipelined; v4 fused q≤8 body present but documented-unreachable — dispatch is v3, ADR-0012), `ols_gram_scratch_bytes` — fused shifted-Gram pass |
| `chol.asm` | `ols_chol_f64_sse2` (factor), `ols_chol_solve_f64_sse2` (R R' solve), `ols_chol_inv_f64_sse2` — stride-aware |
| `stats.asm` | `ols_residual_f64_sse2` (r, SSE, X'r), `ols_pivot_means_f64_sse2` |
| `fit.asm` | `ols_workspace_size`, `ols_fit` — validation, orchestration, statistics |
| `qr.asm` | `ols_qr_f64_sse2`, `ols_qr_solve_f64_sse2`, `ols_qr_pivot_f64_sse2` (+ `ols_qr_build_f64_sse2`) — Householder / column-pivoted QR (ADR-0007) |
| `special.asm` | `ols_exp_f64_sse2`, `ols_log_f64_sse2`, `ols_lgamma_f64_sse2`, `ols_ibeta_f64_sse2`, t/F p-value wrappers, `ols_t_qvalue_f64_sse2`, `ols_confint_f64_sse2` (ADR-0010/0011) |

Non-asm: `src/thread/ols_mt.c` (the one sanctioned C file, ADR-0008) provides `ols_gram_mt`.

Every `global` carries a contract header (Args/Returns/Requires/Memory/
Clobbers — `tools/lint_contracts.py` enforces it in `make verify`); integer
args go through `ARG1..ARG6` (abi.inc) so the same source assembles for SysV
and Win64; struct-arg blocks (`ols_gram_args_t`, `ols_resid_args_t`) carry
the >6-argument kernels; cross-object calls use `wrt ..plt` for PC32
link-safety under `-shared`.

## 2. `ols_fit` call graph (Tier-1 gram path; methods AUTO/GRAM_CHOL/GRAM_CSNE/QR/QR_PIVOT accepted — `OLS_CG` reserved)

```
ols_fit(prob, res, ws, ws_bytes)
 ├─ validate: struct_size/version, n≥q, p≤…, method≤4 (OLS_CG reserved → OLS_E_ARG), ws ≥ ols_workspace_size
 ├─ pivots:   ols_pivot_means_f64_sse2  (c = w-means of first min(n,256) rows; c=0 when no intercept)
 ├─ gram:     ols_gram_f64_{avx2,sse2}  (S, s, W shifted moments + G scratch)
 ├─ center:   G = S − s s'/W            (exact centering algebra, fit.asm loops)
 ├─ solve:    ols_chol_f64_sse2 (p-block, stride q, eps_rel = 8pu)
 │            ols_chol_solve_f64_sse2 (β from G[p][p] column)
 ├─ intercept:b0 = ȳ − x̄'β              (pivot-shifted means, fit.asm loops)
 ├─ stats:    ols_residual_f64_sse2     (r, SSE, X'r — XTR slot reserved for CSNE)
 │            sst = G[p][p] (centered; raw without intercept)
 │            r2, sigma2 = sse/max(1,n−p−int), r2_adj (degenerate→r2), f_stat (v3)
 ├─ se:       p solves of C z = e_j     (z[j] = (X'X)⁻¹_jj; se_j = √(σ²·z[j]))
 │            se_b0 = √(σ²·(1/W + m'C⁻¹m))
 ├─ cond_est: (max R_jj / min R_jj)²    (lower bound, chol path)
 └─ meta:     rank=p, method_used, isa_used (fresh ols_isa), t = β/se (v2)
```

Methods 3-4 (`OLS_QR`, `OLS_QR_PIVOT`) are wired (Phase 3/4, ADR-0007 —
see `qr.asm` and the QR branch in `fit.asm`). `OLS_CG` (method 5) still
returns `OLS_E_ARG` — reserved.

## 3. Workspace layout (doubles; total = 8·(q² + 3q + p + 2) + gram scratch + 64)

| Region | Size | Role |
| --- | --- | --- |
| `C` | q | pivots → column means m (stashed, reused by intercept-se) |
| `G` | q×q | shifted Gram → centered → Cholesky factor (upper, stride q) |
| `S` | q | shifted first moments (destroyed by se solves) |
| `W` | 1 | Σw |
| `RHS` | q | solve buffer (β, then e_j / m RHS) |
| `XTR` | p (+1) | X'r from the residual pass (+ SSE scalar in the slack slot) |
| tail | scratch+64 | gram PKG/GSCR buffers, 64-aligned *before* the call |

The +64 slack exists because the kernel aligns its base up internally; the
fit hands it a pre-aligned tail so the worst-case align-up cannot escape the
caller's buffer (the second corruption lesson).

## 4. Struct geometry (v3, ADR-0006)

- `ols_problem_t` 88 B: `struct_size, version, X, y, w, n, p, ldx, method,
  flags, lambda, rcond, threads, pad`. Version must equal 1 (problem v2 is
  deferred to `ols_predict`, P3.4).
- `ols_result_t` 104 B: `beta* (0), intercept (8), sse (16), sst (24),
  r2 (32), r2_adj (40), sigma2 (48), cond_est (56), se* (64), rank (72),
  method_used (76), isa_used (80), pad (84), t* (88), f_stat (96)`.
  v1 callers (88 B) and v2 (96 B) keep working: v3 fields are the *last*
  append, and the error path zeroes every scalar through offset 96.
- `ols_gram_args_t` 96 B / `ols_resid_args_t` 88 B: caller-built register
  blocks — one pointer deref instead of stack-spilled 7th+ args.

## 5. Test pyramid (what gates what)

| Level | Harness | Catches |
| --- | --- | --- |
| L1 | `tests/py/run_all.py` shapes/offsets | boundary sizes, element-offset misalignment |
| L2 | ABI register probe | calling-convention violations (callee-saved canaries) |
| L3 | guard pages (`tests/guard/*.c`) | OOB reads/writes at page boundaries |
| L4 | kernel differentials + cross-ISA | wrong math; ISA-dependent drift |
| L5 | NIST StRD via `ols_fit` | conditioning-scale accuracy (9 datasets) |
| L6 | adversarial offsets/scales | pivot-shift failures (1e9 offsets) |
| L7 | mutation suite | survivors = missing test power |
| L8 | property sweeps (shift/scale invariance, κ=1e1..1e4) | algebraic identity breaks |

`make verify` = lint + harness + all of the above. Known open issue: the ABI
probe subprocess crashes after reporting (agent-comms 2026-11-21); the probe
itself runs isolated and its mask verdicts stand.
