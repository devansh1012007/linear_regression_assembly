# Phase 5 — session 5 (2026-11-25): QR_PIVOT wiring closed, threads, baseline ladder

Execution: PLAN §Phase 4c + §Phase 5 (partial) · commits 747e390 → HEAD on
`fix/review-issues` · `make verify` ALL GREEN at HEAD (12 asm files + the
sanctioned src/thread/ C layer).

## Delivered

### 1. OLS_QR_PIVOT fit wiring closed (`747e390`)
The phase-4 open item is done. Two bugs found by binary-marker patching
(a technique worth recording):
- **Dispatch**: `cmp OLS_M_QR; jne .gram_path` sent method 4 down the GRAM
  path, whose chol refused the rank-deficient G (`.fsing`) — the mystery
  rc=−7/rank=−1. Method dispatch now routes 3 AND 4 to the QR path.
- **The sentinel**: `xorpd xmm0, xmm4` negated by XOR-ing with the
  THRESHOLD VALUE — when rcond ≤ default, xmm4 == xmm0 bitwise, and
  `xorpd(x, x) = +0.0`: the kernel received +0.0 (no pin, threshold 0,
  never rank-deficient). Fixed to sign-mask negation. Found by: LD_PRELOAD
  argument shims (which first LIED — a varargs mismatch in my own shim,
  worth remembering), then binary patches (err_zero reset → +1, chol
  fsing → −73, OOB rcond load → pointer-bits double, forced-constant
  sentinel).
- Method 4 end-to-end: full-rank mixed-scale LRE 12-14 vs mpmath;
  dup-column refusal with detected rank=3; rcond<0 pin pass-through;
  no-intercept LRE 15; se/t zeroed per contract; plain QR/chol/CSNE
  regressions green. `test_pivot_fit` in the suite.

### 2. Phase 4c threads (`131b694`)
- `ols_gram_mt` (src/thread/ols_mt.c, the ONE sanctioned non-asm source
  under src/ — ADR-0008): T−1 pthreads + main, per-thread partials (same
  pivots, own kernel scratch) in ONE caller block, **fixed-index-order
  merge** (never completion order).
- Honest reproducibility contract (ADR-0008): same T ⇒ bitwise identical
  across runs/schedulings (verified for T=4 reruns); across T within ~n·u
  relative — bitwise 1-vs-N is impossible with a partials merge (different
  float addition order; measured ~1e-15 rel). The PLAN's original wording
  is revised rather than quietly weakened — the ADR explains why.
- Scaling in the quota-capped sandbox: n=2e6, p=7 — 41.8 ms (T1) → 20.3 ms
  (T2) → 19.7 ms (T4): T=2 saturates this allocation; real scaling verdict
  awaits bare metal.

### 3. Phase 5 baseline ladder (ADR-0009)
- `bench/ladder/` (C: naive-O2/O3, AVX2+FMA intrinsics, asm-v3; python:
  NumPy/OpenBLAS Z'Z) across n ∈ {2e5, 1e6} × q ∈ {8, 17};
  `results/ladder.json`.
- **Honest verdicts: H1 RED, G2 RED** [label corrected 2026-11-26: the
  asm-vs-OpenBLAS comparison is PLAN's H1, not H2; the original report
  wrote H2 — see ADR-0009's correction note]. OpenBLAS wins 3 of 4 cases (2.2x at
  q=8 — it uses the whole machine; the ladder's asm rung is single-
  threaded). asm-v3 wins exactly one: q=17, n=1e6 (DRAM-bound), +11%.
  Two recorded surprises: gcc -O3 made the intrinsics rung SLOWER
  (vectorizer broke the broadcast/FMA stripe); asm-v3's edge over compiled
  C grows with q (1.09x → 2.9x) but has no q=8 headroom here.
- Ladder bugs fixed en route: the first draft aliased y into Z's last
  column (the kernel walks y with stride 8, not the row stride — maxdiff
  81 at n=1e5); the bench now copies the column out.

## Not finished (and why)
- Bare-metal `make repro` re-run of the ladder + thread scaling (this
  sandbox is CPU-quota capped; the artifacts are committed for it).
- PR push — still blocked on a GitHub credential (kit ready in
  download/pr-resume-kit; rerun `scripts/make_staging.py` first).
- Freestanding libc-free CLI (PLAN 4a stretch): the CLI links libc;
  the .olsb/CSV loaders are the pieces that would need raw syscalls.
- p-values/CIs and the other COULD items — out of scope this session.

## Open next session
1. Push via the resume kit once a PAT exists.
2. Bare-metal re-run: ladder + ols_gram_mt scaling (G4/H1 verdicts may
   change; the honest wording is already in ADR-0009).
3. Optional: p-values (asm lgamma/incomplete beta) — the next accuracy
   feature in the COULD list.
