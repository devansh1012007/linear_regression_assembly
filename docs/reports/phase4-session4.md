# Phase 4 — session 4 (2026-11-25): PR staging, mutation closure, CLI + I/O, pivoted QR

Execution: PLAN.md §Phase 4 (partial) · commits b05ef06 → 7ec00d6 on
`fix/review-issues` · `make verify` ALL GREEN at 7ec00d6 (12 asm files).

## Delivered

### PR staging regenerated (user item 1)
- `scripts/make_staging.py` (new): rebuilds `staging.json` from git truth —
  the phase-2 kit predated Phase 3 (34 files; HEAD has 92 tracked).
  Result: **56 add/mod files in 10 logical batches (515 KB) + 16 deletes**,
  36 base-identical files validated and intentionally skipped. Completeness
  invariant holds: staged + base-identical == HEAD exactly.
- `gh_flow.py` LIST_DIRS extended to the 13 new directories (its verify step
  would have crashed); PR title/body rewritten for Phases 2+3.
- Resume kit refreshed: staging, driver, title, body, README (current state
  @ de7239c), `asm-ols-phase2-3.patch` + bundle (verified).
- **Push attempted and blocked exactly as documented**: no GITHUB_TOKEN in
  the sandbox (`gh_flow.py sanity` → rc=1 "no token"). The one-command flow
  (`GITHUB_TOKEN=<PAT> python3 gh_flow.py --allow-existing all`) is ready
  in `/home/z/my-project/download/pr-resume-kit/`.

### Mutation survivors closed (user item 2)
- Cancellation battery (math.fsum oracle) added to the mutation suite: kills
  5 of the 7 xmm-swap survivors. **Honest score 28/34 = 0.82.**
- The battery immediately caught the AVX2 dot "failing" it — which is the
  DOCUMENTED uncompensated contract (sse2 = compensated reference, avx2 =
  γ₂ₙ bound); the battery is now contract-aware.
- Remaining 2 survivors analyzed as **provably equivalent** (results/
  mutation-phase3.json): lane-0 compensation masked by the 4×4-channel
  merge; non-negative Σr² where the sequential bound n·u is below tolerance.
- **Runtime upper-half check** (tests/guard/test_upper_half.c): a
  legacy-SSE sink (-mno-avx) times cold-vs-warm after every AVX2 export;
  ratio 1.68 < 3.0 threshold → discipline holds at runtime, not just lint.

### CLI + binary I/O — Phase 4a (user item 3, `434846c`)
- `cli/asm_ols.c`: `asm-ols fit|info` with the `.olsb` binary format
  (magic/version/offsets header, NUL-terminated names, CRC32 trailer) and a
  CSV loader (line-numbered parse errors, non-finite rejection);
  `--target/--features/--method auto|chol|csne|qr/--no-intercept/--lambda/
  --rcond/--json`; human + JSON reports.
- Verified: coefficients exact vs numpy to **4e-15** (JSON path), ridge
  exact vs the mpmath slopes-only contract (3.6e-15), .olsb roundtrip,
  error paths. My first battery had two test bugs (text-parse rounding; a
  numpy ridge check that wrongly penalized the intercept) — both found by
  comparing against the JSON path and the mpmath contract.
- **CRITICAL FIX found by the CLI**: `ols_workspace_size` clobbered the
  caller's rbx (mov-then-push saved the NEW value; the pop "restored" it).
  Every C caller keeping a pointer in rbx crashed after free(). Root-caused
  through: heap-corruption hypothesis → guard pages → canary rings →
  register canaries → `beta` becoming 0x140 → the faulting `free(0x140)`.
  The value now crosses the call IN rbx (callee-saved), the caller's rbx
  crosses on the stack.

### Column-pivoted QR — Phase 4b (user item 3, `7ec00d6`)
- `ols_qr_pivot_f64_sse2` kernel: max-residual-norm column selection,
  **full column swaps** (rows 0..n−1 — the first draft swapped only j..n−1
  and the back-substitution diverged; caught by a python replication),
  perm tracking, rcond<0 pins column 0, |rcond| threshold. Fully tested:
  12–14 LRE vs mpmath on mixed-scale designs, rank-deficiency detection
  (duplicate column → rank 3), pin semantics.
- **ols_fit method-4 wiring attempted and REVERTED** — open item: a
  register-clobber in the rcond-sentinel setup (sentinel math clobbers rcx
  = stride; the re-set attempt crashed the plain-QR path via an
  uninitialized perm stash, itself now fixed). Method 4 returns OLS_E_ARG;
  the kernel + solve + perm-mapping recipe is in tests/py/test_qr.py.
- 4 wiring bugs found and documented (partial swap; comisd polarity
  inverted twice; the −0.0 sentinel never pinning since −0.0 == +0.0).

## Not finished (and why)
- **PR push**: no GitHub credential in the sandbox — kit is ready and
  validated offline; needs a PAT from the user (fine-grained: Contents RW +
  Pull requests RW on devansh1012007/linear_regression_assembly).
- **Phase 4c threads**: not started (time-boxed out after 4a/4b; PLAN
  MoSCoW had it behind CLI + pivoting).
- **Phase 5 baseline ladder**: not started; G2 remains RED, honestly.
- QR_PIVOT fit wiring (above).

## Open next session
1. Push via the resume kit once a PAT exists; regenerate staging first
   (`python3 /home/z/my-project/scripts/make_staging.py`) since HEAD moved.
2. QR_PIVOT fit wiring: fix the sentinel register-clobber (write the
   sentinel BEFORE the arg setup, or stash args in frame slots).
3. Threads (fixed-order merge, reproducible mode), then the baseline
   ladder (C -O2/-O3, intrinsics, NumPy/OpenBLAS; H2 verdict).
