# Phase 9 — session 9 (2026-11-26): PR #4 merged, G4 closure, fuzzer, freestanding CLI, plots

Execution: PLAN §Phase 4/5 leftovers + §9 DoD items 3 & 6 · branch
`phase3-7` · `make verify` ALL GREEN at HEAD on a CLEAN build (pyramid +
cli_regression + freestanding_check suites).

## Session context

PR #3 had merged at ea99e844 BEFORE the H-label correction commit
(e1b011a) landed on the branch. Opened PR #4 carrying that single commit
(CI green: verify run 45783115616) and merged it (c3fa88a8). Main now
carries the corrected hypothesis verdicts.

## The finding that shaped this session

PLAN §9 DoD item 4 (H1-H6 verdicts) was completed by the H-label fix.
Auditing the remaining DoD items surfaced the real gap: the phase-8
report claimed "G4 green", but PLAN's G4 has FOUR clauses and three had
no artifacts anywhere in the repo:

| G4 clause | status at session start | session 9 |
|---|---|---|
| 1e7 x 16 E2E matches numpy pipeline within bounds | not run | **GREEN** (max rel slope diff 2.1e-13) |
| fuzz clean for a fixed budget | no fuzzer existed | **GREEN** after 4 bug fixes |
| static binary, no dynamic section, NX stack | no freestanding build | **GREEN** (112 KB, readelf-verified) |
| thread scaling >= 85% of all-thread BW | = H5 | **RED** (18.0% / 18.7%, 2-vCPU quota host) |

## Delivered

### 1. Evidence plots (DoD item 3: "roofline plots, thread scaling ... generated")
- `tools/make_plots.py` + `results/plots/roofline.png` (ceilings,
  measured kernels, p* = 23.4 crossover), `thread-scaling.png` (1t vs
  mt2 vs ceilings, H5 context), `ladder.png` (4 cases x 6 rungs).
  `make plots`, wired into `make repro`. All values read from committed
  results JSON; nothing hand-typed. Non-blank verified.

### 2. 1e7 x 16 end-to-end (G4 clause 1)
- `tests/py/e2e_1e7.py` + `results/e2e-1e7.json`: streams a 1.36 GB
  .olsb (never materialized in RAM), CLI fit vs a chunked numpy
  normal-equations reference. PASS: max rel slope diff 2.1e-13 (max abs
  diff 2.9e-13; bound 5e-10), intercept 6.7e-16, sigma2 1.2e-13 rel, r2
  5.6e-16.
- Honest numbers recorded: asm 14.4 s vs OpenBLAS-threaded reference
  1.4 s (consistent with H1 RED).
- Debugging note: the first reference implementation was WRONG (used
  X'y where the intercept normal equations need the column sums 1'X;
  intercept off by ~1.19). Caught by the lstsq differential at n=2e5
  before trusting any 1e7 number. The CLI was right all along.

### 3. Parser fuzzing (G4 clause 3) — found 4 real bugs
- `tests/py/fuzz_csv.py` (5 s budget, seeded, CSV structure + .olsb
  byte mutations + argv fuzzing; invariant: rc in {0,1,2}, no crash
  signal, rc=0 => valid JSON) + `results/fuzz-csv.json`.
- Round 1 (21 failures) = 4 distinct CLI bugs, all fixed in
  cli/asm_ols.c:
  1. column names printed unescaped into JSON keys (header `y,"x1",x2`
     -> `{"x1":: ...}` invalid JSON) -> `print_json_string()`.
  2. `-nan`/`inf` stat tokens -> `json_num()` (non-finite -> null).
  3. `--features` values that resolve to no column (getopt ate the next
     flag) ran the fit on UNINITIALIZED X and emitted `{, "(intercept)"
     ...}` -> validation + duplicate/empty entry rejection (rc 2).
  4. mutated .olsb name bytes (raw 0x88) passed to stdout verbatim ->
     `name_ok()` UTF-8/control validation at load (both loaders).
- Round 2 false positive was fuzzer-side (getopt eating `--json` as a
  value after `--rcond`); harness fixed, not the CLI.
- Clean: seed 42 (5,731 cases) and seed 1337 (26,055 cases; the committed artifact results/fuzz-csv-seed1337.json holds a re-run at the same 20 s budget with 23,956 cases — case count is timing-dependent).
- `tests/py/cli_regression.py` locks all 4 fixes; runs in `make verify`.

### 4. Freestanding CLI (G4 clause 4, MoSCoW MUST "libc-free")
- `cli/freestanding.c` -> `build/cli/asm-ols-fs` (112,160 bytes):
  `_start` (pure-asm thunk; the first draft read rsp after the prologue
  had moved it — argc came back garbage), raw syscalls (read/write/
  open/close/mmap/exit), -nostdlib -static, readelf-verified: no
  dynamic section, GNU_STACK RW (NX), no INTERP. Links the PRODUCTION
  CORE_OBJS + libgcc (for __float128 formatting; libgcc is not libc).
- Fit-only scope over .olsb (documented): same JSON field names as the
  libc CLI minus p_values/confint/f_stat.
- **BITWISE parity**: 13/13 shared numeric fields identical to the
  libc CLI (`tests/py/freestanding_check.py`, in `make verify`).
- `%.17g` by hand: double and long-double rescale loops measurably lose
  the 17th digit (192,598 then 5,304 failures in a 300k sweep vs libc;
  numbers in the perf log). Final: __float128 scaling + round-half-even
  -> **300,000/300,000 exact match** (harness: `tests/c/outnum_test.c`).
- Two instructive failures on the way, both documented in source:
  `ols_init(0)` force-selects SSE2 (auto is -1) — every result silently
  differed ~1e-15 from the libc CLI; and an alignment-red-herring
  session where a byte-reversed hex printer made correct data look
  corrupt.

### 5. Repairs + plumbing
- `scripts/to_olsb.py`: CSV/.npy -> .olsb converter (PLAN Phase-4 names
  it; it never existed). Round-trip verified: to_olsb -> CLI fit ==
  numpy lstsq to 1.1e-15.
- `results/nist-phase2.json` regenerated: the committed copy was
  all-zero LRE (stale artifact; a fresh run is green 11.7-14.0). This
  is a REPAIR of a broken artifact, not a measurement change.
- `results/binary-size.json` (DoD item 3).
- Makefile: `build/cli/asm-ols-fs` target with the readelf assertions
  baked in, `plots` target, fs binary in `harness`.

## Gate status after this session

- G0/G1/G3: green (unchanged). G2: RED, CLOSED with evidence (ADR-0012).
- **G4: SPLIT** — E2E GREEN, fuzz GREEN, static-binary GREEN,
  thread-scaling RED (= H5, host-bound). The blanket "G4 green" in the
  phase-8 report was corrected (note added there).
- H1-H6: verdicts unchanged from the session-8 correction (H1 RED,
  H2 GREEN, H3 not built, H4 CONFIRMED, H5 RED host, H6 RED).
- `make verify` ALL GREEN on a clean build; CI green on the PR.

## What I got wrong (this session's own list)

1. The blanket "G4 green" I inherited was unearned; I audited it only
   because the DoD checklist made the missing artifacts impossible to
   unsee. Lesson: a gate with multiple clauses is graded per clause.
2. My first 1e7 reference had a real algebra bug (X'y vs 1'X in the
   intercept rows). The 2e5-vs-lstsq differential caught it; without
   that, the "PASS" would have been a lie against a wrong reference.
3. The freestanding formatter cost three attempts (double, long double,
   u128 bigint abandoned mid-way, __float128 + half-even). The unit
   sweep vs libc is what ended it — every iteration was measured.
4. `ols_init(0)` vs `ols_init(-1)`: I read "0" as "auto". The ISA table
   says 0 = SSE2. The 1e-15-different-results symptom was the contract
   working exactly as documented.

## Files

- cli/freestanding.c, cli/asm_ols.c (4 fixes), Makefile
- scripts/to_olsb.py, tools/make_plots.py
- tests/py/{e2e_1e7,fuzz_csv,cli_regression,freestanding_check,fs_differential}.py
- tests/c/outnum_test.c
- results/{e2e-1e7,fuzz-csv,binary-size}.json, results/plots/{roofline,thread-scaling,ladder}.png
- results/nist-phase2.json (repaired), README, this report
