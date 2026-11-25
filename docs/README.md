# docs/ — the written record

Everything a reviewer needs beyond code and tests lives here. House rules
(AGENTS.md §2): **no number is typed by hand into a doc unless a script
produced it at the current commit** — every claim below cites the results
file or report it came from. Predictions are written into `perf-log.md`
*before* the corresponding measurement.

## Reading order

1. `../AGENTS.md` — mission, hard rules, process
2. `../PLAN.md` — roadmap, pre-registered gates, phase execution checklists
3. `design.md` — architecture, ABI and memory contracts, why it is built this way
4. `architecture.md` — module map, call graph, workspace layouts, kernel inventory
5. `numerical_stability.md` — the numerics stack and its evidence
6. `performance.md` — machine characterization, kernel measurements, honest gate status

## Layout

| Path | Content |
| --- | --- |
| `README.md` | this index |
| `design.md` | architecture rationale: purity, ABI, memory, dispatch honesty |
| `architecture.md` | module map, `ols_fit` call graph, workspace/struct layouts |
| `numerical_stability.md` | pivot-shifted compensated numerics; NIST evidence; guards |
| `performance.md` | machine limits, gram v1→v3 history, gate status, baseline plan |
| `perf-log.md` | append-only predictions → measurements ledger (every kernel) |
| `agent-comms.md` | append-only cross-session log (findings, decisions, handoffs) |
| `adr/` | architecture decision records: 0001-0003, 0005-0007 (0004 never issued) |
| `reports/` | phase reports: `phase0-1/phase0-foundations.md`, `phase1-simple-regression.md`, `phase2-gram-engine.md` |

## Results (script-produced, cited by the docs)

| Path | Producer | Content |
| --- | --- | --- |
| `../results/machine.json` | `bench/machine/characterize.py` | FMA peak, bandwidth ladder, roofline crossover |
| `../results/gram-bench.json` / `gram-bench-v3.json` | gram benchmark | v1/v3 kernel throughput |
| `../results/nist-phase1.json` / `nist-phase2.json` | pyramid L4/L5 | NIST StRD LREs per dataset |
| `../results/mutation-phase1.json` | mutation suite | 31 mutants, survivors analysis |
| `../results/pyramid-summary.json` | `tests/py/run_all.py` | L1-L8 gate record |

## Regenerating

```sh
make verify        # lint + harness + full test pyramid (CI equivalent)
make repro         # verify + bench + bench-report (regenerates results/)
```

## Known documentation debts

- ADR-0004 was skipped in numbering (0003 → 0005); the gap is deliberate
  history, not a lost file — do not renumber.
- `docs/reports/phase3-accuracy.md` is pending (Phase-3 checklist P3.10).
- The ABI-probe subprocess crash warning in the pyramid log is a known open
  issue (agent-comms 2026-11-21); guard/differential coverage stands.
