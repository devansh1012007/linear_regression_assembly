# agent-comms.md — append-only (AGENTS.md §8)

## 2026-11-20T07:30Z session-1 -> ALL [INFO]
Phase 0 + Phase 1 complete; G0 and G1 green. Evidence: docs/reports/phase0-1/,
results/machine.json, results/nist-phase1.json, results/mutation-phase1.json.

## 2026-11-20T07:30Z session-1 -> ALL [DECISION]
PLAN deviation recorded in ADR-0002: dev host is Intel SPR-class KVM (2 vCPU,
4 GiB, AVX-512 present) not Ryzen-class; AVX-512 stays a COULD item. 8x L3
DRAM working set impossible (RAM < 8x L3); DRAM numbers capped at 1.5 GiB and
must be re-measured on a bigger host before publication.

## 2026-11-20T07:30Z session-1 -> next-session [QUESTION]
Two mutation-survivor test gaps are queued as your first tasks: (1) ymm
upper-half / vzeroupper check helper (asm-verification §2b), (2) sse2-axpy
tail case in tests/py/mutation_suite.py. Kernel benchmark measurements for
perf-log #1-#10 predictions are written but unmeasured — run in a quiet
window and via tools/update_perflog.py.
