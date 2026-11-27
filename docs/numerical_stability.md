# numerical_stability.md — the accuracy stack and its evidence

The library's reason to exist is being *right where simple implementations
are wrong*. This doc states the numerical design and the measured evidence;
the producers are the pyramid (L4-L8) and `results/nist-*.json`.

## 1. Pivot-shifted compensated accumulation

Every summation/dot product in the library is Neumaier-compensated (error
carried alongside the accumulator), starting from a *shifted* origin: the
pivots `c` are the weighted means of the first `min(n, 256)` rows
(`ols_pivot_means_f64_sse2`). The Gram pass accumulates shifted moments

```
S = Σ w (z − c)(z − c)',   s = Σ w (z − c),   W = Σ w
```

and the API layer completes the centering algebraically:

```
G = S − s s'/W          (exact for ANY c; the pivot only bounds cancellation)
```

so the centering error is bounded by `|mean − c|` (a first-256-row estimate)
instead of `|mean|` — that is what survives the adversarial-offset case
`X = 1e9 + noise`, where the naive raw Gram loses everything to cancellation.
Measured: the asm fit agrees with the mpmath reference to maxdiff 3.6e-07
on that case (test_fit differential, current commit). Without an intercept
`c = 0` and the raw Gram is required — a shifted solve without an intercept
column returns slopes for `(x − c)`, not `x` (the no-intercept failure).

## 2. Factorization and singularity

The centered slope block is factored by `ols_chol_f64_sse2` with threshold
`eps_rel = 8·p·u` against the max input diagonal (`u = 2^-52`) →
`OLS_E_SINGULAR` when a pivot dies. Two hard lessons are encoded there: the
threshold constant is loaded as the *double* `2^-52` by bit pattern (an
early `movq` of the integer `2^52` passed 4.5e15 and made every
factorization "singular"), and the factor uses stride-aware addressing so
the fit can factor the leading p×p block of the q-wide Gram in place.

`cond_est = (max R_jj / min R_jj)²` is reported as a **lower bound** on the
condition of the centered information matrix — explicitly not an estimate of
`κ(X)`, and documented as such in the header.

## 3. Statistics: conventions and guards

| Quantity | Definition | Guard |
| --- | --- | --- |
| `sst` | centered `Σw(y−ȳ)²` with intercept; raw `Σwy²` without | — |
| `sigma2` | `sse / max(1, n − p − intercept)` | dof clamped at 1 |
| `r2` | `1 − sse/sst` (sst convention as above) | sst ≤ 0 → 1 |
| `r2_adj` | `1 − (1−r2)(n−1)/(n−p−intercept)` | dof ≤ 1 or sst ≤ 0 → r2 |
| `f_stat` (v3) | `((sst−sse)/p) / (sse/dof)`, dof = n−p−intercept | sse ≤ 0, sst ≤ sse, dof < 2 → 0 |
| `se_j` | `√(σ²·(C⁻¹)_jj)`, `(C⁻¹)_jj = z[j]` with `C z = e_j` | — |
| `se_b0` | `√(σ²·(1/W + m'C⁻¹m))` | m re-stashed after solves destroy S |
| `t_j` | `β_j / se_j` | se ≤ 0 → 0 |

The `se_j` definition is `z[j]`, the j-th *component* of `C⁻¹e_j` — an
earlier draft used `‖z‖²` (i.e. `(C⁻²)_jj`), underestimating se; the SciPy
differential caught it and the fix is pinned by that test (phase-2 report).
`f_stat`'s dof was `n − q` in its first draft — wrong without an intercept
(q = p+1 there, but the model has p parameters); fixed to `n − p − intercept`
and pinned by a no-intercept closed-form differential (ADR-0006).

## 4. NIST StRD evidence (method: ols_fit, gram+chol)

Minimum LRE vs certified values over all coefficients and the residual SD
(`results/nist-phase2.json`; phase-1 four-dataset table in
`results/nist-phase1.json`, worst LRE 11.9):

| Dataset | n | p | min LRE | gate | note |
| --- | --- | --- | --- | --- | --- |
| Longley | 16 | 6 | 11.72 | 11.0 | the classic collinearity stress |
| Wampler1 | 21 | 5* | 7.40 | 7.0 | exact polynomial, zero residual |
| Wampler2 | 21 | 5* | 10.10 | 9.0 | noisier variant |
| Wampler3 | 21 | 5* | 7.40 | 7.0 | adversarial y-scaling |
| Filip | 82 | 10 | — | none | `rc = −7` — refusal is the contract |

(\* polynomial basis in one variable, q = 6 columns.) Filip is beyond double
precision for normal equations — κ(X) makes the Gram numerically singular —
so the honest verdict is a documented refusal, not a wrong answer. The
QR-based second opinion (ADR-0007, P3.5) is since built —
`results/nist-phase3.json` covers 11 datasets × {gram_chol, gram_csne,
qr}; it does not square the condition number, and Filip is still refused
at fit level (raw-kernel solve: rank 11, LRE 7.11 vs certified), reported
without a target (G3). [reproducible via tests/py/test_qr.py vs the
certified values; recorded in the P3.x commit messages, not a committed
results artifact]

## 5. Property sweeps and the planned κ study

L8 runs shift/scale-invariance properties and a condition sweep
κ(X) = 1e1..1e4 against mpmath (zero-residual + large-residual variants).
Phase 3 extends the sweep to κ = 1e8 and adds CSNE and QR arms
(P3.7): hypothesis H4 predicts plain Gram+Cholesky degrades around
κ ≈ 1e8 where κ²·u ≈ O(1), CSNE restores QR-like accuracy up to a measured
κ*, and the `auto` threshold is derived from that data — all pre-registered
in `PLAN.md` §Phase 3 before measurement.

## 6. What stability work is *not* claimed

- No accuracy claim survives without its differential: "matches NumPy" cites
  the test and tolerance; "agrees with mpmath" cites the case.
- The κ = 1e8 study ran in Phase 3 (`results/accuracy/kappa-sweep.json`):
  H4 CONFIRMED — plain Gram+Cholesky loses ≈ 2·log10(κ) digits, CSNE is
  QR-grade up to the measured κ* = 1e6, QR loses only log10(κ) digits
  (13.5→8.0 at κ=1e8).
- Weighted fits are differential-tested (`w` path in test_fit); ridge
  (`lambda`) is wired (Phase 3, P3.3b): slopes-only per the documented
  contract, differential-pinned vs the mpmath ridge reference incl.
  weighted designs.
