---
name: numerical-ols-algorithms
description: Algorithm ladder, formulas, error analysis and accuracy-testing rules for ordinary least squares in floating point (centering, Gram+Cholesky, corrected semi-normal equations, Householder/TSQR, ridge, weights, rank deficiency, standard errors, NIST StRD). Load before choosing or implementing any numerical method in asm-ols, writing its reference implementation, or making an accuracy claim.
---

# Numerical OLS algorithms

Problem: given `X` (n x p), `y` (n), find `b = argmin ||y - [1 X] b||` (intercept optional). Everything below is about getting `b`, `SSE`, `R^2` and standard errors **accurately, in one or two passes over X**.

## 1. Algorithm ladder
| Tier | Method | Passes over X | Flops (n >> p) | Accuracy expectation (verify empirically) | Use when |
|---|---|---|---|---|---|
| 0 | Simple regression, centered two-term sums (compensated) | 1 | O(n) | near machine precision | p = 1 |
| 1 | Centered Gram `C = sum (x-xbar)(x-xbar)'` + Cholesky | 1 | n p^2 | error ~ kappa(X)^2 * u; fails (non-positive pivot) near kappa ~ 1e8 | default fast path, well-conditioned |
| 1b | Tier 1 + **CSNE** (corrected semi-normal equations, one refinement) | 2 | ~ 2 n p^2 | typically QR-like for moderate kappa; limits must be measured | default accurate path |
| 2 | Householder QR on `[X y]`, streamed as TSQR blocks | 1 | ~ 2-4 n p^2 | error ~ kappa (small residual), up to kappa^2 (large residual) | ill-conditioned, or Tier 1 fails |
| 2b | Rank-revealing (column-pivoted) QR | 1+ | more | detects rank deficiency | collinearity suspected |
| 3 | CG/LSQR/GD on the Gram or on X | many | per iter O(p^2) on Gram | converges to Tier 1 accuracy at best | teaching, huge p, streaming SGD |
Method `auto`: run Tier 1, estimate conditioning from the Cholesky diagonal; if a pivot is non-positive or the estimate exceeds a threshold fixed *by the kappa-sweep experiment*, switch to Tier 2 and report it. Never silently return a poor answer: always report `cond_est`, `rank`, `method_used`.

## 2. Centering without catastrophic cancellation
Naive `sum x^2 - n xbar^2` loses all digits when |mean| >> std (e.g. x = 1e9 + noise).
- **Shifted one-pass:** pick pivot `c` (column means of the first block, >= a few hundred rows); accumulate `S = sum (x-c)(x-c)'` and `s = sum (x-c)`; then `C = S - s s'/n`. Cancellation is bounded by |mean - c|, not |mean|.
- **Chan/Welford merge** for blocks and threads: with `(n_a, mean_a, M_a)` and `(n_b, mean_b, M_b)` (M = centered co-moment matrix): `n = n_a+n_b; d = mean_b - mean_a; mean = mean_a + d n_b/n; M = M_a + M_b + d d' n_a n_b / n`.
- **Augmented column:** treat `y` as column p+1 of `Z=[X y]` so one kernel yields `C_xx`, `c_xy`, `C_yy`.
- Adversarial test data (mandatory): offset 1e8..1e12 with unit noise; columns with scales 1e-6..1e6; near-collinear pairs.
- Column scaling: equilibrate (divide by column norm/std) before factorization, undo afterwards; report scaling used.

## 3. Cholesky details
`C = L L'`. Slopes `b = C^-1 c_xy` via two triangular solves. Intercept `b0 = ybar - xbar' b`. Fail with `OLS_E_SINGULAR` if any pivot `<= eps * max_diag` (eps documented, e.g. p * u). Condition estimate: `kappa(C) >= (max L_ii / min L_ii)^2` as a cheap *lower bound*; better, a few power iterations on `R'R` / inverse iteration using the factor. Ridge: add `lambda` to the diagonal of `C` (not to the intercept; with centering the intercept is unpenalized for free).

## 4. SSE and R^2 — avoid the cancelling identity
`SSE = y'y - b'X'y` cancels catastrophically when R^2 -> 1. Instead:
- Tier 1b/2: compute **residuals explicitly** in a pass (`r = y - b0 - Xb`, fused with `X'r` for CSNE) and sum `r^2` (compensated).
- Tier 2 with QR of `[X y]`: `R_aug = [[R, z],[0, rho]]`, then `b = R^-1 z` and **`SSE = rho^2`** (exact byproduct).
`SST = sum (y - ybar)^2` from centered moments. `R^2 = 1 - SSE/SST`. For no-intercept models use the **uncentered** R^2 (`1 - SSE / y'y`) — this is what NIST certifies for NoInt1/NoInt2. Adjusted `R^2 = 1 - (1-R^2)(n-1)/(n-p-1)`; `F = (R^2/p) / ((1-R^2)/(n-p-1))`.

## 5. CSNE (corrected semi-normal equations, Bjorck)
With `R` from Cholesky of the (centered) Gram: `b0 = solve(R'R, X'y)`; `r = y - X b0`; `d = solve(R'R, X'r)`; `b = b0 + d`. The two extra quantities fuse into one more pass (residual + `X'r`). It avoids storing Q, costs one extra pass, and in theory recovers most of QR's accuracy for moderately ill-conditioned problems. **Do not claim a kappa range; measure it** with the sweep below and record the measured threshold.

## 6. Householder / TSQR
Maintain `R` (q x q upper triangular, q = p+1). For each row block `B` (rows x q): QR of the stacked `[R; B]`, keep the new `R`. Blocks (or threads) combine by QR of `[R_a; R_b]`. Reflector: `v = x + sign(x_1)||x|| e_1`, `H = I - 2 v v'/(v'v)`; apply to trailing columns. Row-major blocks make trailing-column updates contiguous. Column pivoting (largest remaining column norm, downdated norms with the usual recomputation safeguard) gives rank detection: rank = count of `|R_jj| > rcond * |R_11|`.

## 7. Weights, ridge, rank deficiency
- WLS: scale each row (incl. y) by `sqrt(w_i)` in the packing pass; reject negative/non-finite weights.
- Ridge via QR path: append `sqrt(lambda) I` rows (never form them; fold into the TSQR combine step).
- Rank deficiency policy: default **error** with the detected rank and the offending columns; `--rcond` and `--ridge` opt in to graceful handling. Never return the minimum-norm solution silently.

## 8. Inference quantities
`sigma^2 = SSE/(n-p-1)` (with intercept) or `SSE/(n-p)` (without). Slopes: `Var(b) = sigma^2 C^-1` (diagonal via `R^-1`: form `R^-1` by back-substitution, row norms squared). Intercept: `Var(b0) = sigma^2 (1/n + xbar' C^-1 xbar)`. `t_j = b_j / se_j`. Leverage `h_i = 1/n + (x_i-xbar)' C^-1 (x_i-xbar)` (extra pass), Cook's distance, VIF_j = `(C^-1)_jj * C_jj`. p-values/CIs need the t-distribution (regularized incomplete beta) -> requires `exp/log/lgamma` in asm: a COULD item with its own accuracy test vs MPFR/mpmath; until then report t-statistics only.

## 9. Error-bound language for tests
Define `u = 2^-53`, `gamma_k = k u / (1 - k u)`. Dot-product error <= `gamma_n sum|a_i b_i|`. A well-conditioned solve should satisfy `||b - b_ref|| / ||b_ref|| <= c * kappa^s * u` with `s = 2` (normal equations) or `s = 1` (QR, small residual). Tests assert these with a **recorded** constant `c` and fail if a change makes it grow. Equality tests on floating-point output are banned except in reproducible mode (bitwise across thread counts).

## 10. Reference problems and metrics
- **NIST StRD linear regression:** Norris, Pontius (lower difficulty); NoInt1, NoInt2 (average); Filip, Longley, Wampler1-5 (higher). Metric: `LRE = -log10(|q - c| / |c|)` per coefficient (c = certified value, cap at 15), plus certified SE, residual SD, R^2. Report the full table per method; do **not** pre-announce success on Filip (many solvers in double precision fail it) — report what is achieved and compare with NumPy `lstsq` and statsmodels.
- **Controlled-kappa generator:** `X = U diag(s) V'` with random orthogonal U, V (QR of Gaussians), `s` geometric from 1 to 1/kappa, `kappa in 10^0..10^14`; `y = X b_true + noise` with (a) zero residual and (b) large residual. Ground truth `b_true` for zero-residual; for noisy cases a high-precision reference (mpmath/`fractions` with 50+ digits). Plot error vs kappa for each method (log-log) with reference slopes kappa and kappa^2.
- Pre-register accuracy targets in PLAN.md; if you revise a target, log why. Never edit a target to match a result silently.

## 11. Python reference (tests/ref/ols_ref.py) must provide
(1) float64 NumPy reference; (2) mpmath 60-digit solver for any (X,y,weights,ridge); (3) kappa-controlled generator; (4) NIST file parser; (5) functions returning every statistic the asm library returns, so the differential test compares like with like.
