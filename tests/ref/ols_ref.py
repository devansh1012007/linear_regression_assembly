# ------------------------------------------------------------------------------
# ols_ref.py — Python references for asm-ols (numerical-ols-algorithms §11).
#   (1) float64 NumPy reference
#   (2) mpmath 60-digit solver
#   (3) kappa-controlled generator
#   (4) NIST StRD file parser (data + certified values)
#   (5) every statistic the Phase-1 asm library returns, so the differential
#       test compares like with like.
# ------------------------------------------------------------------------------
"""Python references: float64, mpmath 60-digit, kappa generator, NIST parser."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import mpmath as mp
except ImportError:            # pragma: no cover - mpmath is a test dep
    mp = None

NIST_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "nist"


# ==============================================================================
# (4) NIST StRD parser
# ==============================================================================
@dataclass
class NistData:
    name: str
    y: np.ndarray                    # (n,)
    X: np.ndarray                    # (n, p) predictors WITHOUT intercept col
    n: int
    p: int                           # number of B coefficients incl. intercept
    has_intercept: bool
    certified_b: np.ndarray          # (p,)
    certified_se: np.ndarray         # (p,)
    residual_sd: float
    r_squared: float
    anova_sse: float | None = None
    raw: str = field(default="", repr=False)

    def design(self) -> np.ndarray:
        """Full design matrix [1 X] if intercept else X (n x p)."""
        if self.has_intercept:
            return np.hstack([np.ones((self.n, 1)), self.X])
        return self.X


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?")


def _fortran_floats(line: str) -> list[float]:
    return [float(t.replace("D", "E").replace("d", "e")) for t in _NUM.findall(line)]


def parse_nist(path: str | Path) -> NistData:
    """Parse a NIST StRD .dat file: header, certified values, data rows."""
    raw = Path(path).read_text()
    lines = [ln.rstrip("\r") for ln in raw.splitlines()]   # NIST files are CRLF

    name = ""
    for ln in lines[:5]:
        if "Dataset Name:" in ln:
            name = ln.split(":", 1)[1].strip().split()[0]
            break

    n_obs = int(re.search(r"(\d+)\s+Observations", raw).group(1))
    n_par = int(re.search(r"(\d+)\s+Parameter", raw).group(1))    # singular/plural
    has_intercept = "B0" in raw and "1 Parameter (B1)" not in raw

    # certified block: lines of the form  B0  <est>  <se>  after the first
    # "Certified Regression Statistics" header, until "Residual"
    cert_b, cert_se = [], []
    in_cert = False
    residual_sd = float("nan")
    r2 = float("nan")
    anova_sse = None
    prev_ln = ""
    for ln in lines:
        if "Certified Regression Statistics" in ln:
            in_cert = True
            continue
        if in_cert:
            m = re.match(r"\s*B(\d+)\s+(\S+)\s+(\S+)", ln)
            if m:
                cert_b.append(float(m.group(2)))
                cert_se.append(float(m.group(3)))
                continue
            # "Residual\n     Standard Deviation   <value>" spans two lines
            if ln.strip().startswith("Standard Deviation") and \
               prev_ln.strip() == "Residual":
                residual_sd = float(ln.split()[-1])
                prev_ln = ln
                continue
            if re.search(r"R-Squared\s+(\S+)", ln):
                r2 = float(re.search(r"R-Squared\s+(\S+)", ln).group(1))
                continue
            if re.match(r"\s*Residual\s+\d+", ln):
                nums = _fortran_floats(ln)
                if len(nums) >= 2:
                    anova_sse = nums[1]
        prev_ln = ln

    # data block: the LAST line matching "Data:" (the first "Data:" lines are
    # part of the header describing response/predictor counts)
    data_rows: list[list[float]] = []
    in_data = False
    for ln in reversed(lines):
        if re.match(r"\s*Data:", ln):
            break
    else:
        raise ValueError(f"{name}: no Data: block found")
    data_start = max(i for i, ln in enumerate(lines) if re.match(r"\s*Data:", ln))
    for ln in lines[data_start + 1:]:
        nums = _fortran_floats(ln)
        if not nums:
            continue
        data_rows.append(nums)
        if len(data_rows) == n_obs:
            break
    if len(data_rows) != n_obs:
        raise ValueError(f"{name}: expected {n_obs} data rows, got {len(data_rows)}")

    arr = np.array(data_rows, dtype=np.float64)
    y = arr[:, 0].copy()
    if has_intercept:
        X = arr[:, 1:].copy()                   # (n, p-1)
    else:
        X = arr[:, 1:].copy()                   # still y, x columns; B1 only

    assert len(cert_b) == n_par, f"{name}: {len(cert_b)} certified coefs, want {n_par}"
    return NistData(name=name, y=y, X=X, n=n_obs, p=n_par,
                    has_intercept=has_intercept,
                    certified_b=np.array(cert_b),
                    certified_se=np.array(cert_se),
                    residual_sd=residual_sd, r_squared=r2,
                    anova_sse=anova_sse, raw=raw)


def lre(estimate: float, certified: float, cap: float = 15.0) -> float:
    """Log-relative error, NIST's standard accuracy metric (capped)."""
    if estimate == certified:
        return cap
    if certified == 0.0:
        return 0.0 if estimate != 0 else cap
    return min(cap, max(0.0, -math.log10(abs(estimate - certified) / abs(certified))))


# ==============================================================================
# (1) float64 NumPy references — mirror the Phase-1 asm kernels exactly
# ==============================================================================
def neumaier_sum(x: np.ndarray) -> float:
    """Sequential Neumaier compensated sum (float64). Mirrors ols_sum_f64_sse2."""
    s = 0.0
    c = 0.0
    for v in np.asarray(x, dtype=np.float64).ravel():
        t = s + v
        if abs(s) >= abs(v):
            c += (s - t) + v
        else:
            c += (v - t) + s
        s = t
    return s + c


def neumaier_dot(a: np.ndarray, b: np.ndarray) -> float:
    """Neumaier-compensated dot of fl(a*b) products. Mirrors ols_dot_f64_sse2."""
    s = 0.0
    c = 0.0
    for u, v in zip(np.asarray(a, dtype=np.float64).ravel(),
                    np.asarray(b, dtype=np.float64).ravel()):
        x = u * v
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return s + c


def centered_sums(x: np.ndarray, y: np.ndarray,
                  piv: tuple[float, float] = (0.0, 0.0)) -> dict[str, float]:
    """Shifted sums (what ols_centered_sums_f64_sse2 computes).
    out = [n, s(x-cx), s(y-cy), s((x-cx)^2), s((x-cx)(y-cy)), s((y-cy)^2)],
    all Neumaier-compensated, pivots cx, cy from the caller."""
    cx, cy = piv
    n = float(len(x))
    xs = np.asarray(x, dtype=np.float64) - cx
    ys = np.asarray(y, dtype=np.float64) - cy
    return {
        "n": n,
        "sum_x": neumaier_sum(xs),
        "sum_y": neumaier_sum(ys),
        "sum_xx": neumaier_dot(xs, xs),
        "sum_xy": neumaier_dot(xs, ys),
        "sum_yy": neumaier_dot(ys, ys),
    }


def simple_fit(x: np.ndarray, y: np.ndarray, stats: bool = True) -> dict[str, float]:
    """Simple regression with intercept via centered sums (float64 reference).
    Mirrors ols_simple_fit_f64_sse2 exactly, including error codes as NaN-free
    exceptions with the same semantics."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n == 0:
        raise ValueError("OLS_E_ARG")
    cx, cy = float(x[0]), float(y[0])          # pivot = first element
    s = centered_sums(x, y, (cx, cy))
    Sxx = s["sum_xx"] - s["sum_x"] * s["sum_x"] / n
    Sxy = s["sum_xy"] - s["sum_x"] * s["sum_y"] / n
    if n == 1 or Sxx == 0.0:
        raise ValueError("OLS_E_SINGULAR")
    b1 = Sxy / Sxx
    xbar = cx + s["sum_x"] / n
    ybar = cy + s["sum_y"] / n
    b0 = ybar - b1 * xbar
    out = {"b1": b1, "b0": b0, "sse": 0.0, "sst": 0.0, "r2": 0.0,
           "xbar": xbar, "ybar": ybar}
    if stats:
        r = y - (b0 + b1 * x)                    # residual pass
        sse = neumaier_dot(r, r)
        sst = s["sum_yy"] - s["sum_y"] * s["sum_y"] / n   # centered SST (shifted)
        out["sse"] = sse
        out["sst"] = sst
        out["r2"] = 1.0 - sse / sst if sst != 0.0 else 1.0
    return out


def simple_fit_noint(x: np.ndarray, y: np.ndarray, stats: bool = True) -> dict[str, float]:
    """Regression through the origin (NIST NoInt semantics). Mirrors
    ols_simple_fit_noint_f64_sse2."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n == 0:
        raise ValueError("OLS_E_ARG")
    sxx = neumaier_dot(x, x)
    if sxx == 0.0:
        raise ValueError("OLS_E_SINGULAR")
    b = neumaier_dot(x, y) / sxx
    out = {"b": b, "sse": 0.0, "ySqNorm": 0.0, "r2": 0.0, "n": float(n)}
    if stats:
        r = y - b * x
        sse = neumaier_dot(r, r)
        yy = neumaier_dot(y, y)
        out["sse"] = sse
        out["ySqNorm"] = yy
        out["r2"] = 1.0 - sse / yy if yy != 0.0 else 1.0   # uncentered R^2
    return out


def quad_fit(x: np.ndarray, y: np.ndarray, stats: bool = True) -> dict[str, float]:
    """Quadratic (Pontius) fit y = b0 + b1 x + b2 x^2 via centered normal
    equations solved with the 2x2 symmetric solver — mirrors the asm path:
    center x and z=x^2 by their means, form the 2x2 centered Gram with
    compensated dots, 2x2 solve, back out b0. Used for NIST Pontius."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n == 0:
        raise ValueError("OLS_E_ARG")
    z = x * x
    # column means (Neumaier)
    mx = neumaier_sum(x) / n
    mz = neumaier_sum(z) / n
    my = neumaier_sum(y) / n
    xc, zc, yc = x - mx, z - mz, y - my
    # centered Gram (compensated)
    Sxx = neumaier_dot(xc, xc)
    Sxz = neumaier_dot(xc, zc)
    Szz = neumaier_dot(zc, zc)
    Sxy = neumaier_dot(xc, yc)
    Szy = neumaier_dot(zc, yc)
    det = Sxx * Szz - Sxz * Sxz
    if det == 0.0:
        raise ValueError("OLS_E_SINGULAR")
    b1 = (Szz * Sxy - Sxz * Szy) / det
    b2 = (Sxx * Szy - Sxz * Sxy) / det
    b0 = my - b1 * mx - b2 * mz
    out = {"b0": b0, "b1": b1, "b2": b2, "sse": 0.0, "sst": 0.0, "r2": 0.0}
    if stats:
        r = y - (b0 + b1 * x + b2 * z)
        sse = neumaier_dot(r, r)
        sst = neumaier_dot(yc, yc)
        out["sse"] = sse
        out["sst"] = sst
        out["r2"] = 1.0 - sse / sst if sst != 0.0 else 1.0
    return out


# ==============================================================================
# (2) mpmath 60-digit solver
# ==============================================================================
def mp_solve(X: np.ndarray, y: np.ndarray, intercept: bool = True,
             dps: int = 60) -> dict[str, object]:
    """High-precision OLS via mpmath normal equations with Cholesky (mp.LUProxy
    not needed at 60 dps for these sizes). Returns b (list of mp.mpf),
    sse, sst, r2 as mp.mpf."""
    if mp is None:
        raise ImportError("mpmath not available")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = X.shape[0]
    with mp.workdps(dps):
        A = [[mp.mpf(v) for v in row] for row in
             (np.hstack([np.ones((n, 1)), X]) if intercept else X)]
        bv = [mp.mpf(v) for v in y]
        m = len(A[0])
        # G = A'A, g = A'y (exact products at 60 dps)
        G = [[mp.fsum(A[k][i] * A[k][j] for k in range(n)) for j in range(m)]
             for i in range(m)]
        g = [mp.fsum(A[k][i] * bv[k] for k in range(n)) for i in range(m)]
        # Cholesky
        L = [[mp.mpf(0)] * m for _ in range(m)]
        ok = True
        for i in range(m):
            for j in range(i + 1):
                s = mp.fsum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    d = G[i][i] - s
                    if d <= 0:
                        ok = False
                        break
                    L[i][j] = mp.sqrt(d)
                else:
                    L[i][j] = (G[i][j] - s) / L[j][j]
            if not ok:
                break
        if not ok:
            raise ValueError("OLS_E_SINGULAR (mp)")
        # forward/back substitution
        w = [mp.mpf(0)] * m
        for i in range(m):
            w[i] = (g[i] - mp.fsum(L[i][k] * w[k] for k in range(i))) / L[i][i]
        b = [mp.mpf(0)] * m
        for i in range(m - 1, -1, -1):
            b[i] = (w[i] - mp.fsum(L[k][i] * b[k] for k in range(i + 1, m))) / L[i][i]
        # stats at high precision
        r = [bv[k] - mp.fsum(A[k][j] * b[j] for j in range(m)) for k in range(n)]
        sse = mp.fsum(v * v for v in r)
        ybar = mp.fsum(bv) / n
        sst = mp.fsum((v - ybar) ** 2 for v in bv) if intercept else mp.fsum(v * v for v in bv)
        r2 = 1 - sse / sst if sst != 0 else mp.mpf(1)
        return {"b": b, "sse": sse, "sst": sst, "r2": r2, "dps": dps}


def mp_simple_fit(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    return mp_solve(x.reshape(-1, 1), y, intercept=True)


def mp_simple_fit_noint(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    return mp_solve(x.reshape(-1, 1), y, intercept=False)


def mp_quad_fit(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    X2 = np.column_stack([x, np.asarray(x, dtype=np.float64) ** 2])
    return mp_solve(X2, y, intercept=True)


# ==============================================================================
# (3) kappa-controlled generator
# ==============================================================================
def make_kappa_problem(n: int, p: int, kappa: float, seed: int = 12345,
                       residual: str = "none", noise_rel: float = 1e-8):
    """X = U diag(s) V' with random orthogonal U, V (QR of Gaussians),
    s geometric 1 -> 1/kappa. y = X b_true (+ noise). Returns X, y, b_true,
    and (for noisy cases) the mpmath reference solution.
    cond2(X) = kappa exactly (2-norm singular values 1..1/kappa), so
    cond2(X'X) ~= kappa^2 — the normal-equations loss is 2*log10(kappa)
    digits.
    Noise is relative to ||y|| (noise_rel * ||X b_true||_2): sigma_max is 1,
    so absolute noise_std = noise_rel * ||Xb|| keeps the residual small in
    EVERY singular direction down to 1/kappa, which is the regime where
    kappa^2-vs-kappa accuracy differences between methods are visible."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, p))
    Q, _ = np.linalg.qr(A)
    B = rng.standard_normal((p, p))
    V, _ = np.linalg.qr(B)
    s = np.geomspace(1.0, 1.0 / kappa, p)
    X = (Q[:, :p] * s) @ V.T
    b_true = rng.standard_normal(p)
    y = X @ b_true
    if residual == "large":
        yb = np.linalg.norm(y)
        y = y + (noise_rel * yb) * rng.standard_normal(n) / np.sqrt(n)
        ref = mp_solve(X, y, intercept=False)
        return X, y, b_true, ref
    return X, y, b_true, None


# ==============================================================================
# helpers for the ctypes differential tests
# ==============================================================================
def asm_compatible_simple_fit(x, y) -> dict[str, float]:
    """Same formula order as the asm (Sxx from compensated raw sums) — used
    when bit-comparison of the FORMULA (not the asm) is needed."""
    return simple_fit(x, y)


def mp_ridge(X: np.ndarray, y: np.ndarray, lam: float,
             w: np.ndarray | None = None, dps: int = 60) -> dict[str, object]:
    """High-precision ridge reference for the asm contract (P3.3b): minimize
    sum_i w_i (y_i - b0 - x_i'b)^2 + lam * ||b||^2 — slopes only, intercept
    never penalized. Solved as the augmented normal equations
    (Z'WZ + lam*diag(0, I)) theta = Z'Wy with Z = [1|X] at `dps` digits."""
    if mp is None:
        raise ImportError("mpmath not available")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    wv = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
    with mp.workdps(dps):
        Z = np.hstack([np.ones((n, 1)), X])
        M = mp.zeros(p + 1, p + 1)
        g = mp.zeros(p + 1, 1)
        for i in range(n):
            wi = mp.mpf(float(wv[i]))
            yi = mp.mpf(float(y[i]))
            zi = [mp.mpf(float(v)) for v in Z[i]]
            for a in range(p + 1):
                g[a] += wi * zi[a] * yi
                for b in range(a, p + 1):
                    M[a, b] += wi * zi[a] * zi[b]
                    M[b, a] = M[a, b]
        for j in range(1, p + 1):
            M[j, j] += mp.mpf(float(lam))
        sol = mp.lu_solve(M, g)
        return {"b0": sol[0], "b": [sol[j] for j in range(1, p + 1)]}


def mp_qr_ols(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
              intercept: bool = True, dps: int = 60) -> dict[str, object]:
    """High-precision Householder-QR reference following the qr.asm contract
    exactly (ADR-0007), at `dps` digits:

    1. Aug = [1 | X - 1c_x' | y - c_y] (intercept) or [X | y] (no intercept),
       rows scaled by sqrt(w_i); c are the caller pivots (asm: first-256-row
       weighted means, zeros without intercept).
    2. Design columns (including the ones column) scaled to unit 2-norm.
    3. Householder with the reflected choice alpha = -sign(x0)*||x|| (no
       cancellation); reflectors applied to ALL columns including y.
    4. beta_scaled = backsub(R, Q'y); beta_raw_j = beta_scaled_j / norm_j
       (slopes only; the ones-column coefficient is discarded — b0 comes
       from the exact weighted means: b0 = ybar - xbar'beta_raw).
    5. sse = R[m-1][m-1]^2 (the [X|y]-byproduct). The reference also checks
       this against ||y - X b||^2 directly at `dps` digits.
    6. se from the centered scaled gram G = R_sub'R_sub - ss'/W formed from
       the factor, chol-solved per e_j; se_j = sqrt(sigma2 * z_j)/norm_j,
       intercept se via sigma2*(1/W + m_s'G^-1 m_s) with m_s = m/norm_j.

    Returns dict(beta=[mp.mpf], b0, sse, se=[mp.mpf], rank=dc)."""
    if mp is None:
        raise ImportError("mpmath not available")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    wv = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
    with mp.workdps(dps):
        # pivots: the asm uses weighted means of the first min(n,256) rows;
        # the reference accepts them as inputs to mirror the contract, but
        # for its OWN calls uses the exact weighted means (any c is exact —
        # the ones column makes the formulation exact for any pivot).
        cX = [mp.mpf(0)] * p
        cy = mp.mpf(0)
        # 1. build Aug (n x m), m = p + 1 + intercept
        ic = 1 if intercept else 0
        m = p + 1 + ic
        Aug = [[mp.mpf(0)] * m for _ in range(n)]
        W = mp.mpf(0)
        s = [mp.mpf(0)] * (p + 1)          # pivot-shifted sums (raw units)
        sst_s = mp.mpf(0)                  # sum w (y - cy)^2
        for i in range(n):
            sw = mp.sqrt(mp.mpf(float(wv[i])))
            W += sw * sw
            yi = mp.mpf(float(y[i]))
            if ic:
                Aug[i][0] = sw
            for j in range(p):
                v = mp.mpf(float(X[i][j])) - cX[j]
                Aug[i][j + ic] = sw * v
                s[j] += sw * sw * v
            yv = yi - cy
            Aug[i][p + ic] = sw * yv
            s[p] += sw * sw * yv
            sst_s += sw * sw * yv * yv
        # exact weighted means for the reference b0 (the asm computes the
        # same quantity via c + S/W with its own pivots)
        xbar = [s[j] / W for j in range(p)]
        ybar = s[p] / W
        # 2. scale design columns to unit norm
        norms = []
        for j in range(m - 1):             # design columns = 0..m-2
            ssq = mp.mpf(0)
            for i in range(n):
                ssq += Aug[i][j] * Aug[i][j]
            nj = mp.sqrt(ssq)
            norms.append(nj)
            if nj != 0:
                inv = 1 / nj
                for i in range(n):
                    Aug[i][j] *= inv
        # 3. Householder on columns 0..m-2, applied to all m columns
        dc = m - 1
        for j in range(dc):
            nrm2 = mp.mpf(0)
            for i in range(j, n):
                nrm2 += Aug[i][j] * Aug[i][j]
            nrm = mp.sqrt(nrm2)
            x0 = Aug[j][j]
            alpha = -nrm if x0 >= 0 else nrm
            v = [Aug[i][j] for i in range(j, n)]
            v[0] = x0 - alpha
            vnorm2 = 2 * nrm * (nrm + abs(x0))
            Aug[j][j] = alpha
            for i in range(j + 1, n):
                Aug[i][j] = v[i - j]
            beta_h = 2 / vnorm2
            # w_k = sum_i v_i * Aug[i][k] over the trailing columns, then
            # Aug[i][k] -= beta_h * v_i * w_k
            for k in range(j + 1, m):
                wk = mp.mpf(0)
                for i in range(j, n):
                    wk += v[i - j] * Aug[i][k]
                wk *= beta_h
                for i in range(j, n):
                    Aug[i][k] -= v[i - j] * wk
        # 4. back-substitution: R beta_s = (Q'y)[:dc]
        bs = [mp.mpf(0)] * dc
        for i in range(dc - 1, -1, -1):
            acc = Aug[i][m - 1]
            for k in range(i + 1, dc):
                acc -= Aug[i][k] * bs[k]
            bs[i] = acc / Aug[i][i]
        beta_raw = [bs[ic + j] / norms[ic + j] for j in range(p)]
        # 5. sse byproduct: after dc reflectors the y column rows dc..n-1
        # are the UNPROCESSED residual tail (only dc rows of R exist), so
        # sse = ||tail||^2 = sum_{i>=dc} Aug[i][m-1]^2 — cancellation-free.
        # (An early draft claimed R[m-1][m-1]^2; the reference's own
        # identity check caught it — only the FIRST tail entry is |R_dd|.)
        sse_by = mp.mpf(0)
        for i in range(dc, n):
            sse_by += Aug[i][m - 1] * Aug[i][m - 1]
        r2d = mp.mpf(0)
        for i in range(n):
            acc = mp.mpf(float(y[i]))
            if ic:
                b0c = ybar - sum(xbar[j] * beta_raw[j] for j in range(p))
                acc -= b0c
            for j in range(p):
                acc -= mp.mpf(float(X[i][j])) * beta_raw[j]
            r2d += mp.mpf(float(wv[i])) * acc * acc
        assert abs(sse_by - r2d) <= mp.mpf(10) ** (-dps + 12) * max(1, r2d), \
            "QR sse byproduct R_last^2 != ||r||^2 at reference precision"
        # 6. se via formed gram (slope block p x p, scaled units).
        # gram(A[:,1:]) = R[:,1:]'R[:,1:] — INCLUDING row 0 of R (the
        # ones-column row contributes an outer product; an early draft
        # started the sum at `off` and lost it). For the intercept case the
        # shifted gram is then centered algebraically (G -= ss'/W); without
        # an intercept the se gram is the RAW one (c = 0, no centering —
        # same convention as the chol path).
        off = ic                      # slope block of R_s: cols off..dc-1
        G = [[mp.mpf(0)] * p for _ in range(p)]
        for a in range(p):
            for b in range(a, p):
                acc = mp.mpf(0)
                # sum over ALL rows of R[:,1:] that touch column off+a:
                # rows 0..off+a (row 0 = the ones-column row when ic=1,
                # off+a = the diagonal of column off+a; both were missing
                # in earlier drafts — caught by this file's own
                # reference-vs-numpy validation).
                for i in range(0, off + a + 1):
                    acc += Aug[i][off + a] * Aug[i][off + b]
                G[a][b] = acc
                G[b][a] = acc
        if ic:
            ss = [s[j] / norms[ic + j] for j in range(p)]  # scaled sums
            for a in range(p):
                for b in range(a, p):
                    G[a][b] -= ss[a] * ss[b] / W
                    G[b][a] = G[a][b]
        # chol-solve G z = e_j per slope, intercept via 1/W + m_s'G^-1 m_s
        ms = [xbar[j] / norms[ic + j] for j in range(p)]
        dof = n - p - ic
        sigma2 = sse_by / max(1, dof)
        # chol-solve G z = e_j per slope via an EXPLICIT Cholesky G = L L'
        # (an early draft divided by G_ii as if G were already a factor —
        # only a triangular R may be solved that way; caught by validation)
        L = [[mp.mpf(0)] * p for _ in range(p)]
        for i in range(p):
            for j in range(i + 1):
                acc = G[i][j]
                for k in range(j):
                    acc -= L[i][k] * L[j][k]
                if i == j:
                    L[i][i] = mp.sqrt(acc)
                else:
                    L[i][j] = acc / L[j][j]

        def _chol_solve_vec(rhs):
            t = [mp.mpf(0)] * p
            for i in range(p):                     # L t = rhs
                acc = rhs[i]
                for k in range(i):
                    acc -= L[i][k] * t[k]
                t[i] = acc / L[i][i]
            z = [mp.mpf(0)] * p
            for i in range(p - 1, -1, -1):         # L' z = t
                acc = t[i]
                for k in range(i + 1, p):
                    acc -= L[k][i] * z[k]
                z[i] = acc / L[i][i]
            return z

        se = []
        for j in range(p):
            z = _chol_solve_vec([mp.mpf(1) if i == j else mp.mpf(0)
                                 for i in range(p)])
            se.append(mp.sqrt(sigma2 * z[j]) / norms[ic + j])
        # intercept se
        z = _chol_solve_vec(ms)
        mGm = sum(ms[j] * z[j] for j in range(p))
        se0 = mp.sqrt(sigma2 * (1 / W + mGm))
        se = se + [se0]
        return {"beta": beta_raw, "b0": ybar - sum(xbar[j] * beta_raw[j]
                                                   for j in range(p)),
                "sse": sse_by, "se": se, "rank": dc}
