"""Covariance estimators and variance-forecast evaluation (numpy only).

All estimators take a (T × N) array of past **weekly** returns and return a one-week
covariance forecast; ``horizon_cov`` turns it into an h-week forecast (GARCH term structure
mean-reverts, the others scale linearly).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

ESTIMATORS: list[str] = ["Sample", "Rolling", "EWMA", "LedoitWolf", "GARCH-CCC"]
ROLL_WEEKS: int = 156
EWMA_HALFLIFE: float = 26.0


def sample_cov(R: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance of all rows (expanding window)."""
    return np.cov(R, rowvar=False)


def rolling_cov(R: np.ndarray, window: int = ROLL_WEEKS) -> np.ndarray:
    """Sample covariance of the last ``window`` rows."""
    return np.cov(R[-window:], rowvar=False)


def ewma_cov(R: np.ndarray, halflife: float = EWMA_HALFLIFE) -> np.ndarray:
    """RiskMetrics EWMA on zero-mean returns: Σ_t = λΣ_{t-1} + (1-λ) r r'; λ = 0.5^(1/halflife)."""
    lam = 0.5 ** (1.0 / halflife)
    w = (1 - lam) * lam ** np.arange(len(R) - 1, -1, -1)
    w = w / w.sum()
    return (R * w[:, None]).T @ R


def ledoit_wolf_cc(R: np.ndarray, window: int = ROLL_WEEKS) -> tuple[np.ndarray, float]:
    """Ledoit–Wolf (2004) shrinkage towards constant correlation on the last ``window`` rows.

    Returns the shrunk covariance and the shrinkage intensity δ ∈ [0, 1].
    """
    X = R[-window:]
    T, N = X.shape
    X = X - X.mean(axis=0)
    S = X.T @ X / T
    sd = np.sqrt(np.diag(S))
    corr = S / np.outer(sd, sd)
    rbar = (corr.sum() - N) / (N * (N - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, np.diag(S))
    Y = X ** 2
    pi_mat = (Y.T @ Y) / T - S ** 2
    pi_hat = pi_mat.sum()
    theta = ((X ** 3).T @ X) / T - np.diag(S)[:, None] * S   # θ_ii,ij = E[x_i³ x_j] − s_ii s_ij
    rho_off = (rbar / 2) * ((np.outer(1 / sd, sd) * theta) + (np.outer(sd, 1 / sd) * theta.T))
    np.fill_diagonal(rho_off, 0.0)
    rho_hat = np.trace(pi_mat) + rho_off.sum()
    gamma_hat = np.sum((F - S) ** 2)
    delta = float(np.clip((pi_hat - rho_hat) / gamma_hat / T, 0.0, 1.0))
    Sigma = delta * F + (1 - delta) * S
    return Sigma * T / (T - 1), delta


@dataclass
class Garch11:
    """GARCH(1,1) with variance targeting: σ²_t = ω + α r²_{t-1} + β σ²_{t-1}, ω = v̄(1−α−β)."""
    alpha: float
    beta: float
    vbar: float
    loglik: float

    @property
    def persistence(self) -> float:
        """α + β."""
        return self.alpha + self.beta


def _garch_loglik_grid(r: np.ndarray, a: np.ndarray, b: np.ndarray, vbar: float) -> np.ndarray:
    """Gaussian log-likelihood for many (α, β) pairs at once (vectorised over the grid)."""
    om = vbar * (1 - a - b)
    s2 = np.full(a.shape, vbar)
    ll = np.zeros(a.shape)
    r2 = r ** 2
    for t in range(len(r)):
        ll -= 0.5 * (np.log(s2) + r2[t] / s2)
        s2 = om + a * r2[t] + b * s2
    return ll


def fit_garch11(r: np.ndarray) -> Garch11:
    """MLE by grid search plus two refinement rounds (α ∈ [0.005, 0.35], β ∈ [0.5, 0.995], α+β < 0.999)."""
    r = np.asarray(r, float)
    r = r - r.mean()
    vbar = float(r.var())
    a_lo, a_hi, b_lo, b_hi, k = 0.005, 0.35, 0.50, 0.995, 36
    best = (0.05, 0.90)
    for _ in range(3):
        A, B = np.meshgrid(np.linspace(a_lo, a_hi, k), np.linspace(b_lo, b_hi, k))
        A, B = A.ravel(), B.ravel()
        ok = A + B < 0.999
        A, B = A[ok], B[ok]
        ll = _garch_loglik_grid(r, A, B, vbar)
        i = int(np.argmax(ll))
        best, best_ll = (A[i], B[i]), ll[i]
        da, db = (a_hi - a_lo) / k * 2, (b_hi - b_lo) / k * 2
        a_lo, a_hi = max(0.0005, best[0] - da), best[0] + da
        b_lo, b_hi = max(0.0, best[1] - db), min(0.998, best[1] + db)
    return Garch11(float(best[0]), float(best[1]), vbar, float(best_ll))


def garch_filter(r: np.ndarray, g: Garch11) -> tuple[np.ndarray, float]:
    """Conditional variances σ²_1..σ²_T and the one-step-ahead forecast σ²_{T+1}."""
    r = np.asarray(r, float) - np.mean(r)
    om = g.vbar * (1 - g.persistence)
    s2 = np.empty(len(r))
    v = g.vbar
    for t, x in enumerate(r):
        s2[t] = v
        v = om + g.alpha * x * x + g.beta * v
    return s2, float(v)


def garch_horizon_var(next_var: float, g: Garch11, h: float) -> float:
    """Sum of k-step variance forecasts k = 1..h (fractional h interpolated)."""
    k = np.arange(int(np.floor(h)))
    phi = g.persistence
    total = np.sum(g.vbar + phi ** k * (next_var - g.vbar))
    frac = h - np.floor(h)
    return float(total + frac * (g.vbar + phi ** np.floor(h) * (next_var - g.vbar)))


def garch_ccc_cov(R: np.ndarray, params: list[Garch11], h: float = 1.0) -> np.ndarray:
    """CCC-GARCH h-week covariance: D_h^{1/2} C D_h^{1/2}, C = corr of standardised residuals."""
    n = R.shape[1]
    z = np.empty_like(R)
    hv = np.empty(n)
    for i in range(n):
        s2, nxt = garch_filter(R[:, i], params[i])
        z[:, i] = (R[:, i] - R[:, i].mean()) / np.sqrt(s2)
        hv[i] = garch_horizon_var(nxt, params[i], h)
    C = np.corrcoef(z, rowvar=False)
    sd = np.sqrt(hv)
    return C * np.outer(sd, sd)


def horizon_cov(name: str, R: np.ndarray, h: float, garch_params: list[Garch11] | None = None) -> np.ndarray:
    """h-week covariance forecast for estimator ``name`` from past weekly returns ``R``."""
    if name == "Sample":
        return h * sample_cov(R)
    if name == "Rolling":
        return h * rolling_cov(R)
    if name == "EWMA":
        return h * ewma_cov(R)
    if name == "LedoitWolf":
        return h * ledoit_wolf_cc(R)[0]
    if name == "GARCH-CCC":
        if garch_params is None:
            raise ValueError("GARCH-CCC needs fitted parameters")
        return garch_ccc_cov(R, garch_params, h)
    raise ValueError(f"unknown estimator {name!r}")


def qlike(realised: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """QLIKE loss RV/σ̂² − ln(RV/σ̂²) − 1 (robust to noisy unbiased RV proxies; Patton 2011)."""
    x = np.asarray(realised) / np.asarray(forecast)
    return x - np.log(x) - 1


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 3) -> tuple[float, float]:
    """DM test of equal predictive accuracy with Newey–West variance; returns (stat, two-sided p).

    Negative stat ⇒ model a has lower loss.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    T = len(d)
    dc = d - d.mean()
    lrv = dc @ dc / T
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / T
    stat = d.mean() / math.sqrt(lrv / T)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return float(stat), float(p)


def month_end_schedule(weekly_index: pd.DatetimeIndex, start: str) -> list[tuple[pd.Timestamp, pd.DatetimeIndex]]:
    """(forecast date, weeks of the following calendar month) pairs for the walk-forward."""
    ends = pd.date_range(start, weekly_index[-1], freq="ME")
    out = []
    for t0, t1 in zip(ends[:-1], ends[1:]):
        nxt = weekly_index[(weekly_index > t0) & (weekly_index <= t1)]
        if len(nxt):
            out.append((t0, nxt))
    return out


def walk_forward_forecasts(weekly: pd.DataFrame, start: str = "2010-04-30",
                           refit_month: int = 1) -> dict:
    """Monthly walk-forward covariance forecasts for every estimator.

    At each month-end t only weekly returns dated ≤ t are used. GARCH parameters are
    re-estimated on the expanding window at the first date and every January (``refit_month``);
    in between, the variance filter is updated with the latest data and fixed parameters.

    Returns a dict with ``dates``, ``h`` (weeks in the next month), ``next_weeks`` (index
    arrays), per-estimator ``cov1w`` (1-week) and ``covh`` (h-week) arrays of shape
    (D, N, N), ``lw_delta`` and the GARCH parameter history.
    """
    R_all = weekly.to_numpy()
    idx = weekly.index
    sched = month_end_schedule(idx, start)
    D, N = len(sched), R_all.shape[1]
    cov1 = {e: np.empty((D, N, N)) for e in ESTIMATORS}
    covh = {e: np.empty((D, N, N)) for e in ESTIMATORS}
    hs, deltas, nxt_weeks, garch_hist = np.empty(D), np.empty(D), [], {}
    params: list[Garch11] | None = None
    for k, (t, weeks) in enumerate(sched):
        R = R_all[idx <= t]
        h = float(len(weeks))
        if params is None or t.month == refit_month:
            params = [fit_garch11(R[:, i]) for i in range(N)]
            garch_hist[t] = params
        for e in ESTIMATORS:
            cov1[e][k] = horizon_cov(e, R, 1.0, params)
            covh[e][k] = horizon_cov(e, R, h, params)
        deltas[k] = ledoit_wolf_cc(R)[1]
        hs[k] = h
        nxt_weeks.append(weeks)
    return {"dates": pd.DatetimeIndex([s[0] for s in sched]), "h": hs, "next_weeks": nxt_weeks,
            "cov1w": cov1, "covh": covh, "lw_delta": deltas, "garch": garch_hist,
            "assets": list(weekly.columns)}


def save_forecasts(fc: dict, path) -> None:
    """Store walk-forward covariance forecasts (npz) for notebooks 04–06."""
    arrays = {f"cov1w__{e}": fc["cov1w"][e] for e in ESTIMATORS}
    arrays |= {f"covh__{e}": fc["covh"][e] for e in ESTIMATORS}
    np.savez_compressed(path, dates=fc["dates"].values.astype("datetime64[ns]"), h=fc["h"],
                        lw_delta=fc["lw_delta"], assets=np.array(fc["assets"]), **arrays)


def load_forecasts(path) -> dict:
    """Inverse of ``save_forecasts`` (without the next-week index lists and GARCH history)."""
    z = np.load(path, allow_pickle=False)
    return {"dates": pd.DatetimeIndex(z["dates"]), "h": z["h"], "lw_delta": z["lw_delta"],
            "assets": list(z["assets"]),
            "cov1w": {e: z[f"cov1w__{e}"] for e in ESTIMATORS},
            "covh": {e: z[f"covh__{e}"] for e in ESTIMATORS}}
