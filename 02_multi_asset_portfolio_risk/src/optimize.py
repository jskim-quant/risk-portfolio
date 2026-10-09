"""Portfolio construction rules implemented with numpy only.

* ``solve_qp`` — OSQP-style ADMM for ``min ½xᵀPx + qᵀx  s.t.  l ≤ Ax ≤ u`` (also solves LPs).
* Minimum variance, mean-variance, CVaR (Rockafellar–Uryasev LP) on top of ``solve_qp``.
* Risk parity by cyclical coordinate descent (Griveau-Billion, Richard & Roncalli 2013).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ASSETS: list[str] = ["SPY", "VGK", "EWJ", "SHY", "IEF", "TLT", "EUR_CORE7_10_SYN",
                     "JGB7_10_SYN", "LQD", "HYG", "GLD", "DBC"]
# Global 60/40: equity split ≈ developed-market caps (US 70 / Europe 20 / Japan 10),
# bonds tilted to the EUR home market, all bonds FX-hedged.
W_6040: dict[str, float] = {"SPY": 0.42, "VGK": 0.12, "EWJ": 0.06,
                            "EUR_CORE7_10_SYN": 0.20, "IEF": 0.15, "JGB7_10_SYN": 0.05}


@dataclass
class QPResult:
    """Solution of ``solve_qp``."""
    x: np.ndarray
    iters: int
    prim_res: float
    dual_res: float
    converged: bool


def solve_qp(P: np.ndarray, q: np.ndarray, A: np.ndarray, l: np.ndarray, u: np.ndarray,
             rho: float = 0.1, sigma: float = 1e-6, alpha: float = 1.6, max_iter: int = 50_000,
             eps: float = 1e-8) -> QPResult:
    """ADMM (OSQP algorithm, fixed step) for a convex QP/LP with two-sided linear constraints."""
    n, m = P.shape[0], A.shape[0]
    eq = np.isclose(l, u)
    rho_v = np.where(eq, rho * 1e3, rho)
    K = P + sigma * np.eye(n) + A.T @ (rho_v[:, None] * A)
    K_inv = np.linalg.inv(K)
    x, z, y = np.zeros(n), np.zeros(m), np.zeros(m)
    for k in range(1, max_iter + 1):
        x_t = K_inv @ (sigma * x - q + A.T @ (rho_v * z - y))
        z_t = A @ x_t
        x = alpha * x_t + (1 - alpha) * x
        z_relax = alpha * z_t + (1 - alpha) * z
        z_new = np.clip(z_relax + y / rho_v, l, u)
        y = y + rho_v * (z_relax - z_new)
        z = z_new
        if k % 25 == 0:
            r_p = np.max(np.abs(A @ x - z))
            r_d = np.max(np.abs(P @ x + q + A.T @ y))
            if r_p < eps and r_d < eps:
                return QPResult(x, k, r_p, r_d, True)
    return QPResult(x, max_iter, float(np.max(np.abs(A @ x - z))),
                    float(np.max(np.abs(P @ x + q + A.T @ y))), False)


def _weight_constraints(n: int, cap: float | np.ndarray = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows for Σw = 1 and 0 ≤ w ≤ cap."""
    A = np.vstack([np.ones((1, n)), np.eye(n)])
    l = np.concatenate([[1.0], np.zeros(n)])
    u = np.concatenate([[1.0], np.broadcast_to(cap, n)])
    return A, l, u


def _clean(w: np.ndarray) -> np.ndarray:
    """Remove solver noise: clip tiny negatives and renormalise."""
    w = np.where(np.abs(w) < 1e-7, 0.0, w)
    w = np.clip(w, 0.0, None)
    return w / w.sum()


def min_variance(cov: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """Long-only minimum-variance weights with a per-asset cap."""
    n = cov.shape[0]
    A, l, u = _weight_constraints(n, cap)
    scale = 1.0 / np.mean(np.diag(cov))
    return _clean(solve_qp(cov * scale, np.zeros(n), A, l, u).x)


def gmv_unconstrained(cov: np.ndarray) -> np.ndarray:
    """Closed-form global minimum variance (shorting allowed): Σ⁻¹1 / 1ᵀΣ⁻¹1."""
    x = np.linalg.solve(cov, np.ones(cov.shape[0]))
    return x / x.sum()


def mean_variance(mu: np.ndarray, cov: np.ndarray, gamma: float = 5.0, cap: float = 1.0) -> np.ndarray:
    """Long-only ``max μᵀw − γ/2 wᵀΣw`` (inputs annualised)."""
    n = cov.shape[0]
    A, l, u = _weight_constraints(n, cap)
    return _clean(solve_qp(gamma * cov, -mu, A, l, u).x)


def risk_parity(cov: np.ndarray, budget: np.ndarray | None = None, tol: float = 1e-12,
                max_iter: int = 10_000) -> np.ndarray:
    """Long-only (equal) risk-budget portfolio: w_i (Σw)_i ∝ b_i."""
    n = cov.shape[0]
    b = np.full(n, 1.0 / n) if budget is None else np.asarray(budget, float) / np.sum(budget)
    y = 1.0 / np.sqrt(np.diag(cov))
    y = y / y.sum()
    d = np.diag(cov)
    for _ in range(max_iter):
        y_old = y.copy()
        for i in range(n):
            c = cov[i] @ y - d[i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4 * d[i] * b[i])) / (2 * d[i])
        if np.max(np.abs(y - y_old)) < tol:
            break
    return y / y.sum()


def min_cvar(scenarios: np.ndarray, beta: float = 0.95, cap: float = 1.0) -> tuple[np.ndarray, float]:
    """Rockafellar–Uryasev: minimise CVaR_β of portfolio loss −Rw over S scenarios.

    Variables x = [w (n), a (1), z (S)];  min a + Σz / ((1−β)S)
    s.t. z_s + R_s w + a ≥ 0, z ≥ 0, Σw = 1, 0 ≤ w ≤ cap.
    """
    R = np.asarray(scenarios, float)
    S, n = R.shape
    scale = 1.0 / np.std(R)
    Rs = R * scale
    N = n + 1 + S
    q = np.concatenate([np.zeros(n), [1.0], np.full(S, 1.0 / ((1 - beta) * S))])
    rows = [np.hstack([Rs, np.ones((S, 1)), np.eye(S)]),            # z + Rw + a ≥ 0
            np.hstack([np.zeros((S, n + 1)), np.eye(S)]),             # z ≥ 0
            np.concatenate([np.ones(n), np.zeros(1 + S)])[None, :],    # Σw = 1
            np.hstack([np.eye(n), np.zeros((n, 1 + S))])]             # 0 ≤ w ≤ cap
    A = np.vstack(rows)
    l = np.concatenate([np.zeros(S), np.zeros(S), [1.0], np.zeros(n)])
    u = np.concatenate([np.full(S, np.inf), np.full(S, np.inf), [1.0], np.full(n, cap)])
    res = solve_qp(1e-6 * np.eye(N), q, A, l, u, rho=0.5, max_iter=100_000, eps=1e-7)
    w = _clean(res.x[:n])
    return w, cvar(R @ w, beta)


def cvar(returns: np.ndarray, beta: float = 0.95) -> float:
    """Empirical CVaR_β of losses (positive number = loss), Rockafellar–Uryasev definition."""
    loss = -np.asarray(returns, float)
    var = np.quantile(loss, beta)
    return float(var + np.mean(np.maximum(loss - var, 0.0)) / (1 - beta))


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fraction of portfolio variance from each asset: w_i (Σw)_i / wᵀΣw."""
    mc = cov @ w
    return w * mc / (w @ mc)


def benchmark_6040(assets: list[str] = ASSETS) -> pd.Series:
    """Fixed global 60/40 weights aligned to ``assets``."""
    return pd.Series(W_6040).reindex(assets).fillna(0.0)


METHODS: list[str] = ["EW", "60/40", "MinVar", "RiskParity", "CVaR", "MV-hist"]


def weight_paths(fc: dict, weekly: pd.DataFrame, estimator: str = "GARCH-CCC", cap: float = 0.30,
                 cvar_beta: float = 0.95, scen_weeks: int = 156, gamma: float = 5.0,
                 cache_dir=None, max_new: int | None = None) -> dict[str, pd.DataFrame]:
    """Monthly target weights for every method, using only information up to each forecast date.

    * MinVar / RiskParity use the 1-week covariance forecast of ``estimator``.
    * CVaR uses the last ``scen_weeks`` weekly returns as equally likely scenarios.
    * MV-hist (cautionary) uses rolling 3-year historical means, annualised, with the same Σ.
    Results are cached per method as CSV in ``cache_dir``; at most ``max_new`` new dates are
    computed per call so long runs can be resumed.
    """
    from pathlib import Path
    assets = fc["assets"]
    dates = fc["dates"]
    cache = {}
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        for m in METHODS:
            p = Path(cache_dir) / f"weights_{m.replace('/', '')}.csv"
            cache[m] = pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else pd.DataFrame(columns=assets)
    else:
        cache = {m: pd.DataFrame(columns=assets) for m in METHODS}
    todo = [k for k, t in enumerate(dates) if any(t not in cache[m].index for m in METHODS)]
    if max_new is not None:
        todo = todo[:max_new]
    ew, b6040 = np.full(len(assets), 1 / len(assets)), benchmark_6040(assets).to_numpy()
    for k in todo:
        t = dates[k]
        S = fc["cov1w"][estimator][k]
        hist = weekly.loc[:t, assets].to_numpy()
        rows = {"EW": ew, "60/40": b6040, "MinVar": min_variance(S, cap), "RiskParity": risk_parity(S),
                "CVaR": min_cvar(hist[-scen_weeks:], cvar_beta, cap)[0],
                "MV-hist": mean_variance(hist[-156:].mean(axis=0) * 52, S * 52, gamma, cap)}
        for m, w in rows.items():
            cache[m].loc[t] = w
    out = {}
    for m in METHODS:
        df = cache[m].astype(float).sort_index()
        df.index.name = "date"
        if cache_dir is not None:
            df.to_csv(Path(cache_dir) / f"weights_{m.replace('/', '')}.csv")
        out[m] = df
    return out
