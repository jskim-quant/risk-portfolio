"""One-day VaR / ES forecasts on a position-based pseudo-history and their backtests (numpy only).

For each month the current target weights (times leverage) are applied to the full history of
daily asset returns, giving the P&L series the portfolio *would* have had; every model uses only
observations strictly before the forecast day. Losses are positive numbers.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .risk import fit_garch11

MODELS: list[str] = ["Historical", "Normal", "EWMA", "GARCH-t", "FHS"]
HS_WINDOW, NORMAL_WINDOW, GARCH_WINDOW, EWMA_LAMBDA = 500, 250, 1000, 0.94
_T_CACHE: dict[float, np.ndarray] = {}


def _std_t_draws(nu: float, n: int = 400_000) -> np.ndarray:
    """Sorted draws from a unit-variance Student-t (ν rounded to 0.25, cached)."""
    key = round(nu * 4) / 4
    if key not in _T_CACHE:
        rng = np.random.default_rng(12345)
        x = rng.standard_t(key, n) * math.sqrt((key - 2) / key) if key < 200 else rng.standard_normal(n)
        _T_CACHE[key] = np.sort(x)
    return _T_CACHE[key]


def _tail(x_sorted_or_sample: np.ndarray, alpha: float) -> tuple[float, float]:
    """(VaR, ES) of losses for a sample of *returns* at confidence ``alpha``."""
    loss = -np.asarray(x_sorted_or_sample)
    v = float(np.quantile(loss, alpha))
    return v, float(loss[loss >= v].mean())


def _normal_es_factor(alpha: float) -> tuple[float, float]:
    """(z_α, φ(z_α)/(1−α)) for the standard normal, via bisection on erf."""
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(mid / math.sqrt(2)) > 1 - alpha:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2
    return z, math.exp(-z * z / 2) / math.sqrt(2 * math.pi) / (1 - alpha)


def nu_from_kurtosis(z: np.ndarray) -> float:
    """Method-of-moments Student-t degrees of freedom: ν = 4 + 6 / excess kurtosis (clipped to [4.25, 50])."""
    z = z - z.mean()
    ek = np.mean(z ** 4) / np.mean(z ** 2) ** 2 - 3
    return 50.0 if ek <= 0.12 else float(np.clip(4 + 6 / ek, 4.25, 50.0))


def var_forecasts(asset_daily: pd.DataFrame, targets: pd.DataFrame, leverage: pd.Series | None = None,
                  alphas: tuple[float, ...] = (0.95, 0.99), end: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """Daily VaR/ES forecasts for every model, columns MultiIndex (model, alpha, VaR|ES).

    ``asset_daily`` must start well before the first decision date (pseudo-history).
    """
    R = asset_daily[targets.columns].fillna(0.0)
    dates = R.index
    X = R.to_numpy()
    tdates = targets.index
    end = pd.Timestamp(end) if end is not None else tdates[-1] + pd.offsets.MonthEnd(1)
    lev = leverage.reindex(tdates).fillna(1.0) if leverage is not None else pd.Series(1.0, index=tdates)
    zN = {a: _normal_es_factor(a) for a in alphas}
    rows, idx = [], []
    for k, t in enumerate(tdates):
        t_next = tdates[k + 1] if k + 1 < len(tdates) else end
        days = np.where((dates > t) & (dates <= t_next))[0]
        if len(days) == 0:
            continue
        h = X @ (lev.iloc[k] * targets.iloc[k].to_numpy())
        i0 = days[0]
        # GARCH fitted once per month on the window ending the day before the month starts
        win = h[max(0, i0 - GARCH_WINDOW):i0]          # up to 1000 days (≈800 available in 2010)
        g = fit_garch11(win)
        om = g.vbar * (1 - g.persistence)
        mu = win.mean()
        s2 = g.vbar
        for x in win - mu:
            s2 = om + g.alpha * x * x + g.beta * s2
        s2_path = np.empty(len(win))
        v = g.vbar
        for j, x in enumerate(win - mu):
            s2_path[j] = v
            v = om + g.alpha * x * x + g.beta * v
        z = (win - mu) / np.sqrt(s2_path)
        nu = nu_from_kurtosis(z)
        tdraw = _std_t_draws(nu)
        # EWMA variance at the month start from the last 1000 observations
        past = h[max(0, i0 - 1000):i0]
        wts = (1 - EWMA_LAMBDA) * EWMA_LAMBDA ** np.arange(len(past) - 1, -1, -1)
        e2 = float(wts @ past ** 2 / wts.sum())
        for i in days:
            row = {}
            hs = h[i - HS_WINDOW:i]
            sd = h[i - NORMAL_WINDOW:i].std(ddof=1)
            for a in alphas:
                row[("Historical", a, "VaR")], row[("Historical", a, "ES")] = _tail(hs, a)
                za, ef = zN[a]
                row[("Normal", a, "VaR")], row[("Normal", a, "ES")] = sd * za, sd * ef
                row[("EWMA", a, "VaR")], row[("EWMA", a, "ES")] = math.sqrt(e2) * za, math.sqrt(e2) * ef
                vt, et = _tail(tdraw, a)
                sg = math.sqrt(s2)
                row[("GARCH-t", a, "VaR")], row[("GARCH-t", a, "ES")] = sg * vt - mu, sg * et - mu
                vf, ef_ = _tail(z, a)
                row[("FHS", a, "VaR")], row[("FHS", a, "ES")] = sg * vf - mu, sg * ef_ - mu
            rows.append(row)
            idx.append(dates[i])
            x = h[i]
            e2 = EWMA_LAMBDA * e2 + (1 - EWMA_LAMBDA) * x * x
            s2 = om + g.alpha * (x - mu) ** 2 + g.beta * s2
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["model", "alpha", "measure"])
    return out


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of χ² with 1 or 2 degrees of freedom (closed form)."""
    x = max(x, 0.0)
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    if df == 2:
        return math.exp(-x / 2)
    raise ValueError("df must be 1 or 2")


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * math.log(y)


def kupiec(hits: np.ndarray, p: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures LR test; returns (LR, p-value)."""
    n, x = len(hits), int(hits.sum())
    pi = x / n
    lr = -2 * (_xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - pi) - _xlogy(x, pi))
    return lr, _chi2_sf(lr, 1)


def christoffersen(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) independence LR test (first-order Markov); returns (LR, p-value)."""
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    p0 = n01 / max(n00 + n01, 1)
    p1 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = _xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)
    l1 = _xlogy(n00, 1 - p0) + _xlogy(n01, p0) + _xlogy(n10, 1 - p1) + _xlogy(n11, p1)
    lr = -2 * (l0 - l1)
    return lr, _chi2_sf(lr, 1)


def backtest_table(losses: pd.Series, fc: pd.DataFrame) -> pd.DataFrame:
    """Exception counts, Kupiec, Christoffersen, conditional coverage and an ES ratio per model × alpha."""
    out = {}
    L = losses.reindex(fc.index)
    for m in fc.columns.get_level_values(0).unique():
        for a in fc.columns.get_level_values(1).unique():
            var, es = fc[(m, a, "VaR")], fc[(m, a, "ES")]
            hits = (L > var).to_numpy()
            lr_uc, p_uc = kupiec(hits, 1 - a)
            lr_ind, p_ind = christoffersen(hits)
            exc = L[hits]
            out[(m, f"{a:.0%}")] = {
                "days": len(hits), "exceptions": int(hits.sum()), "expected": (1 - a) * len(hits),
                "rate %": 100 * hits.mean(), "Kupiec p": p_uc, "Christoffersen p": p_ind,
                "cond. coverage p": _chi2_sf(lr_uc + lr_ind, 2),
                "loss / ES on exceptions": float((exc / es[hits]).mean()) if hits.any() else np.nan}
    return pd.DataFrame(out).T


def traffic_light(losses: pd.Series, var99: pd.Series, window: int = 250) -> pd.Series:
    """Share of rolling 250-day windows in the Basel green (≤4), yellow (5–9) and red (≥10) zones."""
    hits = (losses.reindex(var99.index) > var99).astype(int)
    cnt = hits.rolling(window).sum().dropna()
    return pd.Series({"green": (cnt <= 4).mean(), "yellow": ((cnt >= 5) & (cnt <= 9)).mean(), "red": (cnt >= 10).mean()})


def cached_forecasts(path, asset_daily: pd.DataFrame, targets: pd.DataFrame,
                     leverage: pd.Series | None = None) -> pd.DataFrame:
    """``var_forecasts`` with a pickle cache (≈90 s per portfolio on first run)."""
    from pathlib import Path
    p = Path(path)
    if p.exists():
        return pd.read_pickle(p)
    out = var_forecasts(asset_daily, targets, leverage)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(p)
    return out
