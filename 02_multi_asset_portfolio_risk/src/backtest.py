"""Daily walk-forward backtest with weight drift, turnover, costs and optional leverage.

Timeline: target weights decided at the close of month-end ``t`` (information ≤ t) are traded at
that close and earn returns from the next trading day. Between rebalances weights drift with
returns. Any weight not invested in assets (1 − Σw, negative when levered) earns or pays the
EUR overnight rate (ACT/360).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


@dataclass
class BacktestResult:
    """Daily gross/net returns, per-rebalance turnover and costs, leverage used."""
    gross: pd.Series
    net: pd.Series
    turnover: pd.Series
    cost: pd.Series
    leverage: pd.Series


def daily_cash_returns(dates: pd.DatetimeIndex, rate_pct: pd.Series) -> pd.Series:
    """Overnight accrual between consecutive dates: r_{t-1} · days / 360 (rate in percent)."""
    r = rate_pct.reindex(rate_pct.index.union(dates)).ffill().reindex(dates)
    gap = dates.to_series().diff().dt.days.fillna(1).to_numpy()
    return pd.Series(r.shift(1).bfill().to_numpy() / 100 * gap / 360, index=dates)


def run_backtest(daily: pd.DataFrame, targets: pd.DataFrame, cash: pd.Series,
                 cost_bps: float = 10.0, leverage: pd.Series | None = None,
                 charge_initial: bool = False, end: pd.Timestamp | str | None = None) -> BacktestResult:
    """Simulate a monthly-rebalanced portfolio on daily asset returns.

    Args:
        daily: daily EUR asset returns (NaN = market closed → 0 that day).
        targets: target weights indexed by decision date (columns = assets, rows sum to 1).
        cash: daily cash return on the same dates as ``daily`` (see ``daily_cash_returns``).
        cost_bps: cost per unit of traded notional, TC = c · Σ|w_new − w_pre|.
        leverage: optional scale applied to the targets at each decision date.
        charge_initial: whether the first build from cash is charged.
        end: last simulated day; default = month-end after the last decision date.
    """
    targets = targets.sort_index()
    end = pd.Timestamp(end) if end is not None else targets.index[-1] + pd.offsets.MonthEnd(1)
    days = daily.index[(daily.index > targets.index[0]) & (daily.index <= end)]
    R = daily.loc[days, targets.columns].fillna(0.0).to_numpy()
    cr = cash.reindex(days).fillna(0.0).to_numpy()
    tdates = targets.index
    T = targets.to_numpy()
    lev = (leverage.reindex(tdates).fillna(1.0).to_numpy() if leverage is not None else np.ones(len(tdates)))
    c = cost_bps / 1e4
    w = np.zeros(T.shape[1])
    k = 0
    gross, net = np.empty(len(days)), np.empty(len(days))
    to_rec, cost_rec, lev_rec = {}, {}, {}
    for i, d in enumerate(days):
        cost_d = 0.0
        while k < len(tdates) and tdates[k] < d:
            w_new = lev[k] * T[k]
            first = not np.any(w)
            turnover = float(np.abs(w_new - w).sum())
            cost_k = 0.0 if (first and not charge_initial) else c * turnover
            to_rec[tdates[k]], cost_rec[tdates[k]], lev_rec[tdates[k]] = (0.0 if first else turnover), cost_k, lev[k]
            cost_d += cost_k
            w = w_new
            k += 1
        r = R[i]
        g = float(w @ r + (1.0 - w.sum()) * cr[i])
        gross[i] = g
        net[i] = (1 + g) * (1 - cost_d) - 1
        w = w * (1 + r) / (1 + g)
    return BacktestResult(pd.Series(gross, days), pd.Series(net, days), pd.Series(to_rec),
                          pd.Series(cost_rec), pd.Series(lev_rec))


def vol_target_leverage(targets: pd.DataFrame, cov1w: np.ndarray, ref: pd.DataFrame,
                        cap: float = 5.0) -> pd.Series:
    """Leverage that equalises ex-ante volatility with a reference portfolio at each date."""
    out = {}
    for k, t in enumerate(targets.index):
        S = cov1w[k]
        w, b = targets.loc[t].to_numpy(), ref.loc[t].to_numpy()
        out[t] = min(cap, math.sqrt(b @ S @ b) / math.sqrt(w @ S @ w))
    return pd.Series(out)


def perf_stats(net: pd.Series, cash: pd.Series, turnover: pd.Series | None = None,
               cost: pd.Series | None = None) -> dict[str, float]:
    """Annualised performance and risk statistics from daily net returns."""
    r = net.dropna()
    cr = cash.reindex(r.index).fillna(0.0)
    years = (r.index[-1] - r.index[0]).days / 365.25
    wealth = (1 + r).cumprod()
    ann = wealth.iloc[-1] ** (1 / years) - 1
    ex = r - cr
    vol = r.std() * math.sqrt(TRADING_DAYS)
    down = ex[ex < 0]
    mdd = float((wealth / wealth.cummax() - 1).min())
    q = np.quantile(r, 0.05)
    out = {"ann. return": ann, "ann. vol": vol,
           "Sharpe": ex.mean() * TRADING_DAYS / vol,
           "Sortino": ex.mean() * TRADING_DAYS / (math.sqrt((down ** 2).sum() / len(ex)) * math.sqrt(TRADING_DAYS)),
           "max drawdown": mdd, "Calmar": ann / abs(mdd),
           "VaR95 1d": -q, "ES95 1d": -r[r <= q].mean()}
    if turnover is not None:
        out["turnover p.a."] = turnover.sum() / years
    if cost is not None:
        out["cost p.a."] = cost.sum() / years
    return out


def memmel_test(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Jobson–Korkie test with Memmel (2003) correction for equal Sharpe ratios (monthly excess returns).

    Returns (z, two-sided p); positive z ⇒ ``a`` has the higher Sharpe ratio.
    """
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    T = len(df)
    sa, sb = x.std(), y.std()
    ma, mb = x.mean(), y.mean()
    rho = x.corr(y)
    var = (2 * sa**2 * sb**2 - 2 * sa * sb * sa * sb * rho
           + 0.5 * ma**2 * sb**2 + 0.5 * mb**2 * sa**2 - ma * mb * sa * sb * rho**2) / (sa**2 * sb**2) / T
    z = (ma / sa - mb / sb) / math.sqrt(var)
    return float(z), float(math.erfc(abs(z) / math.sqrt(2)))
