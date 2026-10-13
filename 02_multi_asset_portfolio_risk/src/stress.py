"""Historical stress testing: scenario replay of current portfolios and realised window statistics.

Replay = buy-and-hold of today's target weights through a past window, so the P&L decomposes
exactly into asset (and cash/financing) contributions:
    R = Σ_i L w_i (Π(1+r_i) − 1) + (1 − L)(Π(1+r_cash) − 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReplayResult:
    """Cumulative return, drawdown, worst day, value path and exact contributions of a replay."""
    cum_return: float
    max_drawdown: float
    worst_day: float
    path: pd.Series
    contributions: pd.Series


def window_days(index: pd.DatetimeIndex, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Trading days in (start, end]: the return of ``start`` itself is not part of the window."""
    return index[(index > pd.Timestamp(start)) & (index <= pd.Timestamp(end))]


def replay(weights: pd.Series, daily: pd.DataFrame, start, end, cash: pd.Series | None = None,
           leverage: float = 1.0) -> ReplayResult:
    """Buy-and-hold replay of ``weights`` (scaled by ``leverage``) over a historical window.

    Missing daily returns (closed market) count as 0. The unfunded part 1 − L·Σw earns
    (or pays, if negative) the cash return.
    """
    days = window_days(daily.index, start, end)
    R = daily.loc[days, weights.index].fillna(0.0)
    growth = (1 + R).cumprod()
    w = weights.to_numpy() * leverage
    c = (1 + (cash.reindex(days).fillna(0.0) if cash is not None else pd.Series(0.0, index=days))).cumprod()
    cash_w = 1.0 - w.sum()
    value = growth.to_numpy() @ w + cash_w * c.to_numpy()
    path = pd.Series(np.concatenate([[1.0], value]), index=[pd.Timestamp(start), *days])
    contrib = pd.Series(w * (growth.iloc[-1].to_numpy() - 1), index=weights.index)
    contrib["cash/financing"] = cash_w * (c.iloc[-1] - 1)
    daily_ret = path.pct_change().dropna()
    return ReplayResult(float(path.iloc[-1] - 1), float((path / path.cummax() - 1).min()),
                        float(daily_ret.min()), path, contrib)


def realised(net: pd.Series, start, end) -> dict[str, float]:
    """Statistics of realised daily net returns inside (start, end]."""
    r = net.loc[window_days(net.index, start, end)].dropna()
    if r.empty:
        return {}
    w = (1 + r).cumprod()
    q = np.quantile(r, 0.05)
    return {"cum. return": float(w.iloc[-1] - 1), "max drawdown": float((w / w.cummax().clip(lower=1) - 1).min()),
            "ann. vol": float(r.std() * np.sqrt(252)), "worst day": float(r.min()), "ES95 1d": float(-r[r <= q].mean()),
            "days": float(len(r))}


def available(daily: pd.DataFrame, assets: list[str], start) -> list[str]:
    """Assets of ``assets`` whose first return is on or before ``start`` (holiday NaN ignored)."""
    return [a for a in assets if daily[a].first_valid_index() is not None
            and daily[a].first_valid_index() <= pd.Timestamp(start)]
