"""Daily walk-forward backtest with weight drift, turnover, costs and optional leverage.

Timeline: target weights decided at the close of month-end ``t`` (information ≤ t) are traded at
that close and earn returns from the next trading day. Between rebalances weights drift with
returns. Any weight not invested in assets (1 − Σw, negative when levered) earns or pays the
EUR overnight rate (ACT/360).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252  # legacy constant; perf_stats now infers the observation frequency


@dataclass
class BacktestResult:
    """Daily gross/net returns, per-rebalance turnover and costs, leverage used, NAV ledger.

    ``gross`` is the return before transaction costs and *after* cash interest / funding costs;
    ``net`` additionally deducts transaction costs. ``turnover`` and ``cost`` are recorded per
    rebalance decision and exclude the initial build from cash, which is reported separately in
    ``initial_turnover`` / ``initial_cost``. ``ledger`` holds the daily NAV decomposition
    (NAV_open = 1 at the start) and ``ruined`` the first day NAV hit zero (``None`` otherwise).
    """
    gross: pd.Series
    net: pd.Series
    turnover: pd.Series
    cost: pd.Series
    leverage: pd.Series
    initial_turnover: float = 0.0
    initial_cost: float = 0.0
    ledger: pd.DataFrame | None = None
    ruined: pd.Timestamp | None = None
    funding_spread_bps: float = 0.0
    meta: dict = field(default_factory=dict)


def daily_cash_returns(dates: pd.DatetimeIndex, rate_pct: pd.Series) -> pd.Series:
    """Overnight accrual between consecutive dates: r_{t-1} · days / 360 (rate in percent)."""
    r = rate_pct.reindex(rate_pct.index.union(dates)).ffill().reindex(dates)
    gap = dates.to_series().diff().dt.days.fillna(1).to_numpy()
    return pd.Series(r.shift(1).bfill().to_numpy() / 100 * gap / 360, index=dates)


def _cost_vector(cost_bps: float | pd.Series | dict, assets: pd.Index) -> np.ndarray:
    """Per-asset cost per unit traded notional (fraction); a scalar applies to every asset."""
    if np.isscalar(cost_bps):
        return np.full(len(assets), float(cost_bps) / 1e4)
    s = pd.Series(cost_bps, dtype=float)
    missing = [a for a in assets if a not in s.index]
    if missing:
        raise ValueError(f"cost_bps has no value for {missing}")
    return s.reindex(assets).to_numpy() / 1e4


def run_backtest(daily: pd.DataFrame, targets: pd.DataFrame, cash: pd.Series,
                 cost_bps: float | pd.Series | dict = 10.0, leverage: pd.Series | None = None,
                 charge_initial: bool = False, end: pd.Timestamp | str | None = None,
                 funding_spread_bps: float = 0.0) -> BacktestResult:
    """Simulate a monthly-rebalanced, self-financing portfolio on daily asset returns.

    Accounting for day d (weights w are fractions of NAV after the trade at the previous close;
    cash weight c = 1 − Σw, borrowed when negative)::

        TC_d       = NAV_open · Σ_i k_i |w_i^new − w_i^pre|          (only on rebalance days)
        NAV_inv    = NAV_open − TC_d                                   (costs paid pro rata)
        PnL_assets = NAV_inv · Σ_i w_i r_{i,d}
        PnL_cash   = NAV_inv · max(c, 0) · r^cash_d
        Funding    = NAV_inv · max(−c, 0) · (r^cash_d + s · Δdays/360)
        NAV_close  = NAV_inv + PnL_assets + PnL_cash − Funding

    gross_d = NAV_close / NAV_inv − 1 and net_d = NAV_close / NAV_open − 1 = (1 − TC_d/NAV_open)(1 + gross_d) − 1.
    With ``funding_spread_bps = 0`` this reproduces the original engine exactly. Between
    rebalances weights drift as w ← w ∘ (1 + r) / (1 + gross). NAV is floored at zero (limited
    liability): once wiped out the portfolio stays in cash at zero value.

    Args:
        daily: daily EUR asset returns (NaN = market closed → 0 that day; the move is captured
            the next day because returns are built from forward-filled price levels).
        targets: target weights indexed by decision date (columns = assets, rows sum to 1).
        cash: daily cash return on the same dates as ``daily`` (see ``daily_cash_returns``).
        cost_bps: cost per unit of traded notional in bp — scalar or per-asset mapping.
        leverage: optional scale applied to the targets at each decision date.
        charge_initial: whether the first build from cash is charged.
        end: last simulated day; default = month-end after the last decision date.
        funding_spread_bps: annual spread over the overnight rate paid on borrowed cash only
            (ACT/360); positive cash balances earn the overnight rate without spread.
    """
    targets = targets.sort_index()
    end = pd.Timestamp(end) if end is not None else targets.index[-1] + pd.offsets.MonthEnd(1)
    days = daily.index[(daily.index > targets.index[0]) & (daily.index <= end)]
    R = daily.loc[days, targets.columns].fillna(0.0).to_numpy()
    cr = cash.reindex(days).fillna(0.0).to_numpy()
    gap = np.asarray((days - days.insert(0, targets.index[0])[:-1]) / pd.Timedelta(days=1))
    sp = funding_spread_bps / 1e4 * gap / 360
    tdates = targets.index
    T = targets.to_numpy()
    lev = (leverage.reindex(tdates).fillna(1.0).to_numpy() if leverage is not None else np.ones(len(tdates)))
    kvec = _cost_vector(cost_bps, targets.columns)
    n = len(days)
    w = np.zeros(T.shape[1])
    k = 0
    nav = 1.0
    started, dead, ruined = False, False, None
    init_to, init_cost = 0.0, 0.0
    L = {c: np.zeros(n) for c in ["nav_open", "tc", "pnl_assets", "pnl_cash", "funding_cost", "nav_close",
                                  "gross_exposure", "net_exposure", "borrowed"]}
    gross, net = np.zeros(n), np.zeros(n)
    to_rec, cost_rec, lev_rec = {}, {}, {}
    for i, d in enumerate(days):
        tc_frac = 0.0
        while k < len(tdates) and tdates[k] < d:
            if not dead:
                w_new = lev[k] * T[k]
                trade = np.abs(w_new - w)
                to, ck = float(trade.sum()), float(kvec @ trade)
                if not started:
                    init_to = to
                    init_cost = ck if charge_initial else 0.0
                    tc_frac += init_cost
                    to_rec[tdates[k]], cost_rec[tdates[k]] = 0.0, init_cost
                    started = True
                else:
                    tc_frac += ck
                    to_rec[tdates[k]], cost_rec[tdates[k]] = to, ck
                lev_rec[tdates[k]] = lev[k]
                w = w_new
            k += 1
        r = R[i]
        nav_open = nav
        tc = nav_open * tc_frac
        nav_inv = nav_open - tc
        cw = 1.0 - w.sum()
        pa = nav_inv * float(w @ r)
        pc = nav_inv * max(cw, 0.0) * cr[i]
        fc = nav_inv * max(-cw, 0.0) * (cr[i] + sp[i])
        nav = nav_inv + pa + pc - fc
        if nav <= 0.0 and not dead:
            nav, dead, ruined = 0.0, True, d
        g = (nav / nav_inv - 1.0) if nav_inv > 0 else 0.0
        gross[i] = g
        net[i] = (nav / nav_open - 1.0) if nav_open > 0 else 0.0
        for c, v in zip(L, [nav_open, tc, pa, pc, fc, nav, float(np.abs(w).sum()), float(w.sum()), max(-cw, 0.0)]):
            L[c][i] = v
        if dead:
            w = np.zeros_like(w)
        else:
            w = w * (1 + r) / (1 + g)
    ledger = pd.DataFrame(L, index=days)
    return BacktestResult(pd.Series(gross, days), pd.Series(net, days), pd.Series(to_rec, dtype=float),
                          pd.Series(cost_rec, dtype=float), pd.Series(lev_rec, dtype=float),
                          init_to, init_cost, ledger, ruined, funding_spread_bps)


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
               cost: pd.Series | None = None, periods_per_year: float | None = None) -> dict[str, float]:
    """Annualised performance and risk statistics from daily net returns.

    ``periods_per_year`` defaults to the observed frequency len(r) / years. The daily calendar
    here is the union of US, euro-area and Japanese trading days (all weekdays, ≈ 260 per year),
    so the former fixed 252 understated volatility and Sharpe by a factor √(252/260) ≈ 0.984.
    Drawdowns are measured from the initial NAV of 1. Turnover is two-way, Σ|Δw| per year,
    excluding the initial build.
    """
    r = net.dropna()
    cr = cash.reindex(r.index).fillna(0.0)
    years = (r.index[-1] - r.index[0]).days / 365.25
    ppy = periods_per_year if periods_per_year is not None else len(r) / years
    wealth = (1 + r).cumprod()
    ann = wealth.iloc[-1] ** (1 / years) - 1
    ex = r - cr
    vol = r.std() * math.sqrt(ppy)
    down = ex[ex < 0]
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    mdd = float((wealth.to_numpy() / peak - 1).min())
    q = np.quantile(r, 0.05)
    out = {"ann. return": ann, "ann. vol": vol,
           "Sharpe": ex.mean() * ppy / vol,
           "Sortino": ex.mean() * ppy / (math.sqrt((down ** 2).sum() / len(ex)) * math.sqrt(ppy)),
           "max drawdown": mdd, "Calmar": ann / abs(mdd) if mdd < 0 else np.nan,
           "VaR95 1d": -q, "ES95 1d": -r[r <= q].mean(), "periods/yr": ppy}
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
