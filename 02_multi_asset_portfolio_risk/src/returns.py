"""Return construction: synthetic constant-maturity bonds and EUR conversion.

Conventions
-----------
* Yields and short rates are inputs in **percent**; everything returned is in decimals.
* A synthetic constant-maturity bond is re-bought at par every observation date:
  on day ``t-1`` it is a par bond with coupon ``y_{t-1}`` and maturity ``M``; on day ``t``
  it is repriced at the new par yield ``y_t`` with remaining maturity ``M - dt`` and the
  accrued coupon ``y_{t-1} * dt`` is added (Swinkels 2019 style exact repricing).
* FX quotes are **foreign currency per 1 EUR** (EURUSD = USD per EUR, EURJPY = JPY per EUR),
  so an asset index in foreign currency converts to EUR as ``index / fx``.
* FX hedge: at the start of each period the foreign value is sold forward at the
  covered-interest-parity forward ``F = S0 (1 + r_f dt_f) / (1 + r_EUR dt_EUR)``; the
  gain/loss on the asset during the period stays unhedged (converted at spot ``S1``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

YEAR: float = 365.25


def par_bond_price(coupon: np.ndarray | float, yld: np.ndarray | float,
                   maturity: np.ndarray | float, freq: int = 2) -> np.ndarray:
    """Price per 1 nominal of a fixed-coupon bond, ``coupon`` and ``yld`` in decimals.

    Uses the annuity formula with ``n = freq * maturity`` (fractional n allowed) and a
    zero-yield limit so negative and near-zero yields (JGB, Bund) are handled.
    """
    c, y, m = (np.asarray(x, dtype=float) for x in (coupon, yld, maturity))
    n = freq * m
    per = y / freq
    disc = (1.0 + per) ** (-n)
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(np.abs(per) > 1e-10, (1.0 - disc) / per, n)
    return (c / freq) * annuity + disc


def par_bond_duration_convexity(yld: np.ndarray | float, maturity: float,
                                freq: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Modified duration and convexity of a par bond by central finite differences."""
    y = np.asarray(yld, dtype=float)
    h = 1e-4
    p0 = par_bond_price(y, y, maturity, freq)
    up = par_bond_price(y, y + h, maturity, freq)
    dn = par_bond_price(y, y - h, maturity, freq)
    dur = -(up - dn) / (2 * h * p0)
    conv = (up + dn - 2 * p0) / (h ** 2 * p0)
    return dur, conv


def synthetic_bond_returns(yields_pct: pd.Series, maturity: float, freq: int = 2,
                           method: str = "exact") -> pd.Series:
    """Daily total returns of a constant-maturity par bond built from a par-yield series.

    Args:
        yields_pct: par yield of maturity ``maturity`` in percent (own trading calendar).
        maturity: constant maturity in years.
        freq: coupon frequency (2 = UST/JGB semi-annual, 1 = euro govies annual).
        method: ``exact`` (repricing) or ``approx`` (carry − D·Δy + ½·C·Δy²).
    """
    y = (yields_pct.dropna() / 100.0).astype(float)
    dt = y.index.to_series().diff().dt.days.to_numpy() / YEAR
    y_prev, y_now = y.shift(1).to_numpy(), y.to_numpy()
    if method == "exact":
        price = par_bond_price(y_prev, y_now, maturity - dt, freq)
        ret = price - 1.0 + y_prev * dt
    elif method == "approx":
        dur, conv = par_bond_duration_convexity(y_prev, maturity, freq)
        dy = y_now - y_prev
        ret = y_prev * dt - dur * dy + 0.5 * conv * dy ** 2
    else:
        raise ValueError(f"unknown method {method!r}")
    return pd.Series(ret, index=y.index, name=f"SYN_{maturity:g}Y").iloc[1:]


def blend_returns(parts: list[pd.Series], name: str) -> pd.Series:
    """Equal-weight daily rebalanced blend of return series on their common dates."""
    df = pd.concat(parts, axis=1, join="inner")
    return df.mean(axis=1).rename(name)


def index_from_returns(returns: pd.Series, base: float = 100.0) -> pd.Series:
    """Total-return index starting at ``base`` on the day before the first return."""
    r = returns.dropna()
    idx = base * (1.0 + r).cumprod()
    start = r.index[0] - pd.Timedelta(days=1)
    return pd.concat([pd.Series([base], index=[start]), idx]).rename(returns.name)


def period_levels(level: pd.Series, rule: str) -> pd.Series:
    """Last available observation of a level series per period (``W-WED``, ``ME`` or ``D``)."""
    s = level.dropna()
    return s if rule == "D" else s.resample(rule).last().dropna()


def to_eur_unhedged(index_fc: pd.Series, fx_fc_per_eur: pd.Series) -> pd.Series:
    """EUR value of a foreign-currency index; FX forward-filled at most 5 days to the asset's dates."""
    fx = fx_fc_per_eur.reindex(fx_fc_per_eur.index.union(index_fc.index)).ffill(limit=5)
    return (index_fc / fx.reindex(index_fc.index)).dropna().rename(index_fc.name)


def eur_hedged_returns(index_fc: pd.Series, fx_fc_per_eur: pd.Series, r_eur_pct: pd.Series,
                       r_fc_pct: pd.Series, rule: str, dc_eur: float = 360.0,
                       dc_fc: float = 360.0) -> pd.Series:
    """Period returns in EUR of a foreign asset hedged with a one-period CIP forward.

    ``R_h = (1 + r_EUR·τ_EUR) / (1 + r_f·τ_f) − 1 + R_local · S0 / S1`` with S in foreign
    currency per EUR, rates fixed at the period start and money-market day counts.
    """
    lvl = period_levels(index_fc, rule)
    dates = lvl.index

    def at(s: pd.Series) -> pd.Series:
        full = s.reindex(s.index.union(dates)).ffill(limit=5)
        return full.reindex(dates)

    S, re_, rf = at(fx_fc_per_eur), at(r_eur_pct) / 100.0, at(r_fc_pct) / 100.0
    days = dates.to_series().diff().dt.days
    r_loc = lvl.pct_change()
    carry = (1 + re_.shift(1) * days / dc_eur) / (1 + rf.shift(1) * days / dc_fc) - 1
    out = carry + r_loc * S.shift(1) / S
    return out.iloc[1:].rename(index_fc.name)


def tracking_stats(a: pd.Series, b: pd.Series, periods_per_year: int) -> dict[str, float]:
    """Compare two return series on common dates: corr, beta, annualised TE and mean gap."""
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    diff = x - y
    return {
        "n": float(len(df)),
        "corr": float(x.corr(y)),
        "beta (a on b)": float(np.cov(x, y)[0, 1] / y.var()),
        "TE p.a.": float(diff.std() * np.sqrt(periods_per_year)),
        "mean gap p.a.": float(diff.mean() * periods_per_year),
        "vol a p.a.": float(x.std() * np.sqrt(periods_per_year)),
        "vol b p.a.": float(y.std() * np.sqrt(periods_per_year)),
    }
