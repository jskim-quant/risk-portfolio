"""Assemble the v1 investable universe as EUR returns (FX policy applied).

FX policy (EUR base): equity, gold and commodities unhedged; government bonds and
credit hedged monthly/weekly/daily with a one-period CIP forward (see ``returns``).
"""
from __future__ import annotations

import pandas as pd

from . import data_loader as dl
from . import returns as R

UNHEDGED_USD: list[str] = ["SPY", "VGK", "EWJ", "GLD", "DBC"]
HEDGED_USD: list[str] = ["SHY", "IEF", "TLT", "LQD", "HYG"]
ASSETS: list[str] = ["SPY", "VGK", "EWJ", "SHY", "IEF", "TLT", "EUR_CORE7_10_SYN",
                     "JGB7_10_SYN", "LQD", "HYG", "GLD", "DBC"]
SLEEVE: dict[str, str] = {
    "SPY": "equity", "VGK": "equity", "EWJ": "equity",
    "SHY": "govt", "IEF": "govt", "TLT": "govt", "EUR_CORE7_10_SYN": "govt", "JGB7_10_SYN": "govt",
    "LQD": "credit", "HYG": "credit", "GLD": "real", "DBC": "real",
}
RULES: dict[str, str] = {"daily": "D", "weekly": "W-WED", "monthly": "ME"}


def synthetic_indices() -> dict[str, pd.Series]:
    """Local-currency total-return indices of the synthetic 7–10Y bonds (UST, EUR AAA, EUR ALL, JGB)."""
    ust, jgb = dl.load_ust_curve(), dl.load_jgb_curve()
    aaa, allc = dl.load_ecb_curve("AAA", "PY"), dl.load_ecb_curve("ALL", "PY")

    def blend(curve: pd.DataFrame, freq: int, name: str) -> pd.Series:
        parts = [R.synthetic_bond_returns(curve[c], m, freq) for c, m in (("7Y", 7), ("10Y", 10))]
        return R.index_from_returns(R.blend_returns(parts, name))

    return {
        "UST7_10_SYN": blend(ust, 2, "UST7_10_SYN"),
        "EUR_CORE7_10_SYN": blend(aaa, 1, "EUR_CORE7_10_SYN"),
        "EUR_ALL7_10_SYN": blend(allc, 1, "EUR_ALL7_10_SYN"),
        "JGB7_10_SYN": blend(jgb, 2, "JGB7_10_SYN"),
    }


def eur_returns(freq: str = "monthly", hedge_bonds: bool = True,
                extra: tuple[str, ...] = ("EUR_ALL7_10_SYN",)) -> pd.DataFrame:
    """EUR period returns for the v1 universe.

    Args:
        freq: ``daily``, ``weekly`` (Wednesday) or ``monthly`` (month-end).
        hedge_bonds: apply the FX hedge to bonds and credit (policy); ``False`` gives unhedged.
        extra: additional robustness series to append (not part of the investable 12).
    """
    rule = RULES[freq]
    px, fx, sr = dl.load_etf_prices(), dl.load_fx(), dl.load_short_rates()
    syn = synthetic_indices()
    out: dict[str, pd.Series] = {}

    def unhedged(idx: pd.Series, fx_s: pd.Series) -> pd.Series:
        return R.period_levels(R.to_eur_unhedged(idx, fx_s), rule).pct_change().iloc[1:]

    for t in UNHEDGED_USD:
        out[t] = unhedged(px[t].dropna(), fx["EURUSD"])
    for t in HEDGED_USD:
        idx = px[t].dropna()
        out[t] = (R.eur_hedged_returns(idx, fx["EURUSD"], sr["EUR"], sr["USD"], rule)
                  if hedge_bonds else unhedged(idx, fx["EURUSD"]))
    jgb = syn["JGB7_10_SYN"]
    out["JGB7_10_SYN"] = (R.eur_hedged_returns(jgb, fx["EURJPY"], sr["EUR"], sr["JPY"], rule, dc_fc=365.0)
                          if hedge_bonds else unhedged(jgb, fx["EURJPY"]))
    for k in ("EUR_CORE7_10_SYN", *extra):
        out[k] = R.period_levels(syn[k], rule).pct_change().iloc[1:]
    cols = ASSETS + [e for e in extra if e not in ASSETS]
    df = pd.DataFrame(out)[cols]
    if rule != "D":  # drop the last period if it is not complete for every input
        last_obs = min([px[t].last_valid_index() for t in UNHEDGED_USD + HEDGED_USD]
                       + [v.last_valid_index() for v in syn.values()]
                       + [fx["EURUSD"].last_valid_index(), fx["EURJPY"].last_valid_index()])
        df = df.loc[df.index <= last_obs]
    return df
