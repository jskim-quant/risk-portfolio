"""Processed-data invariants (run: python -m pytest tests or python tests/test_universe.py).

Tests that need data/processed are skipped when the files have not been built yet (notebook 02).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data_loader as dl  # noqa: E402
from src import returns as R  # noqa: E402
from src import universe as U  # noqa: E402

PROC = ROOT / "data" / "processed"


def _load(name: str) -> pd.DataFrame | None:
    p = PROC / f"asset_returns_eur_{name}.csv"
    return pd.read_csv(p, index_col="date", parse_dates=True) if p.exists() else None


def test_processed_schema() -> None:
    """All six files pass schema, alignment and missing-data checks."""
    for freq in ["daily", "weekly", "monthly"]:
        for name in [freq, f"{freq}_unhedged"]:
            df = _load(name)
            if df is None:
                return
            assert U.check_processed(df, freq) == [], name


def test_monthly_compounds_daily_for_unhedged() -> None:
    """Unhedged columns: monthly return equals the compounded daily return."""
    d, m = _load("daily"), _load("monthly")
    if d is None or m is None:
        return
    comp = (1 + d.fillna(0)).resample("ME").prod() - 1
    for c in ["SPY", "GLD", "EUR_CORE7_10_SYN"]:
        x = pd.concat([comp[c], m[c]], axis=1, join="inner").loc[U.COMMON_START["monthly"]:]
        assert np.allclose(x.iloc[:, 0], x.iloc[:, 1], atol=1e-10), c


def test_hedged_roll_bias_small() -> None:
    """Hedged columns: daily-rolled vs monthly-rolled hedge differ by < 2bp/month on average."""
    d, m = _load("daily"), _load("monthly")
    if d is None or m is None:
        return
    comp = (1 + d.fillna(0)).resample("ME").prod() - 1
    for c in ["IEF", "HYG", "JGB7_10_SYN"]:
        x = pd.concat([comp[c], m[c]], axis=1, join="inner").loc[U.COMMON_START["monthly"]:]
        assert abs((x.iloc[:, 0] - x.iloc[:, 1]).mean()) < 2e-4, c


def test_jgb_negative_yields_handled() -> None:
    """Synthetic JGB returns are finite on negative-yield days (2016–2020)."""
    if not (ROOT / "data" / "raw" / "rates" / "jgb" / "jgb_curve.csv").exists():
        return
    jgb = dl.load_jgb_curve()
    neg = jgb.index[jgb["10Y"] < 0]
    r = R.synthetic_bond_returns(jgb["10Y"], 10, 2)
    assert len(neg) > 100 and np.isfinite(r.reindex(neg).dropna()).all()


def test_ief_gate() -> None:
    """Validation gate: synthetic UST 7–10Y vs IEF (corr ≥ 0.97, |β−1| ≤ 0.10, TE ≤ 1.5% p.a.)."""
    if not (ROOT / "data" / "raw" / "market" / "etf" / "IEF.csv").exists():
        return
    ust = dl.load_ust_curve()
    syn = R.index_from_returns(R.blend_returns(
        [R.synthetic_bond_returns(ust[m], float(m[:-1]), 2) for m in ("7Y", "10Y")], "UST7_10"))
    a = R.period_levels(syn, "ME").pct_change().loc["2002-09":]
    b = R.period_levels(dl.load_etf_prices(["IEF"])["IEF"].dropna(), "ME").pct_change().loc["2002-09":]
    st = R.tracking_stats(a, b, 12)
    assert st["corr"] >= 0.97 and abs(st["beta (a on b)"] - 1) <= 0.10 and st["TE p.a."] <= 0.015


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
