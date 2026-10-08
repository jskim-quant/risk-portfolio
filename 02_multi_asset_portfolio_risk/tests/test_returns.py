"""Unit tests for src.returns (run: python -m pytest tests or python tests/test_returns.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import returns as R  # noqa: E402


def test_par_bond_prices_at_par() -> None:
    """A bond whose coupon equals its yield prices at 1, including zero and negative yields."""
    for y, f in [(0.04, 2), (0.0, 2), (-0.002, 1)]:
        assert abs(R.par_bond_price(y, y, 10, f) - 1) < 1e-12


def test_duration_of_10y_par_bond() -> None:
    """Modified duration of a 4% 10Y semi-annual par bond is about 8.18."""
    d, c = R.par_bond_duration_convexity(0.04, 10)
    assert 8.1 < float(d) < 8.25 and c > 0


def test_flat_curve_earns_the_yield() -> None:
    """With an unchanged 3% curve the synthetic bond earns ~3% a year (daily accrual)."""
    idx = pd.bdate_range("2020-01-01", periods=520)
    r = R.synthetic_bond_returns(pd.Series(3.0, index=idx), 10)
    ann = (1 + r).prod() ** (365.25 / (idx[-1] - idx[0]).days) - 1
    assert 0.029 < ann < 0.031


def test_rate_rise_loses_about_duration() -> None:
    """A one-day +50bp jump costs roughly duration × 0.5%."""
    idx = pd.bdate_range("2020-01-01", periods=2)
    r = R.synthetic_bond_returns(pd.Series([4.0, 4.5], index=idx), 10)
    assert -0.042 < r.iloc[0] < -0.036


def test_hedged_return_without_fx_or_rate_moves() -> None:
    """With equal rates and constant FX the hedged return equals the local return."""
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    lvl = pd.Series([100, 101, 99, 102.0], index=idx)
    out = R.eur_hedged_returns(lvl, pd.Series(1.2, index=idx), pd.Series(2.0, index=idx),
                               pd.Series(2.0, index=idx), "ME")
    assert np.allclose(out.to_numpy(), lvl.pct_change().iloc[1:].to_numpy())


def test_hedge_removes_fx_on_principal() -> None:
    """If the asset is flat and the foreign currency drops 10%, the hedged loss is only carry."""
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    lvl = pd.Series([100.0, 100.0], index=idx)
    fx = pd.Series([1.0, 1.1], index=idx)  # foreign per EUR rises → foreign currency weaker
    out = R.eur_hedged_returns(lvl, fx, pd.Series(0.0, index=idx), pd.Series(0.0, index=idx), "ME")
    unh = R.to_eur_unhedged(lvl, fx).pct_change().iloc[-1]
    assert abs(out.iloc[0]) < 1e-12 and unh < -0.09


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
