"""Backtest engine and look-ahead tests (run: python -m pytest tests or python tests/test_backtest.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import backtest as B  # noqa: E402
from src import risk as K  # noqa: E402

IDX = pd.bdate_range("2020-01-01", periods=60)
ASSETS = ["a", "b"]


def test_drift_turnover_and_cost() -> None:
    """After 30 days of +1%/day on asset a, rebalancing to 50/50 trades exactly the drift."""
    daily = pd.DataFrame({"a": 0.01, "b": 0.0}, index=IDX)
    tg = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], index=[IDX[0], IDX[30]], columns=ASSETS)
    res = B.run_backtest(daily, tg, pd.Series(0.0, index=IDX), cost_bps=10, end=IDX[-1])
    wa = 0.5 * 1.01 ** 30 / (0.5 * 1.01 ** 30 + 0.5)
    assert abs(res.turnover.iloc[-1] - 2 * abs(wa - 0.5)) < 1e-12
    assert abs(res.cost.iloc[-1] - 1e-3 * res.turnover.iloc[-1]) < 1e-15


def test_leverage_pays_cash_rate() -> None:
    """2× leverage earns twice the asset return minus one unit of financing."""
    daily = pd.DataFrame({"a": 0.01, "b": 0.0}, index=IDX)
    tg = pd.DataFrame([[0.5, 0.5]], index=[IDX[0]], columns=ASSETS)
    res = B.run_backtest(daily, tg, pd.Series(1e-4, index=IDX), 0, leverage=pd.Series([2.0], index=[IDX[0]]), end=IDX[-1])
    assert abs(res.gross.iloc[0] - (2 * 0.5 * 0.01 - 1e-4)) < 1e-15


def test_weights_take_effect_after_decision_date() -> None:
    """A return on the decision date itself must not be earned by the new weights."""
    daily = pd.DataFrame({"a": [0.5] + [0.0] * 59, "b": 0.0}, index=IDX)
    tg = pd.DataFrame([[1.0, 0.0]], index=[IDX[0]], columns=ASSETS)
    res = B.run_backtest(daily, tg, pd.Series(0.0, index=IDX), 0, end=IDX[-1])
    assert res.gross.index[0] == IDX[1] and abs(res.gross.sum()) < 1e-15


def test_forecasts_do_not_look_ahead() -> None:
    """Covariance forecasts up to 2015-11 are identical whether or not later data exists."""
    p = ROOT / "data" / "processed" / "asset_returns_eur_weekly.csv"
    if not p.exists():
        return  # processed data is rebuilt by notebook 02
    from src import universe as U
    weekly = pd.read_csv(p, index_col=0, parse_dates=True)[U.ASSETS].dropna()
    full = K.walk_forward_forecasts(weekly)
    cut = K.walk_forward_forecasts(weekly.loc[:"2015-12-31"])
    n = len(cut["dates"]) - 1
    for e in K.ESTIMATORS:
        assert np.allclose(full["cov1w"][e][:n], cut["cov1w"][e][:n])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
