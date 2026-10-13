"""Stress-replay tests (run: python -m pytest tests or python tests/test_stress.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import stress as St  # noqa: E402

IDX = pd.bdate_range("2020-01-01", periods=30)
RNG = np.random.default_rng(0)
DAILY = pd.DataFrame(RNG.normal(0, 0.01, (30, 3)), index=IDX, columns=["a", "b", "c"])


def test_contributions_add_up_exactly() -> None:
    """Asset + cash contributions equal the cumulative replay return (also with leverage)."""
    w = pd.Series([0.5, 0.3, 0.2], index=DAILY.columns)
    cash = pd.Series(1e-4, index=IDX)
    for lev in (1.0, 3.0):
        r = St.replay(w, DAILY, IDX[0], IDX[-1], cash, lev)
        assert abs(r.contributions.sum() - r.cum_return) < 1e-12


def test_window_excludes_start_day() -> None:
    """The return dated at the window start belongs to the previous period."""
    d = DAILY.copy() * 0
    d.iloc[0] = 0.5
    r = St.replay(pd.Series([1.0, 0, 0], index=d.columns), d, IDX[0], IDX[-1])
    assert abs(r.cum_return) < 1e-15 and r.path.index[0] == IDX[0]


def test_leverage_pays_financing() -> None:
    """2× leverage on a flat asset loses exactly one unit of compounded cash return."""
    d = DAILY * 0
    cash = pd.Series(1e-4, index=IDX)
    r = St.replay(pd.Series([1.0, 0, 0], index=d.columns), d, IDX[0], IDX[-1], cash, 2.0)
    assert abs(r.cum_return + ((1 + 1e-4) ** 29 - 1)) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
