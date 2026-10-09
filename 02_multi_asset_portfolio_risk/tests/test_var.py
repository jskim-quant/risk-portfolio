"""Tests for VaR backtests and distribution helpers (run: python -m pytest tests or python tests/test_var.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import var as V  # noqa: E402


def test_normal_quantiles() -> None:
    """Bisection reproduces z_95 = 1.6449 and z_99 = 2.3263 and the normal ES factor."""
    z95, ef95 = V._normal_es_factor(0.95)
    z99, _ = V._normal_es_factor(0.99)
    assert abs(z95 - 1.644854) < 1e-5 and abs(z99 - 2.326348) < 1e-5 and abs(ef95 - 2.062713) < 1e-5


def test_kupiec_exact_rate_and_excess() -> None:
    """Exactly 1% hits gives LR ≈ 0; 3% hits at a 1% VaR is rejected."""
    exact = np.zeros(1000, bool)
    exact[::100] = True
    assert V.kupiec(exact, 0.01)[1] > 0.99
    bad = np.zeros(1000, bool)
    bad[::33] = True
    assert V.kupiec(bad, 0.01)[1] < 0.001


def test_christoffersen_detects_clustering() -> None:
    """Spread-out hits pass independence; the same number of hits in one block fails."""
    spread = np.zeros(2000, bool)
    spread[::100] = True
    block = np.zeros(2000, bool)
    block[1000:1020] = True
    assert V.christoffersen(spread)[1] > 0.05 and V.christoffersen(block)[1] < 0.001


def test_student_t_draws_unit_variance_and_fat_tails() -> None:
    """Standardised t draws have variance ≈ 1 and a fatter 99% quantile than the normal for ν = 5."""
    x = V._std_t_draws(5.0)
    assert abs(x.var() - 1) < 0.05
    assert V._tail(x, 0.99)[0] > 2.326 and abs(V._tail(V._std_t_draws(300.0), 0.99)[0] - 2.326) < 0.03


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
