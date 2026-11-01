"""NAV accounting identities of the backtest engine on analytically solvable cases."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import backtest as B  # noqa: E402

IDX = pd.bdate_range("2021-01-04", periods=40)
A = ["a", "b"]


def _tg(rows, dates=None) -> pd.DataFrame:
    dates = dates if dates is not None else [IDX[0]]
    return pd.DataFrame(rows, index=dates, columns=A)


def _cash(x: float = 0.0) -> pd.Series:
    return pd.Series(x, index=IDX)


def _check_identity(res: B.BacktestResult) -> None:
    """NAV_close = NAV_open − TC + PnL_assets + PnL_cash − Funding, and NAV chains day to day."""
    L = res.ledger
    rhs = L.nav_open - L.tc + L.pnl_assets + L.pnl_cash - L.funding_cost
    alive = L.nav_close > 0
    assert np.allclose(L.nav_close[alive], rhs[alive], atol=1e-13)
    assert np.allclose(L.nav_open.iloc[1:].to_numpy(), L.nav_close.iloc[:-1].to_numpy(), atol=1e-13)
    assert np.allclose((1 + res.net).cumprod().to_numpy(), L.nav_close.to_numpy(), atol=1e-12)
    assert np.allclose(L.nav_close.to_numpy(), (L.nav_open - L.tc).to_numpy() * (1 + res.gross.to_numpy()), atol=1e-12)


def test_unlevered_two_assets_closed_form() -> None:
    """Buy-and-hold 60/40 with constant returns: NAV_T = 0.6·1.01^T + 0.4·0.99^T."""
    daily = pd.DataFrame({"a": 0.01, "b": -0.01}, index=IDX)
    res = B.run_backtest(daily, _tg([[0.6, 0.4]]), _cash(), 0, end=IDX[-1])
    T = len(res.net)
    assert abs(res.ledger.nav_close.iloc[-1] - (0.6 * 1.01 ** T + 0.4 * 0.99 ** T)) < 1e-12
    _check_identity(res)


def test_levered_pays_rate_plus_spread_on_borrowing_only() -> None:
    """2× leverage: borrowed 1·NAV pays O/N + spread; one day, Δ = 1 calendar day."""
    daily = pd.DataFrame({"a": 0.01, "b": 0.0}, index=IDX)
    rc, s = 1e-4, 100.0
    res = B.run_backtest(daily, _tg([[1.0, 0.0]]), _cash(rc), 0, leverage=pd.Series([2.0], index=[IDX[0]]),
                         end=IDX[-1], funding_spread_bps=s)
    first = res.ledger.iloc[0]
    assert abs(first.borrowed - 1.0) < 1e-15 and abs(first.gross_exposure - 2.0) < 1e-15
    assert abs(res.gross.iloc[0] - (2 * 0.01 - (rc + s / 1e4 / 360))) < 1e-15
    _check_identity(res)


def test_positive_cash_earns_rate_without_spread() -> None:
    """50 % cash earns the O/N rate; the funding spread must not touch a positive balance."""
    daily = pd.DataFrame({"a": 0.0, "b": 0.0}, index=IDX)
    rc = 2e-4
    a = B.run_backtest(daily, _tg([[0.5, 0.0]]), _cash(rc), 0, end=IDX[-1], funding_spread_bps=0)
    b = B.run_backtest(daily, _tg([[0.5, 0.0]]), _cash(rc), 0, end=IDX[-1], funding_spread_bps=500)
    assert np.allclose(a.net, b.net, atol=0) and abs(a.gross.iloc[0] - 0.5 * rc) < 1e-16
    assert (b.ledger.funding_cost == 0).all()
    _check_identity(b)


def test_spread_accrues_over_weekend_gap() -> None:
    """Funding spread accrues ACT/360 over calendar days (Friday → Monday = 3 days)."""
    daily = pd.DataFrame({"a": 0.0, "b": 0.0}, index=IDX)
    res = B.run_backtest(daily, _tg([[1.5, 0.0]]), _cash(0.0), 0, end=IDX[-1], funding_spread_bps=360.0)
    L = res.ledger
    fc = L.funding_cost / ((L.nav_open - L.tc) * L.borrowed)          # rate per unit borrowed
    gaps = np.asarray((res.net.index - res.net.index.insert(0, IDX[0])[:-1]) / pd.Timedelta(days=1))
    assert np.allclose(fc.to_numpy(), 0.036 * gaps / 360, atol=1e-16)


def test_spread_zero_reproduces_original_engine() -> None:
    """With s = 0 the levered return is w·r + (1 − Σw)·r_cash, as in the original engine."""
    rng = np.random.default_rng(0)
    daily = pd.DataFrame(rng.normal(0, 0.02, (len(IDX), 2)), index=IDX, columns=A)
    tg = _tg([[0.7, 0.3], [0.2, 0.8]], [IDX[0], IDX[20]])
    lev = pd.Series([1.8, 2.5], index=tg.index)
    res = B.run_backtest(daily, tg, _cash(5e-5), 0, leverage=lev, end=IDX[-1])
    w, k, g = np.zeros(2), 0, []
    for i, d in enumerate(res.net.index):
        while k < 2 and tg.index[k] < d:
            w = lev.iloc[k] * tg.iloc[k].to_numpy(); k += 1
        r = daily.loc[d].to_numpy()
        gi = w @ r + (1 - w.sum()) * 5e-5
        g.append(gi); w = w * (1 + r) / (1 + gi)
    assert np.allclose(res.gross.to_numpy(), g, atol=1e-15)
    _check_identity(res)


def test_cost_deducted_before_returns() -> None:
    """net = (1 − k·TO)(1 + gross) − 1 on a rebalance day; initial build charged only on request."""
    daily = pd.DataFrame({"a": 0.02, "b": -0.01}, index=IDX)
    tg = _tg([[0.5, 0.5], [0.5, 0.5]], [IDX[0], IDX[10]])
    free = B.run_backtest(daily, tg, _cash(), 20, end=IDX[-1])
    paid = B.run_backtest(daily, tg, _cash(), 20, end=IDX[-1], charge_initial=True)
    d = free.net.index.get_loc(IDX[11])
    to = free.turnover.iloc[-1]
    assert abs(free.net.iloc[d] - ((1 - 0.002 * to) * (1 + free.gross.iloc[d]) - 1)) < 1e-15
    assert free.initial_turnover == 1.0 and free.initial_cost == 0.0 and abs(paid.initial_cost - 0.002) < 1e-16
    assert abs(paid.net.iloc[0] - ((1 - 0.002) * (1 + paid.gross.iloc[0]) - 1)) < 1e-15
    assert free.turnover.iloc[0] == 0.0 and paid.turnover.iloc[0] == 0.0   # same convention
    _check_identity(free); _check_identity(paid)


def test_asset_specific_costs() -> None:
    """Per-asset costs: TC = Σ k_i |Δw_i|; a constant mapping equals the scalar case."""
    daily = pd.DataFrame({"a": 0.01, "b": 0.0}, index=IDX)
    tg = _tg([[0.5, 0.5], [0.5, 0.5]], [IDX[0], IDX[10]])
    vec = B.run_backtest(daily, tg, _cash(), {"a": 5, "b": 50}, end=IDX[-1])
    sca = B.run_backtest(daily, tg, _cash(), 10, end=IDX[-1])
    same = B.run_backtest(daily, tg, _cash(), {"a": 10, "b": 10}, end=IDX[-1])
    half = vec.turnover.iloc[-1] / 2                       # |Δw_a| = |Δw_b| for two assets
    assert abs(vec.cost.iloc[-1] - half * (5e-4 + 5e-3)) < 1e-16
    assert np.allclose(sca.net, same.net, atol=0)


def test_large_drawdown_and_extreme_returns() -> None:
    """−20 % (−40 % levered) then +100 % on a 2× levered position: identities hold, weights drift exactly."""
    r = np.zeros(len(IDX)); r[1], r[2] = -0.20, 1.0          # IDX[0] is the decision date
    daily = pd.DataFrame({"a": r, "b": 0.0}, index=IDX)
    res = B.run_backtest(daily, _tg([[1.0, 0.0]]), _cash(), 0, leverage=pd.Series([2.0], index=[IDX[0]]), end=IDX[-1])
    nav = res.ledger.nav_close
    assert abs(nav.iloc[0] - 0.6) < 1e-15                                      # 2×(−20 %), first day = IDX[1]
    w_after = 2 * 0.8 / 0.6                                                      # drifted exposure
    assert abs(nav.iloc[1] - 0.6 * (1 + w_after * 1.0)) < 1e-12
    assert res.ruined is None
    _check_identity(res)


def test_ruin_floors_nav_at_zero() -> None:
    """3× leverage and a −40 % day wipes out equity: NAV = 0 thereafter, no NaN/inf."""
    r = np.zeros(len(IDX)); r[3] = -0.40; r[10] = 0.5
    daily = pd.DataFrame({"a": r, "b": 0.0}, index=IDX)
    res = B.run_backtest(daily, _tg([[1.0, 0.0]]), _cash(1e-4), 0, leverage=pd.Series([3.0], index=[IDX[0]]), end=IDX[-1])
    assert res.ruined == IDX[3]
    assert np.isfinite(res.net).all() and np.isfinite(res.ledger.to_numpy()).all()
    assert (res.ledger.nav_close.loc[IDX[3]:] == 0).all() and res.net.min() >= -1.0
    _check_identity(res)


def test_zero_returns_only_costs_move_nav() -> None:
    """Zero asset and cash returns: NAV changes only by transaction costs."""
    daily = pd.DataFrame({"a": 0.0, "b": 0.0}, index=IDX)
    tg = _tg([[1.0, 0.0], [0.0, 1.0]], [IDX[0], IDX[5]])
    res = B.run_backtest(daily, tg, _cash(), 25, end=IDX[-1], charge_initial=True)
    assert abs(res.ledger.nav_close.iloc[-1] - (1 - 0.0025) * (1 - 0.0025 * 2)) < 1e-15
    _check_identity(res)


def test_missing_trading_days_are_flat_not_dropped() -> None:
    """NaN (market closed) earns 0 that day; the day stays in the sample and drift is unchanged."""
    daily = pd.DataFrame({"a": 0.01, "b": 0.0}, index=IDX)
    daily.iloc[5, 0] = np.nan
    res = B.run_backtest(daily, _tg([[1.0, 0.0]]), _cash(), 0, end=IDX[-1])
    assert len(res.net) == len(IDX) - 1 and res.net.loc[IDX[5]] == 0.0
    assert abs(res.ledger.nav_close.iloc[-1] - 1.01 ** (len(IDX) - 2)) < 1e-12


def test_perf_stats_infers_frequency_and_initial_drawdown() -> None:
    """Annualisation uses the observed frequency; a loss on day 1 counts as drawdown from NAV 1."""
    idx = pd.date_range("2020-01-01", periods=731, freq="D")      # 2 calendar years, 365.25 / yr
    r = pd.Series(0.0, index=idx); r.iloc[0] = -0.1
    s = B.perf_stats(r, pd.Series(0.0, index=idx))
    assert abs(s["periods/yr"] - 731 / (730 / 365.25)) < 1e-9
    assert abs(s["max drawdown"] + 0.1) < 1e-15


def test_regression_against_stored_backtest() -> None:
    """Refactored engine (s = 0) reproduces notebook 05's stored net returns, incl. levered RP*."""
    proc = ROOT / "data" / "processed"
    need = [proc / "backtest_daily_net.csv", proc / "cov_forecasts.npz", proc / "weights" / "weights_RiskParity.csv",
            ROOT / "data" / "raw" / "short_rates" / "ESTR.csv"]
    if not all(f.exists() for f in need):
        return  # processed data is rebuilt by notebooks 02–05
    from src import data_loader as dl, risk as K
    fc = K.load_forecasts(proc / "cov_forecasts.npz")
    daily = pd.read_csv(proc / "asset_returns_eur_daily.csv", index_col="date", parse_dates=True)[fc["assets"]]
    stored = pd.read_csv(proc / "backtest_daily_net.csv", index_col=0, parse_dates=True)
    cash = B.daily_cash_returns(daily.index, dl.load_short_rates()["EUR"])
    w = {m: pd.read_csv(proc / "weights" / f"weights_{m}.csv", index_col=0, parse_dates=True)[fc["assets"]]
         for m in ("6040", "RiskParity")}
    assert (B.run_backtest(daily, w["6040"], cash).net - stored["60/40"]).abs().max() < 1e-12
    lev = B.vol_target_leverage(w["RiskParity"], fc["cov1w"]["GARCH-CCC"], w["6040"])
    rp = B.run_backtest(daily, w["RiskParity"], cash, leverage=lev)
    assert (rp.net - stored["RiskParity*"]).abs().max() < 1e-12
    _check_identity(rp)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
