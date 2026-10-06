"""Load raw market data for the multi-asset portfolio project.

All loaders read from ``data/raw`` (populated by ``scripts/run_all.sh`` and the
manual downloads documented in ``data/metadata/README.md``) and return pandas
objects indexed by a ``DatetimeIndex`` named ``date``. Nothing here downloads
or transforms data beyond renaming columns and splicing documented series.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT: Path = Path(__file__).resolve().parents[1]
RAW: Path = ROOT / "data" / "raw"

ETFS: list[str] = ["SPY", "VGK", "EWJ", "SHY", "IEF", "TLT", "LQD", "HYG", "GLD", "DBC"]
ECB_CURVES: dict[str, str] = {"AAA": "G_N_A", "ALL": "G_N_C"}
EONIA_ESTR_SWITCH: str = "2019-10-01"


def _read(rel: str) -> pd.DataFrame:
    """Read a raw CSV with the first column parsed as the date index."""
    df = pd.read_csv(RAW / rel, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df.sort_index()


def load_etf_prices(tickers: list[str] | None = None, field: str = "AdjClose") -> pd.DataFrame:
    """Return one price column per ETF (default: split- and dividend-adjusted close).

    Args:
        tickers: ETFs to load; defaults to the full v1 universe.
        field: ``AdjClose`` (total-return proxy) or ``Close`` (price only).
    """
    tickers = tickers or ETFS
    cols = {t: _read(f"market/etf/{t}.csv")[f"{t}_{field}"] for t in tickers}
    return pd.DataFrame(cols)


def load_ust_curve() -> pd.DataFrame:
    """US Treasury constant-maturity par yields in percent, columns renamed to tenors."""
    df = _read("rates/ust/ust_curve.csv")
    return df.rename(columns=lambda c: c.replace("DGS", "").replace("MO", "M") + ("" if "MO" in c else "Y"))


def load_ecb_curve(curve: str = "AAA", kind: str = "PY") -> pd.DataFrame:
    """ECB euro-area government yield curve in percent.

    Args:
        curve: ``AAA`` (AAA-rated issuers) or ``ALL`` (all euro-area central governments).
        kind: ``PY`` (par yield) or ``SR`` (zero-coupon spot rate).
    """
    df = _read(f"rates/ecb/{ECB_CURVES[curve]}_{kind}.csv")
    df = df.rename(columns=lambda c: c.split(f"{kind}_")[-1])
    order = ["3M", "1Y", "2Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
    return df[[m for m in order if m in df.columns]]


def load_jgb_curve(start: str = "1986-07-01") -> pd.DataFrame:
    """Japanese government bond constant-maturity yields in percent.

    Data before mid-1986 contains implausible jumps in short tenors and is dropped by default.
    """
    df = _read("rates/jgb/jgb_curve.csv").rename(columns=lambda c: c.replace("JGB_", ""))
    return df.loc[start:]


def load_short_rates() -> pd.DataFrame:
    """Overnight rates in percent: USD (EFFR), EUR (EONIA spliced into €STR), JPY (BOJ call)."""
    usd = _read("short_rates/DFF.csv").iloc[:, 0].rename("USD")
    eonia = _read("short_rates/EONIA.csv").iloc[:, 0]
    estr = _read("short_rates/ESTR.csv").iloc[:, 0]
    eur = pd.concat([eonia.loc[:pd.Timestamp(EONIA_ESTR_SWITCH) - pd.Timedelta(days=1)],
                     estr.loc[EONIA_ESTR_SWITCH:]]).rename("EUR")
    jpy = _read("short_rates/JPY_CALL_ON.csv").iloc[:, 0].rename("JPY")
    return pd.concat([usd, eur, jpy], axis=1)


def load_fx() -> pd.DataFrame:
    """FX rates: EURUSD (USD per EUR), USDJPY (JPY per USD) and derived EURJPY (JPY per EUR)."""
    df = _read("fx/fx_usd.csv").rename(columns={"DEXUSEU": "EURUSD", "DEXJPUS": "USDJPY"})
    df["EURJPY"] = df["EURUSD"] * df["USDJPY"]
    return df


def load_credit() -> pd.DataFrame:
    """Moody's Aaa/Baa yields and Baa−Aaa spread in percent (long history credit indicator)."""
    df = _read("credit/moodys/moodys_aaa_baa.csv").rename(columns={"DAAA": "AAA", "DBAA": "BAA"})
    df["BAA_AAA"] = df["BAA"] - df["AAA"]
    return df


def load_vix() -> pd.Series:
    """CBOE VIX index level."""
    return _read("volatility/VIXCLS.csv").iloc[:, 0].rename("VIX")


def load_stress_periods() -> pd.DataFrame:
    """Stress-period definitions used by the stress-testing notebook."""
    return pd.read_csv(ROOT / "data" / "metadata" / "stress_periods.csv", parse_dates=["start_date", "end_date"])
