"""Validate data/processed/asset_returns_eur_*.csv (schema, alignment, missing data, consistency).

Usage: python scripts/91_validate_processed.py   → exit code 1 on any failure.
Checks are defined in ``src.universe.check_processed`` (docs/design_02_return_construction.md §6).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import universe as U  # noqa: E402

PROC = ROOT / "data" / "processed"


def load(name: str) -> pd.DataFrame:
    """Read one processed file with a parsed date index."""
    return pd.read_csv(PROC / f"asset_returns_eur_{name}.csv", index_col="date", parse_dates=True)


def main() -> int:
    """Run every check and print a one-line verdict per file."""
    failures = 0
    daily = load("daily")
    for freq in ["daily", "weekly", "monthly"]:
        hedged, unhedged = load(freq), load(f"{freq}_unhedged")
        for name, df, kw in [(freq, hedged, {"unhedged": unhedged, "daily": daily}),
                             (f"{freq}_unhedged", unhedged, {})]:
            fails = U.check_processed(df, freq, **kw)
            core = df.loc[U.COMMON_START[freq]:]
            status = "PASS" if not fails else "FAIL"
            print(f"{status}  {name:18s} {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}  rows {len(df):5d}  "
                  f"common-sample NaN {int(core.isna().sum().sum())}")
            for f in fails:
                print(f"      - {f}")
            failures += len(fails)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
