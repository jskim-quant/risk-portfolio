"""Shared helpers for the raw-data download scripts."""
from __future__ import annotations

from pathlib import Path
from datetime import date
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
META = ROOT / "data/metadata/sources.csv"

def save(df: pd.DataFrame, rel: str, source: str, url: str, unit: str, notes: str = "") -> None:
    """Write ``df`` to data/raw/<rel> and upsert one sources.csv row per column."""
    p = RAW / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p)
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        rows.append(dict(series_id=c, source=source, source_url=url,
            download_date=date.today().isoformat(), frequency="D", unit=unit,
            start_date=s.index.min().date(), end_date=s.index.max().date(),
            n_obs=len(s), missing_pct=round(100 * df[c].isna().mean(), 1),
            file=f"data/raw/{rel}", notes=notes))
    new = pd.DataFrame(rows)
    m = pd.read_csv(META) if META.exists() else pd.DataFrame(columns=new.columns)
    m = pd.concat([m[~m.series_id.isin(new.series_id)], new], ignore_index=True)
    META.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(META, index=False)
    for r in rows:
        print(f"{r['series_id']:40s} {r['start_date']} -> {r['end_date']}  n={r['n_obs']}")

def save_bytes(content: bytes, rel: str) -> None:
    """Write an original downloaded file unchanged to data/raw/<rel>."""
    p = RAW / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
