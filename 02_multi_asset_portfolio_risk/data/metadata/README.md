# Data

Raw files are **not committed** (Yahoo / ICE / Moody's licence terms). Rebuild them with:

```bash
bash scripts/run_all.sh          # Yahoo ETFs, ECB curves + EONIA/€STR, Japan MoF JGB curve
```

plus the manual downloads below, then validate with `python scripts/90_validate_raw.py`.

| Source | How | Series |
|---|---|---|
| FRED | 5 CSV links (`fredgraph.csv?id=...`) | UST curve, DFF, DTB3, DEXUSEU, DEXJPUS, DAAA, DBAA, ICE OAS, VIXCLS |
| Yahoo Finance | `scripts/01_etf.py` | 10 ETFs (OHLC, Close, AdjClose) |
| ECB Data Portal | `scripts/03_ecb.py` | YC (AAA + all, spot + par), €STR, EONIA |
| Japan MoF | `scripts/04_mof.py` | JGB constant-maturity curve 1Y–40Y |
| Bank of Japan | manual (stat-search, `FM01'STRDCLUCON`, daily CSV) | JPY overnight call rate |

## Files

| File | Content |
|---|---|
| `metadata/sources.csv` | one row per series: source URL, download date, start/end, n, missing % |
| `metadata/series_dictionary.csv` | what each series is, its role and FX treatment |
| `metadata/stress_periods.csv` | fixed stress windows used in notebook 07 |
| `metadata/validation_report.md` | output of `scripts/90_validate_raw.py` |

## Data roles

* **Investable** — what the optimiser allocates to: 10 ETFs + synthetic EUR and JGB 7–10Y bonds.
* **Risk factor / conversion** — curves, FX, short rates, credit spreads, VIX: used to build returns,
  hedge carry, risk decomposition and stress scenarios, never allocated to directly.

## Known data issues (from validation)

| Issue | Rule |
|---|---|
| HYG starts 2007-04-11 | common sample start; first out-of-sample month 2010-04 (3y window); GFC is in-sample stress only |
| US / EUR / JP calendars differ (only 4,543 common days since 2007-05) | estimate covariances on weekly data; never inner-join daily data globally |
| ECB 30Y fitted yields jump (e.g. 2010-10-15) | do not use 30Y for synthetic bonds; treat with care in curve PCA / ALM |
| JGB short tenors jump before 1986 | `load_jgb_curve()` starts at 1986-07 |
| ICE OAS has ~3y history | reference only; long-history credit indicator is Moody's Baa−Aaa |
| EONIA → €STR | spliced at 2019-10-01; overlap difference is exactly 8.5bp |
| FRED FX is a noon NY fixing, ETFs close at 16:00 NY | negligible at weekly/monthly frequency |
