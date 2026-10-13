# Processed data schema — `data/processed/asset_returns_eur_{freq}[_unhedged].csv`

Built by notebook 02 (`src.universe.eur_returns`), validated by `scripts/91_validate_processed.py`
(`src.universe.check_processed`). Files are git-ignored and rebuilt from raw data.

## Layout

| | daily | weekly | monthly |
|---|---|---|---|
| index column | `date` (ISO 8601) | `date` | `date` |
| index rule | union of all input trading dates | `W-WED`: last observation up to each Wednesday | `ME`: last observation up to each calendar month-end, labelled with the month-end date |
| first row | 1999-01-05 (first date with any EUR return) | 1999-01-13 | 1999-02-28 |
| common sample (all 13 columns present) | from 2007-04-12 | from 2007-04-18 | from 2007-05-31 |
| last row | last data day | last complete week | last complete month |
| rows (2026-10 build) | 7,221 | 1,447 | 332 |

## Columns (fixed order)

`SPY, VGK, EWJ, SHY, IEF, TLT, EUR_CORE7_10_SYN, JGB7_10_SYN, LQD, HYG, GLD, DBC, EUR_ALL7_10_SYN`

* Values: simple period returns in EUR, decimals (0.01 = 1%).
* The first 12 columns are the investable v1 universe; `EUR_ALL7_10_SYN` is the all-issuer robustness
  series (not investable).
* Hedged file (policy): SHY, IEF, TLT, LQD, HYG and JGB7_10_SYN FX-hedged with a one-period CIP forward
  rolled at the file frequency; SPY, VGK, EWJ, GLD, DBC unhedged; EUR bonds need no hedge.
* `_unhedged` file: the six hedged columns converted at spot instead; all other columns identical.

## Missing values — policy

| Where | Meaning | What consumers do |
|---|---|---|
| before an asset's first return | asset did not exist yet (e.g. HYG before 2007-04-12) | restrict to the common sample |
| daily, inside the common sample | that market was closed (US holiday: 167 days; ECB: 91; Tokyo: 306) | treat as 0 return (`fillna(0)`): no price change while closed; the move is captured on the next open day |
| weekly / monthly, inside the common sample | never — a check fails | — |

No row is entirely NaN, and no row before the first data date is written.

## Checks (`check_processed`)

Index name `date`; strictly increasing, unique index; fixed column order; no leading all-NaN row;
weekly = Wednesdays; monthly = month-ends; daily: no all-NaN row and ≤ 8% holiday NaN per column in the
common sample; weekly/monthly: no NaN in the common sample; |return| below 20% daily, 25% weekly/monthly;
unhedged sleeves identical in hedged and unhedged files; monthly = compounded daily for unhedged columns
(1e-10); hedged daily-roll vs monthly-roll bias < 2bp/month.
