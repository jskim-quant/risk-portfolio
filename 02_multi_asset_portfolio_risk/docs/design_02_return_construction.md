# Design — 02 Return Construction (as built + next commit)

Status: **implemented** in `205aad9` (notebook 02, `src/returns.py`, `src/universe.py`, `tests/test_returns.py`)
on branch `feat/multi-asset-portfolio-risk`, built on top of `864bc12` (data pipeline, `src/data_loader.py`).
This document is the specification of what exists, the acceptance criteria it was held to, and the
remaining gaps (§9) for the next commit.

---

## 1. Scope

| | Minimum shipped scope (v1) | Follow-up (not in v1) |
|---|---|---|
| ETFs | total return from `AdjClose`, EUR conversion | dividend reconstruction check vs `Close` + distributions |
| Synthetic bonds | UST, EUR AAA, EUR all-issuer, JGB 7–10Y (par repricing) | roll-down, coupon-bearing bond baskets, 30Y |
| FX | unhedged (equity, gold, commodities); one-period CIP hedge (bonds, credit) | hedge-ratio optimisation, forward-points data, cost of hedging spreads |
| Validation | synthetic UST vs SHY / IEF / TLT gate | EUR-hedged UCITS ETF comparison, JGB vs a JGB futures proxy |
| Output | daily / weekly / monthly EUR CSVs, hedged + unhedged | parquet with metadata, versioned data snapshots |

## 2. Files

| File | Status | Content |
|---|---|---|
| `src/data_loader.py` | existing (`864bc12`) | `load_etf_prices(field="AdjClose")`, `load_ust_curve()`, `load_ecb_curve(curve, kind)`, `load_jgb_curve(start="1986-07-01")`, `load_short_rates()` (EONIA→€STR splice at `EONIA_ESTR_SWITCH`), `load_fx()` (`EURJPY = EURUSD × USDJPY`) |
| `src/returns.py` | added (`205aad9`) | `par_bond_price`, `par_bond_duration_convexity`, `synthetic_bond_returns`, `blend_returns`, `index_from_returns`, `period_levels`, `to_eur_unhedged`, `eur_hedged_returns`, `tracking_stats` |
| `src/universe.py` | added | `ASSETS`, `SLEEVE`, `RULES = {"daily": "D", "weekly": "W-WED", "monthly": "ME"}`, `synthetic_indices()`, `eur_returns(freq, hedge_bonds, extra)` |
| `tests/test_returns.py` | added | 6 unit tests (§7.1) |
| `notebooks/02_return_construction.ipynb` | executed | method, validation gate, EUR/JGB synthetics, hedging, universe stats, CSV export |
| `scripts/91_validate_processed.py` | **next commit** | processed-data checks (§6) |
| `tests/test_universe.py` | **next commit** | processed-data invariants (§7.2) |
| `data/metadata/processed_schema.md` | **next commit** | CSV schema (§5) |

## 3. Formulas

### 3.1 ETF total return (USD)

Yahoo `AdjClose` is split- and distribution-adjusted, so

    R^USD_t = AdjClose_t / AdjClose_{t-1} − 1      (≈ (Close_t + D_t) / Close_{t-1} − 1)

Sanity check on the raw data (`90_validate_raw.py`): CAGR(AdjClose) − CAGR(Close) ≈ distribution yield
(HYG 6.4%, LQD 4.4%, IEF 3.0%, SPY 1.8%, GLD 0.00%).

### 3.2 Constant-maturity synthetic bond (exact repricing)

Par bond priced off a single yield `y`, coupon `c`, maturity `T`, `f` coupons a year, `n = f·T`:

    P(c, y, T) = (c/f) · [1 − (1 + y/f)^(−n)] / (y/f) + (1 + y/f)^(−n),   annuity → n as y → 0

Each day the position is a new par bond (`c = y_{t−1}`, maturity `M`), repriced at `y_t` with
`T = M − Δt` (`Δt` = calendar days / 365.25), plus coupon accrual:

    R_t = P(y_{t−1}, y_t, M − Δt) − 1 + y_{t−1} · Δt

Cross-check `method="approx"`: `R_t ≈ y_{t−1}Δt − D_{t−1}Δy_t + ½C_{t−1}Δy_t²`
(max daily gap 2.8e-5; 0.0008% p.a.).

| Series | Curve (`data_loader`) | Yield type | f | Construction |
|---|---|---|---|---|
| `UST7_10_SYN` (validation only) | `load_ust_curve()["7Y","10Y"]` | par CMT | 2 | 50/50 7Y + 10Y, daily rebalanced |
| `EUR_CORE7_10_SYN` (investable) | `load_ecb_curve("AAA","PY")` | par | 1 | 50/50 7Y + 10Y |
| `EUR_ALL7_10_SYN` (robustness) | `load_ecb_curve("ALL","PY")` | par | 1 | 50/50 7Y + 10Y |
| `JGB7_10_SYN` (investable) | `load_jgb_curve()` (≥ 1986-07) | par | 2 | 50/50 7Y + 10Y |

Known bias: daily compounding of the accrual adds 2–3 bp p.a. vs semi-annual compounding.

### 3.3 EUR conversion — unhedged

FX quoted as foreign currency per EUR (`EURUSD`, `EURJPY`). For a foreign-currency index `I`:

    I^EUR_t = I_t / S_t,     1 + R^EUR = (1 + R^fc) · S_{t−1} / S_t

`to_eur_unhedged` forward-fills FX at most 5 days onto the asset's own dates.
USD-listed ETFs (VGK, EWJ) already embed their underlying currency → only `EURUSD` is applied.

### 3.4 EUR conversion — FX-hedged (bonds, credit, JGB)

One-period CIP forward set at the period start `t0` (rates fixed at `t0`, money-market day counts
`τ_EUR = days/360`, `τ_USD = days/360`, `τ_JPY = days/365`):

    F = S0 · (1 + r_fc τ_fc) / (1 + r_EUR τ_EUR)
    R^hedged = (1 + r_EUR τ_EUR) / (1 + r_fc τ_fc) − 1 + R^fc · S0 / S1

The principal is hedged; only the period P&L `R^fc` stays exposed to FX. Implemented in
`eur_hedged_returns(index_fc, fx_fc_per_eur, r_eur_pct, r_fc_pct, rule, dc_eur=360, dc_fc=360)`;
the hedge is rolled at the output frequency (daily, weekly or monthly), so frequencies are *not*
exact compounds of each other for hedged assets (§6, check P7).

| Sleeve | Assets | FX treatment | Short rate (fc) |
|---|---|---|---|
| equity | SPY, VGK, EWJ | unhedged, `EURUSD` | — |
| real | GLD, DBC | unhedged, `EURUSD` | — |
| govt | SHY, IEF, TLT | hedged | `DFF` (USD, ACT/360) |
| govt | EUR_CORE7_10_SYN | none (EUR) | — |
| govt | JGB7_10_SYN | hedged, `EURJPY` | BOJ O/N call (ACT/365) |
| credit | LQD, HYG | hedged | `DFF` |

EUR rate: `load_short_rates()["EUR"]` = EONIA until 2019-09-30, €STR from 2019-10-01.

## 4. Validation gate — synthetic UST vs Treasury ETFs

Defined **before** looking at the results; computed with `tracking_stats(syn, etf, periods_per_year)`
on month-end returns from 2002-09:

| Criterion | Threshold | IEF vs 7Y+10Y | SHY vs 2Y | TLT vs 20Y+30Y |
|---|---|---|---|---|
| correlation | ≥ 0.97 | 0.993 ✓ | 0.985 ✓ | 0.994 ✓ |
| β (syn on ETF) | 0.90 – 1.10 | 0.99 ✓ | 1.07 ✓ | 0.98 ✓ |
| tracking error p.a. | ≤ 1.5% | 0.78% ✓ | 0.29% ✓ | 1.52% ✗ |
| mean gap p.a. | reported | −0.65% | −0.07% | +0.07% |

Single maturities fail the β test (IEF vs 10Y: β 1.13; vs 7Y: 0.86) → the 50/50 blend is required.
The gate covers the 7–10Y method (IEF) and the short end (SHY); TLT enters v1 as the ETF itself.
The −0.65% gap is explained by missing roll-down and on-the-run vs off-the-run yields.
Empirical duration (daily OLS on Δ average 7Y/10Y yield, 2004-09 onward): UST 7.38, EUR core 7.71,
EUR all 7.52, JGB 8.00.

```python
gate = R.tracking_stats(syn_monthly.loc["2002-09":], ief_monthly.loc["2002-09":], 12)
assert gate["corr"] >= 0.97 and abs(gate["beta (a on b)"] - 1) <= 0.10 and gate["TE p.a."] <= 0.015
```

## 5. Output schema — `data/processed/asset_returns_eur_{freq}[_unhedged].csv`

| Property | daily | weekly | monthly |
|---|---|---|---|
| `rule` | own trading dates (union) | `W-WED` (Wednesday close) | `ME` (calendar month-end label) |
| index | `date` (ISO), increasing, unique | same, all Wednesdays | same, all month-ends |
| columns (order) | `SPY, VGK, EWJ, SHY, IEF, TLT, EUR_CORE7_10_SYN, JGB7_10_SYN, LQD, HYG, GLD, DBC, EUR_ALL7_10_SYN` | same | same |
| values | simple returns in EUR, decimals | same | same |
| common sample | from 2007-04-12 | from 2007-04-18 | from 2007-05-31 |
| NaN inside common sample | allowed = market closed that day (US holidays etc.) | none | none |
| last row | last data day | last complete week (≤ min last observation) | last complete month |

`_unhedged` files apply §3.3 to SHY, IEF, TLT, LQD, HYG, JGB; all other columns are identical.

## 6. Alignment and missing-data checks (`scripts/91_validate_processed.py`)

Measured on the current files; each line is a direct assert.

| # | Check | Current result |
|---|---|---|
| P1 | index monotonic, no duplicates, all files | ✓ |
| P2 | column list == §5 order | ✓ |
| P3 | weekly index all Wednesdays; monthly index all month-ends | ✓ |
| P4 | weekly/monthly: no NaN from common start | ✓ (0 NaN) |
| P5 | daily: no all-NaN row from 2007-04-12; per-asset NaN = holiday count (167 US-listed, 91 EUR, 306 JGB) | ✓ |
| P6 | unhedged sleeves identical in hedged and unhedged files | ✓ (`np.allclose`) |
| P7 | unhedged columns: monthly return == compounded daily return | ✓ (max \|diff\| 2e-15) |
| P8 | hedged columns: mean(monthly − compounded daily) within ±2bp/month | IEF 0.02bp, JGB −0.68bp, HYG 1.34bp ✓ |
| P9 | plausibility: max \|monthly\| < 25% (max today: DBC 18.4%) | ✓ |
| P10 | last weekly/monthly row ≤ last observation of every input | ✓ |

```python
COMMON = {"daily": "2007-04-12", "weekly": "2007-04-18", "monthly": "2007-05-31"}

def check(df: pd.DataFrame, freq: str) -> None:
    core = df.loc[COMMON[freq]:]
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert list(df.columns) == U.ASSETS + ["EUR_ALL7_10_SYN"]
    if freq == "weekly":
        assert (core.index.dayofweek == 2).all()
    if freq == "monthly":
        assert core.index.is_month_end.all()
    if freq == "daily":
        assert not core.isna().all(axis=1).any()
    else:
        assert core.notna().all().all()
    assert core.abs().max().max() < {"daily": 0.15, "weekly": 0.25, "monthly": 0.25}[freq]
```

## 7. Tests

### 7.1 Existing (`tests/test_returns.py`, all passing)

| Test | Assertion |
|---|---|
| `test_par_bond_prices_at_par` | `P(y, y, 10) = 1` for y = 4%, 0%, −0.2% (annual) |
| `test_duration_of_10y_par_bond` | 4% 10Y semi-annual par: D ∈ (8.10, 8.25), C > 0 |
| `test_flat_curve_earns_the_yield` | unchanged 3% curve → annualised 2.9–3.1% |
| `test_rate_rise_loses_about_duration` | one-day +50bp → R ∈ (−4.2%, −3.6%) |
| `test_hedged_return_without_fx_or_rate_moves` | equal rates, constant FX → hedged = local |
| `test_hedge_removes_fx_on_principal` | flat asset, foreign currency −10% → hedged 0, unhedged < −9% |

### 7.2 Next commit (`tests/test_universe.py`)

| Test | Assertion |
|---|---|
| `test_processed_schema` | P1–P4 for all six files (skips if `data/processed` missing) |
| `test_monthly_compounds_daily_for_unhedged` | P7 for SPY, GLD, EUR_CORE7_10_SYN (`atol=1e-12`) |
| `test_hedged_roll_bias_small` | P8 for IEF, HYG, JGB |
| `test_jgb_negative_yields_handled` | synthetic JGB finite on 2016–2020 negative-yield days |
| `test_ief_gate` | §4 gate re-computed from raw data |

## 8. Implementation order (as executed)

1. `par_bond_price` + zero-yield limit → unit tests 1–2.
2. `synthetic_bond_returns` (exact + approx) → tests 3–4, cross-check gap.
3. **Gate first:** synthetic UST vs SHY/IEF/TLT on 2002-09+ → choose 50/50 blend; do not build EUR/JGB before it passes.
4. `synthetic_indices()` for UST, EUR AAA, EUR all, JGB; empirical-duration check on a common window.
5. `to_eur_unhedged`, `eur_hedged_returns` → tests 5–6.
6. `eur_returns(freq)`; drop incomplete last period (bug found: October 2026 counted as a full month).
7. Export six CSVs; notebook 02 executed and committed.

## 9. Gaps → next commit

| # | Gap (found while writing this spec) | Fix |
|---|---|---|
| G1 | CSV index has no name (blank header) | `df.index.name = "date"` before `to_csv` |
| G2 | hedged files start in 1986 with 3,151 leading all-NaN daily rows; unhedged start 1999 | drop leading all-NaN rows in `eur_returns` |
| G3 | daily holiday NaN policy is implicit (consumers fill with 0) | document in `processed_schema.md`; check P5 |
| G4 | no automated processed-data checks | `scripts/91_validate_processed.py` + `tests/test_universe.py` |
| G5 | hedged daily vs monthly differ by up to 133bp in a month (HYG, Oct 2008) | expected (P&L stays unhedged within the roll period); bound by \|R_fc\|·\|S0/S1 − 1\| and documented |

Commit: `fix: processed return schema, leading-NaN trim and validation checks (notebook 02)`.
