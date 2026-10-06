# Raw Data Validation Report
Generated: 2026-10-06 20:58

## 1. ETFs (Yahoo)

| ETF | start | end | n | max \|daily ret\| (date) | AdjClose≤Close | div-adj gap p.a. |
|---|---|---|---|---|---|---|
| DBC | 2006-02-06 | 2026-10-06 | 5199 | 7.9% (2022-03-09) | 100.0% | 1.13% |
| EWJ | 1996-03-18 | 2026-10-06 | 7688 | 15.8% (2008-10-13) | 100.0% | 1.48% |
- **WARN** EWJ: daily return +15.8% on 2008-10-13
| GLD | 2004-11-18 | 2026-10-06 | 5504 | 11.3% (2008-09-17) | 100.0% | 0.00% |
| HYG | 2007-04-11 | 2026-10-06 | 4904 | 12.3% (2008-10-13) | 100.0% | 6.41% |
| IEF | 2002-07-30 | 2026-10-06 | 6086 | 3.4% (2009-03-18) | 100.0% | 3.00% |
| LQD | 2002-07-30 | 2026-10-06 | 6086 | 9.8% (2008-09-30) | 100.0% | 4.35% |
| SHY | 2002-07-30 | 2026-10-06 | 6086 | 1.0% (2023-03-13) | 100.0% | 1.95% |
- **WARN** SHY: 4 consecutive unchanged values (ends 2012-12-11)
| SPY | 1993-01-29 | 2026-10-06 | 8479 | 14.5% (2008-10-13) | 100.0% | 1.80% |
| TLT | 2002-07-30 | 2026-10-06 | 6086 | 7.5% (2020-03-20) | 100.0% | 3.51% |
| VGK | 2005-03-10 | 2026-10-06 | 5428 | 14.2% (2008-10-13) | 100.0% | 3.65% |

*div-adj gap* = AdjClose CAGR − Close CAGR ≈ dividend yield. Equity ~1.5–3%, bonds ~2–5%, GLD ≈ 0.

## 2. UST curve (FRED, par CMT)

| tenor | start | end | n | min | max | max \|Δ1d\| bp |
|---|---|---|---|---|---|---|
| DGS3MO | 1981-09-01 | 2026-10-05 | 11273 | 0.00 | 17.01 | 169 |
- **WARN** DGS3MO: jump +1.690pp on 1982-02-01
- **WARN** DGS3MO: jump -1.040pp on 1982-02-22
- **WARN** DGS3MO: jump -0.830pp on 1982-07-08
- **WARN** DGS3MO: jump -0.850pp on 1982-08-02
- **WARN** DGS3MO: jump -0.810pp on 2008-09-17
- **WARN** DGS3MO: jump +0.760pp on 2008-09-19
| DGS1 | 1962-01-02 | 2026-10-05 | 16175 | 0.04 | 17.31 | 110 |
- **WARN** DGS1: jump +1.020pp on 1979-10-09
- **WARN** DGS1: jump -0.790pp on 1980-04-07
- **WARN** DGS1: jump -0.760pp on 1980-04-16
- **WARN** DGS1: jump -0.890pp on 1980-05-02
- **WARN** DGS1: jump -1.080pp on 1980-12-19
- **WARN** DGS1: jump -0.970pp on 1981-01-05
- **WARN** DGS1: jump +0.920pp on 1981-05-04
- **WARN** DGS1: jump +1.100pp on 1981-07-20
- **WARN** DGS1: … 12 jumps total > 0.75pp
| DGS2 | 1976-06-01 | 2026-10-05 | 12583 | 0.09 | 16.95 | 89 |
- **WARN** DGS2: jump +0.830pp on 1980-02-19
- **WARN** DGS2: jump -0.840pp on 1980-12-19
- **WARN** DGS2: jump -0.810pp on 1980-12-22
- **WARN** DGS2: jump -0.820pp on 1981-01-05
- **WARN** DGS2: jump +0.890pp on 1981-07-20
- **WARN** DGS2: jump +0.800pp on 1982-02-01
- **WARN** DGS2: jump -0.840pp on 1987-10-20
| DGS5 | 1962-01-02 | 2026-10-05 | 16175 | 0.19 | 16.27 | 77 |
- **WARN** DGS5: jump -0.770pp on 1980-12-19
- **WARN** DGS5: jump -0.770pp on 1987-10-20
| DGS7 | 1969-07-01 | 2026-10-05 | 14305 | 0.36 | 16.05 | 78 |
- **WARN** DGS7: jump -0.780pp on 1980-02-28
- **WARN** DGS7: jump -0.770pp on 1987-10-20
| DGS10 | 1962-01-02 | 2026-10-05 | 16175 | 0.52 | 15.84 | 75 |
| DGS20 | 1962-01-02 | 2026-10-05 | 14486 | 0.87 | 15.78 | 127 |
- **WARN** DGS20: jump -1.270pp on 1993-10-01
- **INFO** DGS20: gap 2466.0d ending 1993-10-01
| DGS30 | 1977-02-15 | 2026-10-05 | 12405 | 0.99 | 15.21 | 76 |
- **WARN** DGS30: jump -0.760pp on 1987-10-20

10Y−2Y inverted on 16.3% of days (latest +47bp)

## 3. ECB curves

- G_N_A_SR: 2004-09-06 → 2026-10-05, n=5645, cols=9, max NaN 0.00%, range [-1.00, 5.18]
- **WARN** G_N_A_SR.G_N_A.SR_30Y: jump -0.564pp on 2008-12-04
- **WARN** G_N_A_SR.G_N_A.SR_3M: jump -0.940pp on 2008-10-07
- G_N_A_PY: 2004-09-06 → 2026-10-05, n=5646, cols=9, max NaN 0.00%, range [-1.00, 4.97]
- **WARN** G_N_A_PY.G_N_A.PY_3M: jump -0.946pp on 2008-10-07
- G_N_C_SR: 2004-09-06 → 2026-10-05, n=5645, cols=9, max NaN 0.00%, range [-0.83, 5.84]
- **WARN** G_N_C_SR.G_N_C.SR_30Y: jump +0.517pp on 2009-01-23
- **WARN** G_N_C_SR.G_N_C.SR_30Y: jump -0.881pp on 2010-10-15
- **WARN** G_N_C_SR.G_N_C.SR_3M: jump -0.661pp on 2008-10-08
- G_N_C_PY: 2004-09-06 → 2026-10-05, n=5645, cols=9, max NaN 0.00%, range [-0.83, 5.54]
- **WARN** G_N_C_PY.G_N_C.PY_30Y: jump -0.557pp on 2010-10-15
- **WARN** G_N_C_PY.G_N_C.PY_3M: jump -0.665pp on 2008-10-08

All-gov − AAA 10Y par spread: mean 54bp, max 190bp (2012-07-23), <0 on 0.0% days
AAA 10Y zero − par: mean 3.8bp, |max| 17bp (should be small, sign follows curve slope)

## 4. JGB curve (MoF)

| tenor | start | n | min | max |
|---|---|---|---|---|
| JGB_1Y | 1974-09-24 | 12664 | -0.371 | 11.237 |
- **WARN** JGB_1Y: jump -0.588pp on 1975-02-22
- **WARN** JGB_1Y: jump +4.507pp on 1980-08-22
- **WARN** JGB_1Y: jump -0.559pp on 1981-01-08
- **WARN** JGB_1Y: jump +0.502pp on 1985-10-25
| JGB_2Y | 1974-09-24 | 12951 | -0.372 | 12.145 |
- **WARN** JGB_2Y: jump +2.241pp on 1979-08-22
- **WARN** JGB_2Y: jump -0.585pp on 1980-04-25
| JGB_3Y | 1974-09-24 | 13236 | -0.374 | 11.701 |
| JGB_4Y | 1974-09-24 | 13312 | -0.382 | 10.562 |
| JGB_5Y | 1974-09-24 | 13312 | -0.389 | 10.656 |
- **WARN** JGB_5Y: jump +0.558pp on 1985-10-25
| JGB_6Y | 1974-09-24 | 13312 | -0.405 | 10.639 |
- **WARN** JGB_6Y: jump +0.614pp on 1985-10-25
| JGB_7Y | 1974-09-24 | 13312 | -0.423 | 10.858 |
- **WARN** JGB_7Y: jump +0.617pp on 1985-10-25
| JGB_8Y | 1974-09-24 | 13312 | -0.393 | 10.940 |
- **WARN** JGB_8Y: jump +0.604pp on 1985-10-25
| JGB_9Y | 1974-09-24 | 13312 | -0.350 | 10.670 |
- **WARN** JGB_9Y: jump +0.586pp on 1985-10-25
| JGB_10Y | 1986-07-05 | 9951 | -0.297 | 8.105 |
| JGB_15Y | 1991-08-30 | 8610 | -0.150 | 6.546 |
| JGB_20Y | 1986-12-01 | 9838 | 0.022 | 7.749 |
| JGB_25Y | 2004-03-22 | 5517 | 0.036 | 4.185 |
| JGB_30Y | 1999-09-02 | 6635 | 0.042 | 4.166 |
| JGB_40Y | 2007-11-06 | 4623 | 0.067 | 4.195 |

10Y negative on 453 days (2016-02-09 – 2020-05-22)

## 5. Short rates

- DFF: 1954-07-01 → 2026-10-05, n=26395, [0.040, 22.360], last 3.880
- **WARN** DFF: jump +1.120pp on 1954-10-07
- **WARN** DFF: jump -1.130pp on 1955-05-17
- **WARN** DFF: jump -1.190pp on 1956-01-04
- **WARN** DFF: jump +1.250pp on 1956-01-05
- **WARN** DFF: jump +1.250pp on 1956-08-30
- **WARN** DFF: jump -1.250pp on 1956-11-28
- **WARN** DFF: jump +1.630pp on 1956-11-29
- **WARN** DFF: jump -2.000pp on 1957-01-02
- **WARN** DFF: … 448 jumps total > 1.0pp
- DTB3: 1954-01-04 → 2026-10-05, n=18181, [-0.050, 17.140], last 4.050
- **WARN** DTB3: jump +1.120pp on 1979-10-09
- **WARN** DTB3: jump -1.270pp on 1980-12-19
- **WARN** DTB3: jump -1.130pp on 1981-01-05
- **WARN** DTB3: jump +1.030pp on 1981-04-06
- **WARN** DTB3: jump +1.340pp on 1981-05-04
- **WARN** DTB3: jump +1.070pp on 1981-07-20
- **WARN** DTB3: jump +1.160pp on 1982-02-01
- EONIA: 1999-01-04 → 2021-12-31, n=5890, [-0.505, 5.750], last -0.505
- **WARN** EONIA: jump +1.160pp on 2000-05-24
- ESTR: 2019-10-01 → 2026-10-05, n=1796, [-0.593, 3.913], last 2.438
- JPY_CALL: 1998-01-05 → 2026-10-02, n=7046, [-0.081, 1.227], last 1.227
- JPY_OECD_M: 1985-07-01 → 2026-08-01, n=494, [-0.071, 8.278], last 0.977

EONIA − €STR overlap: 8.50bp ± 0.00bp (n=579) → splice valid
DTB3 − DFF since 2000: mean -12bp (T-bill discount basis + convenience yield)
BOJ daily (monthly avg) vs OECD monthly: mean |diff| 0.1bp over 344 months
EUR spliced O/N: 1999-01-04 → 2026-10-05, gaps: 0

## 6. FX (FRED H.10, noon NY)

- DEXUSEU: 1999-01-04 → 2026-10-02, [0.827, 1.601], max |Δlog| 4.6% (2009-03-19), ann.vol 9.1%
- DEXJPUS: 1971-01-04 → 2026-10-02, [75.72, 358.4], max |Δlog| 9.5% (1973-02-13), ann.vol 10.1%
- **WARN** DEXJPUS: daily move -5.0% on 1971-08-31
- **WARN** DEXJPUS: daily move -9.5% on 1973-02-13
- **WARN** DEXJPUS: daily move +6.3% on 1974-01-07
- **WARN** DEXJPUS: daily move -5.2% on 1978-04-03
- **WARN** DEXJPUS: daily move -5.6% on 1998-10-07
- **WARN** DEXJPUS: daily move -5.2% on 2008-10-24
- **WARN** DEXJPUS: 10 consecutive unchanged values (ends 1972-09-26)
- EURJPY (derived): 1999-01-04 → last 177.68

## 7. Credit

- Baa − Aaa: mean 95bp, min 40bp, max 350bp (2008-12-03)
- **WARN** DAAA: jump +0.520pp on 2020-03-17
- **WARN** DAAA: jump +0.520pp on 2020-03-18
- **WARN** DAAA: jump -0.550pp on 2020-03-23
- ICE OAS: 2023-10-09 → 2026-10-05 (n=783) — **3y history only, reference use**

## 8. VIX

- 1990-01-02 → 2026-10-05, [9.14, 82.69] (max 2020-03-16), median 17.6

## 9. Cross-source sanity checks

- IEF ret vs ΔDGS10: corr -0.96, empirical duration 7.2 (ref ≈ 7.5)
- TLT ret vs ΔDGS10: corr -0.90, empirical duration 14.2 (ref ≈ 17.0)
- SHY ret vs ΔDGS10: corr -0.74, empirical duration 1.2 (ref ≈ 1.9)
- SPY ret vs ΔVIX: corr -0.71 (expect ≈ −0.7)
- HYG monthly ret vs Δ(Baa−Aaa): corr -0.52 (expect negative)
- Calendars since 2007-05: US 4890, EUR 4966, JP 4752 days; common 4543

## Summary

- FAIL: 0
- WARN: 76
- INFO: 1