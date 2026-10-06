# 02 — Global Multi-Asset Portfolio & Risk Management

EUR-based multi-asset portfolio (US / Euro Area / Japan) comparing covariance estimators and
portfolio construction rules **out of sample**, with transaction costs, VaR/ES backtesting,
historical stress tests and an asset–liability (insurance) extension.

## Questions this project answers with numbers

1. Which covariance estimator (Sample, Rolling, EWMA, Ledoit-Wolf, GARCH) forecasts out-of-sample portfolio risk best?
2. How do Equal Weight, 60/40, Minimum Variance, Risk Parity and CVaR portfolios differ out of sample?
3. Does the edge of optimised portfolios survive turnover and transaction costs?
4. How much did diversification fail in 2022 when the stock–bond correlation turned positive?
5. Which VaR model passes Kupiec / Christoffersen backtests?
6. How does the optimal allocation change once insurance liabilities are added (surplus optimisation)?

## Universe (v1, EUR base)

| Sleeve | Assets | FX |
|---|---|---|
| Equity | SPY, VGK, EWJ | unhedged |
| Government bonds | SHY, IEF, TLT, synthetic EUR 7–10Y, synthetic JGB 7–10Y | hedged |
| Credit | LQD, HYG | hedged |
| Real assets | GLD, DBC | unhedged |

Common sample starts 2007-04-11 (HYG); with a 3-year window the first out-of-sample month is 2010-04.

## Structure

```
data/metadata/   sources, series dictionary, stress periods, validation report (raw data not committed)
scripts/         download (run_all.sh) and raw-data validation (90_validate_raw.py)
src/             data_loader.py
notebooks/       01 data pipeline · 02 returns · 03 risk estimation · 04 optimisation ·
                 05 walk-forward · 06 VaR/ES backtesting · 07 stress · 08 results · 09 ALM
```

## Status

| Step | State |
|---|---|
| Raw data download + validation | done |
| 01 Data pipeline | done |
| 02 Return construction (synthetic UST vs IEF validation first) | next |
| 03–09 | planned |

## Reproduce

```bash
pip install -r requirements.txt
bash scripts/run_all.sh            # + manual downloads, see data/metadata/README.md
python scripts/90_validate_raw.py
```
