# Insurance Risk Engine

Eleven notebooks that follow one question: how do claim frequency, severity, extreme losses, reserve uncertainty, cross-line dependence and reinsurance together set an insurer's risk and capital, and which assumptions matter most. Two public data sets are used (French motor policies, US Schedule P loss development). They are separate case studies, not one insurer, and their currencies are never mixed.

## Architecture

```
Claims (01 calibration) -> Aggregate loss (02) -> Stress (03) -> Reinsurance (04)
                                   |                                  |
                                   +-> Pricing (07)                   |
CAS loss ratios -> Multi-line portfolio (05) -> Dependence and copulas (06, with portfolio reinsurance, RORAC)
CAS paid triangles -> Reserving (08) -> Reserve uncertainty (09)
Illustrative Standard Formula case (10)
All of the above -> Integrated risk, parameter risk, model-risk table, dashboard (11)
```

Each notebook writes its handoff file to `results/<stage>/`; `src/project.py` refuses to load a file that is missing or older than the code and data that produced it.

## Data

| Case | Files | Unit | Notebooks |
|---|---|---|---|
| French motor | `freMTPL2freq.csv`, `freMTPL2sev.csv` | EUR; 678,013 policies, 26,639 claims | 01-04, 07 |
| US multi-line loss ratios | `ppauto`, `comauto`, `wkcomp`, `othliab`, `medmal`, `prodliab` | USD thousands in the data, shown in USD millions | 05-06, 11 |
| US paid triangles | the same CAS files | USD thousands | 08-09, 11 |
| Regulatory illustration | assumptions written in 10 | EUR million | 10 |

The motor files are the extract I was given, not a fresh download ([CASdatasets manual](https://dutangc.perso.math.cnrs.fr/RRepository/pub/web/CASdatasets-manual.pdf)); claim counts and severity records do not fully reconcile and 01 shows the gap. The CAS files are [Schedule P](https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p) data, net of reinsurance, accident years 1998-2007. They are used unchanged and nothing is downloaded by the code; if a CSV is missing, put it back in `data/`. `data/` holds raw files only, so there is no `raw/` and `processed/` split.

05-06 fit lag-10 company-year loss ratios and scale them by median premiums. 08-09 sum companies with complete filings into one paid triangle. Neither is a real single insurer.

## Methodology

- Frequency: exposure-adjusted negative binomial (NB2); Poisson fits worse (01).
- Severity: truncated lognormal body, generalised Pareto tail above the 99th percentile; the closed-form mean is checked against numerical integration and a capped-mean test (02).
- Aggregate loss: Monte Carlo over annual counts and claims. VaR, ES and K = VaR - E[loss] are always reported separately; K is an internal capital proxy, not a regulatory SCR.
- Reinsurance: quota share, per-claim excess of loss, and per-line stop-loss, all settled so that retained + ceded = gross (04, 06, 11).
- Multi-line: Burr XII loss-ratio marginals, Gaussian copula from Spearman (2 sin(pi rho / 6)) and Student-t copula from Kendall (sin(pi tau / 2)), correlation matrices checked for positive definiteness (05-06).
- Reserving: Chain-Ladder, Bornhuetter-Ferguson, Mack standard errors checked against `chainladder`; lognormal quantiles are an extra approximation (08-09). Runoff and one-year risk are kept apart.
- Integration (11): premium risk plus reserve risk under assumed dependence, a project-level bootstrap of the inputs that move capital, and a model-risk table that ranks the assumptions.
- Regulatory: one Standard Formula premium and reserve component, for illustrative inputs only (10).

## Key results

- Motor, one policy-year (EUR): expected loss 297.11, VaR 99.5% 6,247, K 5,950. The GPD shape is 0.852, so the mean is finite and the variance is not. ES 99.5% is 45,792 for seed 42 but runs from 21,111 to 40,162 across 500k-draw replicates; it should never be quoted as one number.
- Cross-line (USD, six lines, gross): VaR 99.5% is 60.3m under independence and 62.2m under the fitted Gaussian copula. VaR diversification is 37% falling to 35%; on the K basis 53% and 50%. Workers' comp takes 69% of the allocated ES.
- The Workers' comp tail is the largest modelling risk for the CAS portfolio: the portfolio VaR runs from 48.1m (AY2001 premium artifact removed) to 137.2m (empirical tail) against 62.2m for the Burr XII baseline.
- Premium plus reserve (USD, moderate premium-reserve dependence): K is 34.9m against 33.9m for premium risk alone; reserve risk adds little because Mack CVs are a few percent for the large lines. After a 25% quota share on the premium block K is 26.6m. This is runoff risk, which overstates one-year risk.
- Parameter risk: Monte Carlo noise is about 0.8% of the integrated VaR, fit uncertainty puts the 95th percentile 18% above the point estimate, almost all of it from the Workers' comp marginal. Motor VaR barely sees the tail model (moving the EVT threshold changes it -1.2%) but ES moves +102%. 20 of 300 resampled severity fits have shape at or above 1 (infinite mean), and a 98th-percentile threshold alone gives shape above 1.
- Pricing (EUR): NB2 beats a constant rate out of sample. Combined ratio 82.70% gross and 76.91% net of an assumed quota share. The capital loading is an assumed scaling constant; k from 0 to 2% moves the average gross premium from EUR 404 to EUR 550.
- Reserving (USD): Workers' comp unpaid is 3,268m on Chain-Ladder and 4,090m on Bornhuetter-Ferguson; the Mack standard error is 102m (3.1%).
- Regulatory illustration (EUR): the premium and reserve underwriting-risk SCR component is 83.346 million for assumed volumes. It is not a total SCR and is not compared with the simulated figures above.

## Model limitations

- Heavy tails: motor severity has infinite variance, so ES and Monte Carlo means are unstable and more simulations do not fix model uncertainty. Scaling exposure with a fixed NB2 dispersion is not the same as summing independent policies.
- Parameter and model uncertainty: the bootstrap covers frequency, severity, the loss-ratio marginals and rank correlations. It does not cover Mack standard errors beyond Mack itself, the copula family, or the Workers' comp marginal family; those are in the model-risk table.
- Historical data: company-year loss ratios are not one insurer; one accident year of premium data drives the Workers' comp tail; only companies with complete filings enter the triangles (wkcomp 83% of companies, comauto 87% of companies); long-tail lines are cut at lag 10 and their reserves are probably understated.
- Premium-reserve dependence is assumed, not estimated.
- Regulatory: all financial inputs in 10 are assumed, one component of one module is implemented, and the 2027 amendments are not applied.
- Reinsurance and pricing: treaty terms, expenses, the capital loading and the 30% expense ratio in the illustrative RORAC are assumptions; reinsurance premium is not modelled.
- Portfolio weights: the CAS portfolio uses median premiums as book sizes, not a real book.
- ALM sits in [multi-asset portfolio risk](../02_multi_asset_portfolio_risk/README.md); there was no ALM code to move here.

## Repository structure

```
notebooks/   01-11, run in order; analysis, plots and interpretation
src/         shared model code: insurance, portfolio, dependence, reinsurance, reserving, pricing, solvency,
             integration and parameter-risk modules, risk measures, project paths and freshness checks
tests/       unit and invariant tests (pytest)
results/     generated handoff files by stage (calibration, aggregate, portfolio, pricing, reserving,
             regulatory, summary) and run reports (validation)
data/        raw CSVs, never modified
run_notebooks.py   runs 01-11 in fresh kernels and writes results/validation/validation.json
```

## Reproduction

From this folder:

```bash
pip install -r requirements.txt
python run_notebooks.py
pytest
```

`run_notebooks.py` deletes the freshness manifest, runs the eleven notebooks in order, saves the outputs into the notebooks and stops at the first failure; `python run_notebooks.py 5 7` runs a range and keeps the manifest. A full run takes about ten minutes, most of it in 11. Package versions of the last run are in `results/validation/validation.json`. Jupyter needs permission to bind local kernel ports. Any change to code or data makes the saved results stale, and the next notebook says what to rerun.
