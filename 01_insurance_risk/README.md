# Insurance Risk Engine

I wanted to see how far one set of simulation code could take a single question: what actually drives an insurer's capital number? Claim frequency, severity, the extreme tail, reserve uncertainty, dependence between lines and reinsurance all feed into it, and I wanted to see which of them moves the answer most and which ones barely matter.

The project is eleven notebooks on top of a small `src/` package. They use two public data sets, French motor policies and US Schedule P loss development. These are two separate case studies, not one company, so the currencies (EUR and USD) are never added together.

## Architecture

```
Claims (01 calibration) -> Aggregate loss (02) -> Stress (03) -> Reinsurance (04)
                                   |
                                   +-> Pricing (07)

CAS loss ratios  -> Multi-line portfolio (05) -> Dependence and copulas (06)
CAS paid triangles -> Reserving (08) -> Reserve uncertainty (09)
Standard Formula illustration (10)

Everything above -> Integrated risk, parameter risk, model-risk table (11)
```

Each notebook saves what the next one needs under `results/<stage>/`. `src/project.py` keeps a manifest of hashes of the data, the `src/` code and the notebook code cells, and refuses to load a result that is older than any of them. So if you edit something and forget to rerun, you get an error rather than a stale number.

## Data

| Case | Files | Unit | Notebooks |
|---|---|---|---|
| French motor | `freMTPL2freq.csv`, `freMTPL2sev.csv` | EUR, 678,013 policies, 26,639 claims | 01-04, 07 |
| US multi-line loss ratios | `ppauto`, `comauto`, `wkcomp`, `othliab`, `medmal`, `prodliab` | USD thousands in the files, USD millions in the output | 05-06, 11 |
| US paid triangles | same CAS files | USD thousands | 08-09, 11 |
| Regulatory illustration | assumptions written out in 10 | EUR million | 10 |

The motor files are the extract I had, not a fresh download (see the [CASdatasets manual](https://dutangc.perso.math.cnrs.fr/RRepository/pub/web/CASdatasets-manual.pdf)). The claim counts and the severity records don't fully reconcile, and notebook 01 shows the gap instead of hiding it. The CAS files come from [Schedule P](https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p), net of reinsurance, accident years 1998-2007. Nothing is downloaded by the code. If a CSV goes missing, put it back in `data/`; the folder only holds raw files, so I did not split it into `raw/` and `processed/`.

Notebooks 05-06 fit lag-10 company-year loss ratios and scale them by median premiums to get book sizes. Notebooks 08-09 add up the companies with complete filings into one paid triangle. Neither is a real insurer, just a convenient stand-in with real data behind it.

## Methodology

Frequency is a negative binomial (NB2) with exposure as an offset; a Poisson fit is clearly worse. Severity is a truncated lognormal body with a generalised Pareto tail above the 99th percentile. The closed-form mean of that spliced distribution is checked against numerical integration and a capped-mean test in notebook 02.

The aggregate loss is simulated by Monte Carlo over annual claim counts and individual claim sizes. I report VaR, expected shortfall and K = VaR minus expected loss side by side. K is an internal capital proxy and shouldn't be read as a regulatory SCR. Reinsurance covers quota share, per-claim excess of loss and per-line stop-loss, always settled so that retained plus ceded equals gross.

For the six CAS lines, each loss ratio gets a Burr XII marginal. Dependence comes from a Gaussian copula calibrated on Spearman correlations (through 2 sin(pi rho / 6)) and a Student-t copula calibrated on Kendall's tau (through sin(pi tau / 2)). Both matrices are checked for positive definiteness; no repair was needed. Diversification is shown on both a VaR basis and a K basis, because they give noticeably different numbers.

Reserving uses Chain-Ladder, Bornhuetter-Ferguson and Mack standard errors, checked against the `chainladder` package. The lognormal quantiles in 09 are an extra approximation on top of Mack. Runoff risk and one-year risk are kept apart, since the first overstates the second.

Notebook 11 ties it together. It adds premium risk and reserve risk under an assumed dependence, bootstraps the inputs that move capital, and ranks the modelling choices in a model-risk table. Notebook 10 is separate: one Standard Formula premium and reserve component, computed from inputs I made up, as an illustration of the regulatory route.

## Key results

The motor book in EUR, one policy-year: expected loss 297.11, VaR at 99.5% of 6,247 and K of 5,950. The fitted GPD shape is 0.852, which gives a finite mean and an infinite variance. That matters for ES. For seed 42 it comes out at 45,792, but across replicates of 500k draws it ranges from 21,111 to 40,162. A single ES figure for this book would be misleading, so the notebooks quote a range.

For the six-line USD portfolio, VaR at 99.5% is 60.3m with independent lines and 62.2m under the fitted Gaussian copula. Diversification benefit falls from 37% to 35% on VaR and from 53% to 50% on K. Workers' comp takes 69% of the allocated ES. It is also the biggest modelling risk in this part: depending on how its tail is handled, portfolio VaR goes from 48.1m (with the AY2001 premium artifact removed) to 137.2m (empirical tail), against 62.2m for the Burr baseline.

Adding reserve risk to premium risk, with moderate dependence between the two, takes K from 33.9m to 34.9m. The increase is small because Mack coefficients of variation are only a few percent for the large lines. A 25% quota share on the premium block brings K down to 26.6m. This is runoff risk, so it overstates a one-year view.

The parameter bootstrap says Monte Carlo noise is about 0.8% of the integrated VaR, and that fit uncertainty pushes the 95th percentile 18% above the point estimate, almost all of it from the Workers' comp marginal. For motor, VaR hardly reacts to the choice of tail model (moving the EVT threshold changes it by -1.2%) while ES moves by +102%. In 20 of 300 resampled severity fits the shape is at or above 1, which means no finite mean; using a 98th-percentile threshold alone is enough to push it above 1.

On pricing, the NB2 model beats a constant rate out of sample. The combined ratio is 82.70% gross and 76.91% after an assumed quota share. The capital loading is just a scaling constant I picked, and moving it from 0 to 2% takes the average gross premium from EUR 404 to EUR 550. In reserving, Workers' comp unpaid is 3,268m on Chain-Ladder and 4,090m on Bornhuetter-Ferguson, with a Mack standard error of 102m (3.1%).

The Standard Formula illustration gives a premium and reserve underwriting-risk component of EUR 83.346 million for the volumes I assumed. It's one component, not a total SCR, and it shouldn't be compared with the simulated numbers above.

## Limitations

The motor severity has an infinite variance, so ES and Monte Carlo means are unstable and running more simulations does not cure the model uncertainty. Scaling exposure with a fixed NB2 dispersion is also not the same as summing independent policies.

The bootstrap covers frequency, severity, the loss-ratio marginals and the rank correlations. It does not cover Mack standard errors beyond what Mack itself gives, the copula family, or the choice of marginal family for Workers' comp; those appear in the model-risk table instead.

The CAS data are company-year loss ratios, not one insurer, and a single accident year of premium drives the Workers' comp tail. Only companies with complete filings enter the triangles (83% of companies for wkcomp, 87% for comauto). Long-tail lines are cut at lag 10, so medical malpractice, other liability and product liability reserves are probably understated.

The dependence between premium and reserve risk is assumed, because the data can't estimate it. Treaty terms, expenses, the capital loading and the 30% expense ratio in the RORAC example are assumptions too, and reinsurance premium is not modelled. Everything financial in notebook 10 is an assumption, only one component of one module is implemented, and the 2027 amendments are not applied.

There was no ALM code to bring over; that work lives in [multi-asset portfolio risk](../02_multi_asset_portfolio_risk/README.md).

## Repository structure

```
notebooks/        01-11, run in order; analysis, plots and commentary
src/              shared model code (insurance, portfolio, dependence, reinsurance, reserving,
                  pricing, solvency, integration, parameter risk, risk measures, paths and freshness checks)
tests/            unit and invariant tests (pytest)
results/          handoff files by stage (calibration, aggregate, portfolio, pricing, reserving,
                  regulatory, summary) plus run reports in validation/
data/             raw CSVs, left untouched
run_notebooks.py  runs 01-11 in fresh kernels and writes results/validation/validation.json
```

## Reproduction

From this folder:

```bash
pip install -r requirements.txt
python run_notebooks.py
pytest
```

`run_notebooks.py` clears the freshness manifest, runs the eleven notebooks in order, saves their outputs and stops at the first failure. `python run_notebooks.py 5 7` runs just notebooks 5 to 7 and keeps the manifest. A full run takes about ten minutes, most of it in notebook 11. The package versions from the last run are saved in `results/validation/validation.json`. Jupyter needs permission to open local kernel ports. If you change code or data, the saved results count as stale and the next notebook tells you what to rerun.
