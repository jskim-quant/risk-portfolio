# Audit & Validation Report — 02 Multi-Asset Portfolio Risk

Status: **Phase 1 (critical validation) complete** · audit date 2026-10-24 · branch `feat/multi-asset-portfolio-risk`
Baseline commit before any change: `e84c733` (notebook 08).

Each claim below is labelled **[verified]** (recomputed during this audit), **[fixed]** (code
changed), **[open]** (known issue, scheduled for a later phase) or **[assumption]**.

---

## 1. Repository audit (before changes)

| Item | Finding |
|---|---|
| Structure | `src/` 9 modules (data_loader, returns, universe, risk, evaluate, optimize, backtest, var, stress) · notebooks 01–08 executed, 09 skeleton · `scripts/` download + validation · `tests/` 6 files |
| Test suite | **29 passed, 0 failed, 8.1 s** (pytest 9.1, numpy 2.2.6, pandas 2.3.3); no warnings with `-W default` [verified] |
| Dependencies | `requirements.txt` listed cvxpy, arch, scipy, scikit-learn, nbformat/nbclient/ipykernel — **none are imported**; it also pinned numpy ≥ 2.4 / pandas ≥ 3.0, although the code runs on 2.2 / 2.3 [fixed: trimmed to what is used] |
| Notebook execution | no Jupyter kernel on the dev machine → added `scripts/run_notebook.py` (stdlib in-process executor) [fixed] |
| Reported results | notebook 08 re-runs every strategy from the cached weights and matches `backtest_daily_net.csv` to 3e-16 [verified] |
| Docs vs code | README "Answers so far" consistent with notebook outputs except the VaR claim (see §5) and the 252-day annualisation (see §2) |

### Bias checks

| Risk | Check | Result |
|---|---|---|
| Look-ahead (covariance) | `test_forecasts_do_not_look_ahead`: forecasts up to 2015-11 identical with data cut at 2015-12 | pass [verified] |
| Look-ahead (execution) | `test_weights_take_effect_after_decision_date`: decision-day return not earned by new weights | pass [verified] |
| Look-ahead (VaR) | `var_forecasts` uses `h[:i0]` (strictly before the first day of the month) | pass [verified, code read] |
| Survivorship / selection | universe = 10 ETFs alive today, chosen ex post; no delisted funds. Low survivorship risk (all are broad, liquid index funds), but **selection bias** remains: GLD and DBC were chosen knowing the full history | **[open]** — disclose; no cure without a point-in-time ETF universe |
| Hyper-parameters | EWMA half-life 26w, window 156w, cap 30 %, γ = 5, β = 0.95 fixed in the design doc before backtests; not tuned on OOS — cannot be proven ex post | [assumption] |
| Data leakage via calendar | holiday NaN → 0 return: daily vs. weekly compounding of **unhedged** series agree to 1e-13 over 2010–2026, so no return is lost (the move is booked the next open day) | pass [verified] |

### Data-frequency consistency (new finding)

Compounding daily vs. weekly vs. monthly returns over the same window:

| Series | daily / weekly − 1 | daily / monthly − 1 |
|---|---|---|
| unhedged (SPY, VGK, EWJ, GLD, DBC, EUR synthetic) | < 1e-13 | < 1e-13 |
| FX-hedged (SHY, IEF, TLT, LQD, HYG, JGB synthetic) | −0.5 % … +0.7 % | −1.75 % … +1.1 % (16 y) |

The hedged series differ because the CIP hedge is **re-struck every period**: the daily
backtest implicitly rolls the FX hedge every day at zero cost, while the weekly/monthly series
roll it weekly/monthly. The effect is ≈ 0.1 % p.a. at most (TLT). It is not an error in the
formula. It is an implicit, unrealistic hedge policy **[open → Phase 4: monthly hedge roll +
hedge transaction costs]**.

---

## 2. Backtest accounting (Phase 1)

### Original engine — what it did

* return on day d: `g = w·r + (1 − Σw)·r_cash`; net = `(1+g)(1−TC) − 1`; drift `w ← w(1+r)/(1+g)`
* borrowing (Σw > 1) paid the EUR O/N rate **with no spread**; the leverage cap applied at rebalance only
* initial build from cash **free** (`charge_initial=False`), turnover two-way Σ|Δw| excluding the build
* no explicit NAV ledger, no ruin guard (1+g ≤ 0 would have produced inf/NaN weights)
* `perf_stats` annualised with **252** days and measured drawdown from the first close

The return algebra was correct [verified]: the Memmel variance formula matches Memmel (2003),
`2 − 2ρ + ½(SR_a² + SR_b² − 2SR_aSR_bρ²)`, and the drift identity is exact.

### Changes [fixed]

| Change | Backward compatible? |
|---|---|
| Explicit daily NAV ledger: `nav_open, tc, pnl_assets, pnl_cash, funding_cost, nav_close, gross_exposure, net_exposure, borrowed` | yes — new field `BacktestResult.ledger` |
| `funding_spread_bps`: spread over O/N, ACT/360 over calendar days, **only on borrowed cash**; positive cash earns O/N without spread | yes — default 0 reproduces the old numbers exactly |
| `cost_bps` accepts a per-asset mapping | yes — scalar still works |
| initial build turnover/cost reported separately (`initial_turnover`, `initial_cost`); turnover convention unchanged | yes |
| limited-liability floor: NAV ≥ 0, strategy stays in cash after ruin, `ruined` timestamp | yes — never triggered in the real backtests (min NAV 0.40, MinVar* at 200 bp spread) |
| `perf_stats`: observed periods per year (≈ 260.3 for the US∪EU∪JP weekday calendar); drawdown from initial NAV 1; `periods/yr` reported | **numbers change**: Sharpe and vol × 1.016, CAGR unchanged |

Accounting identity enforced by the tests (see `src/backtest.py` docstring):
`NAV_close = NAV_open − TC + PnL_assets + PnL_cash − Funding`, with
`gross = NAV_close/NAV_inv − 1` (after funding, before TC) and `net = NAV_close/NAV_open − 1`.

### Tests [added: `tests/test_backtest_accounting.py`, 13 tests]

unlevered closed form · levered (rate + spread on borrowed only) · positive cash (no spread) ·
spread accrual over weekend gaps · s = 0 equals original formula on random paths ·
rebalancing after TC + charge_initial · asset-specific costs · large drawdown and +100 % day
with exact drift · ruin (3× leverage, −40 % day) · zero returns · missing trading days ·
annualisation / initial drawdown · regression vs. stored `backtest_daily_net.csv` (60/40, RP*).

**Suite after Phase 1: 42 passed, 0 failed, 7.1 s.**

---

## 3. Phase 1 results (notebook `05b_frictions_sensitivity.ipynb`, tables in `results/frictions/`)

All cost and spread levels are **scenario assumptions**, not observed costs.

* **Baseline change.** Sharpe +1.5 % for all strategies (60/40 0.789 → 0.801), initial build
  10–50 bp once. No ranking or significance conclusion changes.
* **Transaction costs 0–50 bp.** 60/40 is first at every level. At 0 bp the best alternative is
  RP* at 0.60, so there is no gross edge for costs to remove. ΔSharpe per 10 bp ∝ turnover
  (60/40 −0.003 … MinVar* −0.088). MinVar/MinVar* Sharpe < 0 above ≈ 25 bp.
* **Funding spread × leverage cap.** With s = 0, static caps of 1–3 leave Sharpe unchanged
  (leverage scales risk only). RP* (cap 5) Sharpe 0.54 → 0.38 at 50 bp. Break-even spread
  against the unlevered version: RP* ≈ 105–120 bp, CVaR* ≈ 50–70 bp, MinVar* ≈ 10–20 bp.
  **MinVar\* and CVaR\* only look viable with near-zero financing spreads.**
* **Leverage drift.** With a cap of 5, gross exposure reached 6.1–6.3× between monthly
  rebalances. Caps are not enforced intra-month **[open → document; daily cap check optional]**.

---

## 4. Remaining limitations (scheduled)

| # | Issue | Phase |
|---|---|---|
| L1 | Memmel test assumes iid normal returns → block-bootstrap ΔSR intervals, Holm / SPA for multiple strategies | 2 |
| L2 | Condition numbers, eigenvalue paths, perturbation → weight instability → turnover chain | 2 |
| L3 | Regularised MinVar (L1 turnover, L2) with nested walk-forward λ selection | 3 |
| L4 | Daily hedge roll at zero cost; no hedge transaction costs; calendar alignment sensitivity | 4 |
| L5 | Rebalanced vs buy-and-hold stress; GFC replay is in-sample for estimators | 4 |
| L6 | Leverage cap only at rebalance dates | 4 (document / optional) |
| L7 | ETF universe selected ex post (selection bias) | disclose |
| L8 | Notebook 05 stored series keep the legacy convention (free initial build); notebooks 05b and later charge it | — |

## 5. Corrections to earlier claims

* **VaR (README answer 5):** FHS was described as "passing at 99 %". It passes Kupiec at 99 %,
  but fails Christoffersen independence for 60/40 (p = 0.02) and EW (p = 0.04). Corrected in
  notebook 08 and the README on 2026-10-23.
* **Sharpe ratios** in README/notebooks 05 and 08 were annualised with 252 days. They have been
  updated to the observed frequency (e.g. 60/40 0.79 → 0.80).
