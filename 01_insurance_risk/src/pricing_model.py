"""Frequency GLMs, held-out diagnostics, and annual premium calculations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

VEHAGE_CAP = 30      # modeling cap
BONUSMALUS_CAP = 150  # modeling cap

GLM_FORMULA = (
    "ClaimNb ~ C(Area) + VehPower + VehAge_capped + DrivAge + "
    "BonusMalus_capped + C(VehBrand) + C(VehGas) + np.log(Density) + C(Region)"
)


@dataclass
class PricingCleaningReport:
    n_policies: int
    n_vehage_capped: int
    vehage_max_before: int
    n_bonusmalus_capped: int
    bonusmalus_max_before: int


def clean_for_pricing(freq: pd.DataFrame) -> tuple[pd.DataFrame, PricingCleaningReport]:
    df = freq.copy()
    n_vehage_capped = int((df["VehAge"] > VEHAGE_CAP).sum())
    vehage_max_before = int(df["VehAge"].max())
    df["VehAge_capped"] = df["VehAge"].clip(upper=VEHAGE_CAP)

    n_bonusmalus_capped = int((df["BonusMalus"] > BONUSMALUS_CAP).sum())
    bonusmalus_max_before = int(df["BonusMalus"].max())
    df["BonusMalus_capped"] = df["BonusMalus"].clip(upper=BONUSMALUS_CAP)

    report = PricingCleaningReport(
        n_policies=len(df),
        n_vehage_capped=n_vehage_capped, vehage_max_before=vehage_max_before,
        n_bonusmalus_capped=n_bonusmalus_capped, bonusmalus_max_before=bonusmalus_max_before,
    )
    return df, report


def train_test_split_pricing(df: pd.DataFrame, test_frac: float = 0.25,
                              seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random policy-level split (freMTPL2 has no temporal ordering to
    respect). Fixed seed for reproducibility."""
    if not 0 < test_frac < 1 or df.IDpol.duplicated().any():
        raise ValueError("Use unique policy IDs and 0 < test_frac < 1")
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def fit_poisson_glm(train: pd.DataFrame):
    """Fit Poisson frequency with log exposure offset."""
    model = smf.glm(
        formula=GLM_FORMULA, data=train,
        family=sm.families.Poisson(),
        offset=np.log(train["Exposure"]),
    )
    return model.fit()


def fit_negbin_glm(train: pd.DataFrame, poisson_results=None):
    """Fit NB2 by IRLS with dispersion estimated from training Poisson residuals."""
    if poisson_results is None:
        poisson_results = fit_poisson_glm(train)
    mu = poisson_results.fittedvalues.to_numpy()
    y = train["ClaimNb"].to_numpy()
    aux_y = ((y - mu) ** 2 - mu) / mu
    alpha_hat = float(np.sum(aux_y * mu) / np.sum(mu ** 2))  # OLS through the origin
    alpha_hat = max(alpha_hat, 1e-6)

    nb_model = smf.glm(
        formula=GLM_FORMULA, data=train,
        family=sm.families.NegativeBinomial(alpha=alpha_hat),
        offset=np.log(train["Exposure"]),
    )
    nb_results = nb_model.fit()
    return nb_results, alpha_hat


def pearson_dispersion(glm_results) -> float:
    """Pearson chi-square / degrees of freedom -- >1 indicates overdispersion
    remaining after the covariates (Poisson assumes exactly 1)."""
    return float(glm_results.pearson_chi2 / glm_results.df_resid)


def predict_annualized_frequency_poisson(glm_results, df: pd.DataFrame) -> np.ndarray:
    """Predicted E[N] for this row's actual exposure, divided back down to a
    per-policy-year (Exposure=1) rate -- the quantity premium is built on."""
    pred_count = glm_results.predict(df, offset=np.log(df["Exposure"]))
    return (pred_count / df["Exposure"]).to_numpy()


def predict_annualized_frequency_negbin(nb_results, df: pd.DataFrame) -> np.ndarray:
    pred_count = nb_results.predict(df, offset=np.log(df["Exposure"]))
    return (pred_count / df["Exposure"]).to_numpy()


def calibration_table(df: pd.DataFrame, predicted_annualized_freq: np.ndarray,
                       n_bins: int = 10) -> pd.DataFrame:
    """Bucket policies into deciles of predicted risk, compare actual vs
    predicted claim counts and exposure-weighted rates per bucket -- the
    standard GLM lift/calibration check."""
    d = df.copy()
    d["pred_rate"] = predicted_annualized_freq
    d["pred_count"] = d["pred_rate"] * d["Exposure"]
    d["decile"] = pd.qcut(d["pred_rate"], n_bins, labels=False, duplicates="drop")

    rows = []
    for dec, g in d.groupby("decile"):
        rows.append(dict(
            decile=int(dec) + 1, n_policies=len(g), total_exposure=g["Exposure"].sum(),
            actual_claims=g["ClaimNb"].sum(), predicted_claims=g["pred_count"].sum(),
            actual_rate=g["ClaimNb"].sum() / g["Exposure"].sum(),
            predicted_rate=g["pred_count"].sum() / g["Exposure"].sum(),
        ))
    out = pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)
    out["pct_error"] = 100 * (out["predicted_rate"] / out["actual_rate"] - 1)
    return out


def gini_lift(df: pd.DataFrame, predicted_annualized_freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lorenz-curve-style ordered lift: sort by predicted risk ascending,
    cumulative share of exposure vs. cumulative share of actual claims.
    Returns (cum_exposure_share, cum_claims_share) for plotting."""
    d = df.copy()
    d["pred_rate"] = predicted_annualized_freq
    d = d.sort_values("pred_rate")
    cum_exposure = np.concatenate([[0], np.cumsum(d["Exposure"]) / d["Exposure"].sum()])
    cum_claims = np.concatenate([[0], np.cumsum(d["ClaimNb"]) / d["ClaimNb"].sum()])
    return cum_exposure, cum_claims


def pure_premium(annualized_freq: np.ndarray, severity_mean: float) -> np.ndarray:
    """Annual expected claims times portfolio-wide mean claim severity."""
    return annualized_freq * severity_mean


@dataclass
class PremiumAssumptions:
    fixed_expense: float             # EUR per policy-year, assumed (not fitted)
    variable_expense_ratio: float    # fraction of GROSS premium, assumed
    target_profit_margin: float      # fraction of GROSS premium, assumed
    capital_loading_k: float         # illustrative scaling constant for the capital loading (not estimated)


def gross_premium(pure_prem: np.ndarray, capital_loading: np.ndarray,
                   assumptions: PremiumAssumptions) -> np.ndarray:
    """Gross = (PurePremium + Fixed + CapitalLoading) / (1 - VariableExpenseRatio - ProfitMargin)."""
    denom = 1 - (assumptions.variable_expense_ratio + assumptions.target_profit_margin)
    if denom <= 0:
        raise ValueError("variable_expense_ratio + target_profit_margin must be < 1")
    return (pure_prem + assumptions.fixed_expense + capital_loading) / denom


def capital_loading_pro_rata(pure_prem: np.ndarray, portfolio_var995: float,
                          portfolio_mean: float, k: float) -> np.ndarray:
    """Allocate k * (VaR - analytical mean) / analytical mean times pure premium. Inputs here describe one policy-year, not a diversified book."""
    if portfolio_mean <= 0 or k < 0:
        raise ValueError("Positive expected loss and nonnegative loading required")
    capital_ratio = max(portfolio_var995 - portfolio_mean, 0.0) / portfolio_mean
    return k * capital_ratio * pure_prem


def loss_ratio(expected_losses: np.ndarray, premium: np.ndarray) -> np.ndarray:
    return expected_losses / premium


def combined_ratio(loss_ratio_: np.ndarray, expense_ratio: np.ndarray) -> np.ndarray:
    return loss_ratio_ + expense_ratio
