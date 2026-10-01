"""Paid-triangle Chain-Ladder, BF, and Mack runoff uncertainty."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TriangleBuildReport:
    line: str
    n_companies_total: int
    n_companies_full_square: int
    n_companies_excluded: int


def build_industry_triangle(data_dir: str | Path, line: str) -> tuple[dict, TriangleBuildReport]:
    """Aggregate complete company filing grids into paid and incurred squares."""
    path = Path(data_dir) / f"{line}.csv"
    df = pd.read_csv(path)

    full_codes = full_square_codes(df)
    n_total = df["GRCODE"].nunique()
    n_full = len(full_codes)

    sub = df[df["GRCODE"].isin(full_codes)]

    cum_paid = sub.groupby(["AccidentYear", "DevelopmentLag"])["CumPaidLoss"].sum().unstack()
    cum_incurred = sub.groupby(["AccidentYear", "DevelopmentLag"])["IncurredLosses"].sum().unstack()
    earned_prem = sub[sub["DevelopmentLag"] == 1].groupby("AccidentYear")["EarnedPremNet"].sum()

    cum_paid.columns.name = "DevelopmentLag"
    cum_incurred.columns.name = "DevelopmentLag"

    triangle = dict(cum_paid=cum_paid, cum_incurred=cum_incurred, earned_prem=earned_prem)
    report = TriangleBuildReport(
        line=line, n_companies_total=n_total, n_companies_full_square=n_full,
        n_companies_excluded=n_total - n_full,
    )
    return triangle, report


def selection_summary(data_dir: str | Path, line: str) -> dict:
    """Counts and shares retained by the full-square filter, so the selection effect is visible."""
    df = pd.read_csv(Path(data_dir) / f"{line}.csv")
    keep = set(full_square_codes(df))
    first = df[df["DevelopmentLag"] == 1]
    prem_all = float(first["EarnedPremNet"].sum())
    prem_kept = float(first.loc[first["GRCODE"].isin(keep), "EarnedPremNet"].sum())
    n_total = int(df["GRCODE"].nunique())
    return dict(line=line, companies_total=n_total, companies_retained=len(keep),
                companies_retained_pct=100 * len(keep) / n_total,
                earned_premium_retained_pct=100 * prem_kept / prem_all)


def mask_to_valuation(cum_full: pd.DataFrame, valuation_year: int) -> pd.DataFrame:
    """Keep cells with accident year + development lag - 1 <= valuation year."""
    masked = cum_full.copy()
    for ay in masked.index:
        max_known_lag = valuation_year - ay + 1
        drop_cols = [c for c in masked.columns if c > max_known_lag]
        masked.loc[ay, drop_cols] = np.nan
    return masked


def latest_diagonal(cum_triangle: pd.DataFrame) -> pd.Series:
    """For each AY, the most recent (highest dev lag) non-NaN cumulative
    value -- the 'latest known' amount Chain-Ladder/BF both project from."""
    out = {}
    for ay in cum_triangle.index:
        row = cum_triangle.loc[ay].dropna()
        if len(row) == 0:
            continue
        out[ay] = row.iloc[-1]
    return pd.Series(out, name="latest_cumulative")


def latest_lag(cum_triangle: pd.DataFrame) -> pd.Series:
    """For each AY, the dev lag of the latest known diagonal cell."""
    out = {}
    for ay in cum_triangle.index:
        row = cum_triangle.loc[ay].dropna()
        if len(row) == 0:
            continue
        out[ay] = row.index[-1]
    return pd.Series(out, name="latest_lag")


@dataclass
class TriangleValidationReport:
    n_accident_years: int
    n_negative_incremental: int
    n_nonmonotonic_ay: int
    cumulative_matches_sum_of_incremental: bool


def validate_triangle(cum_triangle: pd.DataFrame) -> TriangleValidationReport:
    """Check negative increments and cumulative/incremental reconstruction."""
    incr = cum_triangle.diff(axis=1)
    first_col = cum_triangle.columns.min()
    incr[first_col] = cum_triangle[first_col]

    negative_mask = incr < 0
    n_negative = int(negative_mask.sum().sum())
    n_nonmono = int((negative_mask.sum(axis=1) > 0).sum())

    rebuilt = incr.cumsum(axis=1)
    diff = (rebuilt - cum_triangle).abs()
    matches = bool((diff.fillna(0) < 1e-6).all().all())

    return TriangleValidationReport(
        n_accident_years=cum_triangle.shape[0],
        n_negative_incremental=n_negative,
        n_nonmonotonic_ay=n_nonmono,
        cumulative_matches_sum_of_incremental=matches,
    )


@dataclass
class ChainLadderResult:
    age_to_age_factors: pd.Series      # index = dev lag j, value = f_j (j -> j+1)
    cdf_to_ultimate: pd.Series         # index = dev lag j, value = product of f_j..f_{n-1}
    ultimate: pd.Series                # index = AY
    latest_diagonal: pd.Series         # index = AY
    unpaid_reserve: pd.Series          # index = AY, = ultimate - latest_diagonal
    latest_development: pd.Series


def age_to_age_factors(cum_triangle: pd.DataFrame) -> pd.Series:
    """Estimate sum(next paid) / sum(current paid) over observed adjacent pairs."""
    lags = sorted(cum_triangle.columns)
    factors = {}
    for j, j1 in zip(lags[:-1], lags[1:]):
        col_j = cum_triangle[j]
        col_j1 = cum_triangle[j1]
        mask = col_j.notna() & col_j1.notna()
        if mask.sum() == 0 or col_j[mask].sum() <= 0:
            raise ValueError(f"No positive observed pairs for lag {j}->{j1}; choose a later valuation or a shorter explicit target.")
        f = col_j1[mask].sum() / col_j[mask].sum()
        factors[j] = f
    return pd.Series(factors, name="age_to_age_factor")


def cdf_to_ultimate(factors: pd.Series, max_lag: int, tail_factor: float = 1.0) -> pd.Series:
    """Multiply remaining development factors and the stated terminal tail factor."""
    lags = sorted(factors.index) + [max_lag]
    cdf = {}
    running = tail_factor
    cdf[max_lag] = running
    for j in sorted(factors.index, reverse=True):
        running = factors[j] * running
        cdf[j] = running
    return pd.Series(cdf, name="cdf_to_ultimate").sort_index()


def assess_tail_factor(factors: pd.Series) -> dict:
    """Report the last three observed development factors."""
    last_factors = factors.sort_index().tail(3)
    last_factor_value = float(factors.sort_index().iloc[-1])
    return dict(
        last_three_factors=last_factors.to_dict(),
        last_factor=last_factor_value,
        pct_above_unity=100 * (last_factor_value - 1),
    )


def chain_ladder(cum_triangle: pd.DataFrame, tail_factor: float = 1.0) -> ChainLadderResult:
    factors = age_to_age_factors(cum_triangle)
    max_lag = max(cum_triangle.columns)
    cdf = cdf_to_ultimate(factors, max_lag, tail_factor=tail_factor)

    latest = latest_diagonal(cum_triangle)
    lag_of_latest = latest_lag(cum_triangle)

    ultimate = {}
    for ay in latest.index:
        lag = lag_of_latest[ay]
        factor = cdf.get(lag, 1.0)
        ultimate[ay] = latest[ay] * factor
    ultimate = pd.Series(ultimate, name="ultimate")

    unpaid = ultimate - latest
    unpaid.name = "unpaid_reserve"

    return ChainLadderResult(
        age_to_age_factors=factors, cdf_to_ultimate=cdf, ultimate=ultimate,
        latest_diagonal=latest, unpaid_reserve=unpaid, latest_development=lag_of_latest,
    )


@dataclass
class BFResult:
    a_priori_elr: float
    expected_ultimate: pd.Series
    pct_unpaid: pd.Series
    ultimate: pd.Series
    unpaid_reserve: pd.Series


def a_priori_elr_from_mature_years(cl_result: ChainLadderResult, earned_prem: pd.Series,
                                    maturity_cdf_threshold: float = 1.05) -> float:
    """Average CL-implied loss ratios for years with remaining factor at most the threshold."""
    lag_of_latest = latest_lag_from_ultimate_calc(cl_result)
    mature_ays = [ay for ay, lag in lag_of_latest.items()
                  if cl_result.cdf_to_ultimate.get(lag, 1.0) <= maturity_cdf_threshold]
    if not mature_ays:
        raise ValueError("No mature years for an internal ELR; supply an explicit prior")
    loss_ratios = cl_result.ultimate[mature_ays] / earned_prem[mature_ays]
    return float(loss_ratios.mean())


def latest_lag_from_ultimate_calc(cl_result: ChainLadderResult) -> pd.Series:
    """Return the observed latest lag retained in the Chain-Ladder result."""
    return cl_result.latest_development.copy()


def bornhuetter_ferguson(cl_result: ChainLadderResult, earned_prem: pd.Series,
                          a_priori_elr: float) -> BFResult:
    """Add prior expected ultimate times the unpaid fraction to paid to date."""
    lag_of_latest = latest_lag_from_ultimate_calc(cl_result)
    expected_ultimate = earned_prem.reindex(cl_result.latest_diagonal.index) * a_priori_elr

    pct_unpaid = {}
    for ay, lag in lag_of_latest.items():
        cdf = cl_result.cdf_to_ultimate.get(lag, 1.0)
        pct_unpaid[ay] = 1 - 1 / cdf
    pct_unpaid = pd.Series(pct_unpaid, name="pct_unpaid")

    bf_ultimate = cl_result.latest_diagonal + expected_ultimate * pct_unpaid
    bf_ultimate.name = "ultimate"
    bf_unpaid = bf_ultimate - cl_result.latest_diagonal
    bf_unpaid.name = "unpaid_reserve"

    return BFResult(
        a_priori_elr=a_priori_elr, expected_ultimate=expected_ultimate,
        pct_unpaid=pct_unpaid, ultimate=bf_ultimate, unpaid_reserve=bf_unpaid,
    )


@dataclass
class MackResult:
    factors: pd.Series                 # f_j
    sigma_sq: pd.Series                 # sigma_j^2, index = dev lag j (extrapolated for the last one)
    ultimate: pd.Series                 # index = AY, same as chain_ladder's point estimate
    reserve: pd.Series                  # index = AY, = ultimate - latest_diagonal
    process_variance: pd.Series         # index = AY
    parameter_variance: pd.Series       # index = AY
    mse: pd.Series                      # index = AY, = process_variance + parameter_variance
    std_error: pd.Series                # index = AY, = sqrt(mse)
    cv: pd.Series                       # index = AY, = std_error / reserve
    total_reserve: float
    total_process_variance: float
    total_parameter_variance: float     # includes the cross-accident-year covariance term
    total_mse: float
    total_std_error: float
    total_cv: float


def _column_sum_at_lag(cum_triangle: pd.DataFrame, lag: int, lag_next: int) -> float:
    """S_j: sum of C_{i,j} over exactly the accident years used to estimate f_j
    (i.e. those with both lag j and lag j+1 known) -- the same denominator
    used inside age_to_age_factors, recovered here for Mack's variance
    formula."""
    col_j = cum_triangle[lag]
    col_j1 = cum_triangle[lag_next]
    mask = col_j.notna() & col_j1.notna()
    return float(col_j[mask].sum())


def _mack_sigma_sq(cum_triangle: pd.DataFrame, factors: pd.Series) -> pd.Series:
    """Estimate Mack development variance; extrapolate the last single-pair variance using the classic minimum rule."""
    lags = sorted(factors.index)
    sigma_sq = {}
    for j in lags:
        j1 = j + 1
        col_j = cum_triangle[j]
        col_j1 = cum_triangle[j1]
        mask = col_j.notna() & col_j1.notna()
        n_j = int(mask.sum())
        if n_j <= 1:
            continue  # extrapolated below
        f_j = factors[j]
        resid_sq = col_j[mask] * (col_j1[mask] / col_j[mask] - f_j) ** 2
        sigma_sq[j] = float(resid_sq.sum() / (n_j - 1))

    last_lag = lags[-1]
    if last_lag not in sigma_sq:
        estimated_lags = sorted(sigma_sq.keys())
        if len(estimated_lags) >= 2:
            s1, s2 = sigma_sq[estimated_lags[-1]], sigma_sq[estimated_lags[-2]]
            sigma_sq[last_lag] = min(s1 ** 2 / s2 if s2 > 0 else s1, min(s1, s2))
        elif len(estimated_lags) == 1:
            sigma_sq[last_lag] = sigma_sq[estimated_lags[-1]]
        else:
            sigma_sq[last_lag] = 0.0
    return pd.Series(sigma_sq).sort_index()


def mack_chain_ladder(cum_triangle: pd.DataFrame) -> MackResult:
    """Compute Mack process and parameter prediction variance, including cross-year shared-factor covariance. Tail factor is one."""
    if (cum_triangle.stack() <= 0).any():
        raise ValueError("Mack requires strictly positive observed cumulative claims")
    factors = age_to_age_factors(cum_triangle)
    sigma_sq = _mack_sigma_sq(cum_triangle, factors)
    lags = sorted(factors.index)
    max_lag = max(cum_triangle.columns)

    cl_result = chain_ladder(cum_triangle)
    ultimate = cl_result.ultimate
    latest = cl_result.latest_diagonal
    lag_of_latest = latest_lag(cum_triangle)
    reserve = cl_result.unpaid_reserve

    s_j = {j: _column_sum_at_lag(cum_triangle, j, j + 1) for j in lags}

    process_var = {}
    param_var = {}
    for ay in latest.index:
        c_i = lag_of_latest[ay]
        relevant_lags = [j for j in lags if j >= c_i]
        proc_sum = 0.0
        param_sum = 0.0
        c_ij = latest[ay]  # C_{i, current known lag}; walk forward lag by lag
        running_c = c_ij
        for j in relevant_lags:
            f_j = factors[j]
            term = sigma_sq[j] / f_j ** 2
            proc_sum += term / running_c
            param_sum += term / s_j[j]
            running_c = running_c * f_j  # projected C_{i,j+1} for the next term's process denominator
        u_i = ultimate[ay]
        process_var[ay] = u_i ** 2 * proc_sum
        param_var[ay] = u_i ** 2 * param_sum

    process_var = pd.Series(process_var, name="process_variance")
    param_var = pd.Series(param_var, name="parameter_variance")
    mse = process_var + param_var
    std_error = np.sqrt(mse)
    std_error.name = "std_error"
    cv = std_error / reserve.replace(0, np.nan)
    cv.name = "cv"

    ays = list(latest.index)
    cross_term = 0.0
    for idx_i in range(len(ays)):
        for idx_k in range(idx_i + 1, len(ays)):
            ay_i, ay_k = ays[idx_i], ays[idx_k]
            c_i_lag, c_k_lag = lag_of_latest[ay_i], lag_of_latest[ay_k]
            start_lag = max(c_i_lag, c_k_lag)  # shared uncertainty starts after both years' observed development
            shared_lags = [j for j in lags if j >= start_lag]
            pair_sum = sum(sigma_sq[j] / factors[j] ** 2 / s_j[j] for j in shared_lags)
            cross_term += 2 * ultimate[ay_i] * ultimate[ay_k] * pair_sum

    total_reserve = float(reserve.sum())
    total_process_variance = float(process_var.sum())
    total_parameter_variance = float(param_var.sum() + cross_term)
    total_mse = total_process_variance + total_parameter_variance
    total_std_error = float(np.sqrt(total_mse))
    total_cv = total_std_error / total_reserve if total_reserve != 0 else float("nan")

    return MackResult(
        factors=factors, sigma_sq=sigma_sq, ultimate=ultimate, reserve=reserve,
        process_variance=process_var, parameter_variance=param_var, mse=mse,
        std_error=std_error, cv=cv,
        total_reserve=total_reserve, total_process_variance=total_process_variance,
        total_parameter_variance=total_parameter_variance, total_mse=total_mse,
        total_std_error=total_std_error, total_cv=total_cv,
    )


def lognormal_prediction_interval(mean: float, std_error: float,
                                   quantiles=(0.75, 0.90, 0.95, 0.995)) -> dict:
    """Moment-match a lognormal to positive reserve mean and Mack standard error. Returned quantiles assume this distribution."""
    if mean <= 0:
        raise ValueError("lognormal moment-matching requires a positive mean reserve")
    if not np.isfinite([mean, std_error]).all() or std_error < 0:
        raise ValueError("Finite mean and nonnegative standard error required")
    if std_error == 0:
        return {"mu": float(np.log(mean)), "sigma": 0.0, **{f"q{q}":float(mean) for q in quantiles}}
    cv_sq = (std_error / mean) ** 2
    sigma_sq_ln = np.log(1 + cv_sq)
    sigma_ln = np.sqrt(sigma_sq_ln)
    mu_ln = np.log(mean) - sigma_sq_ln / 2
    from scipy import stats as _stats
    out = {"mu": float(mu_ln), "sigma": float(sigma_ln)}
    for q in quantiles:
        out[f"q{q}"] = float(_stats.lognorm.ppf(q, s=sigma_ln, scale=np.exp(mu_ln)))
    return out


def backtest_ultimate(cum_full: pd.DataFrame, valuation_year: int, tail_factor: float = 1.0):
    """Compare predictions with final observed development for the same accident years."""
    masked = mask_to_valuation(cum_full, valuation_year)
    cl = chain_ladder(masked, tail_factor=tail_factor)
    max_lag = max(cum_full.columns)
    actual_ultimate = cum_full[max_lag]
    comparison = pd.DataFrame({
        "projected_ultimate": cl.ultimate,
        "actual_ultimate": actual_ultimate,
    })
    comparison["error"] = comparison["projected_ultimate"] - comparison["actual_ultimate"]
    comparison["pct_error"] = 100 * comparison["error"] / comparison["actual_ultimate"]
    return cl, comparison


def full_square_codes(df: pd.DataFrame) -> pd.Index:
    """Select complete AY x lag grids; row counts alone do not detect duplicates."""
    keys = ["GRCODE", "AccidentYear", "DevelopmentLag"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate company / accident-year / development-lag rows")
    if not (df.DevelopmentYear == df.AccidentYear + df.DevelopmentLag - 1).all():
        raise ValueError("DevelopmentYear does not equal AccidentYear + DevelopmentLag - 1")
    years, lags = set(df.AccidentYear.unique()), set(df.DevelopmentLag.unique())
    expected = {(y, lag) for y in years for lag in lags}
    valid = []
    for code, group in df.groupby("GRCODE"):
        if set(zip(group.AccidentYear, group.DevelopmentLag)) == expected:
            if group.groupby("AccidentYear").EarnedPremNet.nunique().max() != 1:
                raise ValueError(f"Earned premium changes across lags for company {code}")
            valid.append(code)
    return pd.Index(valid)


def package_triangle(cumulative):
    """Convert only observed cells to the independent chainladder package format."""
    import chainladder as cl
    rows = [{"origin": pd.Timestamp(int(ay), 1, 1),
             "valuation": pd.Timestamp(int(ay + lag - 1), 12, 31), "paid": value}
            for ay, row in cumulative.iterrows() for lag, value in row.items()
            if pd.notna(value)]
    return cl.Triangle(pd.DataFrame(rows), origin="origin", development="valuation",
                       columns="paid", cumulative=True)
