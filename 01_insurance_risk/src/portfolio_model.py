"""Company-year loss-ratio marginal models for illustrative CAS portfolios."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from risk_measures import var, es, portfolio_summary  # noqa: F401  (shared definitions)

LINES = ["ppauto", "comauto", "wkcomp", "othliab", "medmal", "prodliab"]
MIN_PREMIUM = 1000  # $000s -> $1M credibility floor


@dataclass
class LineCleaningReport:
    line: str
    n_companies_raw: int
    n_companies_full_square: int
    n_ultimate_rows: int
    n_nonpositive_premium: int
    n_after_credibility_filter: int
    n_lr_nonpositive: int
    lr_nonpositive_min: float


def load_and_clean_line(line: str, data_dir: str | Path | None = None, min_premium: float = MIN_PREMIUM):
    """Select complete company grids and premium-credible lag-10 loss-ratio observations."""
    if data_dir is None:
        from project import DATA_DIR
        data_dir = DATA_DIR
    df = pd.read_csv(Path(data_dir) / f"{line}.csv")
    n_companies_raw = df["GRCODE"].nunique()

    from reserving_model import full_square_codes
    full_square_codes = full_square_codes(df)
    df_full = df[df["GRCODE"].isin(full_square_codes)]
    ultimate = df_full[df_full["DevelopmentLag"] == 10].copy()
    ultimate["LossRatio"] = ultimate["IncurredLosses"] / ultimate["EarnedPremNet"]

    n_nonpositive_premium = int((ultimate["EarnedPremNet"] <= 0).sum())
    cred = ultimate[ultimate["EarnedPremNet"] >= min_premium].copy()

    n_lr_nonpositive = int((cred["LossRatio"] <= 0).sum())
    lr_min = float(cred["LossRatio"].min()) if len(cred) else float("nan")

    report = LineCleaningReport(
        line=line, n_companies_raw=int(n_companies_raw),
        n_companies_full_square=len(full_square_codes),
        n_ultimate_rows=len(ultimate), n_nonpositive_premium=n_nonpositive_premium,
        n_after_credibility_filter=len(cred), n_lr_nonpositive=n_lr_nonpositive,
        lr_nonpositive_min=lr_min,
    )
    return cred, report


@dataclass
class LineFit:
    line: str
    book_premium: float
    n_fit: int
    ln_mu: float
    ln_sigma: float
    ln_aic: float
    burr_c: float
    burr_k: float
    burr_scale: float
    burr_aic: float
    chosen: str   # "lognormal" or "burr12" -- whichever wins on AIC


def fit_line_marginal(cred: pd.DataFrame) -> LineFit:
    """Fit lognormal and fixed-location Burr XII to positive loss ratios; select by AIC."""
    lr = cred.loc[cred["LossRatio"] > 0, "LossRatio"].to_numpy()
    n = len(lr)
    book_premium = float(cred["EarnedPremNet"].median())

    log_lr = np.log(lr)
    mu, sigma = log_lr.mean(), log_lr.std()
    ll_ln = stats.lognorm.logpdf(lr, sigma, scale=np.exp(mu)).sum()
    aic_ln = 2 * 2 - 2 * ll_ln

    c_b, k_b, _, scale_b = stats.burr12.fit(lr, floc=0)
    ll_burr = stats.burr12.logpdf(lr, c_b, k_b, loc=0, scale=scale_b).sum()
    aic_burr = 2 * 3 - 2 * ll_burr  # location is fixed, leaving three fitted parameters

    chosen = "burr12" if aic_burr < aic_ln else "lognormal"

    return LineFit(
        line="", book_premium=book_premium, n_fit=n,
        ln_mu=float(mu), ln_sigma=float(sigma), ln_aic=float(aic_ln),
        burr_c=float(c_b), burr_k=float(k_b), burr_scale=float(scale_b), burr_aic=float(aic_burr),
        chosen=chosen,
    )


def simulate_line(fit: LineFit, n_sim: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate company-year loss ratios and scale by the illustrative book premium."""
    if fit.chosen == "burr12":
        lr_sim = stats.burr12.rvs(fit.burr_c, fit.burr_k, loc=0, scale=fit.burr_scale,
                                   size=n_sim, random_state=rng)
    else:
        lr_sim = rng.lognormal(mean=fit.ln_mu, sigma=fit.ln_sigma, size=n_sim)
    return lr_sim * fit.book_premium


def uniform_from_line(fit: LineFit, lr_values: np.ndarray) -> np.ndarray:
    """Map observed/simulated loss RATIOS (not dollar losses) to uniform(0,1)
    via this line's fitted marginal CDF -- used by the copula machinery in
    Notebook 06."""
    if fit.chosen == "burr12":
        return stats.burr12.cdf(lr_values, fit.burr_c, fit.burr_k, loc=0, scale=fit.burr_scale)
    return stats.lognorm.cdf(lr_values, fit.ln_sigma, scale=np.exp(fit.ln_mu))


def quantile_from_line(fit: LineFit, u: np.ndarray) -> np.ndarray:
    """Inverse of uniform_from_line: uniform(0,1) -> loss ratio."""
    if fit.chosen == "burr12":
        return stats.burr12.ppf(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps), fit.burr_c, fit.burr_k, loc=0, scale=fit.burr_scale)
    return stats.lognorm.ppf(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps), fit.ln_sigma, scale=np.exp(fit.ln_mu))


@dataclass
class LossRatioMarginal:
    """A fitted or empirical loss-ratio marginal: quantile and CDF on the loss-ratio scale."""
    name: str
    book_premium: float
    ppf: Callable[[np.ndarray], np.ndarray]
    cdf: Callable[[np.ndarray], np.ndarray]
    info: dict = field(default_factory=dict)

    def losses(self, u: np.ndarray) -> np.ndarray:
        """Loss in USD thousands for uniform draws u."""
        return self.ppf(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps)) * self.book_premium


def marginal_from_fit(fit: LineFit, family: str | None = None) -> LossRatioMarginal:
    """Parametric marginal of a LineFit. family defaults to the AIC-selected one."""
    family = family or fit.chosen
    if family == "burr12":
        dist = stats.burr12(fit.burr_c, fit.burr_k, loc=0, scale=fit.burr_scale)
    elif family == "lognormal":
        dist = stats.lognorm(fit.ln_sigma, scale=np.exp(fit.ln_mu))
    else:
        raise ValueError(f"unknown family {family}")
    return LossRatioMarginal(family, fit.book_premium, dist.ppf, dist.cdf)


def empirical_marginal(lr_values: np.ndarray, book_premium: float) -> LossRatioMarginal:
    """Empirical distribution of the observed loss ratios (linear interpolation between order statistics).

    Never produces a loss ratio above the largest observation.
    """
    x = np.sort(np.asarray(lr_values, dtype=float))
    n = len(x)
    grid = (np.arange(1, n + 1) - 0.5) / n
    ppf = lambda u: np.interp(u, grid, x)
    cdf = lambda v: np.interp(v, x, grid, left=0.0, right=1.0)
    return LossRatioMarginal("empirical", book_premium, ppf, cdf, dict(n=n, max=float(x[-1])))


def empirical_gpd_marginal(lr_values: np.ndarray, book_premium: float, tail_pct: float = 0.90) -> LossRatioMarginal:
    """Empirical body below the tail_pct quantile, GPD fitted to the exceedances above it.

    A semi-parametric alternative that can extrapolate beyond the largest observation.
    """
    x = np.sort(np.asarray(lr_values, dtype=float))
    n = len(x)
    base = empirical_marginal(x, book_premium)
    u_thr = float(np.quantile(x, tail_pct))
    exc = x[x > u_thr] - u_thr
    xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    p_thr = float(base.cdf(np.array([u_thr]))[0])
    gpd = stats.genpareto(xi, scale=sigma)

    def ppf(u):
        u = np.asarray(u, dtype=float)
        tail_u = np.clip((u - p_thr) / (1 - p_thr), 0.0, 1 - 1e-12)
        return np.where(u <= p_thr, base.ppf(u), u_thr + gpd.ppf(tail_u))

    def cdf(v):
        v = np.asarray(v, dtype=float)
        return np.where(v <= u_thr, base.cdf(v), p_thr + (1 - p_thr) * gpd.cdf(v - u_thr))

    return LossRatioMarginal("empirical+GPD tail", book_premium, ppf, cdf,
                             dict(n=n, threshold=u_thr, n_exceed=int(len(exc)), xi=float(xi), sigma=float(sigma),
                                  tail_pct=tail_pct))


def marginal_alternatives(cred: pd.DataFrame, base_fit: LineFit, exclude_year: int = 2001):
    """Alternative marginals for one line, all on the base premium scale.

    Returns (dict name -> LossRatioMarginal, refit LineFit without `exclude_year`).
    The exclusion is a data-quality sensitivity (accident-year premium data that look unreliable),
    not a recommended replacement of the baseline.
    """
    positive = cred.loc[cred["LossRatio"] > 0, "LossRatio"].to_numpy()
    kept = cred[cred["AccidentYear"] != exclude_year]
    refit = fit_line_marginal(kept)
    refit.line = base_fit.line
    refit.book_premium = base_fit.book_premium
    lr_kept = kept.loc[kept["LossRatio"] > 0, "LossRatio"].to_numpy()
    alts = {
        "Burr XII (baseline)": marginal_from_fit(base_fit, "burr12"),
        "Lognormal": marginal_from_fit(base_fit, "lognormal"),
        "Empirical (all years)": empirical_marginal(positive, base_fit.book_premium),
        f"Burr XII, AY{exclude_year} excluded": marginal_from_fit(refit, "burr12"),
        f"Empirical, AY{exclude_year} excluded": empirical_marginal(lr_kept, base_fit.book_premium),
    }
    return alts, refit


@dataclass
class PortfolioSample:
    per_line: dict      # line -> loss sample, USD thousands
    total: np.ndarray


def independent_portfolio(fits: dict, n_sim: int, seed_base: int = 1000, overrides: dict | None = None) -> PortfolioSample:
    """Lines simulated independently, each with its own seed (seed_base + position in LINES).

    overrides maps a line to a LossRatioMarginal that replaces the fitted marginal for that line
    (inverse-transform sampling with the same seed).
    """
    overrides = overrides or {}
    per_line = {}
    for i, line in enumerate(LINES):
        rng = np.random.default_rng(seed_base + i)
        if line in overrides:
            per_line[line] = overrides[line].losses(rng.uniform(size=n_sim))
        else:
            per_line[line] = simulate_line(fits[line], n_sim, rng)
    total = np.zeros(n_sim)
    for v in per_line.values():
        total += v
    return PortfolioSample(per_line, total)


@dataclass
class MultiLineResults:
    lines: list
    fits: dict            # line -> LineFit.__dict__
    unit: str
    standalone_mean: dict
    standalone_var995: dict
    standalone_es995: dict
    sum_standalone_var995: float
    portfolio_mean: float
    portfolio_var995: float
    portfolio_es995: float
    d_var: float          # 1 - VaR(portfolio) / sum VaR_i
    d_k: float            # 1 - K(portfolio) / sum K_i, with K = VaR - E[loss]
    n_sim: int
    seed: int

    def to_json(self, path: str | Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "MultiLineResults":
        return cls(**json.loads(Path(path).read_text()))
