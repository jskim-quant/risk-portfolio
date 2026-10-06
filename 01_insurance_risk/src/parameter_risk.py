"""Bootstrap CAS and motor model parameters separately.

CAS uses common random numbers and fixed reserve inputs. Motor draws
frequency parameters from the asymptotic fit and refits resampled severities.
Separate fixed-parameter runs estimate Monte Carlo variability."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from risk_measures import var, es
from portfolio_model import LINES, fit_line_marginal, marginal_from_fit
from dependence_model import spearman_matrix, latent_from_spearman, valid_correlation
from insurance_model import (CalibratedModel, fit_severity, frequency_parameter_draws,
                             simulate_aggregate_losses, severity_mean)
from integration_model import integrated_losses

Q = 0.995


def _cas_metrics(marginals, reserve_be, reserve_cv, corr, rho_pr, n, seed_prem, seed_res):
    sim = integrated_losses(marginals, reserve_be, reserve_cv, corr, rho_pr, n, seed_prem, seed_res)
    el = float(np.mean(sim["premium_total"]))
    v_total, v_prem = var(sim["total"], Q), var(sim["premium_total"], Q)
    return dict(VaR_integrated=v_total, ES_integrated=es(sim["total"], Q), K_integrated=v_total - el,
                VaR_premium=v_prem, ES_premium=es(sim["premium_total"], Q))


def cas_mc_replicates(base_marginals, reserve_be, reserve_cv, corr, rho_pr, n, n_rep, seed0=10_000) -> pd.DataFrame:
    """Repeat the CAS simulation with fixed parameters and different seeds."""
    rows = [_cas_metrics(base_marginals, reserve_be, reserve_cv, corr, rho_pr, n, seed0 + 2 * r, seed0 + 2 * r + 1)
            for r in range(n_rep)]
    return pd.DataFrame(rows)


def cas_bootstrap(cred: dict, base_fits: dict, panels: dict, base_corr: np.ndarray,
                  reserve_be, reserve_cv, rho_pr, n, n_boot, variant, seed=500) -> pd.DataFrame:
    """Parameter bootstrap of the integrated CAS portfolio with common random numbers.

    variant: 'all_marginals' | 'wkcomp_marginal' | 'dependence' | 'marginals_and_dependence'
    Company-year loss ratios are resampled with replacement per line; matched rows are resampled per pair.
    Book premium and the marginal family are kept at the baseline values.
    """
    if variant not in {"all_marginals", "wkcomp_marginal", "dependence", "marginals_and_dependence"}:
        raise ValueError(variant)
    rng = np.random.default_rng(seed)
    base_marginals = {l: marginal_from_fit(base_fits[l]) for l in LINES}
    rows = []
    for _ in range(n_boot):
        marginals = dict(base_marginals)
        if variant in {"all_marginals", "marginals_and_dependence", "wkcomp_marginal"}:
            targets = ["wkcomp"] if variant == "wkcomp_marginal" else LINES
            for l in targets:
                sample = cred[l].iloc[rng.integers(0, len(cred[l]), len(cred[l]))]
                fit = fit_line_marginal(sample)
                fit.line, fit.book_premium = l, base_fits[l].book_premium
                marginals[l] = marginal_from_fit(fit, base_fits[l].chosen)
        corr = base_corr
        if variant in {"dependence", "marginals_and_dependence"}:
            rho = spearman_matrix(panels, LINES, rng=rng)
            corr, _ = valid_correlation(latent_from_spearman(rho))
        rows.append(_cas_metrics(marginals, reserve_be, reserve_cv, corr, rho_pr, n, 42, 43))
    return pd.DataFrame(rows)


def motor_mc_replicates(model: CalibratedModel, n: int, n_rep: int, seed0=20_000) -> pd.DataFrame:
    mean = model.lambda_freq * severity_mean(model)
    rows = []
    for r in range(n_rep):
        x = simulate_aggregate_losses(n, model, np.random.default_rng(seed0 + r))
        v = var(x, Q)
        rows.append(dict(VaR=v, ES=es(x, Q), K=v - mean))
    return pd.DataFrame(rows)


def motor_bootstrap(model: CalibratedModel, freq: pd.DataFrame, freq_fit, severities: np.ndarray, n: int,
                    n_boot: int, variant: str, seed=700) -> pd.DataFrame:
    """variant: 'frequency' | 'severity' | 'frequency_and_severity'.

    Frequency: asymptotic-normal draws of (lambda, r). Severity: resample claim amounts, refit the spliced
    lognormal/GPD model with the threshold re-chosen at the same percentile of the resample.
    """
    if variant not in {"frequency", "severity", "frequency_and_severity"}:
        raise ValueError(variant)
    rng = np.random.default_rng(seed)
    draws = frequency_parameter_draws(freq, freq_fit, n_boot, seed=seed) if "frequency" in variant else None
    rows = []
    for b in range(n_boot):
        m = model
        if draws is not None:
            m = replace(m, lambda_freq=float(draws[b, 0]), r_freq=float(draws[b, 1]))
        if "severity" in variant:
            sf = fit_severity(severities[rng.integers(0, len(severities), len(severities))])
            m = replace(m, threshold=sf.threshold, alpha=sf.alpha, mu_body=sf.mu_body, sigma_body=sf.sigma_body,
                        xi_gpd=sf.xi_gpd, sigma_gpd=sf.sigma_gpd)
        sev_mean = severity_mean(m)          # infinite when the resampled GPD shape reaches 1
        mean = m.lambda_freq * sev_mean if np.isfinite(sev_mean) else np.nan
        x = simulate_aggregate_losses(n, m, np.random.default_rng(seed * 1000 + b))
        v = var(x, Q)
        rows.append(dict(VaR=v, ES=es(x, Q), K=v - mean, xi=m.xi_gpd, lam=m.lambda_freq, r=m.r_freq,
                         infinite_mean=bool(m.xi_gpd >= 1)))
    return pd.DataFrame(rows)


def interval(x: np.ndarray, lo=5, hi=95) -> dict:
    x = np.asarray(x, dtype=float)
    return dict(p_lo=float(np.percentile(x, lo)), median=float(np.median(x)), p_hi=float(np.percentile(x, hi)),
                sd=float(np.std(x, ddof=1)))


def parameter_sd(total_sd: float, mc_sd: float) -> float:
    """Spread explained by parameters after removing Monte Carlo variance (0 if MC alone explains it)."""
    return float(np.sqrt(max(total_sd ** 2 - mc_sd ** 2, 0.0)))
