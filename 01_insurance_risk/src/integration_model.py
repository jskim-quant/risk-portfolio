"""Internal economic-risk aggregation of premium risk and reserve risk on the CAS (USD) basis.

L_total = L_premium + L_reserve, per line and in total, where
  L_premium  simulated company-year ultimate loss (loss-ratio marginal x book premium), as in notebooks 05-06
  L_reserve  deviation of the unpaid reserve from its best estimate (runoff, lognormal moment-matched to Mack)

The deviation has mean zero, so E[L_total] = E[L_premium] and K = VaR - E[L_total].
This is an internal aggregation with assumed premium-reserve dependence. It is not the Solvency II
Standard Formula aggregation of notebook 10 (fixed sigma factors, prescribed correlations, one-year horizon).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import stats

from risk_measures import var, es


def reserve_lognormal(best_estimate: float, cv: float) -> tuple[float, float]:
    """(mu, sigma) of a lognormal with the given mean and coefficient of variation."""
    sigma2 = np.log1p(cv ** 2)
    return float(np.log(best_estimate) - sigma2 / 2), float(np.sqrt(sigma2))


def coupled_latent(corr: np.ndarray, rho_pr: float, n: int,
                   seed_prem: int = 42, seed_res: int = 43) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian latent vectors for the premium and the reserve block.

    Corr(premium_i, premium_j) = Corr(reserve_i, reserve_j) = corr_ij and Corr(premium_i, reserve_j) = rho_pr * corr_ij
    (a Kronecker structure, positive semi-definite whenever corr is and |rho_pr| <= 1).
    The premium block uses the same stream as gaussian_copula_uniforms(corr, n, seed_prem).
    """
    if not -1.0 <= rho_pr <= 1.0:
        raise ValueError("rho_pr must lie in [-1, 1]")
    chol = np.linalg.cholesky(corr)
    d = corr.shape[0]
    y_p = np.random.default_rng(seed_prem).standard_normal((n, d)) @ chol.T
    y_0 = np.random.default_rng(seed_res).standard_normal((n, d)) @ chol.T
    y_r = rho_pr * y_p + np.sqrt(1.0 - rho_pr ** 2) * y_0
    return y_p, y_r


def integrated_losses(marginals: Mapping[str, object], reserve_be: Mapping[str, float], reserve_cv: Mapping[str, float],
                      corr: np.ndarray, rho_pr: float, n: int, seed_prem: int = 42, seed_res: int = 43) -> dict:
    """Premium losses and reserve deviations per line (USD thousands) and their sum."""
    lines = list(marginals)
    y_p, y_r = coupled_latent(corr, rho_pr, n, seed_prem, seed_res)
    prem = {l: marginals[l].losses(stats.norm.cdf(y_p[:, i])) for i, l in enumerate(lines)}
    res = {}
    for i, l in enumerate(lines):
        mu, sigma = reserve_lognormal(reserve_be[l], reserve_cv[l])
        res[l] = np.exp(mu + sigma * y_r[:, i]) - reserve_be[l]
    prem_total = np.sum(list(prem.values()), axis=0)
    res_total = np.sum(list(res.values()), axis=0)
    return dict(premium=prem, reserve=res, premium_total=prem_total, reserve_total=res_total,
                total=prem_total + res_total)


def risk_row(losses: np.ndarray, expected: float | None = None, q: float = 0.995) -> dict:
    """Expected loss, VaR, ES and K = VaR - E[L] for one simulated loss vector."""
    mean = float(np.mean(losses)) if expected is None else float(expected)
    v = var(losses, q)
    return dict(EL=mean, VaR=v, ES=es(losses, q), K=v - mean)
