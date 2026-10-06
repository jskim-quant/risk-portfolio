"""Combine CAS premium losses with reserve deviations (USD).

Reserves use a lognormal approximation to Mack runoff uncertainty.
Premium-reserve dependence is assumed. This is not a Standard Formula SCR."""
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
    """Draw correlated premium and reserve blocks.

    Each block has correlation corr; cross-block correlation is rho_pr * corr.
    The premium draws match gaussian_copula_uniforms with seed_prem."""
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
