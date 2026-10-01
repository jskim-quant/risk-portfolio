"""Risk measures and diversification metrics shared by all notebooks.

Terminology used throughout the project (keep it consistent):
  VaR      quantile of the simulated loss distribution
  ES       mean loss at or beyond that quantile
  K        economic-capital proxy = VaR - E[loss]   (internal metric, not a regulatory SCR)
  D_VaR    1 - VaR(portfolio) / sum VaR(stand-alone)
  D_K      1 - K(portfolio)  / sum K(stand-alone)
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def var(losses: np.ndarray, q: float) -> float:
    return float(np.percentile(losses, q * 100))


def es(losses: np.ndarray, q: float) -> float:
    v = var(losses, q)
    tail = losses[losses >= v]
    return float(tail.mean()) if len(tail) > 0 else v


def capital_proxy(losses: np.ndarray, q: float = 0.995, mean: float | None = None) -> float:
    """K = VaR - E[L]. Pass the analytical mean when one exists; otherwise the sample mean is used."""
    m = float(np.mean(losses)) if mean is None else float(mean)
    return var(losses, q) - m


def var_diversification(portfolio_var: float, standalone_vars: Sequence[float]) -> float:
    """D_VaR = 1 - VaR(portfolio) / sum(VaR_i)."""
    total = float(np.sum(standalone_vars))
    return 1.0 - portfolio_var / total


def k_diversification(portfolio_var: float, portfolio_mean: float,
                      standalone_vars: Sequence[float], standalone_means: Sequence[float]) -> float:
    """D_K = 1 - [VaR(S) - E(S)] / sum_i [VaR_i - E(X_i)]."""
    k_port = portfolio_var - portfolio_mean
    k_sum = float(np.sum(standalone_vars) - np.sum(standalone_means))
    return 1.0 - k_port / k_sum


def euler_es_allocation(total: np.ndarray, per_line: Mapping[str, np.ndarray], q: float = 0.995) -> dict:
    """Allocate portfolio ES to components by their mean loss in the portfolio tail scenarios.

    Contributions add up to the portfolio ES (up to floating-point error).
    """
    mask = total >= var(total, q)
    return {k: float(np.asarray(v)[mask].mean()) for k, v in per_line.items()}


def portfolio_summary(per_line: Mapping[str, np.ndarray], total: np.ndarray, q: float = 0.995) -> dict:
    """Stand-alone and portfolio VaR/ES/mean plus both diversification metrics.

    per_line: loss samples by component, total: their sum under the chosen dependence.
    Means are sample means of the same scenarios, so D_K is internally consistent.
    """
    names = list(per_line)
    sa_mean = {k: float(np.mean(per_line[k])) for k in names}
    sa_var = {k: var(per_line[k], q) for k in names}
    sa_es = {k: es(per_line[k], q) for k in names}
    p_mean, p_var, p_es = float(np.mean(total)), var(total, q), es(total, q)
    return dict(
        standalone_mean=sa_mean, standalone_var=sa_var, standalone_es=sa_es,
        sum_standalone_var=float(sum(sa_var.values())), sum_standalone_es=float(sum(sa_es.values())),
        portfolio_mean=p_mean, portfolio_var=p_var, portfolio_es=p_es,
        portfolio_k=p_var - p_mean,
        sum_standalone_k=float(sum(sa_var.values()) - sum(sa_mean.values())),
        d_var=var_diversification(p_var, list(sa_var.values())),
        d_k=k_diversification(p_var, p_mean, list(sa_var.values()), list(sa_mean.values())),
        d_es=1.0 - p_es / float(sum(sa_es.values())),
    )
