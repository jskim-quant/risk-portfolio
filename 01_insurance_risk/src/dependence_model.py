"""Copulas for the CAS loss-ratio portfolio.

Gaussian correlations use Spearman: 2 sin(pi * rho / 6).
Student-t correlations use Kendall: sin(pi * tau / 2)."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import stats



def latent_from_spearman(rho_s: np.ndarray) -> np.ndarray:
    out = 2.0 * np.sin(np.pi * np.asarray(rho_s, dtype=float) / 6.0)
    np.fill_diagonal(out, 1.0)
    return out


def latent_from_kendall(tau: np.ndarray) -> np.ndarray:
    out = np.sin(np.pi * np.asarray(tau, dtype=float) / 2.0)
    np.fill_diagonal(out, 1.0)
    return out


def nearest_correlation(a: np.ndarray, max_iter: int = 200, tol: float = 1e-12) -> np.ndarray:
    """Higham (2002) alternating projections onto the PSD cone and the unit-diagonal set."""
    y = np.array(a, dtype=float)
    delta = np.zeros_like(y)
    for _ in range(max_iter):
        r = y - delta
        w, v = np.linalg.eigh((r + r.T) / 2)
        x = (v * np.maximum(w, 0.0)) @ v.T
        delta = x - r
        y_new = x.copy()
        np.fill_diagonal(y_new, 1.0)
        if np.linalg.norm(y_new - y, "fro") < tol:
            y = y_new
            break
        y = y_new
    w, v = np.linalg.eigh(y)
    y = (v * np.maximum(w, 1e-10)) @ v.T          # strictly positive definite for Cholesky
    d = np.sqrt(np.diag(y))
    return y / np.outer(d, d)


def valid_correlation(r: np.ndarray, tol: float = 1e-8) -> tuple[np.ndarray, dict]:
    """Check positive definiteness and repair the matrix if needed."""
    r = np.asarray(r, dtype=float)
    min_eig = float(np.linalg.eigvalsh(r).min())
    if min_eig > tol:
        return r, dict(min_eigenvalue=min_eig, corrected=False, frobenius_change=0.0)
    fixed = nearest_correlation(r)
    return fixed, dict(min_eigenvalue=min_eig, corrected=True,
                       frobenius_change=float(np.linalg.norm(fixed - r, "fro")),
                       min_eigenvalue_after=float(np.linalg.eigvalsh(fixed).min()))



def gaussian_copula_uniforms(corr: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, corr.shape[0]))
    return stats.norm.cdf(z @ chol.T)


def t_copula_uniforms(corr: np.ndarray, dof: float, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, corr.shape[0]))
    chi2 = rng.chisquare(dof, size=n)
    return stats.t.cdf((z @ chol.T) / np.sqrt(chi2 / dof)[:, None], df=dof)


def comonotonic_uniforms(d: int, n: int, seed: int = 42) -> np.ndarray:
    u = np.random.default_rng(seed).uniform(0, 1, size=n)
    return np.repeat(u[:, None], d, axis=1)


def portfolio_from_uniforms(marginals: Mapping[str, object], u: np.ndarray) -> tuple[dict, np.ndarray]:
    """Map copula uniforms to line losses (USD thousands) with each line's marginal; column order = dict order."""
    per_line = {line: m.losses(u[:, i]) for i, (line, m) in enumerate(marginals.items())}
    return per_line, np.sum(list(per_line.values()), axis=0)



def empirical_coexceedance(a: np.ndarray, b: np.ndarray, q: float) -> tuple[float, int]:
    """P(both pseudo-observations > q) from within-pair ranks, and the number of joint exceedances."""
    n = len(a)
    ua = stats.rankdata(a) / (n + 1)
    ub = stats.rankdata(b) / (n + 1)
    joint = int(np.sum((ua > q) & (ub > q)))
    return joint / n, joint


def gaussian_coexceedance(rho_latent: float, q: float) -> float:
    z = stats.norm.ppf(q)
    both_below = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho_latent], [rho_latent, 1]]).cdf([z, z])
    return float(1 - 2 * q + both_below)


def t_coexceedance(rho_latent: float, dof: float, q: float, seed: int = 0) -> float:
    tq = stats.t.ppf(q, df=dof)
    both_below = stats.multivariate_t(loc=[0, 0], shape=[[1, rho_latent], [rho_latent, 1]], df=dof).cdf(
        [tq, tq], random_state=np.random.default_rng(seed))
    return float(1 - 2 * q + both_below)



def spearman_matrix(panels: Mapping[str, "pd.DataFrame"], lines: list, min_matched: int = 100,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Estimate pairwise Spearman correlations on matched company-years.

    Pairs below min_matched are set to zero. Passing rng bootstraps matched
    rows within each pair. Check positive definiteness after transformation."""
    import pandas as pd
    d = len(lines)
    out = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            a, b = lines[i], lines[j]
            merged = pd.merge(panels[a], panels[b], on=["GRCODE", "AccidentYear"], how="inner")
            if len(merged) < min_matched:
                continue
            if rng is not None:
                merged = merged.iloc[rng.integers(0, len(merged), len(merged))]
            rho, _ = stats.spearmanr(merged[a], merged[b])
            out[i, j] = out[j, i] = rho
    return out
