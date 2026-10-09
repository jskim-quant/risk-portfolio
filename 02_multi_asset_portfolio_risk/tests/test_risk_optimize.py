"""Unit tests for src.risk and src.optimize (run: python -m pytest tests or python tests/test_risk_optimize.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import optimize as O  # noqa: E402
from src import risk as K  # noqa: E402

RNG = np.random.default_rng(0)
X = RNG.standard_normal((400, 8)) @ RNG.standard_normal((8, 8)) * 0.01
S = np.cov(X, rowvar=False)


def _lw_brute(R: np.ndarray) -> float:
    """Ledoit–Wolf (2004) constant-correlation intensity with explicit loops."""
    X0 = R - R.mean(axis=0)
    T, N = X0.shape
    Sm = X0.T @ X0 / T
    sd = np.sqrt(np.diag(Sm))
    rbar = ((Sm / np.outer(sd, sd)).sum() - N) / (N * (N - 1))
    pi = rho = 0.0
    for i in range(N):
        for j in range(N):
            pij = np.mean((X0[:, i] * X0[:, j] - Sm[i, j]) ** 2)
            pi += pij
            if i == j:
                rho += pij
            else:
                tii = np.mean((X0[:, i] ** 2 - Sm[i, i]) * (X0[:, i] * X0[:, j] - Sm[i, j]))
                tjj = np.mean((X0[:, j] ** 2 - Sm[j, j]) * (X0[:, i] * X0[:, j] - Sm[i, j]))
                rho += rbar / 2 * (np.sqrt(Sm[j, j] / Sm[i, i]) * tii + np.sqrt(Sm[i, i] / Sm[j, j]) * tjj)
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, np.diag(Sm))
    return float(np.clip((pi - rho) / ((F - Sm) ** 2).sum() / T, 0, 1))


def test_ledoit_wolf_matches_brute_force() -> None:
    """Vectorised LW intensity equals the loop implementation (guards the θ = E[x_i³x_j] fix)."""
    R = X[-156:] + 0.002 * RNG.standard_t(3, size=(156, 8))
    assert abs(K.ledoit_wolf_cc(R)[1] - _lw_brute(R)) < 1e-10


def test_ewma_matches_recursion() -> None:
    """Closed-form EWMA weights equal the recursion Σ_t = λΣ_{t-1} + (1-λ) r r' (normalised start)."""
    lam = 0.5 ** (1 / 26)
    rec = np.zeros((8, 8))
    for r in X:
        rec = lam * rec + (1 - lam) * np.outer(r, r)
    assert np.allclose(rec / (1 - lam ** len(X)), K.ewma_cov(X))


def test_garch_recovers_parameters() -> None:
    """MLE on 3,000 simulated observations lands near the true (α, β) = (0.08, 0.90)."""
    rng = np.random.default_rng(1)
    a, b, v, r, s2 = 0.08, 0.90, 4e-4, np.empty(3000), 4e-4
    for t in range(3000):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = v * (1 - a - b) + a * r[t] ** 2 + b * s2
    g = K.fit_garch11(r)
    assert abs(g.alpha - a) < 0.04 and abs(g.beta - b) < 0.05


def test_qp_matches_closed_form_gmv() -> None:
    """ADMM with only the budget constraint reproduces Σ⁻¹1 / 1'Σ⁻¹1."""
    n = S.shape[0]
    w = O.solve_qp(S / np.mean(np.diag(S)), np.zeros(n), np.ones((1, n)), np.ones(1), np.ones(1)).x
    assert np.abs(w - O.gmv_unconstrained(S)).max() < 1e-6


def test_min_variance_respects_cap_and_kkt() -> None:
    """Capped long-only MinVar: bounds hold and marginal risk is equal across free assets."""
    w = O.min_variance(S, 0.3)
    free = (w > 1e-6) & (w < 0.3 - 1e-6)
    assert w.min() >= 0 and w.max() <= 0.3 + 1e-9 and abs(w.sum() - 1) < 1e-12
    assert np.ptp((S @ w)[free]) < 1e-8


def test_risk_parity_equalises_contributions() -> None:
    """Every asset contributes 1/N of portfolio variance."""
    rc = O.risk_contributions(O.risk_parity(S), S)
    assert np.allclose(rc, 1 / len(rc), atol=1e-10)


def test_min_cvar_beats_random_portfolios() -> None:
    """The LP optimum is at least as good as the best of 5,000 random long-only portfolios."""
    w, cv = O.min_cvar(X[:200], 0.95)
    best = min(O.cvar(X[:200] @ d) for d in np.random.default_rng(2).dirichlet(np.ones(8), 5000))
    assert cv <= best + 1e-6 and abs(w.sum() - 1) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
