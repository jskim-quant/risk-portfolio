"""Copula parameterisation, positive-definiteness and dependence diagnostics."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dependence_model import (latent_from_spearman, latent_from_kendall, valid_correlation, nearest_correlation,
                              gaussian_copula_uniforms, t_copula_uniforms, comonotonic_uniforms, empirical_coexceedance,
                              gaussian_coexceedance, spearman_matrix)


def test_gaussian_latent_from_spearman_is_exact():
    rho_s = 0.4
    r = latent_from_spearman(np.array([[1, rho_s], [rho_s, 1]]))[0, 1]
    u = gaussian_copula_uniforms(np.array([[1, r], [r, 1]]), 400_000, seed=1)
    assert stats.spearmanr(u[:, 0], u[:, 1])[0] == pytest.approx(rho_s, abs=0.004)
    # using the Spearman value directly as the latent correlation would be biased
    naive = gaussian_copula_uniforms(np.array([[1, rho_s], [rho_s, 1]]), 400_000, seed=1)
    assert abs(stats.spearmanr(naive[:, 0], naive[:, 1])[0] - rho_s) > 0.005


def test_t_latent_from_kendall_is_exact():
    tau = 0.3
    r = latent_from_kendall(np.array([[1, tau], [tau, 1]]))[0, 1]
    u = t_copula_uniforms(np.array([[1, r], [r, 1]]), 5, 200_000, seed=2)
    assert stats.kendalltau(u[:500_00, 0], u[:500_00, 1])[0] == pytest.approx(tau, abs=0.01)


def test_valid_correlation_repairs_indefinite_matrix_and_leaves_good_ones():
    good = np.array([[1, .3], [.3, 1]])
    same, report = valid_correlation(good)
    assert not report['corrected'] and np.allclose(same, good)
    bad = np.array([[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1]])
    fixed, report = valid_correlation(bad)
    assert report['corrected'] and np.linalg.eigvalsh(fixed).min() > 0
    np.testing.assert_allclose(np.diag(fixed), 1)
    np.linalg.cholesky(fixed)


def test_comonotonic_uniforms_identical_columns_and_uniform_margins():
    u = comonotonic_uniforms(4, 10_000, seed=3)
    assert np.all(u == u[:, [0]])
    assert stats.kstest(u[:, 0], 'uniform').pvalue > 0.001


def test_coexceedance_matches_independence_and_gaussian_theory():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=20_000), rng.normal(size=20_000)
    share, joint = empirical_coexceedance(a, b, .9)
    assert share == pytest.approx(.01, abs=.0015) and joint == int(round(share * 20_000))
    r = .6
    u = gaussian_copula_uniforms(np.array([[1, r], [r, 1]]), 200_000, seed=5)
    emp, _ = empirical_coexceedance(u[:, 0], u[:, 1], .9)
    assert emp == pytest.approx(gaussian_coexceedance(r, .9), abs=.002)
    assert gaussian_coexceedance(r, .9) > gaussian_coexceedance(0.0, .9)


def test_spearman_matrix_threshold_and_bootstrap():
    rng = np.random.default_rng(6)
    n = 300
    keys = pd.DataFrame({'GRCODE': np.arange(n), 'AccidentYear': 2000})
    x = rng.normal(size=n)
    panels = {'a': keys.assign(a=x), 'b': keys.assign(b=x + rng.normal(size=n)), 'c': keys.iloc[:50].assign(c=rng.normal(size=50))}
    rho = spearman_matrix(panels, ['a', 'b', 'c'])
    assert rho[0, 1] > .4 and rho[0, 2] == 0 and rho[1, 2] == 0           # fewer than 100 matches -> zero
    boot = spearman_matrix(panels, ['a', 'b', 'c'], rng=np.random.default_rng(1))
    assert boot[0, 1] != rho[0, 1] and np.allclose(boot, boot.T)


def test_copula_margins_are_uniform_and_ranks_reproduce_targets():
    R = np.array([[1, .5, .2], [.5, 1, .3], [.2, .3, 1]])
    for u in (gaussian_copula_uniforms(R, 100_000, seed=7), t_copula_uniforms(R, 6, 100_000, seed=7)):
        for j in range(3):
            assert stats.kstest(u[:, j], 'uniform').pvalue > 0.001
    u = gaussian_copula_uniforms(R, 200_000, seed=7)
    target = 6 / np.pi * np.arcsin(R / 2)                                  # Spearman implied by the Gaussian copula
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert stats.spearmanr(u[:, i], u[:, j])[0] == pytest.approx(target[i, j], abs=.01)
