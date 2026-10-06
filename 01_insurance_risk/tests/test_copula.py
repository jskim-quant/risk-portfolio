"""Correlation transforms and copula samples."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dependence_model import (latent_from_spearman, latent_from_kendall, valid_correlation,
                              gaussian_copula_uniforms, t_copula_uniforms, comonotonic_uniforms, empirical_coexceedance,
                              gaussian_coexceedance, spearman_matrix)


def test_gaussian_spearman_correlation():
    rho_s = 0.4
    latent_rho = latent_from_spearman(np.array([[1, rho_s], [rho_s, 1]]))[0, 1]
    u = gaussian_copula_uniforms(np.array([[1, latent_rho], [latent_rho, 1]]), 400_000, seed=1)
    assert stats.spearmanr(u[:, 0], u[:, 1])[0] == pytest.approx(rho_s, abs=0.004)
    # Spearman rho needs converting before drawing the Gaussian samples.
    naive = gaussian_copula_uniforms(np.array([[1, rho_s], [rho_s, 1]]), 400_000, seed=1)
    assert abs(stats.spearmanr(naive[:, 0], naive[:, 1])[0] - rho_s) > 0.005


def test_t_kendall_correlation():
    tau = 0.3
    latent_rho = latent_from_kendall(np.array([[1, tau], [tau, 1]]))[0, 1]
    u = t_copula_uniforms(np.array([[1, latent_rho], [latent_rho, 1]]), 5, 200_000, seed=2)
    assert stats.kendalltau(u[:50_000, 0], u[:50_000, 1])[0] == pytest.approx(tau, abs=0.01)


def test_correlation_repair():
    good = np.array([[1, .3], [.3, 1]])
    same, report = valid_correlation(good)
    assert not report['corrected']
    assert np.allclose(same, good)
    bad = np.array([[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1]])
    fixed, report = valid_correlation(bad)
    assert report['corrected']
    assert np.linalg.eigvalsh(fixed).min() > 0
    np.testing.assert_allclose(np.diag(fixed), 1)
    np.linalg.cholesky(fixed)


def test_comonotonic_draws():
    u = comonotonic_uniforms(4, 10_000, seed=3)
    assert np.all(u == u[:, [0]])
    assert stats.kstest(u[:, 0], 'uniform').pvalue > 0.001


def test_coexceedance():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=20_000), rng.normal(size=20_000)
    share, joint = empirical_coexceedance(a, b, .9)
    assert share == pytest.approx(.01, abs=.0015)
    assert joint == int(round(share * 20_000))
    r = .6
    u = gaussian_copula_uniforms(np.array([[1, r], [r, 1]]), 200_000, seed=5)
    observed, _ = empirical_coexceedance(u[:, 0], u[:, 1], .9)
    assert observed == pytest.approx(gaussian_coexceedance(r, .9), abs=.002)
    assert gaussian_coexceedance(r, .9) > gaussian_coexceedance(0.0, .9)


def test_spearman_matrix_threshold_and_bootstrap():
    rng = np.random.default_rng(6)
    n = 300
    keys = pd.DataFrame({'GRCODE': np.arange(n), 'AccidentYear': 2000})
    x = rng.normal(size=n)
    panels = {
        'a': keys.assign(a=x),
        'b': keys.assign(b=x + rng.normal(size=n)),
        'c': keys.iloc[:50].assign(c=rng.normal(size=50)),
    }
    rho = spearman_matrix(panels, ['a', 'b', 'c'])
    assert rho[0, 1] > .4
    # Line c has too few matches to estimate either correlation.
    assert rho[0, 2] == 0
    assert rho[1, 2] == 0
    resampled = spearman_matrix(panels, ['a', 'b', 'c'], rng=np.random.default_rng(1))
    assert resampled[0, 1] != rho[0, 1]
    assert np.allclose(resampled, resampled.T)


def test_copula_margins_and_ranks():
    correlation = np.array([[1, .5, .2], [.5, 1, .3], [.2, .3, 1]])
    for u in (gaussian_copula_uniforms(correlation, 100_000, seed=7), t_copula_uniforms(correlation, 6, 100_000, seed=7)):
        for j in range(3):
            assert stats.kstest(u[:, j], 'uniform').pvalue > 0.001
    u = gaussian_copula_uniforms(correlation, 200_000, seed=7)
    target = 6 / np.pi * np.arcsin(correlation / 2)  # Gaussian-implied Spearman rho
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert stats.spearmanr(u[:, i], u[:, j])[0] == pytest.approx(target[i, j], abs=.01)
