"""Checks for the motor frequency and severity models."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import integrate, stats

from insurance_model import (CalibratedModel, severity_mean, capped_severity_mean, draw_severities, spliced_severity_cdf,
                             simulate_claims, aggregate, frequency_cleaning_summary, CleaningReport, nb2_negloglik)

# A finite-variance tail keeps the sampling checks stable.
MODEL = CalibratedModel(
    lambda_freq=0.1, r_freq=0.7,
    threshold=2000, alpha=0.95,
    mu_body=6, sigma_body=0.8,
    xi_gpd=0.2, sigma_gpd=1000,
)


def test_spliced_mean_against_integration():
    body = stats.lognorm(s=MODEL.sigma_body, scale=np.exp(MODEL.mu_body))
    body_part = integrate.quad(lambda x: x * body.pdf(x), 0, MODEL.threshold)[0] / body.cdf(MODEL.threshold)
    tail_part = integrate.quad(lambda x: (MODEL.threshold + x) * stats.genpareto.pdf(x, .2, scale=1000), 0, np.inf)[0]
    assert severity_mean(MODEL) == pytest.approx(.95 * body_part + .05 * tail_part, abs=1e-5)
    assert np.isinf(severity_mean(replace(MODEL, xi_gpd=1)))


def test_capped_mean():
    cap = 10_000.0
    grid = np.linspace(0, cap, 200_001)
    survival = 1 - spliced_severity_cdf(grid, MODEL)
    assert capped_severity_mean(MODEL, cap) == pytest.approx(np.trapezoid(survival, grid), rel=1e-4)
    assert capped_severity_mean(MODEL, 1e12) == pytest.approx(severity_mean(MODEL), rel=1e-3)


def test_sampler_matches_spliced_cdf():
    claims = draw_severities(200_000, MODEL, np.random.default_rng(3))
    assert stats.kstest(claims, lambda v: spliced_severity_cdf(v, MODEL)).pvalue > 0.001
    assert np.mean(claims <= MODEL.threshold) == pytest.approx(MODEL.alpha, abs=0.004)


def test_claim_aggregation():
    year_index, claims = simulate_claims(2000, MODEL, np.random.default_rng(5))
    assert len(year_index) == len(claims)
    assert np.all(claims > 0)
    assert np.all(np.diff(year_index) >= 0)
    assert aggregate(year_index, claims, 2000).sum() == pytest.approx(claims.sum())


def test_nb2_likelihood_prefers_true_parameters():
    rng = np.random.default_rng(1)
    exposure = rng.uniform(.2, 1, 40_000)
    rate, dispersion = .1, .8
    counts = rng.negative_binomial(dispersion, dispersion / (dispersion + rate * exposure)).astype(float)
    fitted_loss = nb2_negloglik(np.log([rate, dispersion]), counts, exposure)
    assert fitted_loss < nb2_negloglik(np.log([2 * rate, dispersion]), counts, exposure)
    assert fitted_loss < nb2_negloglik(np.log([rate, 6 * dispersion]), counts, exposure)


def test_frequency_cleaning_summary():
    report = CleaningReport(**{f: 0 for f in CleaningReport.__dataclass_fields__})
    summary = frequency_cleaning_summary(report)
    assert not {k for k in summary if 'severity' in k or k.startswith('freq_sev')}
    assert 'n_severity_orphans' not in summary


def test_nb2_count_moments():
    model = CalibratedModel(.3, .8, 2000, .95, 6, .8, .7, 1000)
    year_index, _ = simulate_claims(400_000, model, np.random.default_rng(8))
    counts = np.bincount(year_index, minlength=400_000)
    r, mu = model.r_freq, model.lambda_freq
    assert counts.mean() == pytest.approx(mu, rel=0.01)
    assert counts.var() == pytest.approx(mu + mu**2 / r, rel=0.03)  # Var(N) = mean + mean² / dispersion
    half_year_index, _ = simulate_claims(400_000, model, np.random.default_rng(8), exposure=.5)
    assert np.bincount(half_year_index, minlength=400_000).mean() == pytest.approx(.5 * mu, rel=0.02)


def test_severity_splice():
    claims = draw_severities(300_000, MODEL, np.random.default_rng(9))
    body, tail = claims[claims <= MODEL.threshold], claims[claims > MODEL.threshold]
    assert body.max() <= MODEL.threshold
    assert tail.min() > MODEL.threshold
    assert len(body) / len(claims) == pytest.approx(MODEL.alpha, abs=0.003)
    excess = tail - MODEL.threshold
    assert stats.kstest(excess, stats.genpareto(MODEL.xi_gpd, scale=MODEL.sigma_gpd).cdf).pvalue > 0.001
