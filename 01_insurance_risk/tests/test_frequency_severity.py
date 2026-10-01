"""Frequency and severity model: moments, sampler and cleaning-report invariants."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import integrate, stats

from insurance_model import (CalibratedModel, severity_mean, capped_severity_mean, draw_severities, spliced_severity_cdf,
                             simulate_claims, aggregate, frequency_cleaning_summary, CleaningReport, nb2_negloglik)

MODEL = CalibratedModel(.1, .7, 2000, .95, 6, .8, .2, 1000)


def test_spliced_mean_against_integration():
    body = stats.lognorm(s=MODEL.sigma_body, scale=np.exp(MODEL.mu_body))
    body_part = integrate.quad(lambda x: x * body.pdf(x), 0, MODEL.threshold)[0] / body.cdf(MODEL.threshold)
    tail_part = integrate.quad(lambda x: (MODEL.threshold + x) * stats.genpareto.pdf(x, .2, scale=1000), 0, np.inf)[0]
    assert severity_mean(MODEL) == pytest.approx(.95 * body_part + .05 * tail_part, abs=1e-5)
    assert np.isinf(severity_mean(replace(MODEL, xi_gpd=1)))


def test_capped_mean_matches_numerical_integral_and_approaches_full_mean():
    cap = 10_000.0
    grid = np.linspace(0, cap, 200_001)
    survival = 1 - spliced_severity_cdf(grid, MODEL)
    assert capped_severity_mean(MODEL, cap) == pytest.approx(np.trapezoid(survival, grid), rel=1e-4)
    assert capped_severity_mean(MODEL, 1e12) == pytest.approx(severity_mean(MODEL), rel=1e-3)


def test_sampler_matches_spliced_cdf():
    x = draw_severities(200_000, MODEL, np.random.default_rng(3))
    assert stats.kstest(x, lambda v: spliced_severity_cdf(v, MODEL)).pvalue > 0.001
    assert np.mean(x <= MODEL.threshold) == pytest.approx(MODEL.alpha, abs=0.004)


def test_simulate_claims_uses_the_common_sampler():
    idx, claims = simulate_claims(2000, MODEL, np.random.default_rng(5))
    assert len(idx) == len(claims)
    assert np.all(claims > 0) and np.all(np.diff(idx) >= 0)
    assert aggregate(idx, claims, 2000).sum() == pytest.approx(claims.sum())


def test_nb2_likelihood_prefers_true_parameters():
    rng = np.random.default_rng(1)
    exposure = rng.uniform(.2, 1, 40_000)
    lam, r = .1, .8
    counts = rng.negative_binomial(r, r / (r + lam * exposure))
    at_truth = nb2_negloglik(np.log([lam, r]), counts.astype(float), exposure)
    assert at_truth < nb2_negloglik(np.log([2 * lam, r]), counts.astype(float), exposure)
    assert at_truth < nb2_negloglik(np.log([lam, 6 * r]), counts.astype(float), exposure)


def test_frequency_cleaning_summary_leaves_out_severity_fields():
    report = CleaningReport(**{f: 0 for f in CleaningReport.__dataclass_fields__})
    summary = frequency_cleaning_summary(report)
    assert not {k for k in summary if 'severity' in k or k.startswith('freq_sev')}
    assert 'n_severity_orphans' not in summary


def test_simulated_annual_claim_count_matches_theory():
    model = CalibratedModel(.3, .8, 2000, .95, 6, .8, .7, 1000)
    idx, _ = simulate_claims(400_000, model, np.random.default_rng(8))
    counts = np.bincount(idx, minlength=400_000)
    r, mu = model.r_freq, model.lambda_freq
    assert counts.mean() == pytest.approx(mu, rel=0.01)
    assert counts.var() == pytest.approx(mu + mu**2 / r, rel=0.03)            # NB2 variance
    exposure_idx, _ = simulate_claims(400_000, model, np.random.default_rng(8), exposure=.5)
    assert np.bincount(exposure_idx, minlength=400_000).mean() == pytest.approx(.5 * mu, rel=0.02)


def test_body_draws_respect_truncation_and_splice_probability():
    x = draw_severities(300_000, MODEL, np.random.default_rng(9))
    body, tail = x[x <= MODEL.threshold], x[x > MODEL.threshold]
    assert body.max() <= MODEL.threshold and tail.min() > MODEL.threshold
    assert len(body) / len(x) == pytest.approx(MODEL.alpha, abs=0.003)
    excess = tail - MODEL.threshold
    assert stats.kstest(excess, stats.genpareto(MODEL.xi_gpd, scale=MODEL.sigma_gpd).cdf).pvalue > 0.001
