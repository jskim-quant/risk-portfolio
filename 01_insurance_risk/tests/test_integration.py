"""Premium and reserve risk on a small two-line portfolio."""
import numpy as np
import pytest
from scipy import stats

from portfolio_model import LossRatioMarginal
from dependence_model import gaussian_copula_uniforms
from integration_model import reserve_lognormal, coupled_latent, integrated_losses, risk_row
import parameter_risk as pr

CORRELATION = np.array([[1, .3], [.3, 1]])
MARGINALS = {k: LossRatioMarginal('lognormal', 100.0, stats.lognorm(.6).ppf, stats.lognorm(.6).cdf) for k in ('a', 'b')}
RESERVES = {'a': 80.0, 'b': 120.0}
RESERVE_CV = {'a': .1, 'b': .2}


def test_reserve_lognormal_moments():
    mu, sigma = reserve_lognormal(500.0, .15)
    reserves = np.random.default_rng(0).lognormal(mu, sigma, 2_000_000)
    assert reserves.mean() == pytest.approx(500, rel=2e-3)
    assert reserves.std() / reserves.mean() == pytest.approx(.15, rel=5e-3)


def test_premium_draws_match_copula():
    for rho in (0.0, .5, 1.0):
        premium, _ = coupled_latent(CORRELATION, rho, 1000, 42, 43)
        np.testing.assert_allclose(stats.norm.cdf(premium), gaussian_copula_uniforms(CORRELATION, 1000, seed=42))


def test_premium_reserve_correlation():
    premium, reserve = coupled_latent(CORRELATION, .4, 400_000)
    correlation = np.corrcoef(np.hstack([premium, reserve]).T)
    np.testing.assert_allclose(correlation[:2, :2], CORRELATION, atol=.01)
    np.testing.assert_allclose(correlation[2:, 2:], CORRELATION, atol=.01)
    np.testing.assert_allclose(correlation[:2, 2:], .4 * CORRELATION, atol=.01)
    with pytest.raises(ValueError):
        coupled_latent(CORRELATION, 1.2, 10)


def test_integrated_loss_totals():
    losses = integrated_losses(MARGINALS, RESERVES, RESERVE_CV, CORRELATION, .25, 300_000)
    np.testing.assert_allclose(losses['total'], losses['premium_total'] + losses['reserve_total'])
    for line in RESERVES:
        assert abs(losses['reserve'][line].mean()) < .01 * RESERVES[line]
    assert risk_row(losses['total'])['EL'] == pytest.approx(losses['premium_total'].mean(), rel=5e-3)


def test_var_in_selected_dependence_scenarios():
    scenario_var = []
    for rho in (0, .5, 1):
        losses = integrated_losses(MARGINALS, RESERVES, RESERVE_CV, CORRELATION, rho, 300_000)
        scenario_var.append(risk_row(losses['total'])['VaR'])

    assert scenario_var[0] < scenario_var[1] < scenario_var[2]

    independent = integrated_losses(MARGINALS, RESERVES, RESERVE_CV, CORRELATION, 0, 300_000)
    premium_var = risk_row(independent['premium_total'])['VaR']
    assert scenario_var[0] > premium_var


def test_risk_row_identities():
    losses = np.random.default_rng(1).lognormal(0, 1, 100_000)
    risk = risk_row(losses)
    assert risk['ES'] >= risk['VaR']
    assert risk['K'] == pytest.approx(risk['VaR'] - risk['EL'])


def test_parameter_sd_removes_monte_carlo_variance():
    assert pr.parameter_sd(5.0, 3.0) == pytest.approx(4.0)
    assert pr.parameter_sd(2.0, 3.0) == 0.0
    interval = pr.interval(np.arange(101))
    assert (interval['p_lo'], interval['median'], interval['p_hi']) == (5.0, 50.0, 95.0)
