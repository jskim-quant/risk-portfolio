"""Premium + reserve integration and parameter-risk helpers."""
import numpy as np
import pytest
from scipy import stats

from portfolio_model import LossRatioMarginal
from dependence_model import gaussian_copula_uniforms
from integration_model import reserve_lognormal, coupled_latent, integrated_losses, risk_row
import parameter_risk as pr

R2 = np.array([[1, .3], [.3, 1]])
MARGINALS = {k: LossRatioMarginal('lognormal', 100.0, stats.lognorm(.6).ppf, stats.lognorm(.6).cdf) for k in ('a', 'b')}
BE = {'a': 80.0, 'b': 120.0}
CV = {'a': .1, 'b': .2}


def test_reserve_lognormal_reproduces_mean_and_cv():
    mu, sigma = reserve_lognormal(500.0, .15)
    x = np.random.default_rng(0).lognormal(mu, sigma, 2_000_000)
    assert x.mean() == pytest.approx(500, rel=2e-3) and x.std() / x.mean() == pytest.approx(.15, rel=5e-3)


def test_premium_block_equals_plain_gaussian_copula_for_any_rho():
    for rho in (0.0, .5, 1.0):
        y_p, _ = coupled_latent(R2, rho, 1000, 42, 43)
        np.testing.assert_allclose(stats.norm.cdf(y_p), gaussian_copula_uniforms(R2, 1000, seed=42))


def test_cross_block_correlation_is_kronecker():
    y_p, y_r = coupled_latent(R2, .4, 400_000)
    c = np.corrcoef(np.hstack([y_p, y_r]).T)
    np.testing.assert_allclose(c[:2, :2], R2, atol=.01)
    np.testing.assert_allclose(c[2:, 2:], R2, atol=.01)
    np.testing.assert_allclose(c[:2, 2:], .4 * R2, atol=.01)
    with pytest.raises(ValueError):
        coupled_latent(R2, 1.2, 10)


def test_total_is_sum_and_reserve_deviation_has_mean_zero():
    sim = integrated_losses(MARGINALS, BE, CV, R2, .25, 300_000)
    np.testing.assert_allclose(sim['total'], sim['premium_total'] + sim['reserve_total'])
    for line in BE:
        assert abs(sim['reserve'][line].mean()) < .01 * BE[line]
    assert risk_row(sim['total'])['EL'] == pytest.approx(sim['premium_total'].mean(), rel=5e-3)


def test_more_dependence_never_lowers_tail_risk_and_reserve_adds_risk():
    v = [risk_row(integrated_losses(MARGINALS, BE, CV, R2, rho, 300_000)['total'])['VaR'] for rho in (0, .5, 1)]
    assert v[0] < v[1] < v[2]
    premium_only = risk_row(integrated_losses(MARGINALS, BE, CV, R2, 0, 300_000)['premium_total'])['VaR']
    assert v[0] > premium_only


def test_risk_row_identities():
    x = np.random.default_rng(1).lognormal(0, 1, 100_000)
    r = risk_row(x)
    assert r['ES'] >= r['VaR'] and r['K'] == pytest.approx(r['VaR'] - r['EL'])


def test_parameter_sd_removes_monte_carlo_variance():
    assert pr.parameter_sd(5.0, 3.0) == pytest.approx(4.0)
    assert pr.parameter_sd(2.0, 3.0) == 0.0
    iv = pr.interval(np.arange(101))
    assert (iv['p_lo'], iv['median'], iv['p_hi']) == (5.0, 50.0, 95.0)
