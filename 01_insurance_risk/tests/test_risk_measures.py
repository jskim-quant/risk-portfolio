"""Risk-measure definitions and diversification invariants (VaR, ES, K = VaR - E, D_VaR, D_K)."""
import numpy as np
import pytest

from risk_measures import var, es, capital_proxy, var_diversification, k_diversification, euler_es_allocation, portfolio_summary


@pytest.fixture(scope='module')
def lines():
    rng = np.random.default_rng(11)
    return {k: rng.lognormal(0, s, 200_000) for k, s in [('a', .5), ('b', .8), ('c', 1.1)]}


def test_es_is_at_least_var_and_k_identity(lines):
    x = lines['c']
    assert es(x, .995) >= var(x, .995) >= np.mean(x)
    assert capital_proxy(x, .995) == pytest.approx(var(x, .995) - x.mean())
    assert capital_proxy(x, .995, mean=2.0) == pytest.approx(var(x, .995) - 2.0)


def test_diversification_zero_when_comonotonic_and_positive_when_independent(lines):
    total_como = sum(np.sort(v) for v in lines.values())
    summary = portfolio_summary({k: np.sort(v) for k, v in lines.items()}, total_como)
    assert summary['d_var'] == pytest.approx(0, abs=1e-9) and summary['d_k'] == pytest.approx(0, abs=1e-9)
    indep = portfolio_summary(lines, sum(lines.values()))
    assert 0 < indep['d_var'] < 1 and 0 < indep['d_k'] < 1
    assert indep['d_k'] > indep['d_var']          # K subtracts the mean, which diversifies nothing


def test_d_var_and_d_k_formulas():
    assert var_diversification(80, [50, 50]) == pytest.approx(0.2)
    assert k_diversification(80, 20, [50, 50], [10, 10]) == pytest.approx(1 - 60 / 80)


def test_euler_es_contributions_add_to_portfolio_es(lines):
    total = sum(lines.values())
    parts = euler_es_allocation(total, lines, .995)
    assert sum(parts.values()) == pytest.approx(es(total, .995), rel=1e-12)
