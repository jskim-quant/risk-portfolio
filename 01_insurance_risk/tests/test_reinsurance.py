"""Reinsurance settlement: retained + ceded = gross, monotonicity, limits."""
import numpy as np
import pytest

from insurance_model import CalibratedModel, simulate_claims, aggregate
from reinsurance_model import quota_share, per_occurrence_xol, aggregate_xol


@pytest.mark.parametrize('freq_mult,sev_mult', [(1, 1), (1.5, 1), (1, 1.5), (1.5, 1.5)])
def test_treaties_settle_in_stress(freq_mult, sev_mult):
    model = CalibratedModel(.3, .8, 2000, .95, 6, .8, .7, 1000)
    idx, claims = simulate_claims(20_000, model, np.random.default_rng(2), freq_mult, sev_mult)
    gross = aggregate(idx, claims, 20_000)
    for retained in (claims * .75, np.minimum(claims, 5000)):
        net = aggregate(idx, retained, 20_000)
        ceded = aggregate(idx, claims - retained, 20_000)
        np.testing.assert_allclose(gross, net + ceded, rtol=1e-12, atol=1e-8)


def test_quota_share_and_layer_properties():
    x = np.random.default_rng(0).lognormal(0, 1, 10_000)
    kept, ceded = quota_share(x, .25)
    np.testing.assert_allclose(kept + ceded, x)
    np.testing.assert_allclose(ceded, .25 * x)
    for attach, limit in [(1.0, None), (1.0, 2.0), (0.0, 5.0)]:
        kept, ceded = aggregate_xol(x, attach, limit)
        np.testing.assert_allclose(kept + ceded, x)
        assert np.all(ceded >= 0) and np.all(kept <= x + 1e-12)
        if limit is not None:
            assert ceded.max() <= limit + 1e-12
    _, small = per_occurrence_xol(x, 2.0)
    _, large = per_occurrence_xol(x, 1.0)
    assert np.all(large >= small)                    # lower retention cedes more


def test_invalid_treaty_arguments():
    with pytest.raises(ValueError):
        quota_share(np.ones(3), 1.2)
    with pytest.raises(ValueError):
        per_occurrence_xol(np.ones(3), -1)


def test_per_occurrence_retention_is_a_hard_cap_before_aggregation():
    claims = np.random.default_rng(1).pareto(1.5, 50_000) * 1000
    kept, ceded = per_occurrence_xol(claims, 5000)
    assert kept.max() <= 5000 + 1e-9
    np.testing.assert_allclose(ceded, np.clip(claims - 5000, 0, None))
    # an annual total of retained claims is bounded by retention x claim count
    idx = np.arange(len(claims)) // 5
    assert np.bincount(idx, weights=kept).max() <= 5 * 5000 + 1e-6


def test_quota_share_scales_var_and_es_exactly():
    from risk_measures import var, es
    x = np.random.default_rng(2).lognormal(0, 1.2, 100_000)
    kept, _ = quota_share(x, .3)
    assert var(kept, .995) == pytest.approx(.7 * var(x, .995), rel=1e-12)
    assert es(kept, .995) == pytest.approx(.7 * es(x, .995), rel=1e-12)
