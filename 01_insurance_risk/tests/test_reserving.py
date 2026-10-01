"""Triangle construction, Chain-Ladder, Mack against the chainladder package, selection summary."""
import numpy as np
import pandas as pd
import pytest

import reserving_model as rm
from conftest import ROOT


def test_triangle_mask_and_conversion():
    full = pd.DataFrame([[100, 150, 180], [120, 180, 216], [140, 210, 252]], index=[2000, 2001, 2002], columns=[1, 2, 3])
    triangle = rm.mask_to_valuation(full, 2002)
    assert triangle.notna().sum().sum() == 6
    assert rm.validate_triangle(triangle).cumulative_matches_sum_of_incremental
    result = rm.chain_ladder(triangle)
    np.testing.assert_allclose(result.ultimate, [180, 216, 252])
    np.testing.assert_allclose(result.unpaid_reserve, [0, 36, 112])
    modified = full.copy(); modified.loc[2002, 2:] = 999999          # future cells must not leak into the fit
    np.testing.assert_allclose(rm.chain_ladder(rm.mask_to_valuation(modified, 2002)).ultimate, result.ultimate)
    with pytest.raises(ValueError):
        rm.chain_ladder(rm.mask_to_valuation(full, 2001))


@pytest.mark.parametrize('line', ['wkcomp', 'comauto'])
def test_mack_against_reference(line):
    import chainladder as cl
    data, _ = rm.build_industry_triangle(ROOT / 'data', line)
    tri = rm.mask_to_valuation(data['cum_paid'], 2007)
    mine = rm.mack_chain_ladder(tri)
    ref = cl.MackChainladder().fit(cl.Development(average='volume', sigma_interpolation='mack').fit_transform(rm.package_triangle(tri)))
    np.testing.assert_allclose(mine.total_std_error, ref.total_mack_std_err_.values.ravel()[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(mine.ultimate, ref.ultimate_.values.ravel(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.sqrt(mine.total_process_variance), ref.total_process_risk_.values.ravel()[-1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.sqrt(mine.total_parameter_variance), ref.total_parameter_risk_.values.ravel()[-1], rtol=1e-6, atol=1e-6)
    assert mine.total_std_error**2 == pytest.approx(mine.total_process_variance + mine.total_parameter_variance, rel=1e-9)


def test_selection_summary_is_consistent_with_triangle_report():
    for line in ('wkcomp', 'comauto'):
        s = rm.selection_summary(ROOT / 'data', line)
        _, report = rm.build_industry_triangle(ROOT / 'data', line)
        assert (s['companies_total'], s['companies_retained']) == (report.n_companies_total, report.n_companies_full_square)
        assert 0 < s['companies_retained_pct'] <= 100 and 0 < s['earned_premium_retained_pct'] <= 100


def test_ultimate_equals_paid_to_date_plus_reserve_and_development_is_monotone():
    for line in ('wkcomp', 'comauto'):
        data, _ = rm.build_industry_triangle(ROOT / 'data', line)
        tri = rm.mask_to_valuation(data['cum_paid'], 2007)
        cl = rm.chain_ladder(tri)
        np.testing.assert_allclose(cl.ultimate, cl.latest_diagonal + cl.unpaid_reserve, rtol=1e-12)
        steps = tri.diff(axis=1).to_numpy()
        assert np.all(steps[~np.isnan(steps)] >= 0)                             # cumulative paid never falls here
        assert np.all(cl.unpaid_reserve >= -1e-9) and np.all(cl.age_to_age_factors >= 1)
        bf = rm.bornhuetter_ferguson(cl, data['earned_prem'], rm.a_priori_elr_from_mature_years(cl, data['earned_prem']))
        np.testing.assert_allclose(bf.ultimate, cl.latest_diagonal + bf.unpaid_reserve, rtol=1e-12)
