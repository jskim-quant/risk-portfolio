"""Pricing identities: expense basis, exposure weighting, capital loading, held-out split."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import pricing_model as pm

ASSUMPTIONS = pm.PremiumAssumptions(30, .15, .05, .01)


def test_expense_and_profit_are_shares_of_gross_premium():
    pure, load, exposure = np.array([100., 300.]), np.array([10., 30.]), np.array([.2, .8])
    premium = pm.gross_premium(pure, load, ASSUMPTIONS)
    expense = 30 + .15 * premium
    np.testing.assert_allclose(premium - pure - expense, load + .05 * premium)
    assert np.sum(pure * exposure) == pytest.approx(260)
    with pytest.raises(ValueError):
        pm.gross_premium(pure, load, replace(ASSUMPTIONS, variable_expense_ratio=.95))


def test_capital_loading_is_proportional_to_k_and_pure_premium():
    pure = np.array([100., 300.])
    np.testing.assert_allclose(pm.capital_loading_pro_rata(pure, 200, 100, .1), pure * .1)
    np.testing.assert_allclose(pm.capital_loading_pro_rata(pure, 200, 100, 0.0), 0.0)
    premiums = [pm.gross_premium(pure, pm.capital_loading_pro_rata(pure, 6000, 300, k), replace(ASSUMPTIONS, capital_loading_k=k))
                for k in (0, .005, .01, .02)]
    assert all(np.all(b > a) for a, b in zip(premiums, premiums[1:]))
    with pytest.raises(ValueError):
        pm.capital_loading_pro_rata(pure, 200, 100, -.1)


def test_policy_split_is_disjoint_and_rejects_duplicates():
    frame = pd.DataFrame({'IDpol': range(100), 'Exposure': np.ones(100)})
    train, test = pm.train_test_split_pricing(frame)
    assert len(test) == 25 and not set(train.IDpol) & set(test.IDpol)
    with pytest.raises(ValueError):
        pm.train_test_split_pricing(pd.concat([frame, frame]))
