"""Out-of-sample evaluation of covariance forecasts on test portfolios."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import optimize as O
from . import risk as K


def test_portfolios(assets: list[str], n_random: int = 100, seed: int = 7) -> pd.DataFrame:
    """EW, global 60/40 and ``n_random`` Dirichlet(1) long-only portfolios (rows = portfolios)."""
    rng = np.random.default_rng(seed)
    rows = {"EW": np.full(len(assets), 1 / len(assets)), "60/40": O.benchmark_6040(assets).to_numpy()}
    for k in range(n_random):
        rows[f"rand{k:03d}"] = rng.dirichlet(np.ones(len(assets)))
    return pd.DataFrame(rows, index=assets).T


def variance_forecast_table(fc: dict, weekly: pd.DataFrame, ports: pd.DataFrame) -> pd.DataFrame:
    """Long table: date × portfolio × estimator with forecast and realised next-month variance.

    Realised proxy = Σ of squared weekly portfolio returns over the next month (unbiased for the
    h-week variance; QLIKE/MSE rankings stay consistent with a noisy proxy — Patton 2011).
    """
    W = ports.to_numpy()
    rows = []
    for k, (t, weeks) in enumerate(zip(fc["dates"], fc["next_weeks"])):
        Rn = weekly.loc[weeks].to_numpy()
        rv = ((Rn @ W.T) ** 2).sum(axis=0)
        for e in K.ESTIMATORS:
            f = np.einsum("pi,ij,pj->p", W, fc["covh"][e][k], W)
            rows.append(pd.DataFrame({"date": t, "portfolio": ports.index, "estimator": e,
                                      "forecast": f, "realised": rv}))
    return pd.concat(rows, ignore_index=True)


def min_variance_oos(fc: dict, weekly: pd.DataFrame, cap: float = 1.0) -> pd.DataFrame:
    """Engle–Colacito style test: next-month realised vol (annualised) of the min-variance
    portfolio built from each estimator, long-only (``cap``) and unconstrained GMV."""
    out = []
    for k, (t, weeks) in enumerate(zip(fc["dates"], fc["next_weeks"])):
        Rn = weekly.loc[weeks].to_numpy()
        for e in K.ESTIMATORS:
            S = fc["cov1w"][e][k]
            for kind, w in (("long-only", O.min_variance(S, cap)), ("GMV unconstrained", O.gmv_unconstrained(S))):
                out.append({"date": t, "estimator": e, "type": kind,
                            "realised_var": float(((Rn @ w) ** 2).sum()), "weeks": len(weeks),
                            "gross_exposure": float(np.abs(w).sum())})
    return pd.DataFrame(out)
