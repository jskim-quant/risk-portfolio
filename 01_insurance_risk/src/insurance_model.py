"""Exposure-adjusted NB2 frequency and spliced lognormal/GPD claim severity."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, optimize

from risk_measures import var, es, capital_proxy  # noqa: F401  (shared definitions, re-exported)

CANDIDATE_THRESHOLD_PCTS = [90, 92.5, 95, 96, 97, 97.5, 98, 98.5, 99, 99.5]
FINAL_THRESHOLD_PCT = 99.0
CLAIMNB_CAP = 4          # modeling cap; sensitivity in notebook 01
EXPOSURE_CAP = 1.0        # normalize exposure to at most one policy-year


@dataclass
class CleaningReport:
    n_policies: int
    n_exposure_capped: int
    exposure_max_before: float
    n_claimnb_capped: int
    claimnb_max_before: int
    n_severity_records: int
    n_severity_orphans: int          # sev rows whose IDpol isn't in freq at all
    total_claims_freq_table: float
    total_severity_records: int
    freq_sev_gap: float
    freq_sev_gap_pct: float


def clean_frequency(freq: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the frequency table on its own terms. Two real, documented
    freMTPL2 data issues are fixed here; nothing about claim counts is
    inferred from the severity table."""
    if not np.isfinite(freq[["Exposure", "ClaimNb"]]).all().all() or (freq.Exposure <= 0).any() or (freq.ClaimNb < 0).any():
        raise ValueError("Frequency data require positive exposure and finite nonnegative counts")
    if freq.IDpol.duplicated().any():
        raise ValueError("Policy IDs must be unique")
    freq = freq.copy()
    n_exposure_capped = int((freq["Exposure"] > EXPOSURE_CAP).sum())
    exposure_max_before = float(freq["Exposure"].max())
    freq["Exposure"] = freq["Exposure"].clip(upper=EXPOSURE_CAP)

    n_claimnb_capped = int((freq["ClaimNb"] > CLAIMNB_CAP).sum())
    claimnb_max_before = int(freq["ClaimNb"].max())
    freq["ClaimNb"] = freq["ClaimNb"].clip(upper=CLAIMNB_CAP)

    report = CleaningReport(
        n_policies=len(freq),
        n_exposure_capped=n_exposure_capped,
        exposure_max_before=exposure_max_before,
        n_claimnb_capped=n_claimnb_capped,
        claimnb_max_before=claimnb_max_before,
        n_severity_records=0, n_severity_orphans=0,
        total_claims_freq_table=float(freq["ClaimNb"].sum()),
        total_severity_records=0, freq_sev_gap=0.0, freq_sev_gap_pct=0.0,
    )
    return freq, report


def document_frequency_severity_gap(freq: pd.DataFrame, sev: pd.DataFrame,
                                     report: CleaningReport) -> CleaningReport:
    """Report count-table and severity-table mismatches without reconciling missing claims."""
    total_claims = float(freq["ClaimNb"].sum())
    n_sev = len(sev)
    orphans = int((~sev["IDpol"].isin(freq["IDpol"])).sum())
    report.n_severity_records = n_sev
    report.n_severity_orphans = orphans
    report.total_severity_records = n_sev
    report.freq_sev_gap = total_claims - n_sev
    report.freq_sev_gap_pct = 100.0 * (total_claims - n_sev) / total_claims
    return report


@dataclass
class FrequencyFit:
    lambda_pois: float
    loglik_pois: float
    aic_pois: float
    bic_pois: float
    lambda_nb: float
    r_nb: float
    loglik_nb: float
    aic_nb: float
    bic_nb: float
    lr_stat: float
    lr_pvalue: float
    n: int
    total_claims: float
    total_exposure: float


def nb2_negloglik(theta, N: np.ndarray, E: np.ndarray) -> float:
    """Negative NB2 log-likelihood in (log lambda, log r) with exposure E."""
    log_lam, log_r = theta
    lam, r = np.exp(log_lam), np.exp(log_r)
    mu = lam * E
    p = r / (r + mu)
    if not np.all(np.isfinite(p)) or np.any(p <= 0) or np.any(p >= 1):
        return 1e10
    return -np.sum(stats.nbinom.logpmf(N, r, p))


def frequency_parameter_draws(freq: pd.DataFrame, fit: "FrequencyFit", n_draws: int,
                              seed: int = 0, step: float = 1e-3) -> np.ndarray:
    """Draws of (lambda, r) from the asymptotic normal law of the NB2 MLE in log parameters.

    The observed information is a central finite-difference Hessian of the log-likelihood at the fit.
    A full policy-level bootstrap would refit 678k rows per draw; this is the cheaper approximation.
    """
    N = freq["ClaimNb"].to_numpy(dtype=float)
    E = freq["Exposure"].to_numpy(dtype=float)
    x0 = np.log([fit.lambda_nb, fit.r_nb])
    f = lambda t: nb2_negloglik(t, N, E)
    h = step
    H = np.zeros((2, 2))
    f0 = f(x0)
    for i in range(2):
        ei = np.zeros(2); ei[i] = h
        H[i, i] = (f(x0 + ei) - 2 * f0 + f(x0 - ei)) / h**2
    e0, e1 = np.array([h, 0.0]), np.array([0.0, h])
    H[0, 1] = H[1, 0] = (f(x0 + e0 + e1) - f(x0 + e0 - e1) - f(x0 - e0 + e1) + f(x0 - e0 - e1)) / (4 * h**2)
    cov = np.linalg.inv(H)
    draws = np.random.default_rng(seed).multivariate_normal(x0, cov, size=n_draws)
    return np.exp(draws)


def fit_frequency(freq: pd.DataFrame) -> FrequencyFit:
    """Fit exposure-adjusted Poisson and NB2 count likelihoods."""
    N = freq["ClaimNb"].to_numpy(dtype=float)
    E = freq["Exposure"].to_numpy(dtype=float)
    n = len(N)

    lam_pois = N.sum() / E.sum()
    ll_pois = float(np.sum(stats.poisson.logpmf(N, mu=lam_pois * E)))
    aic_pois, bic_pois = 2 * 1 - 2 * ll_pois, 1 * np.log(n) - 2 * ll_pois

    res = optimize.minimize(
        nb2_negloglik, x0=[np.log(lam_pois), 0.0], args=(N, E), method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 5000},
    )
    if not res.success:
        raise RuntimeError(f"NB2 calibration failed: {res.message}")
    lam_nb, r_nb = np.exp(res.x)
    ll_nb = -res.fun
    aic_nb, bic_nb = 2 * 2 - 2 * ll_nb, 2 * np.log(n) - 2 * ll_nb

    lr_stat = 2 * (ll_nb - ll_pois)
    lr_p = float(stats.chi2.sf(lr_stat, df=1))

    return FrequencyFit(
        lambda_pois=float(lam_pois), loglik_pois=ll_pois, aic_pois=float(aic_pois), bic_pois=float(bic_pois),
        lambda_nb=float(lam_nb), r_nb=float(r_nb), loglik_nb=float(ll_nb), aic_nb=float(aic_nb), bic_nb=float(bic_nb),
        lr_stat=float(lr_stat), lr_pvalue=lr_p,
        n=n, total_claims=float(N.sum()), total_exposure=float(E.sum()),
    )


def observed_vs_fitted_counts(freq: pd.DataFrame, fit: FrequencyFit, k_max: int = 5) -> pd.DataFrame:
    """Mix fitted count probabilities over observed policy exposures."""
    N = freq["ClaimNb"].to_numpy(dtype=float)
    E = freq["Exposure"].to_numpy(dtype=float)
    ks = np.arange(0, k_max + 1)
    obs = np.array([(N == k).mean() for k in ks])
    obs = np.append(obs, (N > k_max).mean())

    pois = np.array([np.mean(stats.poisson.pmf(k, mu=fit.lambda_pois * E)) for k in ks])
    pois = np.append(pois, 1 - pois.sum())

    p_vec = fit.r_nb / (fit.r_nb + fit.lambda_nb * E)
    nb = np.array([np.mean(stats.nbinom.pmf(k, fit.r_nb, p_vec)) for k in ks])
    nb = np.append(nb, 1 - nb.sum())

    labels = [str(k) for k in ks] + [f"{k_max + 1}+"]
    return pd.DataFrame({"ClaimNb": labels, "Observed": obs, "Poisson_fit": pois, "NB2_fit": nb})


def threshold_diagnostics(x: np.ndarray, pcts=CANDIDATE_THRESHOLD_PCTS, min_exceed: int = 20) -> pd.DataFrame:
    """Report mean excess and GPD stability across candidate thresholds."""
    rows = []
    for pct in pcts:
        u = np.percentile(x, pct)
        exceed = x[x > u] - u
        ne = len(exceed)
        me = exceed.mean() if ne > 0 else np.nan
        if ne >= min_exceed:
            xi, _, sigma = stats.genpareto.fit(exceed, floc=0)
        else:
            xi, sigma = np.nan, np.nan
        rows.append(dict(pct=pct, threshold=u, n_exceed=ne, mean_excess=me, xi=xi, sigma=sigma))
    return pd.DataFrame(rows)


def mean_residual_life(x: np.ndarray, lo_pct=80, hi_pct=99.5, n_points=60, min_exceed=6):
    u_grid = np.linspace(np.percentile(x, lo_pct), np.percentile(x, hi_pct), n_points)
    mrl = []
    for u in u_grid:
        exc = x[x > u] - u
        mrl.append(exc.mean() if len(exc) > min_exceed else np.nan)
    return u_grid, np.array(mrl)


@dataclass
class SeverityFit:
    threshold: float
    threshold_pct: float
    alpha: float          # P(X <= threshold), i.e. body probability
    n_body: int
    n_tail: int
    mu_body: float
    sigma_body: float
    xi_gpd: float
    sigma_gpd: float


def fit_truncated_lognormal(body: np.ndarray, threshold: float) -> tuple[float, float, float]:
    """Fit the conditional lognormal likelihood below the threshold."""
    log_body = np.log(body)
    n_body = len(body)
    mu0, sigma0 = log_body.mean(), log_body.std()

    def negloglik(theta):
        mu, log_sigma = theta
        sigma = np.exp(log_sigma)
        if sigma <= 1e-8:
            return 1e10
        log_flnx = stats.norm.logpdf(log_body, loc=mu, scale=sigma) - log_body
        Fu = stats.norm.cdf((np.log(threshold) - mu) / sigma)
        if Fu <= 1e-12:
            return 1e10
        return -(log_flnx.sum() - n_body * np.log(Fu))

    res = optimize.minimize(negloglik, x0=[mu0, np.log(sigma0)], method="Nelder-Mead",
                             options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 5000})
    if not res.success:
        raise RuntimeError(f"Severity body calibration failed: {res.message}")
    mu_hat, sigma_hat = res.x[0], float(np.exp(res.x[1]))
    return float(mu_hat), sigma_hat, float(-res.fun)


def fit_severity(sev_amounts: np.ndarray, threshold_pct: float = FINAL_THRESHOLD_PCT) -> SeverityFit:
    x = np.asarray(sev_amounts, dtype=float)
    u = float(np.percentile(x, threshold_pct))
    alpha = float(np.mean(x <= u))
    body = x[x <= u]
    tail_exceed = x[x > u] - u

    mu_hat, sigma_hat, _ = fit_truncated_lognormal(body, u)
    xi_gpd, _, sigma_gpd = stats.genpareto.fit(tail_exceed, floc=0)

    return SeverityFit(
        threshold=u, threshold_pct=threshold_pct, alpha=alpha,
        n_body=len(body), n_tail=len(tail_exceed),
        mu_body=mu_hat, sigma_body=sigma_hat,
        xi_gpd=float(xi_gpd), sigma_gpd=float(sigma_gpd),
    )


@dataclass
class CalibratedModel:
    lambda_freq: float   # NB2 mean rate at E=1
    r_freq: float        # NB2 dispersion
    threshold: float
    alpha: float
    mu_body: float
    sigma_body: float
    xi_gpd: float
    sigma_gpd: float
    freq_model: str = "NB2 (exposure-adjusted)"
    severity_model: str = "Truncated-Lognormal body + GPD tail (proper POT splice)"

    def to_json(self, path: str | Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "CalibratedModel":
        return cls(**json.loads(Path(path).read_text()))


def build_calibrated_model(freq_fit: FrequencyFit, sev_fit: SeverityFit) -> CalibratedModel:
    return CalibratedModel(
        lambda_freq=freq_fit.lambda_nb, r_freq=freq_fit.r_nb,
        threshold=sev_fit.threshold, alpha=sev_fit.alpha,
        mu_body=sev_fit.mu_body, sigma_body=sev_fit.sigma_body,
        xi_gpd=sev_fit.xi_gpd, sigma_gpd=sev_fit.sigma_gpd,
    )


def severity_mean(model: "CalibratedModel") -> float:
    """Return the spliced severity mean; infinity when GPD shape is at least one."""
    mu, sigma, u = model.mu_body, model.sigma_body, model.threshold
    alpha, xi, sigma_gpd = model.alpha, model.xi_gpd, model.sigma_gpd
    if xi >= 1:
        return float("inf")
    Fu = stats.norm.cdf((np.log(u) - mu) / sigma)
    e_body = np.exp(mu + sigma**2 / 2) * stats.norm.cdf((np.log(u) - mu - sigma**2) / sigma) / Fu
    e_tail = u + sigma_gpd / (1 - xi)
    return float(alpha * e_body + (1 - alpha) * e_tail)


def capped_severity_mean(model: CalibratedModel, cap: float) -> float:
    """E[min(X, cap)] for the spliced severity (closed form, finite for every xi).

    Capping removes the infinite-variance tail, so a Monte Carlo mean of min(X, cap)
    converges at the usual 1/sqrt(n) rate. That makes it a clean check of the sampler.
    """
    u, mu, sigma = model.threshold, model.mu_body, model.sigma_body
    if cap <= u:
        raise ValueError("cap must exceed the splice threshold")
    Fu = stats.norm.cdf((np.log(u) - mu) / sigma)
    e_body = np.exp(mu + sigma**2 / 2) * stats.norm.cdf((np.log(u) - mu - sigma**2) / sigma) / Fu
    xi, s_g, y = model.xi_gpd, model.sigma_gpd, cap - u
    if abs(xi - 1.0) < 1e-12:
        raise ValueError("xi = 1 not supported")
    e_tail = u + s_g / (1 - xi) * (1 - (1 + xi * y / s_g) ** (1 - 1 / xi))
    return float(model.alpha * e_body + (1 - model.alpha) * e_tail)


def sample_truncated_lognormal(n: int, mu: float, sigma: float, threshold: float,
                                rng: np.random.Generator) -> np.ndarray:
    """Correct inverse-CDF sampling of X | X <= threshold. No point mass at
    the threshold (contrast with the old np.minimum(lognormal_draw, u))."""
    Fu = stats.norm.cdf((np.log(threshold) - mu) / sigma)
    u_unif = rng.uniform(0.0, Fu, size=n)
    z = stats.norm.ppf(u_unif)
    return np.exp(mu + sigma * z)


def draw_severities(n: int, model: CalibratedModel, rng: np.random.Generator) -> np.ndarray:
    """Draw n independent claim severities from the spliced truncated-lognormal / GPD model.

    Draw order (body indicator, then body values, then tail values) is part of the
    contract: changing it changes every seeded result in the project.
    """
    is_body = rng.random(n) < model.alpha
    severities = np.empty(n)
    n_body = int(is_body.sum())
    if n_body > 0:
        severities[is_body] = sample_truncated_lognormal(
            n_body, model.mu_body, model.sigma_body, model.threshold, rng)
    n_tail = n - n_body
    if n_tail > 0:
        severities[~is_body] = model.threshold + stats.genpareto.rvs(
            model.xi_gpd, scale=model.sigma_gpd, size=n_tail, random_state=rng)
    return severities


def spliced_severity_cdf(x: np.ndarray, model: CalibratedModel) -> np.ndarray:
    """CDF of the spliced severity: truncated lognormal below the threshold, GPD above it."""
    x = np.asarray(x, dtype=float)
    u = model.threshold
    Fu = stats.norm.cdf((np.log(u) - model.mu_body) / model.sigma_body)
    body = model.alpha * stats.norm.cdf((np.log(np.clip(x, 1e-300, None)) - model.mu_body) / model.sigma_body) / Fu
    tail = model.alpha + (1 - model.alpha) * stats.genpareto.cdf(x - u, model.xi_gpd, scale=model.sigma_gpd)
    return np.where(x <= u, body, tail)


def frequency_cleaning_summary(report: CleaningReport) -> dict:
    """Frequency-table fields of the cleaning report only.

    The severity fields are filled in by document_frequency_severity_gap() in notebook 01
    and are intentionally not loaded in notebooks that only use the frequency table.
    """
    skip = {"n_severity_records", "n_severity_orphans", "total_severity_records",
            "freq_sev_gap", "freq_sev_gap_pct"}
    return {k: v for k, v in asdict(report).items() if k not in skip}


def simulate_claims(n_years: int, model: CalibratedModel, rng: np.random.Generator,
                     freq_mult: float = 1.0, sev_mult: float = 1.0,
                     exposure: float = 1.0):
    """Draw NB2 counts and independent spliced severities; return year indices and claims. Exposure scales the mean with fixed dispersion, not the number of independent policies."""
    mu_shocked = model.lambda_freq * exposure * freq_mult
    p_shocked = model.r_freq / (model.r_freq + mu_shocked)
    n_claims_per_year = rng.negative_binomial(model.r_freq, p_shocked, size=n_years)
    total_claims = int(n_claims_per_year.sum())
    sim_index = np.repeat(np.arange(n_years), n_claims_per_year)

    if total_claims == 0:
        return sim_index, np.array([])

    severities = draw_severities(total_claims, model, rng)

    if sev_mult != 1.0:
        severities = severities * sev_mult

    return sim_index, severities


def aggregate(sim_index: np.ndarray, severities: np.ndarray, n_years: int) -> np.ndarray:
    if len(severities) == 0:
        return np.zeros(n_years)
    return np.bincount(sim_index, weights=severities, minlength=n_years)


def simulate_aggregate_losses(n_years: int, model: CalibratedModel, rng: np.random.Generator,
                               freq_mult: float = 1.0, sev_mult: float = 1.0,
                               exposure: float = 1.0) -> np.ndarray:
    """Convenience wrapper used where only the annual total is needed."""
    sim_index, severities = simulate_claims(n_years, model, rng, freq_mult, sev_mult, exposure)
    return aggregate(sim_index, severities, n_years)


