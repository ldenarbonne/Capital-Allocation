"""Stand-alone loss models for each risk type, driven by a common systemic driver X.

Convention: every model maps a high value of its driver X to a high loss, so that the
correlation between drivers translates into tail dependence between business units.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- credit
def credit_losses(x: np.ndarray, rng: np.random.Generator, exposure: float, pd_: float,
                  lgd: float, asset_corr: float, n_obligors: int) -> np.ndarray:
    """One-factor Vasicek / Gaussian copula portfolio with finite granularity.

    Conditional on the systemic factor X, defaults are Binomial(n, p(X)) with
    p(X) = Phi((Phi^-1(PD) + sqrt(rho) X) / sqrt(1 - rho)).
    """
    n_obligors = max(1, int(n_obligors))
    cond_pd = stats.norm.cdf((stats.norm.ppf(pd_) + np.sqrt(asset_corr) * x)
                             / np.sqrt(1.0 - asset_corr))
    defaults = rng.binomial(n_obligors, cond_pd)
    return defaults * (exposure / n_obligors) * lgd


def asrf_capital(exposure: float, pd_: float, lgd: float, asset_corr: float,
                 alpha: float = 0.999) -> float:
    """Analytic ASRF (Basel IRB-style, no maturity adjustment) unexpected loss, for reference."""
    cond_pd = stats.norm.cdf((stats.norm.ppf(pd_) + np.sqrt(asset_corr) * stats.norm.ppf(alpha))
                             / np.sqrt(1.0 - asset_corr))
    return float(exposure * lgd * (cond_pd - pd_))


# --------------------------------------------------------------------------- market
def market_losses(x: np.ndarray, exposure: float, volatility: float, tail_df: float,
                  horizon_years: float = 1.0) -> np.ndarray:
    """Fat-tailed P&L: Student-t shock (unit variance) mapped from the driver's quantile."""
    u = np.clip(stats.norm.cdf(x), 1e-12, 1 - 1e-12)
    if tail_df and tail_df > 2:
        shock = stats.t.ppf(u, tail_df) * np.sqrt((tail_df - 2.0) / tail_df)
    else:
        shock = stats.norm.ppf(u)
    return exposure * volatility * np.sqrt(horizon_years) * shock


# ---------------------------------------------------------------------- operational
def lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """Lognormal (mu, sigma) from a severity mean and coefficient of variation."""
    sigma2 = np.log1p(cv ** 2)
    return float(np.log(mean) - sigma2 / 2.0), float(np.sqrt(sigma2))


def operational_losses(x: np.ndarray, rng: np.random.Generator, frequency: float,
                       severity_mean: float, severity_cv: float) -> np.ndarray:
    """Loss Distribution Approach: Poisson frequency x lognormal severity.

    Aggregate annual losses are simulated independently, then re-ordered so that their
    ranks follow the ranks of the driver X (Iman-Conover style rank coupling).
    """
    n = x.size
    counts = rng.poisson(frequency, n)
    mu, sigma = lognormal_params(severity_mean, severity_cv)
    severities = rng.lognormal(mu, sigma, counts.sum())
    agg = np.bincount(np.repeat(np.arange(n), counts), weights=severities, minlength=n)
    out = np.empty(n)
    out[np.argsort(x)] = np.sort(agg)
    return out
