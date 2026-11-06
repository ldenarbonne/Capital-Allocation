"""Dependence structure between business units: a Gaussian copula on risk drivers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RISK_TYPES = ("credit", "market", "operational")


@dataclass
class CorrelationAssumptions:
    """Driver correlations within and between risk types (inter-risk correlation)."""
    credit_credit: float = 0.50
    market_market: float = 0.60
    operational_operational: float = 0.30
    credit_market: float = 0.40
    credit_operational: float = 0.20
    market_operational: float = 0.20

    def pair(self, a: str, b: str) -> float:
        a, b = sorted((a, b), key=RISK_TYPES.index)
        return getattr(self, f"{a}_{b}")


def nearest_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix onto the nearest valid correlation matrix (eigenvalue clipping)."""
    sym = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    fixed = vecs @ np.diag(np.clip(vals, eps, None)) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def build_correlation(risk_types: list[str], assumptions: CorrelationAssumptions) -> pd.DataFrame:
    """Correlation matrix of the business-unit risk drivers, from risk-type-level assumptions."""
    k = len(risk_types)
    corr = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            corr[i, j] = corr[j, i] = assumptions.pair(risk_types[i], risk_types[j])
    return pd.DataFrame(nearest_psd(corr))


def simulate_drivers(corr: np.ndarray, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Correlated standard normal drivers, shape (n_sims, k). High value = high loss."""
    chol = np.linalg.cholesky(corr)
    return rng.standard_normal((n_sims, corr.shape[0])) @ chol.T
