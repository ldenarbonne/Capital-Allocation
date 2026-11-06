"""Risk measures computed on simulated loss vectors (losses are positive numbers)."""
from __future__ import annotations

import numpy as np


def var(losses: np.ndarray, alpha: float) -> float:
    """Value-at-Risk: the alpha-quantile of the loss distribution."""
    return float(np.quantile(losses, alpha, method="higher"))


def expected_shortfall(losses: np.ndarray, alpha: float) -> float:
    """Expected Shortfall (CVaR): mean loss in the worst (1 - alpha) share of scenarios."""
    losses = np.asarray(losses)
    n_tail = max(1, int(np.ceil(round((1.0 - alpha) * losses.size, 9))))
    tail = np.partition(losses, losses.size - n_tail)[-n_tail:]
    return float(tail.mean())


def risk_measure(losses: np.ndarray, alpha: float, measure: str = "ES") -> float:
    measure = measure.upper()
    if measure == "VAR":
        return var(losses, alpha)
    if measure == "ES":
        return expected_shortfall(losses, alpha)
    raise ValueError(f"Unknown risk measure: {measure}")


def economic_capital(losses: np.ndarray, alpha: float, measure: str = "ES",
                     unexpected_only: bool = True) -> float:
    """Economic capital = risk measure, net of expected loss when `unexpected_only`."""
    rm = risk_measure(losses, alpha, measure)
    return rm - float(np.mean(losses)) if unexpected_only else rm


def tail_indices(total: np.ndarray, alpha: float) -> np.ndarray:
    """Indices of the scenarios that make up the ES tail of the total loss."""
    n_tail = max(1, int(np.ceil(round((1.0 - alpha) * total.size, 9))))
    return np.argpartition(total, total.size - n_tail)[-n_tail:]
