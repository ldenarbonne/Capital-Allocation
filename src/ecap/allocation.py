"""Economic capital allocation methods on a matrix of simulated losses.

All methods allocate EC(total) = rho(T) - E[T] (or rho(T) when `unexpected_only=False`),
where T is the sum of the business-unit losses L_k.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from .measures import economic_capital, tail_indices, var

METHODS = {
    "standalone": "Stand-alone (no diversification)",
    "proportional": "Pro-rata of stand-alone",
    "covariance": "Covariance (variance contribution)",
    "euler": "Euler (marginal contributions to the chosen risk measure)",
    "incremental": "Incremental (discrete marginal), rescaled",
    "shapley": "Shapley value",
}


class Allocator:
    def __init__(self, losses: pd.DataFrame, alpha: float = 0.999, measure: str = "ES",
                 unexpected_only: bool = True, euler_var_window: float = 0.001):
        self.losses = losses
        self.L = losses.to_numpy()
        self.T = self.L.sum(axis=1)
        self.alpha, self.measure, self.ul = alpha, measure.upper(), unexpected_only
        self.window = euler_var_window
        self.means = self.L.mean(axis=0)

    # ------------------------------------------------------------------ helpers
    def ec(self, losses: np.ndarray) -> float:
        return economic_capital(losses, self.alpha, self.measure, self.ul)

    @property
    def total_ec(self) -> float:
        return self.ec(self.T)

    def _net(self, contrib: np.ndarray) -> np.ndarray:
        return contrib - self.means if self.ul else contrib

    def _series(self, values) -> pd.Series:
        return pd.Series(np.asarray(values, dtype=float), index=self.losses.columns)

    # ------------------------------------------------------------------ methods
    def standalone(self) -> pd.Series:
        return self._series([self.ec(self.L[:, k]) for k in range(self.L.shape[1])])

    def proportional(self) -> pd.Series:
        sa = self.standalone()
        return sa / sa.sum() * self.total_ec

    def covariance(self) -> pd.Series:
        Tc = self.T - self.T.mean()
        cov = (self.L - self.means).T @ Tc / (len(Tc) - 1)
        return self._series(cov / Tc.var(ddof=1) * self.total_ec)

    def euler(self) -> pd.Series:
        """Euler allocation consistent with the chosen risk measure (fully additive)."""
        return self.euler_var() if self.measure == "VAR" else self.euler_es()

    def euler_es(self) -> pd.Series:
        """E[L_k | T >= VaR_alpha(T)] (- E[L_k]): exact Euler allocation for ES."""
        idx = tail_indices(self.T, self.alpha)
        return self._series(self._net(self.L[idx].mean(axis=0)))

    def euler_var(self) -> pd.Series:
        """E[L_k | T = VaR_alpha(T)], estimated on a window of scenarios around the quantile,
        then rescaled so that contributions add up exactly to the VaR-based capital."""
        n = self.T.size
        half = max(10, int(self.window * n))
        order = np.argsort(self.T)
        q = int(np.ceil(self.alpha * n)) - 1
        idx = order[max(0, q - half): min(n, q + half + 1)]
        raw = self._net(self.L[idx].mean(axis=0))
        target = var(self.T, self.alpha) - (self.T.mean() if self.ul else 0.0)
        return self._series(raw / raw.sum() * target)


    def incremental(self) -> pd.Series:
        """EC(T) - EC(T - L_k), rescaled to add up to total EC."""
        inc = np.array([self.total_ec - self.ec(self.T - self.L[:, k])
                        for k in range(self.L.shape[1])])
        return self._series(inc / inc.sum() * self.total_ec)

    def shapley(self, max_units: int = 12) -> pd.Series:
        """Exact Shapley value of the cooperative game v(S) = EC(sum of L_k, k in S)."""
        k = self.L.shape[1]
        if k > max_units:
            raise ValueError(f"Exact Shapley limited to {max_units} units (got {k}).")
        value = {(): 0.0}
        for size in range(1, k + 1):
            for subset in combinations(range(k), size):
                value[subset] = self.ec(self.L[:, list(subset)].sum(axis=1))
        phi = np.zeros(k)
        for subset, v in value.items():
            s = len(subset)
            for i in range(k):
                if i in subset:
                    continue
                with_i = tuple(sorted(subset + (i,)))
                w = factorial(s) * factorial(k - s - 1) / factorial(k)
                phi[i] += w * (value[with_i] - v)
        return self._series(phi)

    def run(self, methods: list[str] | None = None) -> pd.DataFrame:
        methods = methods or list(METHODS)
        out = {m: getattr(self, m)() for m in methods}
        return pd.DataFrame(out)


def performance(allocated: pd.Series, expected_loss: pd.Series, revenue: pd.Series,
                costs: pd.Series, hurdle: float, tax_rate: float = 0.0,
                capital_return: float = 0.0) -> pd.DataFrame:
    """Risk-adjusted performance per business unit.

    Risk-adjusted net income = (revenue - costs - EL) x (1 - tax) + capital_return x EC
    RAROC = risk-adjusted net income / EC ; EVA = net income - hurdle x EC
    """
    ec = allocated.clip(lower=1e-9)
    net = (revenue - costs - expected_loss) * (1 - tax_rate) + capital_return * ec
    df = pd.DataFrame({
        "revenue": revenue, "costs": costs, "expected_loss": expected_loss,
        "risk_adj_net_income": net, "economic_capital": allocated,
        "raroc": net / ec, "eva": net - hurdle * ec,
    })
    df["creates_value"] = df["raroc"] >= hurdle
    return df
