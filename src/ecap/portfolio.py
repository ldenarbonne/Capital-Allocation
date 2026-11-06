"""Portfolio definition and joint Monte Carlo simulation of business-unit losses."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .dependence import CorrelationAssumptions, build_correlation, simulate_drivers
from .risk_models import credit_losses, market_losses, operational_losses

REQUIRED = {
    "credit": ["exposure", "pd", "lgd", "asset_corr", "n_obligors"],
    "market": ["exposure", "volatility", "tail_df"],
    "operational": ["op_frequency", "op_severity_mean", "op_severity_cv"],
}
COLUMNS = ["unit", "risk_type", "exposure", "pd", "lgd", "asset_corr", "n_obligors",
           "volatility", "tail_df", "op_frequency", "op_severity_mean", "op_severity_cv",
           "revenue", "costs"]

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_portfolio.csv"


def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def validate(portfolio: pd.DataFrame) -> list[str]:
    """Return a list of human-readable errors (empty if the portfolio is valid)."""
    errors = []
    missing = {"unit", "risk_type"} - set(portfolio.columns)
    if missing:
        return [f"Missing columns: {sorted(missing)}"]
    if portfolio["unit"].duplicated().any():
        errors.append("Business unit names must be unique.")
    for _, row in portfolio.iterrows():
        rt = str(row["risk_type"]).lower()
        if rt not in REQUIRED:
            errors.append(f"{row['unit']}: unknown risk_type '{row['risk_type']}'.")
            continue
        for col in REQUIRED[rt]:
            if col not in row or pd.isna(row[col]):
                errors.append(f"{row['unit']}: '{col}' is required for {rt} risk.")
        if rt == "credit" and not pd.isna(row.get("pd")) and not 0 < row["pd"] < 1:
            errors.append(f"{row['unit']}: pd must be in (0, 1).")
        if rt == "credit" and not pd.isna(row.get("asset_corr")) and not 0 < row["asset_corr"] < 1:
            errors.append(f"{row['unit']}: asset_corr must be in (0, 1).")
    return errors


@dataclass
class SimulationResult:
    losses: pd.DataFrame          # (n_sims, n_units) loss scenarios per business unit
    driver_corr: pd.DataFrame     # correlation of the systemic drivers

    @property
    def total(self) -> np.ndarray:
        return self.losses.sum(axis=1).to_numpy()


def simulate(portfolio: pd.DataFrame, n_sims: int = 50_000, seed: int = 42,
             corr: CorrelationAssumptions | None = None) -> SimulationResult:
    errors = validate(portfolio)
    if errors:
        raise ValueError("; ".join(errors))
    corr = corr or CorrelationAssumptions()
    rng = np.random.default_rng(seed)
    risk_types = [str(r).lower() for r in portfolio["risk_type"]]
    units = list(portfolio["unit"])

    corr_df = build_correlation(risk_types, corr)
    corr_df.index = corr_df.columns = units
    drivers = simulate_drivers(corr_df.to_numpy(), n_sims, rng)

    losses = {}
    for k, (_, row) in enumerate(portfolio.iterrows()):
        x = drivers[:, k]
        rt = risk_types[k]
        if rt == "credit":
            losses[row["unit"]] = credit_losses(x, rng, row["exposure"], row["pd"], row["lgd"],
                                                row["asset_corr"], row["n_obligors"])
        elif rt == "market":
            losses[row["unit"]] = market_losses(x, row["exposure"], row["volatility"], row["tail_df"])
        else:
            losses[row["unit"]] = operational_losses(x, rng, row["op_frequency"],
                                                     row["op_severity_mean"], row["op_severity_cv"])
    return SimulationResult(pd.DataFrame(losses), corr_df)
