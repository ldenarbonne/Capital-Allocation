import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecap import (
    Allocator,
    CorrelationAssumptions,
    asrf_capital,
    expected_shortfall,
    load_sample,
    performance,
    simulate,
    validate,
    var,
)
from ecap.dependence import build_correlation, nearest_psd


@pytest.fixture(scope="module")
def sim():
    return simulate(load_sample(), n_sims=20_000, seed=7)


def test_var_es_on_uniform_grid():
    x = np.arange(1, 1001, dtype=float)
    assert var(x, 0.99) == pytest.approx(991, abs=1)
    assert expected_shortfall(x, 0.99) == pytest.approx(995.5)


def test_es_exceeds_var(sim):
    assert expected_shortfall(sim.total, 0.99) >= var(sim.total, 0.99)


def test_sample_portfolio_is_valid():
    assert validate(load_sample()) == []


def test_validation_catches_errors():
    bad = pd.DataFrame({"unit": ["A"], "risk_type": ["credit"], "exposure": [1.0], "pd": [1.5]})
    errors = validate(bad)
    assert any("pd" in e for e in errors) and any("lgd" in e for e in errors)


def test_simulation_shape_and_reproducibility():
    pf = load_sample()
    a = simulate(pf, n_sims=5_000, seed=1).losses
    b = simulate(pf, n_sims=5_000, seed=1).losses
    assert a.shape == (5_000, len(pf))
    pd.testing.assert_frame_equal(a, b)


def test_credit_expected_loss_matches_pd_lgd_ead(sim):
    pf = load_sample().set_index("unit")
    for unit in pf.index[pf["risk_type"] == "credit"]:
        r = pf.loc[unit]
        assert sim.losses[unit].mean() == pytest.approx(r.exposure * r.pd * r.lgd, rel=0.1)


def test_correlation_matrix_is_valid():
    corr = build_correlation(["credit", "market", "operational", "credit"], CorrelationAssumptions())
    assert np.all(np.linalg.eigvalsh(corr.to_numpy()) > 0)
    fixed = nearest_psd(np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]]))
    assert np.all(np.linalg.eigvalsh(fixed) > 0) and np.allclose(np.diag(fixed), 1)


@pytest.mark.parametrize("measure", ["ES", "VaR"])
@pytest.mark.parametrize("method", ["proportional", "covariance", "euler", "incremental", "shapley"])
def test_full_allocation(sim, measure, method):
    alloc = Allocator(sim.losses, alpha=0.995, measure=measure)
    assert getattr(alloc, method)().sum() == pytest.approx(alloc.total_ec, rel=1e-6)


def test_diversification_benefit(sim):
    alloc = Allocator(sim.losses, alpha=0.999)
    assert alloc.standalone().sum() > alloc.total_ec


def test_shapley_two_units_closed_form(sim):
    sub = sim.losses.iloc[:, :2]
    alloc = Allocator(sub, alpha=0.99)
    sa = alloc.standalone()
    expected0 = 0.5 * sa.iloc[0] + 0.5 * (alloc.total_ec - sa.iloc[1])
    assert alloc.shapley().iloc[0] == pytest.approx(expected0)


def test_asrf_known_value():
    # PD 1%, LGD 45%, rho 19.25% (Basel corporate), EAD 100 -> K ~ 5.9% of EAD at 99.9% (no maturity adjustment)
    k = asrf_capital(100, 0.01, 0.45, 0.1925)
    assert 5.6 < k < 6.2


def test_performance_metrics():
    idx = ["A", "B"]
    perf = performance(pd.Series([100.0, 50.0], idx), pd.Series([5.0, 5.0], idx),
                       pd.Series([30.0, 10.0], idx), pd.Series([10.0, 5.0], idx), hurdle=0.10)
    assert perf.loc["A", "raroc"] == pytest.approx(0.15)
    assert perf.loc["B", "eva"] == pytest.approx(-5.0)
    assert list(perf["creates_value"]) == [True, False]
