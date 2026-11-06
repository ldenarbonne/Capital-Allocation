"""Economic Capital Allocation Engine — Streamlit front-end.

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecap import (
    METHODS,
    Allocator,
    CorrelationAssumptions,
    asrf_capital,
    load_sample,
    performance,
    simulate,
    validate,
)
from ecap.portfolio import COLUMNS

st.set_page_config(page_title="Economic Capital Allocation Engine", page_icon="📊", layout="wide")

PALETTE = ["#1f4e79", "#2e86c1", "#76b7e5", "#c0392b", "#e67e22", "#16a085", "#7d3c98",
           "#95a5a6", "#2c3e50", "#d4ac0d"]


# ------------------------------------------------------------------ cached engine calls
@st.cache_data(show_spinner="Running Monte Carlo simulation…")
def run_simulation(portfolio: pd.DataFrame, n_sims: int, seed: int, corr: dict):
    res = simulate(portfolio, n_sims=n_sims, seed=seed, corr=CorrelationAssumptions(**corr))
    return res.losses, res.driver_corr


@st.cache_data(show_spinner="Allocating capital…")
def run_allocation(losses: pd.DataFrame, alpha: float, measure: str, ul: bool, methods: tuple):
    alloc = Allocator(losses, alpha=alpha, measure=measure, unexpected_only=ul)
    return alloc.run(list(methods)), alloc.total_ec


def fmt(x: float) -> str:
    return f"{x:,.1f}"


# ------------------------------------------------------------------------------ sidebar
st.sidebar.title("⚙️ Parameters")
st.sidebar.subheader("Risk measure")
measure = st.sidebar.radio("Measure", ["ES", "VaR"], horizontal=True,
                           help="Expected Shortfall is coherent and its Euler allocation is exact.")
alpha = st.sidebar.select_slider("Confidence level", options=[0.99, 0.995, 0.999, 0.9995],
                                 value=0.999, format_func=lambda a: f"{a:.2%}")
ul = st.sidebar.toggle("Capital = unexpected loss (net of EL)", value=True)

st.sidebar.subheader("Monte Carlo")
n_sims = st.sidebar.select_slider("Scenarios", options=[10_000, 25_000, 50_000, 100_000, 200_000],
                                  value=50_000, format_func=lambda n: f"{n:,}")
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.subheader("Inter-risk correlation of drivers")
c = CorrelationAssumptions()
corr = {
    "credit_credit": st.sidebar.slider("Credit – Credit", 0.0, 0.95, c.credit_credit, 0.05),
    "market_market": st.sidebar.slider("Market – Market", 0.0, 0.95, c.market_market, 0.05),
    "operational_operational": st.sidebar.slider("Op – Op", 0.0, 0.95, c.operational_operational, 0.05),
    "credit_market": st.sidebar.slider("Credit – Market", 0.0, 0.95, c.credit_market, 0.05),
    "credit_operational": st.sidebar.slider("Credit – Op", 0.0, 0.95, c.credit_operational, 0.05),
    "market_operational": st.sidebar.slider("Market – Op", 0.0, 0.95, c.market_operational, 0.05),
}

st.sidebar.subheader("Performance")
hurdle = st.sidebar.slider("Hurdle rate (cost of equity)", 0.0, 0.30, 0.12, 0.01, format="%.2f")
tax = st.sidebar.slider("Tax rate", 0.0, 0.40, 0.25, 0.01, format="%.2f")
cap_ret = st.sidebar.slider("Return on allocated capital", 0.0, 0.10, 0.03, 0.005, format="%.3f")

# ------------------------------------------------------------------------------- header
st.title("📊 Economic Capital Allocation Engine")
st.caption("Integrated credit, market and operational risk · Gaussian-copula dependence · "
           "Monte Carlo · Euler, Shapley, covariance and incremental allocation · RAROC / EVA")

tab_pf, tab_dist, tab_alloc, tab_perf, tab_method = st.tabs(
    ["📁 Portfolio", "📉 Loss distribution", "🧮 Allocation", "💹 Performance", "📘 Methodology"])

# ---------------------------------------------------------------------------- portfolio
with tab_pf:
    st.subheader("Business units")
    st.markdown("Edit the table directly or upload a CSV with the same columns. "
                "Amounts are in millions; leave non-applicable fields empty.")
    up = st.file_uploader("Upload portfolio CSV", type="csv")
    base = pd.read_csv(up) if up is not None else load_sample()
    base = base.reindex(columns=COLUMNS)
    portfolio = st.data_editor(
        base, num_rows="dynamic", width="stretch", key="pf",
        column_config={
            "risk_type": st.column_config.SelectboxColumn(
                options=["credit", "market", "operational"], required=True),
            "pd": st.column_config.NumberColumn(format="%.4f"),
            "lgd": st.column_config.NumberColumn(format="%.2f"),
            "asset_corr": st.column_config.NumberColumn(format="%.2f"),
            "volatility": st.column_config.NumberColumn(format="%.3f"),
        })
    st.download_button("⬇️ Download portfolio CSV", portfolio.to_csv(index=False),
                       "portfolio.csv", "text/csv")
    with st.expander("Column dictionary"):
        st.markdown("""
| Risk type | Required fields | Meaning |
|---|---|---|
| credit | `exposure`, `pd`, `lgd`, `asset_corr`, `n_obligors` | EAD, 1-year PD, LGD, Vasicek asset correlation, number of (equal-sized) obligors |
| market | `exposure`, `volatility`, `tail_df` | Position, annual volatility, Student-t degrees of freedom (fat tails) |
| operational | `op_frequency`, `op_severity_mean`, `op_severity_cv` | Poisson annual frequency, lognormal severity mean and coefficient of variation |
| all | `revenue`, `costs` | Annual revenue and operating costs (for RAROC / EVA) |
""")

portfolio = portfolio.dropna(subset=["unit", "risk_type"]).reset_index(drop=True)
errors = validate(portfolio)
if errors:
    for tab in (tab_dist, tab_alloc, tab_perf):
        with tab:
            st.error("Portfolio is invalid:\n\n- " + "\n- ".join(errors))
    st.stop()
if len(portfolio) < 2:
    st.warning("Add at least two business units to measure diversification.")
    st.stop()

losses, driver_corr = run_simulation(portfolio, int(n_sims), int(seed), corr)
methods = tuple(m for m in METHODS if m != "shapley" or len(portfolio) <= 12)
alloc_df, total_ec = run_allocation(losses, alpha, measure, ul, methods)
total = losses.sum(axis=1)
standalone_sum = alloc_df["standalone"].sum()
div_benefit = 1 - total_ec / standalone_sum

# ---------------------------------------------------------------------- distribution
with tab_dist:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected loss", fmt(total.mean()))
    k2.metric(f"Total economic capital ({measure} {alpha:.2%})", fmt(total_ec))
    k3.metric("Sum of stand-alone EC", fmt(standalone_sum))
    k4.metric("Diversification benefit", f"{div_benefit:.1%}")

    var_t = float(np.quantile(total, alpha))
    es_t = float(total[total >= var_t].mean())
    fig = px.histogram(total, nbins=200, color_discrete_sequence=[PALETTE[1]],
                       labels={"value": "Total annual loss"}, log_y=True)
    for x, name, col in [(total.mean(), "EL", PALETTE[5]), (var_t, f"VaR {alpha:.2%}", PALETTE[4]),
                         (es_t, f"ES {alpha:.2%}", PALETTE[3])]:
        fig.add_vline(x=x, line_dash="dash", line_color=col, annotation_text=name)
    fig.update_layout(showlegend=False, height=420, yaxis_title="Scenarios (log scale)",
                      title="Distribution of the total portfolio loss")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stand-alone statistics by business unit**")
        stats_df = pd.DataFrame({
            "risk_type": portfolio.set_index("unit")["risk_type"],
            "EL": losses.mean(), "Std dev": losses.std(),
            f"VaR {alpha:.2%}": losses.quantile(alpha),
            "Stand-alone EC": alloc_df["standalone"],
        })
        st.dataframe(stats_df.style.format(precision=1, thousands=","), width="stretch")
    with c2:
        st.markdown("**Correlation of risk drivers (after PSD correction)**")
        st.plotly_chart(px.imshow(driver_corr, text_auto=".2f", color_continuous_scale="Blues",
                                  zmin=0, zmax=1, aspect="auto").update_layout(height=380),
                        width="stretch")

    credit = portfolio[portfolio["risk_type"] == "credit"]
    if not credit.empty:
        with st.expander("Credit: Monte Carlo vs analytic ASRF (infinitely granular, 99.9%)"):
            comp = pd.DataFrame({
                "MC stand-alone EC": alloc_df.loc[credit["unit"], "standalone"].values,
                "ASRF capital (99.9%)": [asrf_capital(r.exposure, r.pd, r.lgd, r.asset_corr)
                                         for r in credit.itertuples()],
            }, index=credit["unit"])
            st.dataframe(comp.style.format(precision=1, thousands=","), width="stretch")
            st.caption("Differences come from the chosen measure/confidence, finite granularity "
                       "(name concentration) and Monte Carlo noise.")

# ------------------------------------------------------------------------ allocation
with tab_alloc:
    st.subheader("Allocated economic capital by method")
    labels = {m: METHODS[m] for m in alloc_df.columns}
    show = st.multiselect("Methods to compare", list(alloc_df.columns), default=list(alloc_df.columns),
                          format_func=lambda m: labels[m])
    long = alloc_df[show].reset_index(names="unit").melt("unit", var_name="method", value_name="ec")
    long["method"] = long["method"].map(labels)
    fig = px.bar(long, x="unit", y="ec", color="method", barmode="group",
                 color_discrete_sequence=PALETTE, labels={"ec": "Economic capital", "unit": ""})
    fig.update_layout(height=450, legend_title_text="")
    st.plotly_chart(fig, width="stretch")

    table = alloc_df[show].copy()
    table.loc["TOTAL"] = table.sum()
    st.dataframe(table.rename(columns=labels).style.format(precision=1, thousands=","),
                 width="stretch")

    st.markdown("**Share of total capital and diversification factor**")
    ref = st.selectbox("Reference method", [m for m in show if m != "standalone"] or ["euler"],
                       format_func=lambda m: labels.get(m, m))
    if ref in alloc_df:
        share = pd.DataFrame({
            "Allocated EC": alloc_df[ref],
            "Share of total": alloc_df[ref] / total_ec,
            "Diversification factor (alloc / stand-alone)": alloc_df[ref] / alloc_df["standalone"],
        })
        st.dataframe(share.style.format({"Allocated EC": "{:,.1f}", "Share of total": "{:.1%}",
                                         "Diversification factor (alloc / stand-alone)": "{:.2f}"}),
                     width="stretch")
        fig = go.Figure(go.Waterfall(
            x=list(alloc_df.index) + ["Total EC"],
            measure=["relative"] * len(alloc_df) + ["total"],
            y=list(alloc_df[ref]) + [0],
            connector={"line": {"color": "#bbb"}},
            increasing={"marker": {"color": PALETTE[0]}},
            decreasing={"marker": {"color": PALETTE[5]}},
            totals={"marker": {"color": PALETTE[3]}}))
        fig.update_layout(title=f"Build-up of total capital — {labels[ref]}", height=400)
        st.plotly_chart(fig, width="stretch")

    st.download_button("⬇️ Download allocation CSV", alloc_df.to_csv(), "allocation.csv", "text/csv")

# ----------------------------------------------------------------------- performance
with tab_perf:
    st.subheader("Risk-adjusted performance (RAROC / EVA)")
    method = st.selectbox("Allocation method used for capital charges",
                          [m for m in alloc_df.columns if m != "standalone"],
                          index=list(alloc_df.columns).index("euler") - 1,
                          format_func=lambda m: METHODS[m])
    pf = portfolio.set_index("unit")
    perf = performance(alloc_df[method], losses.mean(), pf["revenue"].fillna(0),
                       pf["costs"].fillna(0), hurdle, tax, cap_ret)
    k1, k2, k3 = st.columns(3)
    k1.metric("Portfolio RAROC", f"{perf['risk_adj_net_income'].sum() / total_ec:.1%}")
    k2.metric("Portfolio EVA", fmt(perf["eva"].sum()))
    k3.metric("Units below hurdle", int((~perf["creates_value"]).sum()))

    fig = px.bar(perf.reset_index(names="unit"), x="unit", y="raroc",
                 color=perf["creates_value"].map({True: "Above hurdle", False: "Below hurdle"}).values,
                 color_discrete_map={"Above hurdle": PALETTE[5], "Below hurdle": PALETTE[3]},
                 labels={"raroc": "RAROC", "unit": "", "color": ""})
    fig.add_hline(y=hurdle, line_dash="dash", annotation_text=f"Hurdle {hurdle:.0%}")
    fig.update_layout(yaxis_tickformat=".0%", height=420)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(perf.style.format({c: "{:,.1f}" for c in ["revenue", "costs", "expected_loss",
                                                           "risk_adj_net_income",
                                                           "economic_capital", "eva"]}
                                   | {"raroc": "{:.1%}"}), width="stretch")
    st.caption("Risk-adjusted net income = (revenue − costs − EL) × (1 − tax) + return on capital × EC. "
               "RAROC = net income / EC. EVA = net income − hurdle × EC.")

# ----------------------------------------------------------------------- methodology
with tab_method:
    st.markdown((Path(__file__).resolve().parents[1] / "docs" / "methodology.md").read_text())
