# Economic Capital Allocation Engine

[![CI](https://github.com/ldenarbonne/ecap-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/ldenarbonne/ecap-engine/actions/workflows/ci.yml)

An integrated **economic capital** model for a bank's business units — credit, market and operational
risk simulated jointly by Monte Carlo — with a **Streamlit** front-end to compare capital allocation
methods and measure **risk-adjusted performance (RAROC / EVA)**.

> *FR — Moteur de calcul et d'allocation du capital économique : simulation Monte Carlo intégrée des
> risques de crédit, de marché et opérationnel, agrégation par copule gaussienne, allocation par
> contributions d'Euler, Shapley, covariance et incrémentale, mesure de la performance ajustée du
> risque (RAROC / EVA).*

## Features

| Layer | What it does |
|---|---|
| **Credit risk** | One-factor Vasicek / Gaussian copula pools with finite granularity (Binomial conditional defaults); analytic ASRF (Basel IRB) benchmark |
| **Market risk** | Fat-tailed Student-t P&L per trading desk |
| **Operational risk** | Loss Distribution Approach — Poisson frequency × lognormal severity |
| **Aggregation** | Gaussian copula on business-unit risk drivers, inter-risk correlation by risk-type pair, nearest-PSD correction |
| **Risk measures** | VaR or Expected Shortfall, any confidence level, capital as unexpected loss (net of EL) or total |
| **Allocation** | Stand-alone, pro-rata, covariance, **Euler** (ES or VaR), incremental, **Shapley** |
| **Performance** | RAROC, EVA, hurdle-rate test per business unit |
| **App** | Editable portfolio, CSV upload/download, interactive charts, methodology page |

## Quick start

```bash
git clone https://github.com/ldenarbonne/ecap-engine.git
cd ecap-engine
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Use the engine directly from Python:

```python
from ecap import Allocator, load_sample, simulate

result = simulate(load_sample(), n_sims=100_000, seed=42)
alloc = Allocator(result.losses, alpha=0.999, measure="ES")
print(alloc.total_ec)
print(alloc.run())          # one column per allocation method
```

## Project structure

```
ecap-engine/
├── app/streamlit_app.py         # Streamlit user interface
├── src/ecap/
│   ├── risk_models.py           # credit (Vasicek), market (Student-t), operational (LDA)
│   ├── dependence.py            # correlation assumptions, PSD fix, correlated drivers
│   ├── portfolio.py             # portfolio validation and joint simulation
│   ├── measures.py              # VaR, ES, economic capital
│   └── allocation.py            # allocation methods + RAROC / EVA
├── data/sample_portfolio.csv    # illustrative 7-unit bank
├── docs/methodology.md          # formulas and assumptions (also shown in the app)
├── tests/                       # pytest suite incl. full-allocation and app smoke tests
└── .github/workflows/ci.yml     # lint + tests on every push
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Select the repository, branch `main` and main file path `app/streamlit_app.py`.
4. Click **Deploy**.

## Methodology

See [docs/methodology.md](docs/methodology.md). Key point: the Euler allocation with Expected Shortfall,
$\mathbb{E}[L_k \mid T \ge VaR_\alpha(T)] - \mathbb{E}[L_k]$, is fully additive and the only allocation
compatible with RORAC-based performance steering (Tasche, 1999).

## Limitations

Illustrative model: one-year horizon, homogeneous credit pools, deterministic LGD, Gaussian copula,
static correlations. Not intended for regulatory use without validation.

## References

- Tasche, D. (1999). *Risk contributions and performance measurement.* Working paper, TU München.
- McNeil, A., Frey, R., Embrechts, P. (2015). *Quantitative Risk Management*, 2nd ed., Princeton UP.
- Denault, M. (2001). Coherent allocation of risk capital. *Journal of Risk*, 4(1).
- Basel Committee on Banking Supervision (2005). *An Explanatory Note on the Basel II IRB Risk Weight Functions.*

## Author

Laurent Motais de Narbonne — [LinkedIn](https://www.linkedin.com/in/laurent-motais-de-narbonne/) ·
[Publications](https://ldenarbonne.github.io/)

Licence: MIT
