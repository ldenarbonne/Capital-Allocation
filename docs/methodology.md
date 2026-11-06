## Methodology

### 1. Loss models by risk type

Each business unit *k* has a systemic risk driver $X_k \sim N(0,1)$; a high $X_k$ means a high loss.

**Credit — one-factor Vasicek with finite granularity.** For a pool of $n$ equal obligors with
exposure $EAD$, default probability $PD$, loss given default $LGD$ and asset correlation $\rho$:

$$p(X) = \Phi\left(\frac{\Phi^{-1}(PD) + \sqrt{\rho}\,X}{\sqrt{1-\rho}}\right),\qquad
D \mid X \sim \mathrm{Bin}(n, p(X)),\qquad L = D \cdot \frac{EAD}{n}\cdot LGD$$

As $n \to \infty$, the 99.9% unexpected loss converges to the ASRF formula behind the Basel IRB approach
(shown in the app as a benchmark).

**Market — fat-tailed P&L.** $L = \text{Exposure}\cdot\sigma\cdot\sqrt{h}\cdot \varepsilon$, where
$\varepsilon = t_\nu^{-1}(\Phi(X))\sqrt{(\nu-2)/\nu}$ is a unit-variance Student-t shock.

**Operational — Loss Distribution Approach.** Annual loss $= \sum_{i=1}^{N} S_i$ with
$N \sim \mathrm{Poisson}(\lambda)$ and lognormal severities $S_i$ parameterised by mean and coefficient
of variation. Simulated annual losses are re-ordered to follow the ranks of $X_k$ (rank coupling), which
imposes the copula dependence without distorting the marginal distribution.

### 2. Dependence and aggregation

The drivers $(X_1,\dots,X_K)$ follow a Gaussian copula whose correlation matrix is built from
risk-type-level assumptions (credit–credit, credit–market, …) and projected onto the nearest positive
semi-definite matrix. Total loss is the scenario-wise sum $T = \sum_k L_k$, so aggregation is a full
simulation-based approach rather than a variance–covariance shortcut.

### 3. Economic capital

$$EC = \rho_\alpha(T) - \mathbb{E}[T]\quad\text{(unexpected loss)}\qquad\text{or}\qquad EC = \rho_\alpha(T)$$

with $\rho_\alpha$ = VaR or Expected Shortfall at confidence $\alpha$.

### 4. Allocation methods

| Method | Contribution of unit *k* | Properties |
|---|---|---|
| Stand-alone | $\rho(L_k) - \mathbb{E}[L_k]$ | Ignores diversification; sum > total EC |
| Pro-rata | $EC \cdot \dfrac{EC_k^{SA}}{\sum_j EC_j^{SA}}$ | Simple, but blind to correlation |
| Covariance | $EC \cdot \dfrac{\mathrm{Cov}(L_k, T)}{\mathrm{Var}(T)}$ | Additive; only exact for elliptical distributions |
| **Euler (ES)** | $\mathbb{E}[L_k \mid T \ge VaR_\alpha(T)] - \mathbb{E}[L_k]$ | Additive, RORAC-compatible, exact for ES |
| Euler (VaR) | $\mathbb{E}[L_k \mid T = VaR_\alpha(T)] - \mathbb{E}[L_k]$ | Estimated on a window of scenarios around the quantile (noisy), rescaled |
| Incremental | $EC(T) - EC(T - L_k)$, rescaled | "With / without" view; raw sum < total EC |
| Shapley | Average marginal contribution over all orderings of units | Fair (game theory), additive; $2^K$ evaluations |

The Euler principle is the only allocation compatible with RORAC/RAROC performance measurement
(Tasche): increasing exposure in a unit whose RAROC exceeds the portfolio's improves the portfolio's
RAROC.

### 5. Performance

$$\text{RAROC}_k = \frac{(\text{Revenue}_k - \text{Costs}_k - EL_k)(1-\tau) + r_c\,EC_k}{EC_k},\qquad
\text{EVA}_k = \text{Net income}_k - h \cdot EC_k$$

A unit creates value when $\text{RAROC}_k \ge h$ (hurdle rate / cost of equity).

### 6. Limitations

Illustrative model: one-year horizon, homogeneous credit pools, deterministic LGD, Gaussian copula
(no tail dependence beyond that induced by the marginals), static correlations, no model risk buffer.
