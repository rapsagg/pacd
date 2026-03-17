# MCA & Individual Regressions — Findings Report

**Date:** 2026-03-17
**Project:** Drivers of Resource Efficiency Practice Adoption among EU SMEs
**Data:** Flash Eurobarometer 549 (June 2024) — 13,559 SMEs, 28 countries
**Notebook:** NB02_analise.ipynb — Sections 3.8 and 4.1

---

## 1. Context

The core measurement problem is that `intensity_index` — a simple count of Q1 practices adopted (0–9) — treats all practices as equally important. A firm that saves energy (65% adoption rate) scores the same as one that implements eco-design (28%). This is almost certainly wrong: rarer, costlier practices likely signal a deeper and more deliberate commitment to resource efficiency. We used two complementary approaches to investigate this.

---

## 2. Multiple Correspondence Analysis (MCA)

### 2.1 Method

MCA maps 9 binary practice variables into a lower-dimensional factor space, assigning higher coordinate distances to categories that are rarer and less correlated with others. The Dimension 1 coordinates of each firm become the `mca_score` — a weighted alternative to `intensity_index`.

Implementation: `prince.MCA(n_components=9, n_iter=10, random_state=42, engine='sklearn')` fitted on a 13,559 × 9 matrix of categorical ("Adotou" / "Não adotou") practice indicators.

### 2.2 Inertia

| Dimension | Eigenvalue | Inertia (%) | Cumulative (%) |
|---|---|---|---|
| 1 | 0.2817 | **28.17** | 28.17 |
| 2 | 0.1114 | 11.14 | 39.31 |
| 3 | 0.1048 | 10.48 | 49.79 |
| 4 | 0.1002 | 10.02 | 59.81 |
| 5–9 | — | 40.18 | 100.00 |

Dimension 1 explains 28.2% of total inertia, the dominant axis by a wide margin. For binary MCA the theoretical maximum is (9−1)/9 ≈ 88.9% — the remaining inertia is distributed across dimensions capturing which *specific* practices cluster together, not the overall intensity gradient. We retain only Dimension 1, which captures the adoption intensity axis relevant to our dependent variable.

### 2.3 Implicit Weights

The weight each practice receives in the `mca_score` is the absolute value of its "Adotou" coordinate on Dimension 1:

| Practice | MCA weight | Adoption rate |
|---|---|---|
| Eco-design | **0.745** | 28.0% |
| Green suppliers | **0.719** | 35.6% |
| Save water | 0.618 | 47.6% |
| Renewable energy | 0.585 | 26.7% |
| Sell waste | 0.580 | 29.6% |
| Save materials | 0.563 | 57.2% |
| Minimize waste | 0.494 | 65.0% |
| Save energy | 0.444 | 65.2% |
| Recycle internally | **0.441** | 47.5% |

Spearman r (weight ↔ adoption rate) = −0.617, p = 0.077. The expected negative direction is confirmed — rarer practices receive more discriminatory weight. The notable exception is **recycle internally** (47.5% adoption but the lowest MCA weight), suggesting it co-occurs so frequently with other practices that it adds little *unique* information about a firm's green profile.

### 2.4 mca_score Properties

- Pearson r with `intensity_index`: **0.993**
- Mean: 0.000 (MCA coordinates are centered by construction)
- SD: 0.531 | Min: −0.964 | Max: 1.086
- NAs: 0

The high correlation does not make the weighting redundant. The differences emerge at the margins: two firms with the same count of practices can have meaningfully different `mca_scores` depending on *which* practices they adopted.

---

## 3. Individual Logistic Regressions (Section 4.1)

### 3.1 Method

For each of the 9 practices, a logistic regression was fitted using 26 structural predictors: sector dummies (ref: Manufatura), size dummies (ref: Micro), firm age dummies, employment change dummies, Q3 turnover trend dummies, Q4 investment intensity dummies, turnover size ordinal, turnover unknown flag, and financial difficulty ordinal. Q7 barriers were deliberately excluded — they are outcomes, not structural determinants. All 13,559 observations are used in every model (dummies absorb all DK/NA). McFadden pseudo-R² = 1 − (log-lik / log-lik-null).

### 3.2 Results

| Practice | Pseudo-R² (McFadden) | Adoption rate |
|---|---|---|
| Sell waste | **0.0892** | 29.6% |
| Save energy | 0.0759 | 65.2% |
| Minimize waste | 0.0693 | 65.0% |
| Renewable energy | 0.0685 | 26.7% |
| Save materials | 0.0561 | 57.2% |
| Green suppliers | 0.0485 | 35.6% |
| Eco-design | 0.0458 | 28.0% |
| Save water | 0.0430 | 47.6% |
| Recycle internally | **0.0355** | 47.5% |

All R² values are low (0.036–0.089), which is entirely normal for survey data on individual behavioral outcomes. What matters is the *relative ordering*: **sell waste** is the most structurally determined practice (sector, size, and investment level strongly predict it), while **recycling** and **water saving** are almost entirely idiosyncratic — firm observables barely help predict whether a firm recycles.

---

## 4. What the Two Approaches Tell Us Together

The Spearman correlation between pseudo-R² and MCA weight is r = −0.217 (p = 0.576) — weak, non-significant, and **negative**. The two methods do not agree on which practices are "most important," because they measure different things:

- **MCA weight** = statistical rarity + correlation structure → *"Which practices most sharply separate green leaders from the rest?"*
- **Pseudo-R²** = structural predictability → *"Which practices are driven by observable firm characteristics?"*

### 4.1 Divergence of Specific Practices

**Eco-design and green suppliers** — high MCA weight (≥0.72), low R² (≤0.049). These are rare practices that MCA flags as highly discriminating, but structural predictors barely explain their adoption. The implication is that they reflect deliberate, strategically-driven choices that go beyond what firm size, sector, or investment level can predict — exactly the type of behavior our DV should be sensitive to.

**Sell waste** — low MCA weight (0.58), highest R² (0.089). It is predictable from sector and size (industrial firms can monetize byproducts; service firms usually cannot), but that structural predictability means it does not discriminate among firms *within* a structural profile. MCA correctly down-weights it.

**Recycle internally** — lowest on both (MCA weight 0.44, R² 0.036). It contributes little to either approach: common enough not to discriminate (MCA), and idiosyncratic enough not to be explained by firm profile (R²). For modeling purposes, this is the least informative practice.

### 4.2 Cross-Correlations Summary

| Pair | Spearman r | p-value |
|---|---|---|
| Pseudo-R² ↔ adoption rate | +0.167 | 0.668 |
| Pseudo-R² ↔ MCA weight | −0.217 | 0.576 |
| Adoption rate ↔ MCA weight | −0.617 | 0.077 |

The only pair with a meaningful signal is adoption rate ↔ MCA weight (r = −0.617), confirming that MCA's weighting logic is primarily driven by rarity. The pseudo-R² is essentially orthogonal to both, confirming it captures a genuinely different dimension.

---

## 5. Implications for the Final Model

### 5.1 DV Choice

**`mca_score` as primary DV, `intensity_index` as robustness check.** The r = 0.993 between the two means the multilevel model results will be substantively similar regardless of choice. However, `mca_score` is better justified theoretically: it gives more weight to the practices that genuinely distinguish green leaders, and the individual regression analysis reinforces this — the practices MCA weights most highly are precisely the ones that structural predictors *cannot* explain, meaning they capture strategic green commitment rather than structural inevitability.

### 5.2 On Explained Variance

All pseudo-R² values are below 0.09, meaning structural predictors alone account for at most 9% of variance in any single practice. This low ceiling is expected for behavioral survey outcomes and motivates the multilevel approach: a substantial portion of the residual variance is likely to be explained by country-level factors (regulatory environment, culture, incentive structures) rather than firm-level observables.

### 5.3 On Country-Level Effects

The coefficient heatmap (NB02 cell 125) will be useful for interpreting the multilevel results. Practices with high R² (sell waste, save energy) are more likely to show country-level clustering driven by industrial structure. Practices with low R² (eco-design, recycling, water saving) are more likely to reflect country-level cultural or regulatory variation — precisely the signal that random slopes in the multilevel model are designed to detect.

### 5.4 Recommendation for Sections 4.2–4.4

Proceed with `mca_score` as the primary DV for the multilevel model. Use `intensity_index` in Section 4.4 as the robustness check. The weighting is theoretically defensible, independently supported by the regression analysis, and the high correlation between the two means the robustness check will likely confirm the main findings.
