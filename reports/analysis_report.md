# SME Resource Efficiency Analysis — End-to-End Execution Report

**Date:** 2026-03-10
**Project:** Drivers of Resource Efficiency (RE) Practice Adoption among EU SMEs
**Data source:** Eurobarometer SME survey, ~18,159 observations across 27 EU member states + UK
**Notebooks executed:** NB01 → NB02 → NB03 → NB04 → NB05 (all ran without errors)

---

## NB01 — Business Understanding

**What it does:** Frames the research problem in CRISP-DM terms. All six cells are markdown (no code).

**Interpretation:**
The notebook establishes a two-part research agenda: (1) identify *which firm-level and contextual factors drive RE adoption*, and (2) test whether adoption translates into *business performance improvements*. The modelling pivot is correctly documented — the 9 RE action dummies and their total count are the **outcome variables**, while firm characteristics (size, age, financial capacity, strategic orientation, market scope, and perceived barriers) are the **predictors**. The grouping structure (country → macro-region → sector) is defined here, setting up the three-level hierarchical approach used in NB04.

---

## NB02 — Data Understanding

**Key outputs:**
- Raw dataset: **18,159 rows × 236 columns**
- Survey design: stratified probability sample with post-stratification weight `w1_sme` (mean = 1.00, SD = 1.31; range ~0.05–8.9)
- Adoption rates (unweighted): energy saving **63.9%**, waste minimisation **64.5%**, recycling **60.4%**; renewables **36.3%** and green design **41.0%** trail
- Geographic heterogeneity: Northern/Western EU firms show broadly higher adoption; Portugal is notably low across most actions
- Top perceived barriers: lack of finance (~35% of adopters who planned more), lack of information (~29%)
- `re_action_count` distribution: right-skewed, mean ≈ 4.0, with a hard zero cluster

**Interpretation:**
The data quality is good for a large cross-national survey. Two design features are critical for downstream modelling:

1. **Survey weights matter.** The sample heavily over-represents medium firms; micro firms account for 43.8% of the raw sample but 93.5% once weighted, a ~2× reweighting. All descriptive statistics and sample characterisations in NB05 should use weighted estimates.

2. **Barrier variables are MNAR.** The `q7` barrier battery was only asked of firms that *either* adopted at least one RE action or planned to. This means including barrier variables in models of adoption would silently discard all zero-adopters, corrupting the inference sample — this is the root cause of the cascading failures fixed in NB04.

3. **Country × action heterogeneity** is large enough visually (NB02 heatmap) to justify a multilevel country random intercept in the full models.

---

## NB03 — Data Preparation

**Sample construction:**
| Filter | N remaining |
|--------|-------------|
| Raw | 18,159 |
| EU member states only (drop UK etc.) | 14,549 |
| SME filter (≤249 employees) | 13,559 |
| Analytic sample (non-missing core vars) | ~12,800–13,500 (cell-dependent) |

**Engineered variables:**
- `re_action_count` (0–9 integer sum of 9 binary dummies), mean = **4.02** unweighted / **3.65** survey-weighted
- `re_any_action` (binary ≥1), prevalence ≈ 85%
- Four bundle scores: `bundle_energy`, `bundle_circular`, `bundle_supply`, `bundle_inputs`
- `difficulty_count` (0–7 count of active barriers, MNAR-safe use: conditional models only)
- `market_scope` → binary dummies `ms_products` / `ms_services` (reference = 'both')
- Ordinal predictors: `firm_size_ord`, `firm_age_ord`, `turnover_bracket_ord` (0-indexed)

**Interpretation:**
The preparation pipeline is well-structured. The mean action count dropping from 4.02 to 3.65 after survey-weighting confirms that smaller, less active firms are underrepresented in the raw sample — all headline adoption statistics should cite the weighted figure. The hard-zero mass (≈15% of the analytic sample never adopts any action) motivates the hurdle model in NB04. The parquet output is clean and ready for modelling.

---

## NB04 — Modelling

### 4.1 Null models — ICC justification

| Grouping level | ICC | Interpretation |
|----------------|-----|----------------|
| Country (`country_label`) | **13.4%** | Strong country-clustering; multilevel essential |
| Macro-region (`macro_region`) | **5.3%** | Moderate; EU-bloc context adds incremental info |
| Sector (`sector_label`) | **5.2%** | Moderate; sector norms matter beyond country |

All three ICCs exceed the conventional 0.05 threshold, confirming that a substantial share of adoption variance is contextual rather than firm-level. Country accounts for by far the largest share (~13%), consistent with national policy environments (energy regulations, green subsidy schemes) being strong determinants of SME behaviour.

### 4.2 M1 — Firm characteristics → count (country random intercept)

Selected predictors after bivariate screening and VIF diagnostics:

| Predictor | Coef. | Significance | Interpretation |
|-----------|-------|-------------|----------------|
| `has_green_products` | +0.547 | *** | Firms selling green products adopt ~0.55 more actions |
| `has_climate_strategy` | +0.506 | *** | Having a formal climate strategy is strongly predictive |
| `uses_renewables` | +0.961 | *** | Largest effect; renewable use co-occurs with broad RE adoption |
| `firm_size_ord` | +0.271 | *** | Each size step (micro→small→medium) adds ~0.27 actions |
| `ms_services` | −0.528 | *** | Service firms adopt fewer RE actions than mixed-scope firms |
| `ms_products` | −0.182 | *** | Products-only firms also adopt fewer actions, but less severely |

**Interpretation:**
Strategic orientation variables (`has_green_products`, `has_climate_strategy`, `uses_renewables`) dominate the firm-level predictors, with combined coefficients exceeding firm size by roughly 2:1. This suggests that *managerial commitment to sustainability* is a stronger driver of broad RE adoption than structural capacity alone. Firm size retains a modest positive effect (larger SMEs have more resources and face more regulatory pressure), but the effect is dwarfed by strategic factors.

The negative market-scope effects are noteworthy: service-only firms adopt about half an action fewer than firms with mixed product/service offerings. Plausible mechanism — many RE actions (materials efficiency, waste minimisation, recycling) are more physically salient in product supply chains than in service delivery.

### 4.3 M2 — Macro-region fixed effects

Adding EU-bloc dummies alongside the country random intercept reduced the country ICC marginally, indicating that the broad North/South/East/West gradient is already partially captured by country-level clustering. Macro-region provides statistically significant additional variance decomposition but does not substantially change the fixed-effect coefficient pattern.

### 4.3 M3 — Cross-classified (country × sector)

Combined ICC(country:sector) ≈ 16–18%, confirming that both contextual dimensions contribute independently. Sector explains roughly equal variance to macro-region when treated as a crossed random effect, supporting the interpretation that sector norms (e.g., manufacturing vs. retail vs. construction) shape adoption independently of national policy.

### 4.4 M4 — Random slope on top predictor

The random slope on `uses_renewables` (the largest M1 coefficient) shows meaningful cross-country variance in the slope — the renewable-use effect on RE adoption breadth is larger in Western EU countries than in Eastern EU. This likely reflects both subsidy availability and installed capacity differences across national energy markets.

### 4.5 M5 — Binary "any adoption" GLMM (logistic)

The logistic model (M5) replicates the direction of all M1 predictors. Strategic factors remain dominant for the adoption/non-adoption boundary. ICC(country) in M5 ≈ 8–10% (lower than in the count model), suggesting country context shapes *how many* actions a firm takes more than *whether* it takes any.

### 4.6 Per-action models and heatmap

Nine separate logistic mixed models (one per RE action) reveal:

- `uses_renewables` and `has_climate_strategy` are positive and significant predictors of nearly every individual action — they function as broad sustainability orientation proxies
- `has_green_products` is especially strong for green supply chain and design actions (supply-chain coherence effect)
- `firm_size_ord` has the most heterogeneous profile: positive for capital-intensive actions (energy retrofits, renewable installation) but near-zero for waste-related behaviours
- `ms_services` is most negative for materials efficiency and green purchasing actions, confirming the product-supply-chain mechanism

### 4.7 Bundle models

The energy bundle is best explained by `uses_renewables` + `has_climate_strategy`; the circular economy bundle is additionally sensitive to `has_green_products`. The supply chain bundle shows the strongest firm-size gradient, consistent with supply chain management requiring scale. Bundle models provide a useful dimensionality reduction: the 4-bundle structure captures most of the heterogeneity in the 9-action heatmap.

### 4.8 Sensitivity checks

| Check | Finding |
|-------|---------|
| **Mundlak device** | All country-mean terms p > 0.29; random-effects estimates are RE-consistent |
| **With `re_investment_ord`** | Investment intensity partially mediates firm-size effect (~20% coefficient reduction); endogeneity caveat documented |
| **Poisson GLM (no RE)** | IRRs closely track LMM coefficients; confirms LMM robustness |
| **Subsample by size** | Micro firms: strategic variables dominate; `uses_renewables` coef larger. Medium firms: firm-age and turnover become more significant (resources enabling sustained adoption) |
| **Hurdle model** | Zero-inflation ratio 2.43 (observed zeros 2.43× Poisson-predicted); Stage 1 (logistic) mirrors M5; Stage 2 (count | adopter): `difficulty_count` coef = +0.044** — conditional on adopting, barrier count *increases* actions, consistent with selection (barrier-aware firms are also more strategically engaged) |

---

## NB05 — Evaluation

### 5.1 RE adoption → business outcomes (performance effects)

| Outcome | RE action count coef. | p-value | Interpretation |
|---------|----------------------|---------|----------------|
| `cost_impact` (cost savings) | **+0.032** | <0.001 | Each additional RE action associated with modestly better cost outcomes |
| `emp_change` (employment growth) | +0.0048 | 0.225 | No significant employment effect |
| `cost_impact` (ordinal logit) | **+0.0274** | <0.001 | Consistent sign under ordinal specification |

ICC(country) for cost_impact model: **6.2%** — country context shapes reported cost outcomes (likely reflecting energy price environments).

**Micro-firm subgroup:** `cost_impact` coef = **+0.0185** (p = 0.007) — the cost-savings benefit of RE adoption is present even among the smallest firms, though magnitude is smaller than the full-sample estimate, consistent with weaker economies of scale.

**Breusch-Pagan test:** Heteroscedasticity detected in cost_impact residuals. This does not invalidate the point estimates but inflates standard errors if uncorrected; robust SEs (HC3) should be used in final reporting. Residual diagnostics flag a right-skewed tail consistent with a small share of firms reporting very large cost benefits — possibly firms with high energy intensity who adopted renewables.

### 5.2 Model diagnostics summary

| Diagnostic | Result | Flag |
|-----------|--------|------|
| Cook's distance | No influential outliers beyond expected | Clean |
| Residual normality (NB04 M1) | Near-normal with mild right skew | Acceptable |
| Convergence (all models) | All converged | Clean |
| VIF (PREDICTORS) | All < 2.5 | No collinearity concern |
| Random effects caterpillar (country) | Wide spread; confirms ICC interpretation | Consistent |

---

## Overall Conclusions

1. **Multilevel structure is essential.** Country-level clustering (ICC ≈ 13%) means firm-level predictors alone would understate standard errors and overstate significance. The three-level structure (country / macro-region / sector) is jointly warranted.

2. **Strategic orientation is the dominant predictor of RE adoption.** Having a climate strategy, selling green products, and using renewables each predict ~0.5–1.0 additional RE actions — larger than the effect of moving from micro to small firm size. Policy implication: soft infrastructure (strategy frameworks, green certification, renewable energy access) may be more effective levers than direct subsidies targeting size.

3. **Firm size has a real but moderate effect.** A one-step increase in size (micro→small or small→medium) adds approximately 0.27 actions. This effect is partially mediated by investment capacity.

4. **Service firms are systematically lower adopters.** The market-scope effect is robust across all model specifications and deserves attention in sector-targeted policy design.

5. **RE adoption yields modest but significant cost benefits.** The `re_action_count → cost_impact` effect is small in magnitude (+0.032 per action) but statistically robust across OLS and ordinal logit specifications. No employment effect is detected — adoption appears to be a cost-rationalisation strategy rather than a growth catalyst for most SMEs.

6. **Barrier data must be treated with care.** The MNAR structure of the barrier battery is the key data quality risk. The adopted strategy (barriers excluded from primary models; `difficulty_count` used only in conditional/Stage 2 models) is methodologically sound but should be prominently disclosed.

---

*All figures saved to `reports/figures/`. Notebooks are reproducible end-to-end from the raw parquet. No convergence failures, index errors, or deprecation warnings remain after fixes applied in this session.*
