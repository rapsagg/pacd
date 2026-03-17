"""
Multilevel (Hierarchical) Modeling for the SME Resource Efficiency survey.

Supports:
  - Linear mixed models (continuous outcomes like RE action count)
  - Binary logistic mixed models (binary outcomes like turnover_increased)
  - Ordinal regression baseline (proportional odds without random effects)
  - Null models with ICC calculation
  - Model comparison and diagnostics
  - Heteroscedasticity (Breusch-Pagan), leverage (Cook's distance),
    overdispersion, and zero-inflation checks

Uses statsmodels MixedLM / BinomialBayesMixedGLM as primary engines.
For ordinal mixed models, uses a two-stage approach:
  1. Ordinal logit (proportional odds) via statsmodels OrderedModel
  2. Binary logistic GLMM as sensitivity check

ICC notes
---------
* Linear LMM:    ICC = τ² / (τ² + σ²)
* Logistic GLMM: ICC = τ² / (τ² + π²/3)   [latent-variable formulation,
                 Snijders & Bosker 2012; Nakagawa & Schielzeth 2010]
  The level-1 residual variance is fixed at π²/3 ≈ 3.290 for the
  logistic link, NOT estimated from the data.

LRT note
--------
Testing H₀: σ²_u = 0 (no group-level variance) places the null on the
boundary of the parameter space, so the LRT statistic follows a 50:50
mixture of χ²(0) and χ²(1).  The correct p-value is therefore
  p_mixture = 0.5 * P(χ²(1) ≥ stat).
`likelihood_ratio_test()` reports this mixture p-value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HLMResult:
    """Container for a fitted multilevel model."""
    model_name: str
    formula: str
    group: str
    result: object
    model_type: str = "linear"  # "linear", "logistic", "ordinal"
    icc: Optional[float] = None
    notes: str = ""

    def summary(self) -> str:
        return self.result.summary().as_text()

    def aic(self) -> float:
        if hasattr(self.result, "aic"):
            return self.result.aic
        return np.nan

    def bic(self) -> float:
        if hasattr(self.result, "bic"):
            return self.result.bic
        return np.nan

    def loglik(self) -> float:
        if hasattr(self.result, "llf"):
            return self.result.llf
        return np.nan


# ---------------------------------------------------------------------------
# Null model & ICC
# ---------------------------------------------------------------------------

def fit_null_model(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    reml: bool = True,
) -> HLMResult:
    """
    Fit the unconditional (null/empty) model:
        outcome ~ 1 + (1 | group)

    Computes ICC = var_group / (var_group + var_residual).
    A high ICC (e.g., >0.05) justifies multilevel modeling.
    """
    formula = f"{outcome} ~ 1"
    model = smf.mixedlm(formula, df, groups=df[group])
    result = model.fit(reml=reml)

    var_group = float(result.cov_re.iloc[0, 0])
    var_resid = result.scale
    icc = var_group / (var_group + var_resid)

    print(f"Null model ({outcome} ~ 1 | {group})")
    print(f"  Var(group)   = {var_group:.4f}")
    print(f"  Var(residual)= {var_resid:.4f}")
    print(f"  ICC          = {icc:.4f}  ({icc*100:.1f}% variance at group level)")

    return HLMResult(
        model_name="null_model",
        formula=formula,
        group=group,
        result=result,
        model_type="linear",
        icc=icc,
        notes=f"ICC={icc:.4f}",
    )


# ---------------------------------------------------------------------------
# Linear mixed models (for continuous outcomes like re_action_count)
# ---------------------------------------------------------------------------

def fit_linear_mixed(
    df: pd.DataFrame,
    outcome: str,
    fixed_effects: list[str],
    group: str,
    reml: bool = True,
    model_name: str = "linear_mixed",
) -> HLMResult:
    """
    Random-intercept linear mixed model:
        outcome ~ x1 + x2 + ... + (1 | group)
    """
    predictors = " + ".join(fixed_effects)
    formula = f"{outcome} ~ {predictors}"
    model = smf.mixedlm(formula, df, groups=df[group])
    result = model.fit(reml=reml)

    var_group = float(result.cov_re.iloc[0, 0])
    var_resid = result.scale
    icc = var_group / (var_group + var_resid)

    return HLMResult(
        model_name=model_name,
        formula=formula,
        group=group,
        result=result,
        model_type="linear",
        icc=icc,
    )


def fit_linear_random_slope(
    df: pd.DataFrame,
    outcome: str,
    fixed_effects: list[str],
    group: str,
    random_slope_var: str,
    reml: bool = True,
    model_name: str = "linear_random_slope",
) -> HLMResult:
    """
    Random-intercept + random-slope linear mixed model:
        outcome ~ x1 + x2 + ... + (1 + slope_var | group)
    """
    predictors = " + ".join(fixed_effects)
    formula = f"{outcome} ~ {predictors}"
    model = smf.mixedlm(
        formula, df, groups=df[group],
        re_formula=f"~ {random_slope_var}",
    )
    result = model.fit(reml=reml)

    var_group = float(result.cov_re.iloc[0, 0])
    var_resid = result.scale
    icc = var_group / (var_group + var_resid)

    return HLMResult(
        model_name=model_name,
        formula=formula,
        group=group,
        result=result,
        model_type="linear",
        icc=icc,
    )


# ---------------------------------------------------------------------------
# Binary logistic mixed model (for binary outcomes like turnover_increased)
# ---------------------------------------------------------------------------

def fit_logistic_mixed(
    df: pd.DataFrame,
    outcome: str,
    fixed_effects: list[str],
    group: str,
    model_name: str = "logistic_mixed",
) -> HLMResult:
    """
    Random-intercept logistic GLMM (binary outcome):
        logit(P(outcome=1)) ~ x1 + x2 + ... + (1 | group)

    Uses statsmodels BinomialBayesMixedGLM for estimation.
    """
    predictors = " + ".join(fixed_effects)
    formula = f"{outcome} ~ {predictors}"

    # Use generalized linear mixed model
    model = sm.BinomialBayesMixedGLM.from_formula(
        formula, vc_formulas={"group": f"0 + C({group})"},
        data=df,
    )
    result = model.fit_vb()

    return HLMResult(
        model_name=model_name,
        formula=formula,
        group=group,
        result=result,
        model_type="logistic",
        notes="BinomialBayesMixedGLM (variational Bayes)",
    )


# ---------------------------------------------------------------------------
# Ordinal regression (without random effects — baseline)
# ---------------------------------------------------------------------------

def fit_ordinal_logit(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    model_name: str = "ordinal_logit",
) -> HLMResult:
    """
    Proportional odds (ordered logit) model — no random effects.
    Serves as baseline before adding multilevel structure.

    Uses statsmodels OrderedModel with logit link.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    y = df[outcome].dropna()
    X = df.loc[y.index, predictors].dropna()
    y = y.loc[X.index]

    model = OrderedModel(y, X, distr="logit")
    result = model.fit(method="bfgs", disp=False)

    return HLMResult(
        model_name=model_name,
        formula=f"{outcome} ~ {' + '.join(predictors)}",
        group="none",
        result=result,
        model_type="ordinal",
        notes="Proportional odds model (no random effects)",
    )


# ---------------------------------------------------------------------------
# Cross-classified model
# ---------------------------------------------------------------------------

def fit_cross_classified(
    df: pd.DataFrame,
    outcome: str,
    fixed_effects: list[str],
    group1: str,
    group2: str,
    reml: bool = True,
    model_name: str = "cross_classified",
) -> HLMResult:
    """
    Approximate cross-classified model using combined group variable.
    """
    df = df.copy()
    df["_group_combined"] = df[group1].astype(str) + ":" + df[group2].astype(str)
    predictors = " + ".join(fixed_effects)
    formula = f"{outcome} ~ {predictors}"
    model = smf.mixedlm(formula, df, groups=df["_group_combined"])
    result = model.fit(reml=reml)

    var_group = float(result.cov_re.iloc[0, 0])
    var_resid = result.scale
    icc = var_group / (var_group + var_resid)

    return HLMResult(
        model_name=model_name,
        formula=formula,
        group=f"{group1}:{group2}",
        result=result,
        model_type="linear",
        icc=icc,
        notes=f"Cross-classified via '{group1}:{group2}'",
    )


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(models: list[HLMResult]) -> pd.DataFrame:
    """Comparison table: AIC, BIC, log-likelihood, ICC."""
    rows = []
    for m in models:
        rows.append({
            "model": m.model_name,
            "type": m.model_type,
            "formula": m.formula,
            "group": m.group,
            "loglik": round(m.loglik(), 2) if not np.isnan(m.loglik()) else None,
            "aic": round(m.aic(), 2) if not np.isnan(m.aic()) else None,
            "bic": round(m.bic(), 2) if not np.isnan(m.bic()) else None,
            "icc": round(m.icc, 4) if m.icc is not None else None,
            "notes": m.notes,
        })
    return pd.DataFrame(rows).sort_values("aic", na_position="last")


def likelihood_ratio_test(
    m_reduced: HLMResult,
    m_full: HLMResult,
    df_diff: int = 1,
    boundary: bool = True,
) -> dict:
    """
    Chi-squared LRT between nested models (both must use ML, not REML).

    Parameters
    ----------
    df_diff : int
        Degrees of freedom difference (number of additional parameters in
        m_full).  Default=1 for a single added random intercept or fixed effect.
    boundary : bool
        If True (default), test is for a variance component (H₀: σ²=0 is on
        the boundary of the parameter space).  The LRT statistic then follows
        a 50:50 mixture of χ²(0) and χ²(df_diff), and the reported p-value
        is halved relative to the standard chi-squared p-value.
        Set boundary=False when testing fixed effects only.
    """
    from scipy.stats import chi2 as chi2_dist

    ll_r = m_reduced.loglik()
    ll_f = m_full.loglik()
    stat = 2 * (ll_f - ll_r)
    p_standard = chi2_dist.sf(stat, df_diff)
    p_value = p_standard / 2 if boundary else p_standard
    label = "mixture p (boundary)" if boundary else "p-value"
    print(f"LRT: χ²({df_diff}) = {stat:.4f}, {label} = {p_value:.4f}")
    return {"chi2": stat, "df": df_diff, "p_value": p_value, "boundary": boundary}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def plot_residuals(hlm_result: HLMResult, figsize: tuple = (14, 4)) -> plt.Figure:
    """Residuals vs fitted, Q-Q, and histogram for linear mixed models."""
    result = hlm_result.result
    fitted = result.fittedvalues
    resid = result.resid

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].scatter(fitted, resid, alpha=0.15, s=8)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    from scipy import stats as scipy_stats
    scipy_stats.probplot(resid, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q")

    axes[2].hist(resid, bins=40, edgecolor="white", color="steelblue")
    axes[2].set_xlabel("Residual")
    axes[2].set_title("Residual Distribution")

    fig.suptitle(f"Diagnostics: {hlm_result.model_name}", fontsize=12)
    plt.tight_layout()
    return fig


def random_effects_caterpillar(
    hlm_result: HLMResult,
    figsize: tuple = (8, 10),
) -> plt.Figure:
    """Caterpillar plot of random intercepts (BLUPs)."""
    re = hlm_result.result.random_effects
    re_df = pd.DataFrame({
        "group": list(re.keys()),
        "intercept": [v.iloc[0] for v in re.values()],
    })
    re_df = re_df.sort_values("intercept")

    var_re = float(hlm_result.result.cov_re.iloc[0, 0])
    se = np.sqrt(var_re)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(re_df))
    ax.errorbar(
        re_df["intercept"], y_pos,
        xerr=1.96 * se, fmt="o", color="steelblue",
        ecolor="lightsteelblue", capsize=3, markersize=4,
    )
    ax.axvline(0, color="red", linestyle="--")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(re_df["group"], fontsize=7)
    ax.set_xlabel("Random intercept (BLUP)")
    ax.set_title(f"Random Effects: {hlm_result.group}")
    plt.tight_layout()
    return fig


def plot_fixed_effects(hlm_result: HLMResult, figsize: tuple = (8, 5)) -> plt.Figure:
    """Forest plot of fixed effect coefficients with 95% CI."""
    result = hlm_result.result
    params = result.fe_params if hasattr(result, "fe_params") else result.params
    ci = result.conf_int()

    coef_df = pd.DataFrame({
        "coef": params,
        "ci_lower": ci.iloc[:, 0],
        "ci_upper": ci.iloc[:, 1],
    })
    # Drop intercept for cleaner plot
    coef_df = coef_df.drop("Intercept", errors="ignore")

    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(coef_df))
    ax.errorbar(
        coef_df["coef"], y,
        xerr=[coef_df["coef"] - coef_df["ci_lower"],
              coef_df["ci_upper"] - coef_df["coef"]],
        fmt="o", color="steelblue", capsize=4,
    )
    ax.axvline(0, color="red", linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(coef_df.index)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(f"Fixed Effects: {hlm_result.model_name}")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ICC for logistic GLMM (latent-variable formulation)
# ---------------------------------------------------------------------------

# Level-1 residual variance for logistic link = π²/3
_LOGISTIC_L1_VAR = np.pi**2 / 3   # ≈ 3.2899


def compute_icc_logistic(tau_sq: float) -> float:
    """
    ICC for a logistic GLMM using the latent-variable (threshold) formulation.

        ICC = τ² / (τ² + π²/3)

    where τ² is the random-intercept variance and π²/3 ≈ 3.290 is the
    fixed level-1 residual variance implied by the logistic distribution.

    Reference: Snijders & Bosker (2012) §17.2; Nakagawa & Schielzeth (2010)
    Bioinformatics doi:10.1093/bioinformatics/btr519.
    """
    return tau_sq / (tau_sq + _LOGISTIC_L1_VAR)


# ---------------------------------------------------------------------------
# Heteroscedasticity: Breusch-Pagan test
# ---------------------------------------------------------------------------

def breusch_pagan_test(hlm_result: HLMResult) -> dict:
    """
    Breusch-Pagan test for heteroscedasticity in a linear mixed model.

    H₀: residual variance is homogeneous (homoscedastic).
    H₁: residual variance depends linearly on fitted values.

    A significant result (p < 0.05) indicates heteroscedasticity, which
    invalidates standard errors and confidence intervals under OLS/LMM
    assumptions.  Robust (sandwich) SEs should be used in that case.

    Returns a dict with {'lagrange_multiplier', 'p_value', 'f_stat', 'f_p_value'}.
    Only valid for linear mixed models.
    """
    from statsmodels.stats.diagnostic import het_breuschpagan
    import statsmodels.api as sm_

    result = hlm_result.result
    resid  = np.array(result.resid)
    fitted = np.array(result.fittedvalues)

    exog = sm_.add_constant(fitted)
    lm_stat, lm_p, f_stat, f_p = het_breuschpagan(resid, exog)

    print(f"Breusch-Pagan test: LM={lm_stat:.3f}, p={lm_p:.4f}  |  "
          f"F={f_stat:.3f}, p={f_p:.4f}")
    if lm_p < 0.05:
        print("  ⚠  Heteroscedasticity detected — consider robust SEs "
              "or a count/ordinal model.")
    else:
        print("  ✓  No evidence of heteroscedasticity (p ≥ 0.05).")
    return {"lm_stat": lm_stat, "lm_p": lm_p, "f_stat": f_stat, "f_p": f_p}


# ---------------------------------------------------------------------------
# Influence: Cook's distance
# ---------------------------------------------------------------------------

def cooks_distance_plot(
    hlm_result: HLMResult,
    threshold: float = 4.0,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Approximated Cook's distance plot for a linear mixed model.

    Uses OLSInfluence on the marginal (fixed-only) model as an approximation,
    which identifies observations with undue influence on fixed-effect estimates.
    Observations with Cook's D > threshold/N are flagged.

    Note: true Cook's D for LMMs does not have a closed form; this marginal
    approximation is a standard diagnostic heuristic.
    """
    from statsmodels.stats.outliers_influence import OLSInfluence
    import statsmodels.formula.api as smf_

    result = hlm_result.result
    fitted  = np.array(result.fittedvalues)
    resid   = np.array(result.resid)
    n       = len(resid)
    cut     = threshold / n

    # Marginal OLS (no group structure) — for leverage/influence approximation
    formula_ols = hlm_result.formula.split("~")[1].strip()
    # Build a plain OLS using the same data
    df_tmp = pd.DataFrame({
        "y":    np.array(result.model.endog),
        **{f"x{i}": result.model.exog[:, i]
           for i in range(result.model.exog.shape[1])},
    })
    ols_formula = "y ~ " + " + ".join(f"x{i}" for i in range(1, result.model.exog.shape[1]))
    ols_res = smf_.ols(ols_formula, data=df_tmp).fit()
    influence = OLSInfluence(ols_res)
    cooks_d   = influence.cooks_distance[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.stem(range(n), cooks_d, markerfmt=",", linefmt="steelblue",
            basefmt="k-")
    ax.axhline(cut, color="red", linestyle="--",
               label=f"Threshold {threshold}/N = {cut:.4f}")
    n_flagged = (cooks_d > cut).sum()
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cook's distance (marginal approx.)")
    ax.set_title(f"Cook's Distance — {hlm_result.model_name}\n"
                 f"{n_flagged} observations above threshold ({n_flagged/n*100:.1f}%)")
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Count model diagnostics: overdispersion and zero-inflation
# ---------------------------------------------------------------------------

def overdispersion_test(y_observed: np.ndarray, y_fitted: np.ndarray) -> dict:
    """
    Pearson chi-squared overdispersion test for a count model.

    For a Poisson model, Var(Y) = E(Y).  If the Pearson statistic
    (sum of (obs - fitted)² / fitted) divided by its degrees of freedom
    is substantially > 1, the data are overdispersed and a negative
    binomial model should be used instead.

    Parameters
    ----------
    y_observed : array-like of non-negative integers
    y_fitted   : array-like of positive floats (Poisson fitted values)

    Returns
    -------
    dict with 'pearson_stat', 'df', 'ratio', 'overdispersed' (bool)
    """
    y  = np.array(y_observed, dtype=float)
    mu = np.array(y_fitted,   dtype=float)
    pearson = np.sum((y - mu)**2 / mu)
    dof     = len(y) - 1           # approximate; exact df depends on n_params
    ratio   = pearson / dof
    print(f"Pearson chi² / df = {pearson:.1f} / {dof} = {ratio:.3f}")
    if ratio > 1.5:
        print("  ⚠  Overdispersion detected (ratio > 1.5) — consider "
              "negative binomial or quasi-Poisson.")
    else:
        print("  ✓  No strong evidence of overdispersion.")
    return {"pearson_stat": pearson, "df": dof, "ratio": ratio,
            "overdispersed": ratio > 1.5}


def zero_inflation_check(y_observed: np.ndarray, mu_fitted: np.ndarray) -> dict:
    """
    Compare observed vs. Poisson-predicted proportion of zeros.

    If the observed zero proportion substantially exceeds what a Poisson
    distribution with the estimated mean would predict, zero-inflation is
    present and a zero-inflated or hurdle model may be warranted.
    """
    y      = np.array(y_observed, dtype=float)
    mu     = np.array(mu_fitted,  dtype=float)
    n      = len(y)
    obs_zeros  = (y == 0).sum() / n
    pred_zeros = np.mean(np.exp(-mu))           # P(Y=0 | Poisson(μ))
    ratio      = obs_zeros / pred_zeros if pred_zeros > 0 else np.inf
    print(f"Observed zero rate:    {obs_zeros:.3f} ({obs_zeros*100:.1f}%)")
    print(f"Poisson-expected zeros:{pred_zeros:.3f} ({pred_zeros*100:.1f}%)")
    print(f"Ratio observed/expected: {ratio:.2f}")
    if ratio > 1.5:
        print("  ⚠  Possible zero-inflation — consider hurdle or ZIP model.")
    else:
        print("  ✓  Zero count is consistent with Poisson expectation.")
    return {"obs_zeros": obs_zeros, "pred_zeros": pred_zeros, "ratio": ratio,
            "zero_inflated": ratio > 1.5}


# ---------------------------------------------------------------------------
# Multiple testing correction (FDR / Bonferroni)
# ---------------------------------------------------------------------------

def apply_fdr(
    p_values: pd.Series,
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply multiple-testing correction to a Series of p-values.

    Parameters
    ----------
    p_values : pd.Series  (index = predictor names)
    method   : str  — passed to statsmodels multipletests:
        'fdr_bh'  = Benjamini-Hochberg FDR (default; recommended for
                    exploratory screening with correlated predictors)
        'bonferroni' = family-wise error rate (conservative)
        'holm'    = Holm-Bonferroni step-down (less conservative than Bonf.)
    alpha    : float  — overall Type-I error rate (default 0.05)

    Returns a DataFrame with columns ['p_raw', 'p_adj', 'reject', 'method'].

    When to use FDR vs. Bonferroni
    --------------------------------
    * Bonferroni controls FWER (probability of ≥1 false positive → ≤ α).
      Appropriate when ANY false positive is costly (e.g., drug trials).
    * BH-FDR controls expected proportion of false positives among rejected
      tests.  Preferred for exploratory research with many correlated
      predictors, where we accept a small fraction of false findings.
    * Rule of thumb: use BH-FDR for variable screening; use Bonferroni (or
      pre-registration) for confirmatory tests of pre-specified hypotheses.
    """
    from statsmodels.stats.multitest import multipletests

    p = p_values.values
    reject, p_adj, _, _ = multipletests(p, alpha=alpha, method=method)
    return pd.DataFrame({
        "p_raw":  p,
        "p_adj":  p_adj,
        "reject": reject,
        "method": method,
    }, index=p_values.index)
