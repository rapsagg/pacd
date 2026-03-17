"""
Feature engineering for the Flash Eurobarometer SME Resource Efficiency survey.

Constructs:
  - Resource efficiency action count and binary flags
  - Composite indices (RE breadth, green orientation)
  - Outcome variable recoding (turnover change)
  - Firm-level control variable recoding
  - Country-level and sector-level label mappings
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Resource efficiency action variables
# ---------------------------------------------------------------------------

# Q1.1–Q1.12: "What actions is your company undertaking to be more resource efficient?"
# Each is binary: 0 = Not mentioned, 1 = Mentioned
# Q1.1–Q1.9 are the 9 substantive actions; Q1.10=Other, Q1.11=None, Q1.12=DK/NA
RE_ACTION_COLS = [f"q1.{i}" for i in range(1, 13)]

# Labels for Q1 sub-items — exact order from the FL549 questionnaire
RE_ACTION_LABELS = {
    "q1.1":  "Saving water",
    "q1.2":  "Saving energy",
    "q1.3":  "Using predominantly renewable energy",
    "q1.4":  "Saving materials",
    "q1.5":  "Switching to greener suppliers of materials",
    "q1.6":  "Minimising waste",
    "q1.7":  "Selling residues/waste to another company",
    "q1.8":  "Recycling (reusing material/waste within company)",
    "q1.9":  "Designing products easier to maintain/repair/reuse",
    "q1.10": "Other",
    "q1.11": "None",
    "q1.12": "Don't know/No answer",
}

# Q2.1–Q2.12: planned future actions (identical structure to Q1)
RE_PLANNED_COLS = [f"q2.{i}" for i in range(1, 13)]

# Substantive action columns only (excludes Other / None / DK)
RE_ACTION_COLS_CORE   = [f"q1.{i}" for i in range(1, 10)]   # q1.1–q1.9
RE_PLANNED_COLS_CORE  = [f"q2.{i}" for i in range(1, 10)]   # q2.1–q2.9

# Readable names for the 9 substantive RE action dummies
RE_ACTION_DUMMY_NAMES = {
    "q1.1": "re_water",
    "q1.2": "re_energy",
    "q1.3": "re_renewables",
    "q1.4": "re_materials",
    "q1.5": "re_green_suppliers",
    "q1.6": "re_waste_min",
    "q1.7": "re_waste_sell",
    "q1.8": "re_recycling",
    "q1.9": "re_design",
}

# Ordered list of the named dummy columns (convenience)
RE_DUMMY_COLS = list(RE_ACTION_DUMMY_NAMES.values())

# Theoretically motivated action bundles
# Groups capture distinct RE strategies firms might pursue together
RE_BUNDLES: dict[str, list[str]] = {
    # Switching to lower-carbon energy sources
    "bundle_energy":     ["re_energy", "re_renewables"],
    # Circular-economy actions: reduce, sell, reuse, design-for-reuse
    "bundle_circular":   ["re_waste_min", "re_waste_sell", "re_recycling", "re_design"],
    # Green upstream supply chain
    "bundle_supply":     ["re_green_suppliers", "re_materials"],
    # Resource-input savings (water, energy, materials)
    "bundle_inputs":     ["re_water", "re_energy", "re_materials"],
}

BUNDLE_COLS = list(RE_BUNDLES.keys())


def compute_re_action_count(
    df: pd.DataFrame,
    cols: list[str] = RE_ACTION_COLS_CORE,
    col_name: str = "re_action_count",
) -> pd.DataFrame:
    """
    Count of substantive resource efficiency actions undertaken (0–9).
    Uses q1.1–q1.9 by default, excluding Other (q1.10), None (q1.11), DK (q1.12).
    """
    df = df.copy()
    df[col_name] = df[cols].sum(axis=1)
    return df


def compute_re_any(
    df: pd.DataFrame,
    col_name: str = "re_any_action",
) -> pd.DataFrame:
    """
    Binary: has the firm taken at least one RE action?
    True when q1.11 (None) is NOT selected, i.e. firm picked at least one of q1.1–q1.9.
    """
    df = df.copy()
    df[col_name] = (df["q1.11"] != 1).astype(int)
    return df


def compute_re_planned_count(
    df: pd.DataFrame,
    cols: list[str] = RE_PLANNED_COLS_CORE,
    col_name: str = "re_planned_count",
) -> pd.DataFrame:
    """Count of planned future substantive RE actions (0–9)."""
    df = df.copy()
    df[col_name] = df[cols].sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# Green orientation composite
# ---------------------------------------------------------------------------

def compute_green_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite green orientation score combining:
    - re_action_count (0–9)
    - green products offered (q9 == 1 → 1, else 0)
    - climate strategy (q14 == 1 → 1, else 0)
    - renewable energy use (n1a in [1,2] → 1, else 0)

    Score range: 0–12 (higher = more green-oriented).

    ⚠  CIRCULARITY WARNING ⚠
    green_orientation directly incorporates re_action_count.  It MUST NOT
    be used as a predictor of re_action_count, re_any_action, or any of the
    individual RE action dummies — doing so would be predicting a variable
    from a composite that already contains it, inflating associations to near-
    perfect.

    Safe uses:
      - Descriptive profiling of the sample
      - As a secondary outcome variable (e.g., green_orientation ~ firm_size)

    The three non-circular components — has_green_products, has_climate_strategy,
    uses_renewables — are the intended predictors in the main models.
    """
    df = df.copy()
    if "re_action_count" not in df.columns:
        df = compute_re_action_count(df)

    df["has_green_products"] = (df["q9"] == 1).astype(int)
    df["has_climate_strategy"] = (df["q14"] == 1).astype(int)
    df["uses_renewables"] = df["n1a"].isin([1, 2]).astype(int)

    df["green_orientation"] = (
        df["re_action_count"]
        + df["has_green_products"]
        + df["has_climate_strategy"]
        + df["uses_renewables"]
    )
    return df


# ---------------------------------------------------------------------------
# Outcome variable recoding
# ---------------------------------------------------------------------------

def recode_turnover_change(
    df: pd.DataFrame,
    col: str = "scr13a",
) -> pd.DataFrame:
    """
    Recode turnover change (scr13a) into usable forms.

    Original coding:
        1 = Increased ≥10% annually
        2 = Increased <10% annually
        3 = Remained unchanged
        4 = Decreased

    Creates:
        turnover_change_ord : ordinal 1–4 (reversed: 1=decreased → 4=increased ≥10%)
        turnover_increased  : binary (1 = increased, 0 = unchanged/decreased)
        turnover_change_3cat: 3-category (decreased / unchanged / increased)
    """
    df = df.copy()

    # Reverse so higher = better performance
    reverse_map = {1: 4, 2: 3, 3: 2, 4: 1}
    df["turnover_change_ord"] = df[col].map(reverse_map)

    # Binary: increased (1,2) vs not (3,4)
    df["turnover_increased"] = df[col].isin([1, 2]).astype(float)
    df.loc[df[col].isna(), "turnover_increased"] = np.nan

    # 3-category
    cat_map = {1: "increased", 2: "increased", 3: "unchanged", 4: "decreased"}
    df["turnover_change_3cat"] = df[col].map(cat_map)

    return df


def recode_cost_impact(
    df: pd.DataFrame,
    col: str = "q3",
) -> pd.DataFrame:
    """
    Recode production cost impact (q3).

    Original: 1=sig decreased, 2=slightly decreased, 3=slightly increased,
              4=sig increased, 5=not changed
    Creates:
        cost_decreased : binary (1 if costs decreased, 0 otherwise)
        cost_impact_ord: ordinal 1–5 (1=sig increased → 5=sig decreased)
    """
    df = df.copy()

    # Reverse so higher = better (costs decreased more)
    reverse_map = {1: 5, 2: 4, 5: 3, 3: 2, 4: 1}
    df["cost_impact_ord"] = df[col].map(reverse_map)

    # Binary
    df["cost_decreased"] = df[col].isin([1, 2]).astype(float)
    df.loc[df[col].isna(), "cost_decreased"] = np.nan

    return df


def recode_employee_change(
    df: pd.DataFrame,
    col: str = "scr11a",
) -> pd.DataFrame:
    """
    Recode employee change (scr11a).

    Original: 1=increased ≥10%, 2=increased <10%, 3=unchanged, 4=decreased
    Creates:
        emp_change_ord : ordinal reversed (1=decreased → 4=increased ≥10%)
        emp_increased  : binary
    """
    df = df.copy()
    reverse_map = {1: 4, 2: 3, 3: 2, 4: 1}
    df["emp_change_ord"] = df[col].map(reverse_map)
    df["emp_increased"] = df[col].isin([1, 2]).astype(float)
    df.loc[df[col].isna(), "emp_increased"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Firm-level control variables
# ---------------------------------------------------------------------------

def recode_firm_size(
    df: pd.DataFrame,
    col: str = "scr10",
) -> pd.DataFrame:
    """
    Recode firm size categories.
    1=1-9, 2=10-49, 3=50-249 → micro/small/medium labels + ordinal.
    """
    df = df.copy()
    size_labels = {1: "micro", 2: "small", 3: "medium"}
    df["firm_size_cat"] = df[col].map(size_labels)
    df["firm_size_ord"] = df[col]  # already ordinal 1–3 for SMEs
    return df


def recode_firm_age(
    df: pd.DataFrame,
    col: str = "scr12",
) -> pd.DataFrame:
    """
    Recode firm age.
    1=before 2016, 2=2016-2018, 3=2019-2023, 4=after 2023
    Reversed so higher = older.
    """
    df = df.copy()
    age_map = {1: 4, 2: 3, 3: 2, 4: 1}
    df["firm_age_ord"] = df[col].map(age_map)
    age_labels = {1: "pre-2016", 2: "2016-2018", 3: "2019-2023", 4: "post-2023"}
    df["firm_age_cat"] = df[col].map(age_labels)
    return df


def recode_turnover_bracket(
    df: pd.DataFrame,
    col: str = "scr14",
) -> pd.DataFrame:
    """
    Recode turnover bracket (ordinal 1–9).
    Already ordinal from <25k to >50M. Keep as-is for models.
    """
    df = df.copy()
    labels = {
        1: "<25k", 2: "25-50k", 3: "50-100k", 4: "100-250k",
        5: "250-500k", 6: "500k-2M", 7: "2-10M", 8: "10-50M", 9: ">50M",
    }
    df["turnover_bracket_cat"] = df[col].map(labels)
    df["turnover_bracket_ord"] = df[col]
    return df


def recode_market_scope(
    df: pd.DataFrame,
    col: str = "scr16",
) -> pd.DataFrame:
    """Recode what the firm sells: products, services, or both."""
    df = df.copy()
    labels = {1: "products", 2: "services", 3: "both"}
    df["market_scope"] = df[col].map(labels)
    return df


# ---------------------------------------------------------------------------
# Difficulty index
# ---------------------------------------------------------------------------

# Q7.1–Q7.13: barriers to RE actions
# Q7.1–Q7.12 are substantive barriers; Q7.13 = "None" (no barriers encountered)
DIFFICULTY_COLS = [f"q7.{i}" for i in range(1, 14)]
DIFFICULTY_COLS_CORE = [f"q7.{i}" for i in range(1, 13)]  # q7.1–q7.12 (all substantive barriers)

# Readable names for the 12 substantive barrier dummies
Q7_BARRIER_NAMES: dict[str, str] = {
    "q7.1":  "barrier_finance",      # Lack of finance / investment
    "q7.2":  "barrier_cost",         # High upfront cost
    "q7.3":  "barrier_expertise",    # Lack of technical expertise
    "q7.4":  "barrier_time",         # Lack of time / staff
    "q7.5":  "barrier_info",         # Lack of information
    "q7.6":  "barrier_admin",        # Too much administration / complexity
    "q7.7":  "barrier_regulation",   # Regulations or incentives absent
    "q7.8":  "barrier_support",      # No support available
    "q7.9":  "barrier_roi",          # Uncertain return on investment
    "q7.10": "barrier_demand",       # Customers don't demand it
    "q7.11": "barrier_suppliers",    # Supplier constraints
    "q7.12": "barrier_tech",         # Lack of suitable technology
}

BARRIER_COLS = list(Q7_BARRIER_NAMES.values())

BARRIER_LABELS: dict[str, str] = {
    "barrier_finance":    "Lack of finance",
    "barrier_cost":       "High upfront cost",
    "barrier_expertise":  "Lack of expertise",
    "barrier_time":       "Lack of time/staff",
    "barrier_info":       "Lack of information",
    "barrier_admin":      "Too much admin",
    "barrier_regulation": "No regulation/incentive",
    "barrier_support":    "No support available",
    "barrier_roi":        "Uncertain ROI",
    "barrier_demand":     "No customer demand",
    "barrier_suppliers":  "Supplier constraints",
    "barrier_tech":       "Lack of suitable tech",
}


def compute_difficulty_count(
    df: pd.DataFrame,
    cols: list[str] = DIFFICULTY_COLS_CORE,
    col_name: str = "difficulty_count",
) -> pd.DataFrame:
    """Count of substantive barrier types encountered (0–12, summing q7.1–q7.12)."""
    df = df.copy()
    df[col_name] = df[cols].sum(axis=1)
    return df


def rename_barrier_dummies(
    df: pd.DataFrame,
    name_map: dict[str, str] = Q7_BARRIER_NAMES,
) -> pd.DataFrame:
    """
    Create readable named copies of the 12 barrier dummies.

    Maps q7.1–q7.12 → barrier_finance, barrier_cost, …, barrier_tech.
    Original q7.x columns are preserved; named columns added alongside.
    """
    df = df.copy()
    for src, dst in name_map.items():
        if src in df.columns:
            df[dst] = df[src]
    return df


def recode_n2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode N2 (share of recycled materials in products) into two columns:

        recycled_materials_ord : 0=none, 1=<5%, 2=5-19%, 3=20-49%, 4=50-74%, 5=75%+
        uses_recycled_materials: binary (1 if any recycled materials)

    Code 7 (DK) and 9 (inapplicable — non-product firms) → NaN.
    ~37% of firms are inapplicable (services / non-manufacturing sectors).
    """
    df = df.copy()
    recode = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4, 6.0: 5, 7.0: np.nan, 9.0: np.nan}
    df["recycled_materials_ord"] = df["n2"].map(recode)
    df["uses_recycled_materials"] = (df["recycled_materials_ord"] > 0).astype("float")
    df.loc[df["recycled_materials_ord"].isna(), "uses_recycled_materials"] = np.nan
    return df


def recode_n3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode N3 (explicit product lifespan extension policy) into:

        has_lifespan_policy : binary (1 = yes for any products)
        lifespan_policy_ord : 0=no, 1=yes for some products, 2=yes for all products

    Code 4 (DK) and 9 (inapplicable) → NaN.
    ~37% of firms are inapplicable (same skip pattern as n2).
    """
    df = df.copy()
    # has_lifespan_policy
    df["has_lifespan_policy"] = np.where(
        df["n3"].isin([1.0, 2.0]), 1.0,
        np.where(df["n3"] == 3.0, 0.0, np.nan),
    )
    # lifespan_policy_ord: 0=no, 1=some, 2=all
    ord_map = {1.0: 2, 2.0: 1, 3.0: 0, 4.0: np.nan, 9.0: np.nan}
    df["lifespan_policy_ord"] = df["n3"].map(ord_map)
    return df


# ---------------------------------------------------------------------------
# Industry sub-sector labels (D1b)
# ---------------------------------------------------------------------------

D1B_LABELS: dict[float, str] = {
    1.0:   "Aerospace & defence",
    2.0:   "Agri-food",
    3.0:   "Construction",
    4.0:   "Cultural & creative",
    5.0:   "Digital",
    6.0:   "Electronics",
    7.0:   "Energy — renewables",
    8.0:   "Energy-intensive industries",
    9.0:   "Health",
    10.0:  "Mobility / transport",
    11.0:  "Proximity / social economy",
    12.0:  "Retail",
    13.0:  "Textile",
    14.0:  "Tourism",
    997.0: np.nan,
    998.0: np.nan,
}


def rename_re_action_dummies(
    df: pd.DataFrame,
    name_map: dict[str, str] = RE_ACTION_DUMMY_NAMES,
) -> pd.DataFrame:
    """
    Create readable named copies of the 9 substantive RE action dummies.

    Maps q1.1–q1.9 → re_water, re_energy, re_renewables, re_materials,
    re_green_suppliers, re_waste_min, re_waste_sell, re_recycling, re_design.

    Original q1.x columns are preserved; named columns are added alongside.
    Missing q1.x values (NaN) are propagated.
    """
    df = df.copy()
    for src, dst in name_map.items():
        if src in df.columns:
            df[dst] = df[src]
    return df


def compute_action_bundles(
    df: pd.DataFrame,
    bundles: dict[str, list[str]] = RE_BUNDLES,
) -> pd.DataFrame:
    """
    Compute sum scores for each theoretically motivated RE action bundle.

    Each bundle score is the count of that bundle's actions the firm has taken.
    Named dummies (re_water, re_energy, …) must already exist in df.

    Bundle definitions:
        bundle_energy   : re_energy + re_renewables            (0–2)
        bundle_circular : re_waste_min + re_waste_sell
                          + re_recycling + re_design            (0–4)
        bundle_supply   : re_green_suppliers + re_materials     (0–2)
        bundle_inputs   : re_water + re_energy + re_materials   (0–3)
    """
    df = df.copy()
    for name, cols in bundles.items():
        avail = [c for c in cols if c in df.columns]
        df[name] = df[avail].sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# RE investment recoding
# ---------------------------------------------------------------------------

def recode_re_investment(
    df: pd.DataFrame,
    col: str = "q4",
) -> pd.DataFrame:
    """
    Recode RE investment level.
    1=nothing, 2=<1%, 3=1-5%, 4=6-10%, 5=11-30%, 6=>30%
    """
    df = df.copy()
    labels = {
        1: "nothing", 2: "<1%", 3: "1-5%",
        4: "6-10%", 5: "11-30%", 6: ">30%",
    }
    df["re_investment_cat"] = df[col].map(labels)
    df["re_investment_ord"] = df[col]
    df["re_invests"] = (df[col] >= 2).astype(float)
    df.loc[df[col].isna(), "re_invests"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Country & sector labels
# ---------------------------------------------------------------------------

COUNTRY_LABELS = {
    1: "FR",  2: "BE",  3: "NL",  4: "DE",  5: "IT",  6: "LU",
    7: "DK",  8: "IE",  9: "GB",  11: "GR", 12: "ES", 13: "PT",
    16: "FI", 17: "SE", 18: "AT", 19: "CY", 20: "CZ",
    21: "EE", 22: "HU", 23: "LV", 24: "LT", 25: "MT",
    26: "PL", 27: "SK", 28: "SI", 29: "BG", 30: "RO", 32: "HR",
}

# Supranational macro-region groupings (UN Statistics Division / EU cohesion classification)
# Four blocs used as Level 3 in the multilevel model.
# UK grouped with Northern Europe given geographic/institutional proximity.
MACRO_REGION_MAP: dict[str, str] = {
    # Northern Europe
    "GB": "Northern", "DK": "Northern", "FI": "Northern", "SE": "Northern",
    "EE": "Northern", "LV": "Northern", "LT": "Northern", "IE": "Northern",
    # Western Europe
    "AT": "Western", "BE": "Western", "DE": "Western",
    "FR": "Western", "LU": "Western", "NL": "Western",
    # Southern Europe
    "CY": "Southern", "ES": "Southern", "GR": "Southern",
    "IT": "Southern", "MT": "Southern", "PT": "Southern",
    # Central & Eastern Europe
    "BG": "CEE", "CZ": "CEE", "HR": "CEE", "HU": "CEE",
    "PL": "CEE", "RO": "CEE", "SI": "CEE", "SK": "CEE",
}

SECTOR_LABELS = {
    1: "Manufacturing",
    2: "Industry",
    3: "Retail",
    4: "Services",
}


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable country, sector, and industry sub-sector labels."""
    df = df.copy()
    df["country_label"] = df["country"].map(COUNTRY_LABELS)
    df["sector_label"]  = df["nace_b"].map(SECTOR_LABELS)
    df["industry_label"] = df["d1b"].map(D1B_LABELS)
    return df


def add_macro_region(
    df: pd.DataFrame,
    iso_col: str = "isocntry",
    out_col: str = "macro_region",
) -> pd.DataFrame:
    """
    Assign each firm to one of four EU supranational macro-regions using the
    UN Statistics Division / EU cohesion classification:

        Northern : GB DK FI SE IE EE LV LT
        Western  : AT BE DE FR LU NL
        Southern : CY ES GR IT MT PT
        CEE      : BG CZ HR HU PL RO SI SK

    UK (GB) is grouped with Northern Europe given geographic and institutional
    proximity. Observations not in the map (non-EU/UK countries) receive NaN.
    """
    df = df.copy()
    df[out_col] = df[iso_col].map(MACRO_REGION_MAP)
    unmapped = df[out_col].isna().sum()
    if unmapped:
        print(f"add_macro_region: {unmapped:,} rows unmapped (non-EU/UK) → NaN")
    return df
