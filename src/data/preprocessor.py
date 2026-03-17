"""
Data cleaning and preprocessing for the Flash Eurobarometer SME survey.

Covers:
  - Recoding DK/NA/Inap values to NaN
  - Applying SPSS value labels
  - Data quality diagnostics
  - Filtering to SMEs only
  - Subsetting to EU countries
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SPSS coded missing values
# ---------------------------------------------------------------------------

# Common DK/NA/Inap codes used across the survey
DK_NA_CODES = {998.0, 999.0, 997.0, 999997.0, 999998.0}


def recode_dk_na(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    dk_na_codes: set = DK_NA_CODES,
) -> pd.DataFrame:
    """
    Replace DK/NA/Inap coded values with NaN.

    Parameters
    ----------
    cols : columns to process. If None, applies to all numeric columns.
    dk_na_codes : set of values treated as missing (default covers common
                  Eurobarometer codes: 997, 998, 999, 999997, 999998).
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    for col in cols:
        if col in df.columns:
            df.loc[df[col].isin(dk_na_codes), col] = np.nan
    return df


def recode_inap(
    df: pd.DataFrame,
    cols: list[str],
    inap_code: float = 9.0,
) -> pd.DataFrame:
    """
    Replace 'Inapplicable' coded values (typically 9) with NaN.
    Use for skip-pattern questions (e.g., q3 is only asked if q1.x == 1).
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df.loc[df[col] == inap_code, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column quality summary: dtype, n_missing, pct_missing, n_unique, value range.
    """
    report = pd.DataFrame({
        "dtype": df.dtypes,
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    for col in df.select_dtypes(include="number").columns:
        report.loc[col, "min"] = df[col].min()
        report.loc[col, "max"] = df[col].max()
    return report


def value_distribution(
    df: pd.DataFrame,
    col: str,
    meta=None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Frequency table for a single variable.
    If meta is provided, includes SPSS value labels.
    """
    counts = df[col].value_counts(dropna=False).sort_index()
    result = pd.DataFrame({"value": counts.index, "count": counts.values})
    if normalize:
        result["pct"] = (result["count"] / len(df) * 100).round(1)
    if meta is not None:
        val_labels = meta.variable_value_labels.get(col, {})
        result["label"] = result["value"].map(val_labels).fillna("")
    return result


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

# EU-27 country codes (post-Brexit)
EU27_CODES = {
    1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32,
}

# UK code (GB) — included in the analysis because of post-Brexit proximity
UK_CODE = {9}

# Primary analysis sample: EU-27 + UK (28 countries)
EU27_UK_CODES = EU27_CODES | UK_CODE


def filter_eu27(df: pd.DataFrame, country_col: str = "country") -> pd.DataFrame:
    """Keep only EU-27 member state observations (no UK)."""
    before = len(df)
    df = df[df[country_col].isin(EU27_CODES)].copy()
    print(f"Filtered to EU-27: {before:,} → {len(df):,} rows")
    return df


def filter_eu_uk(df: pd.DataFrame, country_col: str = "country") -> pd.DataFrame:
    """Keep EU-27 + UK observations (28 countries, primary analysis sample)."""
    before = len(df)
    df = df[df[country_col].isin(EU27_UK_CODES)].copy()
    print(f"Filtered to EU-27 + UK: {before:,} → {len(df):,} rows")
    return df


def filter_sme(df: pd.DataFrame, size_col: str = "scr10") -> pd.DataFrame:
    """
    Keep only SMEs (1-249 employees).
    scr10 codes: 1 = 1-9, 2 = 10-49, 3 = 50-249, 4 = 250-499, 5 = 500+
    """
    before = len(df)
    df = df[df[size_col].isin([1, 2, 3])].copy()
    print(f"Filtered to SMEs (1-249 employees): {before:,} → {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Type handling
# ---------------------------------------------------------------------------

def set_categorical(
    df: pd.DataFrame,
    cols: list[str],
    ordered: bool = False,
) -> pd.DataFrame:
    """Convert columns to categorical dtype."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
            if ordered:
                df[col] = df[col].cat.as_ordered()
    return df


def drop_high_missing(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Drop columns where fraction of missing values exceeds threshold."""
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > threshold].index.tolist()
    if to_drop:
        print(f"Dropping {len(to_drop)} columns with >{threshold*100:.0f}% missing: {to_drop[:10]}...")
    return df.drop(columns=to_drop)


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """Drop duplicate rows and report count removed."""
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate rows.")
    return df.reset_index(drop=True)
