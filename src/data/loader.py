"""
Data loading utilities for the SME Resource Efficiency project.

Supports SPSS (.sav), CSV, and Excel files.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def list_available_sources() -> dict:
    """Return all data files found in data/raw/."""
    sources = {
        "spss": list(DATA_RAW.glob("*.sav")),
        "csv": list(DATA_RAW.glob("*.csv")),
        "excel": list(DATA_RAW.glob("*.xlsx")) + list(DATA_RAW.glob("*.xls")),
    }
    return sources


# ---------------------------------------------------------------------------
# SPSS (.sav) loading
# ---------------------------------------------------------------------------

def load_sav(
    filename: Optional[str] = None,
    apply_value_labels: bool = False,
) -> tuple[pd.DataFrame, object]:
    """
    Load an SPSS .sav file from data/raw/.

    Parameters
    ----------
    filename : name of .sav file. If None, auto-detects the first .sav file.
    apply_value_labels : if True, replace coded values with SPSS value labels.

    Returns
    -------
    (df, meta) : DataFrame and pyreadstat metadata object.
                 meta.column_names_to_labels  → variable labels
                 meta.variable_value_labels   → value label mappings
    """
    import pyreadstat

    if filename is None:
        sav_files = list(DATA_RAW.glob("*.sav"))
        if not sav_files:
            raise FileNotFoundError(f"No .sav file found in {DATA_RAW}")
        path = sav_files[0]
    else:
        path = DATA_RAW / filename

    if apply_value_labels:
        df, meta = pyreadstat.read_sav(path, apply_value_formats=True)
    else:
        df, meta = pyreadstat.read_sav(path)

    print(f"Loaded {path.name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df, meta


def get_variable_info(meta) -> pd.DataFrame:
    """
    Build a reference table of all variables from SPSS metadata.

    Returns DataFrame with columns: variable, label, has_value_labels, n_values.
    """
    rows = []
    for var in meta.column_names:
        label = meta.column_names_to_labels.get(var, "")
        val_labels = meta.variable_value_labels.get(var, {})
        rows.append({
            "variable": var,
            "label": label,
            "has_value_labels": len(val_labels) > 0,
            "n_values": len(val_labels),
        })
    return pd.DataFrame(rows)


def get_value_labels(meta, variable: str) -> dict:
    """Return the value label mapping for a specific variable."""
    return meta.variable_value_labels.get(variable, {})


# ---------------------------------------------------------------------------
# CSV / Excel loading
# ---------------------------------------------------------------------------

def load_csv(filename: str, **kwargs) -> pd.DataFrame:
    """Load a CSV from data/raw/."""
    path = DATA_RAW / filename
    return pd.read_csv(path, **kwargs)


def load_excel(filename: str, sheet_name=0, **kwargs) -> pd.DataFrame:
    """Load an Excel sheet from data/raw/."""
    path = DATA_RAW / filename
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)


# ---------------------------------------------------------------------------
# Interim / processed data
# ---------------------------------------------------------------------------

def _pkl_path(base: Path, filename: str) -> Path:
    """Replace any extension with .pkl."""
    return base / (Path(filename).stem + ".pkl")


def save_interim(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame to data/interim/ as pickle."""
    out = _pkl_path(PROJECT_ROOT / "data" / "interim", filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    print(f"Saved → {out}")
    return out


def load_interim(filename: str) -> pd.DataFrame:
    """Load a pickle file from data/interim/."""
    path = _pkl_path(PROJECT_ROOT / "data" / "interim", filename)
    return pd.read_pickle(path)


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame to data/processed/ as pickle."""
    out = _pkl_path(PROJECT_ROOT / "data" / "processed", filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    print(f"Saved → {out}")
    return out


def load_processed(filename: str) -> pd.DataFrame:
    """Load a pickle file from data/processed/."""
    path = _pkl_path(PROJECT_ROOT / "data" / "processed", filename)
    return pd.read_pickle(path)
