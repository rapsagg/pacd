# European SMEs: Resource Efficiency & Stock Value Analysis

A CRISP-DM structured data science project examining how **resource efficiency** and **management practices** affect the **stock value** of European small and medium-sized enterprises (SMEs).

## Research Objective

Investigate the relationship between resource efficiency metrics (energy, material, waste, water), management quality indicators, and firm stock market performance across European SMEs using **multilevel (hierarchical linear) modeling** to account for nested data structures (firms within industries, industries within countries).

---

## CRISP-DM Phases

| Phase | Notebook | Description |
|-------|----------|-------------|
| 1 | `01_business_understanding.ipynb` | Research questions, KPIs, success criteria |
| 2 | `02_data_understanding.ipynb` | EDA, data quality checks, distributions |
| 3 | `03_data_preparation.ipynb` | Cleaning, feature engineering, encoding |
| 4 | `04_modeling.ipynb` | OLS baseline + multilevel models (HLM) |
| 5 | `05_evaluation.ipynb` | Model diagnostics, validation, interpretation |

---

## Project Structure

```
pacd/
├── data/
│   ├── raw/          # Original, immutable data (DB or flat files)
│   ├── interim/      # Intermediate cleaned/transformed data
│   └── processed/    # Final datasets ready for modeling
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py          # DB/file ingestion utilities
│   │   └── preprocessor.py    # Cleaning & transformation pipelines
│   ├── features/
│   │   └── engineer.py        # Feature construction & selection
│   └── models/
│       └── multilevel.py      # HLM / mixed-effects model wrappers
├── reports/
│   └── figures/               # Generated plots and tables
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url> && cd pacd

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your raw data in data/raw/
#    (SQLite .db file, CSVs, or Excel files)

# 5. Launch Jupyter
jupyter notebook notebooks/
```

---

## Multilevel Modeling Rationale

SME data is inherently **nested**:

```
Country
  └── Industry sector
        └── Firm (repeated observations / years)
```

Ignoring this structure violates OLS independence assumptions and underestimates standard errors. We use **Hierarchical Linear Models (HLM)** via `statsmodels.MixedLM` to:
- Partition variance across levels
- Allow random intercepts (and slopes) per country/sector
- Properly estimate firm-level effects of resource efficiency

---

## Key Variables (expected)

| Variable | Type | Level | Description |
|----------|------|-------|-------------|
| `stock_return` / `tobin_q` | Outcome | Firm | Stock performance |
| `energy_intensity` | Predictor | Firm | Energy per unit output |
| `material_efficiency` | Predictor | Firm | Material productivity |
| `waste_ratio` | Predictor | Firm | Waste generated per revenue |
| `mgmt_score` | Predictor | Firm | Management quality index |
| `firm_size` | Control | Firm | Log(assets) or employees |
| `leverage` | Control | Firm | Debt-to-equity ratio |
| `sector` | Group | Industry | NACE / SIC code |
| `country` | Group | Country | ISO country code |
| `year` | Time | - | Fiscal year |
