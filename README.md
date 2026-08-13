# Early identification of higher-education withdrawal risk

Code for COSC2669/COSC2816 Individual Task 1, Part 1.3. Logistic Regression and a small neural network are compared on two public datasets to flag students at risk of dropout or later withdrawal.

This is a proposed application relevant to an education-assessment data science role. It is not an existing ACER project.

## What you need

- Python 3.10 or later
- The two source ZIP files listed below (not stored in this repo)

```bash
python -m pip install -r requirements.txt
```

## Datasets

Download the ZIPs and place them in `data/raw/` with these exact names:

| File | Source | Role |
|---|---|---|
| `predict+students+dropout+and+academic+success.zip` | [UCI dropout and academic success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) | Enrolment-time dropout |
| `open+university+learning+analytics+dataset.zip` | [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) | Later withdrawal using day-0–30 information only |

The first notebook/script run builds leakage-safe tables in `data/processed/`. Later runs reuse those CSVs.

## How to run

**Option A — notebook (recommended for the appendix)**

1. Open `notebooks/01_student_risk_analysis.ipynb`.
2. Run all cells. The kernel may start in the repo root or in `notebooks/`; the first cell resolves the project root either way.

**Option B — script**

From the repository root:

```bash
python src/data_prep.py --uci-zip data/raw/predict+students+dropout+and+academic+success.zip --oulad-zip data/raw/open+university+learning+analytics+dataset.zip --output-dir data/processed
python scripts/run_part13_analysis.py
```

Skip `data_prep.py` if `data/processed/uci_enrolment_model.csv` and `data/processed/oulad_day30_model.csv` already exist.

Figures are written to `figures/`. Metric and insight tables are written to `results/`.

## Reproducibility

Splits use `random_state=42`. UCI uses a stratified hold-out; OULAD uses a group-aware split on `id_student` so the same learner does not appear in both train and test.

Thresholds are chosen on validation data only (maximum F1). The test set is scored once.

## Project layout

```
data/raw/            # source ZIPs (you add these)
data/processed/      # leakage-safe modelling tables
docs/analysis_plan.md
figures/
notebooks/01_student_risk_analysis.ipynb
results/
scripts/run_part13_analysis.py
src/data_prep.py     # ZIP → modelling tables
src/modeling.py      # models, metrics, plots
```
