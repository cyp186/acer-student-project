# Early identification of higher-education withdrawal risk

This is a proposed application relevant to an education-assessment data science role. It is not an existing ACER project.

## Requirements to run

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

The first notebook run builds leakage-safe tables in `data/processed/`. Later runs reuse those CSVs.

## How to run

1. Open `notebooks/01_student_risk_analysis.ipynb`.
2. Run all cells. The kernel may start in the repo root or in `notebooks/`; the first cell resolves the project root either way.

If the processed CSVs are missing, put the ZIPs in `data/raw/` first. The notebook will call `build_processed_tables`. You can also build them from the repository root:

```bash
python src/data_prep.py --uci-zip data/raw/predict+students+dropout+and+academic+success.zip --oulad-zip data/raw/open+university+learning+analytics+dataset.zip --output-dir data/processed
```

Figures are written to `figures/`. Metric and insight tables are written to `results/`.

## Reproducibility

Splits use `random_state=42`. UCI uses a stratified hold-out; OULAD uses a group-aware split on `id_student` so the same learner does not appear in both train and test.

Thresholds are chosen on validation data only (maximum F1). The test set is scored once.

