# Reproducible Experiments for Early Dyslexia Risk Screening

This repository contains the technical materials used in Sofia Isabela Sandoval Vasquez's thesis experiments on machine-learning models for early dyslexia risk screening. It includes data organization, the original Orange Data Mining workflow, reproducible Python experiments, results, figures, and methodological documentation.

## Research workflow

The work has two clearly identified stages:

1. **Original Orange stage.** Orange Data Mining was used to design the initial balanced and non-balanced workflows and compare Logistic Regression, Decision Tree, Random Forest, and Neural Network models.
2. **Revised Python stage.** After methodological revisions of the original experiments, the same improved combined dataset was evaluated with a nested, leakage-resistant protocol in Python. This stage strengthened the evaluation through training-only preprocessing and oversampling, model-specific threshold selection, uncertainty estimates, AUPRC, referral rate, a flag-all baseline, and a paired McNemar comparison.

The Python input is not a different dataset. `analysis_input.csv` is a direct export of `dataset_combined_improved.xlsx`; the two were verified to contain the same 5,039 rows, 197 columns, values, and missing entries.

## Repository structure

```text
.
├── data/
│   ├── raw/              Original downloaded and combined files
│   ├── processed/        Manually improved desktop, tablet, and combined files
│   ├── dictionary/       Variable-description workbook
│   └── README.md         Provenance and publication notes
├── orange/
│   ├── dislexia_mejorado.ows
│   └── README.md
├── src/
│   ├── export_analysis_input.py
│   ├── run_nested_evaluation.py
│   └── generate_results.py
├── results/
│   ├── figures/
│   ├── tables/
│   ├── predictions/
│   ├── statistics/
│   └── metadata.json
├── docs/
│   ├── DATA_PREPARATION.md
│   ├── EXPERIMENTAL_PROTOCOL.md
│   └── RESULTS.md
├── CITATION.cff
├── LICENSE_STATUS.md
└── requirements.txt
```

## Dataset summary

- Records: 5,039
- Predictors: 196
- Target: `Dyslexia`
- Negative (`No`) records: 4,499
- Positive (`Yes`) records: 540
- Desktop records: 3,644
- Tablet records: 1,395

The original datasets accompany Rello et al., "Predicting Risk of Dyslexia With an Online Gamified Test," *PLOS ONE*, 2020, DOI: [10.1371/journal.pone.0241687](https://doi.org/10.1371/journal.pone.0241687). See `data/README.md` before publishing or redistributing data files.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproducing the analysis

From the repository root:

```bash
python src/export_analysis_input.py
python src/run_nested_evaluation.py
python src/generate_results.py
```

The first command validates and exports the manually improved workbook. It does **not** claim to recreate the earlier manual harmonization. The second command reruns all eight model/oversampling configurations. The third command summarizes predictions, performs the McNemar comparison, and generates figures.

The full evaluation can take a substantial amount of time because it trains four model families under two sampling conditions across nested folds, including a 500-tree Random Forest and neural networks.

## Main evaluation design

- Fixed random seed: `20260829`
- Outer evaluation: stratified five-fold cross-validation
- Inner threshold selection: stratified three-fold cross-validation
- Candidate thresholds: 0.05 through 0.95 in increments of 0.01
- Threshold objective: maximum F2; higher threshold breaks ties
- Scaling: Logistic Regression and Neural Network only
- Oversampling: training partitions only
- Principal reported metrics: Accuracy, Precision, Recall, F1, F2, AUPRC, AUROC, and referral rate

## Principal result

Random Forest without oversampling achieved the highest mean F2 (`0.592 ± 0.015`) and AUPRC (`0.464 ± 0.020`), with Recall `0.774 ± 0.041` and referral rate `0.272 ± 0.024`. The paired McNemar comparison with Logistic Regression did not establish universal superiority (`p = 0.276`); the recommendation follows the predeclared F2 objective and workload profile.

## Reuse and licensing

No open-source license has been selected for the thesis code yet. Until the authors choose one and confirm the dataset redistribution terms, the contents should not be assumed to permit unrestricted reuse. See `LICENSE_STATUS.md`.
