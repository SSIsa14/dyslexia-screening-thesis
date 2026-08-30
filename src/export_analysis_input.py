"""Validate the manually improved combined dataset and export the analysis CSV.

This script does not claim to recreate the manual feature-renaming and dataset-
harmonization work. Those steps are documented in docs/DATA_PREPARATION.md.
It verifies the final workbook used in the thesis and exports the exact input
expected by run_nested_evaluation.py.
"""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "data/processed/dataset_combined_improved.xlsx"
OUTPUT = REPO_ROOT / "data/processed/analysis_input.csv"

EXPECTED_ROWS = 5039
EXPECTED_COLUMNS = 197  # 196 predictors plus the Dyslexia target
EXPECTED_POSITIVES = 540
EXPECTED_NEGATIVES = 4499


def main() -> None:
    frame = pd.read_excel(SOURCE)

    if frame.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(
            f"Unexpected dataset shape {frame.shape}; "
            f"expected {(EXPECTED_ROWS, EXPECTED_COLUMNS)}"
        )
    if "Dyslexia" not in frame.columns:
        raise ValueError("The target column 'Dyslexia' is missing")

    counts = frame["Dyslexia"].value_counts(dropna=False).to_dict()
    if counts.get("Yes") != EXPECTED_POSITIVES or counts.get("No") != EXPECTED_NEGATIVES:
        raise ValueError(f"Unexpected class counts: {counts}")

    frame.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"Rows: {len(frame)}")
    print(f"Predictors: {len(frame.columns) - 1}")
    print(f"Class counts: {counts}")


if __name__ == "__main__":
    main()
