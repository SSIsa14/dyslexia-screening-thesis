# Dataset preparation

## Status of this documentation

The improved datasets were prepared manually during the thesis. No historical script exists for those operations. This document separates verified facts from steps that still require confirmation against the thesis Word document and the author's records.

## Verified file lineage

```text
dyt_desktop.csv (3,644 records) ─┐
                                 ├─ dataset_combined_original.xlsx (5,039 records)
dyt_tablet.csv  (1,395 records) ─┘

dataset_desktop_improved.xlsx (3,644 records) ─┐
                                               ├─ dataset_combined_improved.xlsx (5,039 records)
dataset_tablet_improved.xlsx  (1,395 records) ─┘
```

The final improved combined workbook contains 5,039 records and 197 columns: 196 predictors and the `Dyslexia` target.

## Verified structural changes

- Original generic column names were replaced with descriptive names.
- Demographic names were standardized, including `NativeLang` and `LangFail`.
- Repeated task columns such as clicks, hits, misses, score, accuracy, and miss rate were renamed by task/domain.
- Desktop and tablet files were aligned to a common 197-column schema.
- Missing tablet measurements were retained as missing values; they are imputed later inside each training fold, not globally in the workbook.
- Desktop and tablet improved files were concatenated to create the final combined workbook.

## Missing data

The desktop improved workbook contains no missing cells. The tablet portion contains 45,650 missing cells because age-customized tablet tests did not administer every task to every participant. These account for approximately 4.60% of all cells in the combined dataset.

The evaluation pipeline handles these values within validation splits:

- numerical values: training-partition mean;
- categorical values: training-partition most frequent value.

## Analysis input

`src/export_analysis_input.py` validates `dataset_combined_improved.xlsx` and exports `analysis_input.csv`. The exported CSV was verified to be exactly equal to the workbook in shape, column order, values, labels, and missing entries.

## Items to confirm when revising the thesis

When the thesis Word document is available, this document should be expanded with the author's exact manual procedure, including:

- how the original semicolon-delimited files were imported;
- the precise variable-name mapping source;
- whether any data types or category spellings were normalized;
- whether any rows were reordered;
- the exact software used for manual editing;
- quality-control checks performed before consolidation.

