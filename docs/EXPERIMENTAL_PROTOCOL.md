# Revised experimental protocol

## Purpose

The Python experiment was added after the original Orange stage to provide a leakage-resistant and reproducible comparison of four classifiers under a screening-oriented F2 objective.

## Configurations

Four model families are evaluated both without and with minority-class oversampling:

- Logistic Regression (LR)
- Decision Tree (DT)
- Random Forest (RF)
- Neural Network / Multilayer Perceptron (NN)

The labels `without oversampling` and `with oversampling` are preferred over `unbalanced` and `balanced`. LR and RF also use class weighting in their estimator definitions, so `without oversampling` does not mean that no imbalance-handling mechanism is present.

## Nested evaluation

1. A stratified five-fold outer loop estimates generalization performance.
2. For each outer training partition, a stratified three-fold inner loop generates out-of-fold probabilities.
3. Candidate thresholds from 0.05 to 0.95 are evaluated by F2.
4. The threshold with maximum F2 is selected; the higher threshold breaks ties.
5. Preprocessing and the model are refitted on the complete outer training partition.
6. The selected threshold is applied once to the untouched outer fold.

## Leakage controls

- Imputation parameters are learned from training data only.
- LR and NN scaling parameters are learned from training data only.
- Oversampling is performed after splitting and on training partitions only.
- Threshold selection never uses the corresponding outer evaluation fold.

## Outputs

- fold-level metrics at the selected threshold and at 0.5;
- out-of-fold probabilities and predictions;
- means and standard deviations across outer folds;
- pooled diagnostic counts and figures;
- exact paired McNemar comparison of RF and LR.

