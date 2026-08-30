# Orange Data Mining workflow

`dislexia_mejorado.ows` is the original thesis workflow used for exploratory modeling in Orange Data Mining.

It contains balanced and non-balanced branches for Logistic Regression, Decision Tree, Random Forest, and Neural Network models. It should be understood as the original graphical workflow, not as the source of the final nested-evaluation metrics reported in `results/`.

The final reported metrics were generated in Python because the revised protocol required explicit control over:

- outer and inner cross-validation folds;
- training-only preprocessing and oversampling;
- model-specific threshold selection;
- fixed random seeds;
- fold-level predictions and uncertainty estimates.

Open the `.ows` file with Orange Data Mining. File paths may need to be reconnected to `data/processed/dataset_combined_improved.xlsx` after cloning the repository.

