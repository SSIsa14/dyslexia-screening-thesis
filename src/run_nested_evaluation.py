from __future__ import annotations

import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn


SEED = 20260829
OUTER_FOLDS = 5
INNER_FOLDS = 3
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data/processed/analysis_input.csv"
OUT_DIR = REPO_ROOT / "results/generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_preprocessor(X: pd.DataFrame, scaled: bool) -> ColumnTransformer:
    categorical = [c for c in ["Gender", "NativeLang", "LangFail"] if c in X.columns]
    numerical = [c for c in X.columns if c not in categorical]
    num_steps = [("imputer", SimpleImputer(strategy="mean", keep_empty_features=True))]
    if scaled:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numerical),
        ("cat", cat_pipe, categorical),
    ], remainder="drop")


def oversample_minority(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = np.bincount(y)
    majority_n = int(counts.max())
    pieces_x, pieces_y = [], []
    for cls in range(len(counts)):
        idx = np.flatnonzero(y == cls)
        sampled = rng.choice(idx, size=majority_n, replace=True)
        pieces_x.append(X[sampled])
        pieces_y.append(y[sampled])
    Xb = np.concatenate(pieces_x, axis=0)
    yb = np.concatenate(pieces_y, axis=0)
    order = rng.permutation(len(yb))
    return Xb[order], yb[order]


def metric_row(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "referral_rate": float(pred.mean()),
        "auprc": average_precision_score(y, prob),
        "auroc": roc_auc_score(y, prob),
        "tp": int(np.sum((pred == 1) & (y == 1))),
        "fp": int(np.sum((pred == 1) & (y == 0))),
        "fn": int(np.sum((pred == 0) & (y == 1))),
        "tn": int(np.sum((pred == 0) & (y == 0))),
    }


def choose_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    scores = np.array([fbeta_score(y, prob >= t, beta=2, zero_division=0) for t in THRESHOLDS])
    best = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12))
    return float(THRESHOLDS[best[-1]])  # higher threshold breaks ties, reducing referrals


def evaluate_configuration(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    estimator,
    scaled: bool,
    balanced: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    fold_rows = []
    oof_rows = []
    started = time.time()
    print(f"START {model_name} balanced={balanced}", flush=True)

    for outer_fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + outer_fold)
        inner_prob = np.zeros(len(train_idx), dtype=float)
        for inner_fold, (itr, ival) in enumerate(inner.split(X_train, y_train), start=1):
            pre = make_preprocessor(X_train.iloc[itr], scaled)
            Xi_train = pre.fit_transform(X_train.iloc[itr])
            Xi_val = pre.transform(X_train.iloc[ival])
            yi_train = y_train[itr]
            if balanced:
                Xi_train, yi_train = oversample_minority(
                    Xi_train, yi_train, SEED + outer_fold * 100 + inner_fold
                )
            fitted = clone(estimator)
            fitted.fit(Xi_train, yi_train)
            inner_prob[ival] = fitted.predict_proba(Xi_val)[:, 1]

        threshold = choose_threshold(y_train, inner_prob)
        pre = make_preprocessor(X_train, scaled)
        Xo_train = pre.fit_transform(X_train)
        Xo_test = pre.transform(X_test)
        yo_train = y_train
        if balanced:
            Xo_train, yo_train = oversample_minority(
                Xo_train, yo_train, SEED + outer_fold * 1000
            )
        fitted = clone(estimator)
        fitted.fit(Xo_train, yo_train)
        prob = fitted.predict_proba(Xo_test)[:, 1]

        selected = metric_row(y_test, prob, threshold)
        default = metric_row(y_test, prob, 0.5)
        base = {
            "model": model_name,
            "balanced": balanced,
            "fold": outer_fold,
            "n_test": len(test_idx),
            "n_positive": int(y_test.sum()),
        }
        fold_rows.append({**base, "operating_point": "inner_f2", **selected})
        fold_rows.append({**base, "operating_point": "default_0.5", **default})
        for local_i, global_i in enumerate(test_idx):
            oof_rows.append({
                "row_index": int(global_i),
                "fold": outer_fold,
                "model": model_name,
                "balanced": balanced,
                "y_true": int(y_test[local_i]),
                "probability": float(prob[local_i]),
                "selected_threshold": threshold,
            })
        print(
            f"  fold={outer_fold} threshold={threshold:.2f} "
            f"F2={selected['f2']:.3f} R={selected['recall']:.3f} "
            f"P={selected['precision']:.3f} Ref={selected['referral_rate']:.3f}",
            flush=True,
        )

    folds = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)
    key = f"{model_name}_{'balanced' if balanced else 'unbalanced'}"
    folds.to_csv(OUT_DIR / f"folds_{key}.csv", index=False)
    oof.to_csv(OUT_DIR / f"oof_{key}.csv", index=False)
    print(f"DONE {key} in {time.time() - started:.1f}s", flush=True)
    return folds, oof


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    y = df.pop("Dyslexia").map({"No": 0, "Yes": 1}).to_numpy(dtype=int)
    X = df
    models = {
        "LR": (LogisticRegression(C=1, penalty="l2", class_weight="balanced", max_iter=3000, random_state=SEED), True),
        "DT": (DecisionTreeClassifier(max_depth=100, min_samples_split=5, min_samples_leaf=2, random_state=SEED), False),
        "RF": (RandomForestClassifier(n_estimators=500, class_weight="balanced", min_samples_split=2, n_jobs=-1, random_state=SEED), False),
        "NN": (MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="sgd", alpha=0.0002, max_iter=1000, random_state=SEED, early_stopping=True, n_iter_no_change=20), True),
    }
    all_folds, all_oof = [], []
    for model_name, (estimator, scaled) in models.items():
        for balanced in (False, True):
            folds, oof = evaluate_configuration(X, y, model_name, estimator, scaled, balanced)
            all_folds.append(folds)
            all_oof.append(oof)

    folds = pd.concat(all_folds, ignore_index=True)
    oof = pd.concat(all_oof, ignore_index=True)
    folds.to_csv(OUT_DIR / "all_fold_metrics.csv", index=False)
    oof.to_csv(OUT_DIR / "all_oof_predictions.csv", index=False)

    metric_cols = ["threshold", "accuracy", "precision", "recall", "f1", "f2", "referral_rate", "auprc", "auroc"]
    summary = folds.groupby(["model", "balanced", "operating_point"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(["operating_point", "f2_mean"], ascending=[True, False])
    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    metadata = {
        "seed": SEED,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "threshold_candidates": [float(THRESHOLDS[0]), float(THRESHOLDS[-1]), 0.01],
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "positive_rows": int(y.sum()),
        "negative_rows": int((y == 0).sum()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nFINAL SUMMARY\n", summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
