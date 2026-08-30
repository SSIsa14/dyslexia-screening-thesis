from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, fbeta_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "results/generated"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
oof = pd.read_csv(ROOT / "all_oof_predictions.csv")
folds = pd.read_csv(ROOT / "all_fold_metrics.csv")

rows = []
for (model, balanced), g in oof.groupby(["model", "balanced"]):
    g = g.sort_values("row_index")
    y = g.y_true.to_numpy()
    prob = g.probability.to_numpy()
    pred = (prob >= g.selected_threshold.to_numpy()).astype(int)
    rows.append({
        "model": model, "balanced": balanced,
        "threshold_mean": g.groupby("fold").selected_threshold.first().mean(),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "referral_rate": pred.mean(),
        "auprc": average_precision_score(y, prob),
        "auroc": roc_auc_score(y, prob),
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
    })
aggregate = pd.DataFrame(rows).sort_values("f2", ascending=False)
aggregate.to_csv(ROOT / "aggregate_oof_metrics.csv", index=False)

selected_folds = folds[folds.operating_point == "inner_f2"]
metric_cols = ["threshold", "accuracy", "precision", "recall", "f1", "f2", "referral_rate", "auprc", "auroc"]
mean_std = selected_folds.groupby(["model", "balanced"])[metric_cols].agg(["mean", "std"])
mean_std.columns = [f"{a}_{b}" for a, b in mean_std.columns]
mean_std.reset_index().to_csv(ROOT / "selected_mean_std.csv", index=False)

def selected_group(model, balanced):
    return oof[(oof.model == model) & (oof.balanced == balanced)].sort_values("row_index")

rf = selected_group("RF", False)
lr = selected_group("LR", False)
y = rf.y_true.to_numpy()
rf_pred = (rf.probability.to_numpy() >= rf.selected_threshold.to_numpy()).astype(int)
lr_pred = (lr.probability.to_numpy() >= lr.selected_threshold.to_numpy()).astype(int)
rf_correct = rf_pred == y
lr_correct = lr_pred == y
b = int((rf_correct & ~lr_correct).sum())
c = int((~rf_correct & lr_correct).sum())
pvalue = binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue
(ROOT / "mcnemar_rf_vs_lr.txt").write_text(
    f"RF correct / LR wrong: {b}\nRF wrong / LR correct: {c}\nExact McNemar p-value: {pvalue:.8g}\n",
    encoding="utf-8",
)

best = {"LR": False, "DT": True, "RF": False, "NN": True}
# Grayscale styles remain distinguishable in print and on screen.
line_styles = {
    "LR": ("black", "-"),
    "DT": ("0.35", "--"),
    "RF": ("0.15", "-."),
    "NN": ("0.55", ":"),
}

fig, ax = plt.subplots(figsize=(5.2, 3.5), dpi=220)
for model, balanced in best.items():
    g = selected_group(model, balanced)
    yy, pp = g.y_true.to_numpy(), g.probability.to_numpy()
    precision, recall, _ = precision_recall_curve(yy, pp)
    ap = average_precision_score(yy, pp)
    color, style = line_styles[model]
    ax.plot(recall, precision, lw=1.8, color=color, ls=style,
            label=f"{model} (AUPRC={ap:.3f})")
ax.axhline(y.mean(), color="0.65", ls=(0, (5, 3)), lw=1,
           label=f"Prevalence={y.mean():.3f}")
ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
ax.grid(alpha=.2)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "precision_recall_curve.jpg", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.5), dpi=220)
for model, balanced in best.items():
    g = selected_group(model, balanced)
    yy, pp = g.y_true.to_numpy(), g.probability.to_numpy()
    fpr, tpr, _ = roc_curve(yy, pp)
    auc = roc_auc_score(yy, pp)
    color, style = line_styles[model]
    ax.plot(fpr, tpr, lw=1.8, color=color, ls=style,
            label=f"{model} (AUROC={auc:.3f})")
ax.plot([0, 1], [0, 1], color="0.65", ls=(0, (5, 3)), lw=1)
ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
ax.grid(alpha=.2)
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "roc_curve.jpg", bbox_inches="tight")
plt.close(fig)

plot_df = selected_folds.groupby(["model", "balanced"])[["recall", "f2"]].mean().reset_index()
order = ["LR", "DT", "RF", "NN"]
x = np.arange(len(order))
width = .34
fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=220)
for j, balanced in enumerate([False, True]):
    d = plot_df[plot_df.balanced == balanced].set_index("model").reindex(order)
    ax.bar(x + (j-.5)*width, d.f2, width,
           label="Balanced" if balanced else "Unbalanced",
           color="0.78" if balanced else "white", edgecolor="black",
           linewidth=.8, hatch="///" if balanced else "...")
ax.axhline(0.375, color="black", ls="--", lw=1.2, label="Flag-all F2")
ax.set_xticks(x, order)
ax.set(ylabel="$F_2$", ylim=(0, .7))
ax.grid(axis="y", alpha=.2)
ax.legend(fontsize=7, ncol=3, loc="upper center")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "f2_comparison.png", bbox_inches="tight")
plt.close(fig)

print(aggregate.to_string(index=False))
print(f"McNemar b={b}, c={c}, p={pvalue:.8g}")
