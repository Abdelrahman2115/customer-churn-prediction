"""Plotting helpers with one consistent, colorblind-safe style.

Colors: "Retained" is always blue, "Churned" is always orange, and model colors
follow a fixed order, so the same entity has the same color in every figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import FIGURES_DIR

# Validated categorical palette (fixed order) + neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RETAINED, CHURNED = SERIES[0], SERIES[1]
NEUTRAL = "#8a8985"
# Each model keeps the same color in every figure (validated as a set for color-vision deficiency).
MODEL_COLORS = {
    "Logistic Regression": "#1baf7a",
    "Random Forest": "#eda100",
    "Hist Gradient Boosting": "#6250d6",
}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE_SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#eef4fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def save(fig, name: str, out_dir: Path = FIGURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    return path


# --------------------------------------------------------------------------- EDA
def plot_target_distribution(y: pd.Series):
    counts = y.value_counts().reindex([0, 1])
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.barh(["Retained", "Churned"], counts.values, color=[RETAINED, CHURNED], height=0.55)
    for i, (c, p) in enumerate(zip(counts.values, pct.values)):
        ax.text(c + counts.max() * 0.01, i, f"{c:,}  ({p:.1f}%)", va="center", color=INK_2)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Customers")
    ax.set_title("Target distribution: about 1 in 4 customers churned")
    return fig


def plot_churn_rate_by(df: pd.DataFrame, columns: list[str], target: str = "Churn", ncols: int = 3):
    """Churn rate per category, one small panel per feature; dashed line = overall rate."""
    overall = df[target].mean()
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, columns):
        rates = df.groupby(col, observed=True)[target].mean().sort_values()
        ax.barh(rates.index.astype(str), rates.values, color=CHURNED, height=0.6)
        ax.axvline(overall, color=NEUTRAL, linestyle="--", linewidth=1)
        for i, v in enumerate(rates.values):
            ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9, color=INK_2)
        ax.set_xlim(0, max(0.75, rates.max() + 0.12))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", visible=False)
        ax.set_title(col, fontsize=11)
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    fig.suptitle(
        f"Churn rate by category (dashed line = overall rate {overall:.1%})",
        x=0.01, ha="left", fontweight="bold", fontsize=13,
    )
    fig.tight_layout()
    return fig


def plot_numeric_by_target(df: pd.DataFrame, columns: list[str], target: str = "Churn"):
    """Overlaid density histograms (step outlines) for retained vs churned customers."""
    fig, axes = plt.subplots(1, len(columns), figsize=(4.6 * len(columns), 3.2))
    for ax, col in zip(np.atleast_1d(axes), columns):
        bins = np.histogram_bin_edges(df[col].dropna(), bins=30)
        for label, color, name in [(0, RETAINED, "Retained"), (1, CHURNED, "Churned")]:
            ax.hist(df.loc[df[target] == label, col], bins=bins, density=True,
                    histtype="step", linewidth=2, color=color, label=name)
        ax.set_title(col)
        ax.set_ylabel("Density")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_correlation(df: pd.DataFrame, columns: list[str]):
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr.values, cmap=BLUE_SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 else INK, fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation between numeric features")
    return fig


# -------------------------------------------------------------------- Evaluation
def plot_cv_comparison(cv_table: pd.DataFrame, metric: str = "cv_roc_auc"):
    t = cv_table.sort_values(f"{metric}_mean")
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(t.index, t[f"{metric}_mean"], xerr=t[f"{metric}_std"], color=RETAINED,
            height=0.55, error_kw={"ecolor": INK_2, "capsize": 3, "linewidth": 1})
    for i, v in enumerate(t[f"{metric}_mean"]):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", color=INK_2)
    ax.set_xlim(0.4, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ROC-AUC (5-fold CV on training set, mean ± std)")
    ax.set_title("Model comparison: cross-validated ROC-AUC")
    return fig


def plot_roc_pr_curves(curves: dict, prevalence: float):
    """curves: {model_name: (y_true, y_proba)} evaluated on the same data.
    The dummy baseline is shown as the dashed reference lines instead of a curve."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, (y_true, proba) in curves.items():
        if name not in MODEL_COLORS:
            continue
        color = MODEL_COLORS[name]
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax1.plot(fpr, tpr, color=color, label=f"{name} (AUC {roc_auc_score(y_true, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ax2.plot(rec, prec, color=color, label=f"{name} (AP {average_precision_score(y_true, proba):.3f})")
    ax1.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", linewidth=1, label="Random guess (AUC 0.500)")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax2.axhline(prevalence, color=NEUTRAL, linestyle="--", linewidth=1,
                label=f"Random guess (AP {prevalence:.3f})")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (test set)", ylim=(0, 1.02))
    ax1.legend(loc="lower right", fontsize=9)
    ax2.legend(loc="lower left", bbox_to_anchor=(0, 0.3), fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap=BLUE_SEQ)
    labels = ["Retained", "Churned"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / total:.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.55 else INK, fontsize=11)
    ax.grid(False)
    ax.set_title(title)
    return fig


def plot_feature_importance(importances: pd.DataFrame, top_n: int = 15):
    t = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.2))
    ax.barh(t.index, t["importance_mean"], xerr=t["importance_std"], color=RETAINED, height=0.6,
            error_kw={"ecolor": INK_2, "capsize": 2, "linewidth": 1})
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled (test set)")
    ax.set_title("Permutation feature importance — final model")
    return fig


def plot_threshold_tradeoff(y_true, proba, chosen_threshold: float):
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(thr, prec[:-1], color=SERIES[0], label="Precision")
    ax.plot(thr, rec[:-1], color=SERIES[1], label="Recall")
    ax.plot(thr, f1, color=SERIES[2], label="F1")
    ax.axvline(chosen_threshold, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.text(chosen_threshold + 0.01, 0.05, f"chosen = {chosen_threshold:.2f}", color=INK_2)
    ax.set(xlabel="Decision threshold", ylabel="Score", ylim=(0, 1.02),
           title="Threshold trade-off (out-of-fold predictions, training set)")
    ax.legend(loc="center right")
    return fig
