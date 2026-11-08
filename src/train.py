"""End-to-end training: split -> tune (CV on train only) -> select -> threshold -> test.

Run from the project root:
    python -m src.train
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed when run as a script
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from src import plots
from src.config import (
    CV_FOLDS,
    FIGURES_DIR,
    MODEL_PATH,
    MODELS_DIR,
    PRIMARY_METRIC,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.data import load_clean_data, split_data
from src.evaluate import (
    best_f1_threshold,
    churners_captured_in_top,
    compute_metrics,
    compute_permutation_importance,
    get_confusion_matrix,
)
from src.features import build_pipeline
from src.models import get_model_candidates

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": "recall",
    # zero_division=0: the dummy baseline never predicts "churn", which is expected
    "precision": make_scorer(precision_score, zero_division=0),
}


@dataclass
class TrainingResults:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    searches: dict = field(default_factory=dict)
    cv_table: pd.DataFrame | None = None
    test_table: pd.DataFrame | None = None
    best_name: str = ""
    best_pipeline: object = None
    threshold: float = 0.5
    oof_proba: object = None
    test_proba: dict = field(default_factory=dict)
    final_metrics: dict = field(default_factory=dict)
    importance: pd.DataFrame | None = None


def get_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def tune_models(X_train, y_train, verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    """Grid-search every candidate with stratified K-fold CV on the TRAINING set only."""
    cv = get_cv()
    searches, rows = {}, []
    for name, (estimator, grid) in get_model_candidates().items():
        start = time.time()
        search = GridSearchCV(
            build_pipeline(estimator),
            param_grid=grid,
            scoring=SCORING,
            refit=PRIMARY_METRIC,
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        searches[name] = search
        i = search.best_index_
        row = {"model": name}
        for m in SCORING:
            row[f"cv_{m}_mean"] = search.cv_results_[f"mean_test_{m}"][i]
            row[f"cv_{m}_std"] = search.cv_results_[f"std_test_{m}"][i]
        row["best_params"] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        rows.append(row)
        if verbose:
            print(f"{name:<24} CV ROC-AUC = {row['cv_roc_auc_mean']:.4f} ± {row['cv_roc_auc_std']:.4f}"
                  f"   ({time.time() - start:.1f}s)  best params: {row['best_params']}")
    cv_table = pd.DataFrame(rows).set_index("model").sort_values("cv_roc_auc_mean", ascending=False)
    return searches, cv_table


def run_training(save: bool = True, verbose: bool = True) -> TrainingResults:
    plots.set_style()
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)
    res = TrainingResults(X_train, X_test, y_train, y_test)
    if verbose:
        print(f"Train: {X_train.shape}, churn rate {y_train.mean():.3f} | "
              f"Test: {X_test.shape}, churn rate {y_test.mean():.3f}\n")

    # 1) Tune all candidates with CV on the training set.
    res.searches, res.cv_table = tune_models(X_train, y_train, verbose)

    # 2) Select the final model by CV score (NOT by test score).
    res.best_name = res.cv_table.index[0]
    res.best_pipeline = res.searches[res.best_name].best_estimator_
    if verbose:
        print(f"\nSelected model (highest CV ROC-AUC): {res.best_name}")

    # 3) Choose the decision threshold on out-of-fold TRAINING predictions.
    res.oof_proba = cross_val_predict(
        res.best_pipeline, X_train, y_train, cv=get_cv(), method="predict_proba", n_jobs=-1
    )[:, 1]
    res.threshold = best_f1_threshold(y_train, res.oof_proba)
    if verbose:
        print(f"F1-optimal threshold from out-of-fold predictions: {res.threshold:.3f}")

    # 4) Evaluate every tuned model once on the held-out test set (for reporting only).
    rows = []
    for name, search in res.searches.items():
        proba = search.best_estimator_.predict_proba(X_test)[:, 1]
        res.test_proba[name] = proba
        m = compute_metrics(y_test, proba, threshold=0.5)
        m["model"] = name
        rows.append(m)
    res.test_table = (pd.DataFrame(rows).set_index("model")
                      .drop(columns="threshold").loc[res.cv_table.index])

    best_proba = res.test_proba[res.best_name]
    res.final_metrics = {
        "model": res.best_name,
        "best_params": res.cv_table.loc[res.best_name, "best_params"],
        "cv_roc_auc_mean": float(res.cv_table.loc[res.best_name, "cv_roc_auc_mean"]),
        "cv_roc_auc_std": float(res.cv_table.loc[res.best_name, "cv_roc_auc_std"]),
        "test_at_default_threshold": compute_metrics(y_test, best_proba, 0.5),
        "test_at_tuned_threshold": compute_metrics(y_test, best_proba, res.threshold),
        "test_churners_captured_top20pct": churners_captured_in_top(y_test, best_proba, 0.2),
        "test_churn_rate": float(y_test.mean()),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
    }

    # 5) Interpretability: permutation importance on the test set.
    res.importance = compute_permutation_importance(res.best_pipeline, X_test, y_test)

    if verbose:
        print("\nTest-set metrics of the selected model:")
        for k in ("test_at_default_threshold", "test_at_tuned_threshold"):
            m = res.final_metrics[k]
            print(f"  threshold={m['threshold']:.3f}: ROC-AUC {m['roc_auc']:.4f} | PR-AUC {m['pr_auc']:.4f} | "
                  f"Acc {m['accuracy']:.4f} | Precision {m['precision']:.4f} | "
                  f"Recall {m['recall']:.4f} | F1 {m['f1']:.4f}")
        print(f"  Top-20% riskiest customers contain "
              f"{res.final_metrics['test_churners_captured_top20pct']:.1%} of all churners")

    if save:
        save_artifacts(res)
    return res


def save_artifacts(res: TrainingResults) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"pipeline": res.best_pipeline, "threshold": res.threshold, "model_name": res.best_name},
        MODEL_PATH,
    )

    comparison = res.cv_table.drop(columns="best_params").join(
        res.test_table.add_prefix("test_")
    )
    comparison["best_params"] = res.cv_table["best_params"].astype(str)
    comparison.round(4).to_csv(REPORTS_DIR / "model_comparison.csv")
    res.importance.round(5).to_csv(REPORTS_DIR / "feature_importance.csv")
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(res.final_metrics, f, indent=2, default=float)

    # Evaluation figures
    figs = {
        "cv_model_comparison": plots.plot_cv_comparison(res.cv_table),
        "roc_pr_curves": plots.plot_roc_pr_curves(
            {n: (res.y_test, p) for n, p in res.test_proba.items()}, prevalence=res.y_test.mean()
        ),
        "threshold_tradeoff": plots.plot_threshold_tradeoff(res.y_train, res.oof_proba, res.threshold),
        "confusion_matrix_tuned_threshold": plots.plot_confusion_matrix(
            get_confusion_matrix(res.y_test, res.test_proba[res.best_name], res.threshold),
            f"{res.best_name} — test set (threshold {res.threshold:.2f})",
        ),
        "feature_importance": plots.plot_feature_importance(res.importance),
    }
    for name, fig in figs.items():
        plots.save(fig, name)
        plt.close(fig)
    print(f"\nSaved model to {MODEL_PATH.relative_to(MODEL_PATH.parents[1])} and reports to reports/")


if __name__ == "__main__":
    run_training()
