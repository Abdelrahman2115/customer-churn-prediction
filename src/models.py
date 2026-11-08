"""Candidate models and their hyper-parameter search grids."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE


def get_model_candidates(random_state: int = RANDOM_STATE) -> dict:
    """Return {name: (estimator, param_grid)}.

    Class imbalance (~26.5% churners) is handled with class weights rather than
    resampling: simple, no synthetic rows, and nothing to leak across folds.
    Parameter names are prefixed with ``clf__`` because each estimator sits in
    the ``clf`` step of the pipeline.
    """
    return {
        "Dummy (baseline)": (
            DummyClassifier(strategy="prior"),
            {},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,  # parallelism is handled by GridSearchCV
            ),
            {
                "clf__max_depth": [4, 6, 8, 12],
                "clf__min_samples_leaf": [5, 10, 20],
                "clf__max_features": ["sqrt", 0.5],
            },
        ),
        "Hist Gradient Boosting": (
            HistGradientBoostingClassifier(
                class_weight="balanced",
                max_iter=200,
                early_stopping=False,
                random_state=random_state,
            ),
            {
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_leaf_nodes": [4, 7, 15],
                "clf__min_samples_leaf": [20, 50],
                "clf__l2_regularization": [1.0, 5.0],
            },
        ),
    }
