"""Preprocessing pipeline.

Everything that learns from data (imputation statistics, scaling parameters,
one-hot categories) lives inside a scikit-learn ColumnTransformer, which is
placed inside the model Pipeline. During cross-validation it is re-fit on each
training fold only, so no information from validation/test rows leaks in.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # safety net for unseen data
            ("scaler", StandardScaler()),  # needed for Logistic Regression; harmless for trees
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    """Preprocessing + classifier as one object (fit, tune, save and serve together)."""
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("clf", estimator)])
