"""Fast sanity checks for data cleaning and the preprocessing pipeline.

Run from the project root:  python -m pytest -q
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import ALL_FEATURES, ID_COL, TARGET
from src.data import clean, load_raw, split_data
from src.features import build_pipeline, build_preprocessor


def test_clean_fixes_total_charges_and_target():
    df = clean(load_raw())
    assert df["TotalCharges"].dtype.kind == "f"
    assert df["TotalCharges"].isna().sum() == 0
    assert set(df[TARGET].unique()) == {0, 1}


def test_split_is_stratified_and_disjoint():
    X_train, X_test, y_train, y_test = split_data(clean(load_raw()))
    assert ID_COL not in X_train.columns and TARGET not in X_train.columns
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_preprocessor_outputs_finite_matrix_and_handles_unseen_category():
    X_train, X_test, *_ = split_data(clean(load_raw()))
    pre = build_preprocessor().fit(X_train)
    Xt = pre.transform(X_test)
    assert np.isfinite(Xt).all()

    weird = X_test.head(2).copy()
    weird["PaymentMethod"] = "Crypto"  # category never seen in training
    weird.loc[weird.index[0], "TotalCharges"] = np.nan
    assert np.isfinite(pre.transform(weird)).all()


def test_pipeline_fits_and_predicts_probabilities():
    X_train, X_test, y_train, _ = split_data(clean(load_raw()))
    pipe = build_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    proba = pipe.predict_proba(X_test[ALL_FEATURES])[:, 1]
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_clean_works_without_target_column():
    raw = load_raw().drop(columns=[TARGET]).head(5)
    out = clean(raw)
    assert TARGET not in out.columns
    assert isinstance(out, pd.DataFrame)
