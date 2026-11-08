"""Loading, cleaning and splitting the Telco churn data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ALL_FEATURES,
    DATA_PATH,
    ID_COL,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def load_raw(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, row-wise cleaning (no statistics learned -> no leakage).

    * TotalCharges is stored as text; 11 rows contain a blank string.
      All of them have tenure == 0 (brand-new customers who have not been
      billed yet), so the correct value is 0, not a dataset-wide average.
    * Churn "Yes"/"No" -> 1/0 (only if the column is present, so the same
      function works for new, unlabeled customers at prediction time).
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_customer = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customer, "TotalCharges"] = 0.0

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target. customerID is never a feature."""
    return df[ALL_FEATURES].copy(), df[TARGET].copy()


def split_data(df: pd.DataFrame):
    """Stratified train/test split. The test set is touched only once, at the end."""
    X, y = get_X_y(df)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def load_clean_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


__all__ = ["load_raw", "clean", "get_X_y", "split_data", "load_clean_data", "ID_COL"]
