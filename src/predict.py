"""Score new customers with the saved model.

Usage (from the project root):
    python -m src.predict --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
                          --output reports/predictions.csv
The input must have the same columns as the raw dataset; the Churn column is optional.
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src.config import ALL_FEATURES, ID_COL, MODEL_PATH
from src.data import clean


def predict(df_raw: pd.DataFrame, model_path=MODEL_PATH) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    df = clean(df_raw)
    proba = bundle["pipeline"].predict_proba(df[ALL_FEATURES])[:, 1]
    out = pd.DataFrame({
        "churn_probability": proba.round(4),
        "churn_prediction": (proba >= bundle["threshold"]).astype(int),
    })
    if ID_COL in df.columns:
        out.insert(0, ID_COL, df[ID_COL].values)
    return out.sort_values("churn_probability", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict customer churn.")
    parser.add_argument("--input", required=True, help="CSV with raw customer records")
    parser.add_argument("--output", default="reports/predictions.csv")
    args = parser.parse_args()

    preds = predict(pd.read_csv(args.input))
    preds.to_csv(args.output, index=False)
    print(f"Scored {len(preds):,} customers -> {args.output}")
    print(f"Flagged as likely churners: {preds['churn_prediction'].sum():,}")
    print(preds.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
