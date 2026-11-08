# Data

`raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` — IBM Telco Customer Churn sample dataset
(also available on Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

- 7,043 rows, one per customer; 21 columns.
- Target: `Churn` (Yes/No) — whether the customer left within the last month.
- Known quirk: `TotalCharges` is stored as text and is blank for 11 customers with `tenure == 0`;
  `src/data.py::clean` converts it to numeric and sets those values to 0.

The raw file is never modified; all cleaning happens in code.
