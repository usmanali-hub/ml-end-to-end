"""Load the raw Telco churn file, fix types, and validate it before anything else touches it.

Cleaning is deliberately minimal. Missing values are *kept* as NaN: filling them with a statistic
computed on the whole dataset would leak information from the test rows into training, so imputation
belongs inside a model pipeline that is fitted on training data only.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
TARGET = "churned"
ID_COLUMN = "customerID"

# Allowed values for every categorical column; used for validation and the data card.
CATEGORIES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
}
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURES = [c for c in [*CATEGORIES, *NUMERIC]]  # the 19 model inputs (no ID, no target)


class DataError(ValueError):
    """The data violates the schema the rest of the project relies on."""


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the CSV as-is: every column as text, nothing converted, nothing dropped."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Fix types only. ``TotalCharges`` holds blank strings for brand-new customers; those become NaN."""
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"]).astype(int)
    df["tenure"] = pd.to_numeric(df["tenure"]).astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"]).astype(float)
    total = df["TotalCharges"].str.strip()
    df["TotalCharges"] = pd.to_numeric(total.where(total != ""), errors="coerce").astype(float)
    df[TARGET] = df["Churn"].map({"No": 0, "Yes": 1})
    if df[TARGET].isna().any():
        raise DataError("Churn must be 'Yes' or 'No' on every row")
    df[TARGET] = df[TARGET].astype(int)
    return df.drop(columns=["Churn"])


def validate(df: pd.DataFrame) -> None:
    """Raise :class:`DataError` on the first schema violation; return quietly when the data is sound."""
    expected = {ID_COLUMN, TARGET, *FEATURES}
    missing = expected - set(df.columns)
    if missing:
        raise DataError(f"missing columns: {sorted(missing)}")
    if df[ID_COLUMN].duplicated().any():
        raise DataError("customerID must be unique")
    for col, allowed in CATEGORIES.items():
        bad = set(df[col].dropna().unique()) - set(allowed)
        if bad:
            raise DataError(f"{col} has unexpected values: {sorted(bad)}")
        if df[col].isna().any():
            raise DataError(f"{col} has missing values")
    if not set(df["SeniorCitizen"].unique()) <= {0, 1}:
        raise DataError("SeniorCitizen must be 0 or 1")
    if (df["tenure"] < 0).any():
        raise DataError("tenure cannot be negative")
    if (df["MonthlyCharges"] <= 0).any() or df["MonthlyCharges"].isna().any():
        raise DataError("MonthlyCharges must be present and positive")
    if (df["TotalCharges"].dropna() < 0).any():
        raise DataError("TotalCharges cannot be negative")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise DataError(f"{TARGET} must be 0 or 1")


def load(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load, clean and validate in one step; this is what the rest of the project calls."""
    df = clean(load_raw(path))
    validate(df)
    return df
