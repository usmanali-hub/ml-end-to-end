"""Generate ``docs/data_card.md`` from the data itself, so the numbers cannot drift from the file."""

import pandas as pd

from .data import CATEGORIES, FEATURES, ID_COLUMN, NUMERIC, TARGET
from .split import SEED, TEST_SIZE, split

COLUMN_DOCS = {
    "customerID": "Unique customer identifier. Not a feature.",
    "gender": "Female or Male as recorded by the company.",
    "SeniorCitizen": "1 if the customer is a senior citizen, else 0.",
    "Partner": "Whether the customer has a partner.",
    "Dependents": "Whether the customer has dependents.",
    "tenure": "Months the customer has been with the company.",
    "PhoneService": "Whether the customer has phone service.",
    "MultipleLines": "Multiple phone lines; 'No phone service' when there is no phone service.",
    "InternetService": "Internet provider type: DSL, Fiber optic or none.",
    "OnlineSecurity": "Online security add-on; 'No internet service' when there is no internet.",
    "OnlineBackup": "Online backup add-on; same convention.",
    "DeviceProtection": "Device protection add-on; same convention.",
    "TechSupport": "Tech support add-on; same convention.",
    "StreamingTV": "Streaming TV add-on; same convention.",
    "StreamingMovies": "Streaming movies add-on; same convention.",
    "Contract": "Contract term: month-to-month, one year or two year.",
    "PaperlessBilling": "Whether the customer uses paperless billing.",
    "PaymentMethod": "How the customer pays.",
    "MonthlyCharges": "Current monthly charge.",
    "TotalCharges": "Total charged over the customer's lifetime; blank in the raw file for brand-new customers.",
    TARGET: "Target. 1 if the customer left the company (raw column 'Churn' = Yes), else 0.",
}


def quality_facts(df: pd.DataFrame) -> dict[str, float | int]:
    """Numbers quoted in the data-quality section; each is asserted by a test."""
    blank = df[df["TotalCharges"].isna()]
    expected = df["MonthlyCharges"] * df["tenure"]
    gap = (df["TotalCharges"] - expected).abs().dropna()
    relative = (gap / expected[gap.index]).replace([float("inf")], float("nan")).dropna()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "churned": int(df[TARGET].sum()),
        "churn_rate": float(df[TARGET].mean()),
        "blank_total_charges": len(blank),
        "blank_all_tenure_zero": int((blank["tenure"] == 0).all()),
        "blank_churned": int(blank[TARGET].sum()),
        "tenure_zero_rows": int((df["tenure"] == 0).sum()),
        "no_internet_rows": int((df["InternetService"] == "No").sum()),
        "no_phone_rows": int((df["PhoneService"] == "No").sum()),
        "duplicate_feature_rows": int(df.duplicated(subset=FEATURES).sum()),
        "median_charge_gap": float(gap.median()),
        "max_charge_gap": float(gap.max()),
        "median_relative_gap": float(relative.median()),
    }


def _column_rows(df: pd.DataFrame) -> list[str]:
    rows = ["| Column | Type | Values | Missing | Meaning |", "|---|---|---|---|---|"]
    for col in [ID_COLUMN, *CATEGORIES, *NUMERIC, TARGET]:
        s = df[col]
        if col == ID_COLUMN:
            values, kind = f"{s.nunique():,} unique", "text"
        elif col in CATEGORIES:
            values, kind = ", ".join(CATEGORIES[col]), "category"
        elif col == "SeniorCitizen" or col == TARGET:
            values, kind = "0, 1", "binary"
        else:
            values, kind = f"{s.min():g} to {s.max():g}", "integer" if str(s.dtype).startswith("int") else "float"
        rows.append(f"| `{col}` | {kind} | {values} | {int(s.isna().sum())} | {COLUMN_DOCS[col]} |")
    return rows


def render(df: pd.DataFrame) -> str:
    q = quality_facts(df)
    train, test = split(df)
    lines = [
        "# Data card: Telco customer churn",
        "",
        "## Summary",
        "",
        "- **Source**: IBM's public \"Telco Customer Churn\" sample dataset; licence and provenance in [`data/NOTICE.md`](../data/NOTICE.md).",
        f"- **Size**: {q['rows']:,} customers, {q['columns']} columns after cleaning (one row per customer, `customerID` is unique).",
        f"- **Target**: `{TARGET}`. {q['churned']:,} of {q['rows']:,} customers churned ({q['churn_rate']:.1%}), so the classes are imbalanced.",
        f"- **Split**: stratified {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, seed {SEED}: {len(train):,} training rows and {len(test):,} test rows, both at the same churn rate. The test rows are reserved for the final model.",
        "",
        "## Intended use",
        "",
        "Learning and demonstrating an end-to-end classification workflow: exploration, preprocessing, model comparison, calibration and deployment. It is a public sample of a telecom company, not evidence about any real customer base.",
        "",
        "**Not suitable for**: decisions about real people. In particular `gender` and `SeniorCitizen` describe protected groups and should not drive retention, pricing or service decisions; later work checks model behaviour across them rather than assuming it is fair.",
        "",
        "## Columns",
        "",
        *_column_rows(df),
        "",
        "## Data quality notes",
        "",
        f"- **Blank `TotalCharges`**: {q['blank_total_charges']} rows. All have `tenure` = 0 ({'yes' if q['blank_all_tenure_zero'] else 'no'}) and {q['blank_churned']} of them churned: they are customers who joined but have not been billed yet. The raw file stores them as blank text, which makes the whole column load as strings; the loader converts them to NaN and leaves imputation to the model pipeline, fitted on training data only.",
        f"- **Redundant categories**: {q['no_internet_rows']:,} customers have no internet service, so six add-on columns say 'No internet service' for the same rows; {q['no_phone_rows']:,} have no phone service, which `MultipleLines` repeats as 'No phone service'. Seven columns therefore restate two facts.",
        f"- **`TotalCharges` is almost `MonthlyCharges` x `tenure`**: the median gap is ${q['median_charge_gap']:.2f} ({q['median_relative_gap']:.1%} of the expected amount) and the largest ${q['max_charge_gap']:.2f}. It carries little beyond tenure and price. The remaining gap is probably plan or price changes over time, which cannot be checked because the data has no dates.",
        f"- **Identical feature rows**: {q['duplicate_feature_rows']} customers share every feature value with another customer. With so many categorical columns this is plausible for different people, so they are kept.",
        "- **Types**: `SeniorCitizen` is 0/1 while every other yes/no column is text 'Yes'/'No'.",
        "",
        "## Limitations",
        "",
        "- **A snapshot, not a history.** There are no dates. `tenure` for a churned customer is how long they stayed, while for an active customer it is how long they have stayed so far, so tenure partly encodes the outcome. Treat results as descriptive; a deployed model would need features known *before* the decision point.",
        "- **One company, one moment.** Nothing here says how a model would behave on another region, product mix or year.",
        "- **No cost information.** Whether a missed churner or a wasted retention offer is worse is a business decision this dataset cannot make, so threshold choices stay explicit and are not tuned to a made-up cost.",
        "",
    ]
    return "\n".join(lines)
