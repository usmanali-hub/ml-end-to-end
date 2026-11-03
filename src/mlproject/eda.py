"""Exploratory analysis helpers. Everything the EDA notebook quotes is computed here, so it can be tested.

Only the *training* split should ever be passed in: looking at the test rows while exploring would let
what is seen there steer feature and model choices, which defeats holding them out.
"""

import math
from collections.abc import Sequence

import pandas as pd
from scipy.stats import chi2_contingency

from .data import CATEGORIES, FEATURES, NUMERIC, TARGET


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; behaves sensibly for small n and rates near 0 or 1."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def rate_table(df: pd.DataFrame, by: str, bins: Sequence[float] | None = None, labels: Sequence[str] | None = None) -> pd.DataFrame:
    """Churn rate per group with a Wilson interval. ``bins`` cuts a numeric column first (right-closed)."""
    key = pd.cut(df[by], bins=list(bins), labels=list(labels) if labels else None) if bins is not None else df[by]
    grouped = pd.DataFrame(df.groupby(key, observed=True)[TARGET].agg(n="size", churned="sum"))
    grouped["rate"] = grouped["churned"] / grouped["n"]
    bounds = [wilson_interval(int(k), int(n)) for k, n in zip(grouped["churned"], grouped["n"], strict=True)]
    grouped["lo"] = [b[0] for b in bounds]
    grouped["hi"] = [b[1] for b in bounds]
    grouped.index.name = by
    return grouped


def stratified_rates(df: pd.DataFrame, rows: str, cols: str, min_n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Churn rate and group size for every (rows, cols) cell; cells under ``min_n`` customers are blanked."""
    rate = df.pivot_table(index=rows, columns=cols, values=TARGET, aggfunc="mean", observed=True)
    size = df.pivot_table(index=rows, columns=cols, values=TARGET, aggfunc="size", observed=True)
    return rate.where(size >= min_n), size


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Association between two categorical variables, 0 (none) to 1 (perfect), from the chi-square statistic."""
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = int(table.to_numpy().sum())
    return float(math.sqrt(chi2 / (n * (min(table.shape) - 1))))


def association_ranking(df: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Every feature's association with churn as Cramér's V. Numeric columns are cut into quantile bins first,
    so all 19 features sit on one scale. A screening tool, not a model: it ignores interactions."""
    rows = []
    for col in FEATURES:
        s = df[col]
        if col in NUMERIC and s.nunique() > 2:
            s = pd.qcut(s, q=quantiles, duplicates="drop")
        rows.append({"feature": col, "cramers_v": cramers_v(s.astype(str), df[TARGET])})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    return pd.DataFrame({"missing": counts[counts > 0]})


TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]
CHARGE_BINS = [0, 35, 70, 90, 120]
CHARGE_LABELS = ["<=35", "36-70", "71-90", "91+"]

__all__ = [
    "CATEGORIES",
    "CHARGE_BINS",
    "CHARGE_LABELS",
    "TENURE_BINS",
    "TENURE_LABELS",
    "association_ranking",
    "cramers_v",
    "missing_summary",
    "rate_table",
    "stratified_rates",
    "wilson_interval",
]
