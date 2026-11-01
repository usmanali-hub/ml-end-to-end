"""Baselines: the numbers any real model has to beat before it is worth keeping.

Four of them, from naive to a business rule someone could write on a whiteboard:

* ``majority``       never predicts churn; carries no ranking information at all
* ``random_prior``   guesses at the training churn rate; a coin weighted like the data
* ``month_to_month`` flags customers on a month-to-month contract; scores by each contract's churn rate
* ``short_tenure``   flags customers in their first year; scores by 1 / (1 + tenure)

They are compared with 5-fold stratified cross-validation on the *training* split only. The test split
is left alone until the final model, so it cannot influence any choice made along the way.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .data import TARGET
from .metrics import METRIC_NAMES, evaluate

SEED = 42


@dataclass
class Majority:
    name: str = "majority"
    rate: float = 0.0

    def fit(self, df: pd.DataFrame) -> "Majority":
        self.rate = float(df[TARGET].mean())
        return self

    def scores(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.rate)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=int)


@dataclass
class RandomPrior:
    name: str = "random_prior"
    seed: int = SEED
    rate: float = 0.0
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(SEED))

    def fit(self, df: pd.DataFrame) -> "RandomPrior":
        self.rate = float(df[TARGET].mean())
        self._rng = np.random.default_rng(self.seed)
        return self

    def scores(self, df: pd.DataFrame) -> np.ndarray:
        return self._rng.random(len(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self._rng.random(len(df)) < self.rate).astype(int)


@dataclass
class MonthToMonth:
    name: str = "month_to_month"
    rates: dict[str, float] = field(default_factory=dict)
    fallback: float = 0.0

    def fit(self, df: pd.DataFrame) -> "MonthToMonth":
        self.rates = {str(k): float(v) for k, v in df.groupby("Contract")[TARGET].mean().items()}
        self.fallback = float(df[TARGET].mean())
        return self

    def scores(self, df: pd.DataFrame) -> np.ndarray:
        return df["Contract"].map(self.rates).fillna(self.fallback).to_numpy(dtype=float)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (df["Contract"] == "Month-to-month").to_numpy(dtype=int)


@dataclass
class ShortTenure:
    name: str = "short_tenure"
    months: int = 12

    def fit(self, df: pd.DataFrame) -> "ShortTenure":
        return self

    def scores(self, df: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + df["tenure"].to_numpy(dtype=float))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (df["tenure"] <= self.months).to_numpy(dtype=int)


def make_baselines() -> list[Majority | RandomPrior | MonthToMonth | ShortTenure]:
    return [Majority(), RandomPrior(), MonthToMonth(), ShortTenure()]


def cross_validate(train: pd.DataFrame, n_splits: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Per-fold metrics for every baseline: one row per (baseline, fold)."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []
    for fold, (fit_idx, val_idx) in enumerate(folds.split(train, train[TARGET])):
        fit_df, val_df = train.iloc[fit_idx], train.iloc[val_idx]
        for model in make_baselines():
            model.fit(fit_df)
            result = evaluate(val_df[TARGET], model.scores(val_df), model.predict(val_df))
            rows.append({"baseline": model.name, "fold": fold, **result})
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over folds, one row per baseline, in the order they were defined."""
    order = [m.name for m in make_baselines()]
    grouped = folds.groupby("baseline")[METRIC_NAMES]
    mean, std = grouped.mean(), grouped.std(ddof=1)
    out = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    return out.loc[order]
