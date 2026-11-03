"""Baselines: the numbers any real model has to beat before it is worth keeping.

Five of them, from naive to a plain linear model:

* ``majority``       never predicts churn; carries no ranking information at all
* ``random_prior``   guesses at the training churn rate; a coin weighted like the data
* ``month_to_month`` flags customers on a month-to-month contract; scores by each contract's churn rate
* ``short_tenure``   flags customers in their first year; scores by 1 / (1 + tenure)
* ``logistic_reference`` a standard logistic regression on every feature (imputation, scaling and
                     one-hot encoding are fitted inside each fold, never on the validation rows)

They are compared with 5-fold stratified cross-validation on the *training* split only. The test split
is left alone until the final model, so it cannot influence any choice made along the way.
"""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .data import CATEGORIES, NUMERIC, TARGET
from .metrics import METRIC_NAMES, evaluate
from .preprocess import DEFAULT_OPTIONS, VARIANTS, Options, build_preprocessor

SEED = 42
MAX_ITER = 2000


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


@dataclass
class LogisticReference:
    """Logistic regression with its preprocessing inside the model, so every fold fits it on training rows only.

    ``drop`` removes input columns, which is how the tenure ablation is run. Hard predictions use 0.5.
    """

    name: str = "logistic_reference"
    drop: tuple[str, ...] = ()
    threshold: float = 0.5
    options: Options = DEFAULT_OPTIONS
    _pipeline: Pipeline | None = None

    def _columns(self) -> tuple[list[str], list[str]]:
        numeric = [c for c in NUMERIC if c not in self.drop]
        categorical = [c for c in CATEGORIES if c not in self.drop]
        return numeric, categorical

    def fit(self, df: pd.DataFrame) -> "LogisticReference":
        numeric, categorical = self._columns()
        prep = build_preprocessor(numeric, categorical, self.options)
        self._pipeline = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=MAX_ITER))])
        self._pipeline.fit(df[numeric + categorical], df[TARGET])
        return self

    def scores(self, df: pd.DataFrame) -> np.ndarray:
        assert self._pipeline is not None, "call fit first"
        numeric, categorical = self._columns()
        return np.asarray(self._pipeline.predict_proba(df[numeric + categorical])[:, 1], dtype=float)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.scores(df) >= self.threshold).astype(int)


class Model(Protocol):
    name: str

    def fit(self, df: pd.DataFrame) -> "Model": ...
    def scores(self, df: pd.DataFrame) -> np.ndarray: ...
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


def make_baselines() -> list[Model]:
    return [Majority(), RandomPrior(), MonthToMonth(), ShortTenure(), LogisticReference()]


def cross_validate(
    train: pd.DataFrame,
    n_splits: int = 5,
    seed: int = SEED,
    make_models: Callable[[], Sequence[Model]] = make_baselines,
) -> pd.DataFrame:
    """Per-fold metrics for every model: one row per (model, fold)."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []
    for fold, (fit_idx, val_idx) in enumerate(folds.split(train, train[TARGET])):
        fit_df, val_df = train.iloc[fit_idx], train.iloc[val_idx]
        for model in make_models():
            model.fit(fit_df)
            result = evaluate(val_df[TARGET], model.scores(val_df), model.predict(val_df))
            rows.append({"baseline": model.name, "fold": fold, **result})
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame, order: list[str] | None = None) -> pd.DataFrame:
    """Mean and standard deviation over folds, one row per model, in the order they were defined."""
    order = order or [m.name for m in make_baselines()]
    grouped = folds.groupby("baseline")[METRIC_NAMES]
    mean, std = grouped.mean(), grouped.std(ddof=1)
    out = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    return out.loc[order]


ABLATIONS: dict[str, tuple[str, ...]] = {
    "all_features": (),
    "without_tenure": ("tenure",),
    "without_tenure_and_total_charges": ("tenure", "TotalCharges"),
}


def tenure_ablation(train: pd.DataFrame) -> pd.DataFrame:
    """The same logistic regression with the time-related columns removed, to show how much it leans on them."""

    def make() -> list[Model]:
        return [LogisticReference(name=name, drop=drop) for name, drop in ABLATIONS.items()]

    return summarize(cross_validate(train, make_models=make), order=list(ABLATIONS))


def compare_variants(train: pd.DataFrame) -> pd.DataFrame:
    """The same logistic regression under each preprocessing recipe in ``preprocess.VARIANTS``.

    Besides the usual summary, each recipe is compared with the reference *fold by fold* (every recipe sees
    identical folds, so the paired difference is much less noisy than comparing two averages), and the model
    is refitted on the whole training split to record whether the solver actually converged.
    """

    def make() -> list[Model]:
        return [LogisticReference(name=name, options=opts) for name, opts in VARIANTS.items()]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # convergence is recorded explicitly below
        folds = cross_validate(train, make_models=make)
        out = summarize(folds, order=list(VARIANTS))
        pr = folds.pivot(index="fold", columns="baseline", values="pr_auc")
        paired = pr.sub(pr["reference"], axis=0)
        out["delta_mean"] = paired.mean()
        out["delta_std"] = paired.std(ddof=1)
        out["folds_better"] = (paired > 0).sum()
        converged = {}
        for name, opts in VARIANTS.items():
            fitted = LogisticReference(name=name, options=opts).fit(train)
            assert fitted._pipeline is not None
            converged[name] = bool(fitted._pipeline.named_steps["model"].n_iter_[0] < MAX_ITER)
        out["converged"] = pd.Series(converged)
    return out
