"""Gradient boosting (LightGBM) for churn, tuned on the training split and compared with the earlier models.

A random forest grows many deep trees independently and averages them. Gradient boosting grows many *shallow* trees
one after another, each fitted to the mistakes of the ensemble so far, and adds them up with a small learning rate.
It has more knobs than a forest (tree size, learning rate, number of rounds) and, unlike a forest, it keeps
overfitting if you keep adding rounds, so the number of rounds has to be chosen.

All comparisons use the same 5-fold stratified splits of the training split as the earlier models. Choosing the
rounds and tree size on those folds and then reporting the same folds is optimistic, so ``nested`` repeats the whole
tuning procedure inside each outer fold to measure how optimistic. The test split has not been used.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import trees
from .baseline import SEED
from .data import TARGET
from .preprocess import build_preprocessor
from .trees import COLUMNS, TREE_OPTIONS, ForestSpec, cross_validate, folds_of, summarize

LEAVES = (4, 8, 16)
RATES = (0.02, 0.05, 0.1)
ROUNDS = (10, 25, 50, 100, 150, 200, 300, 400, 600)
MIN_CHILD = 40  # minimum rows in a leaf, as for the forest
ROW_SAMPLE = 0.8
COLUMN_SAMPLE = 0.8
INNER_FOLDS = 3
TOLERANCE = 0.001


@dataclass(frozen=True)
class BoostSpec:
    leaves: int
    rate: float
    rounds: int

    @property
    def label(self) -> str:
        return f"leaves={self.leaves}, rate={self.rate}, rounds={self.rounds}"


def booster(leaves: int, rate: float, rounds: int) -> LGBMClassifier:
    """A LightGBM classifier with fixed regularisation; single-threaded and deterministic so reports reproduce."""
    return LGBMClassifier(
        n_estimators=rounds,
        learning_rate=rate,
        num_leaves=leaves,
        min_child_samples=MIN_CHILD,
        subsample=ROW_SAMPLE,
        subsample_freq=1,
        colsample_bytree=COLUMN_SAMPLE,
        random_state=SEED,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )


def library_default() -> LGBMClassifier:
    """LightGBM with its own defaults (31 leaves, rate 0.1, 100 rounds, no row or column sampling)."""
    return LGBMClassifier(random_state=SEED, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)


def make_boosted(spec: BoostSpec) -> Pipeline:
    return Pipeline(
        [("prep", build_preprocessor(options=TREE_OPTIONS)), ("model", booster(spec.leaves, spec.rate, spec.rounds))]
    )


def make_default() -> Pipeline:
    return Pipeline([("prep", build_preprocessor(options=TREE_OPTIONS)), ("model", library_default())])


def make_sklearn_hist(spec: BoostSpec) -> Pipeline:
    """scikit-learn's own histogram boosting with the same tree size, rate and rounds, as a cross-check."""
    model = HistGradientBoostingClassifier(
        max_leaf_nodes=spec.leaves,
        learning_rate=spec.rate,
        max_iter=spec.rounds,
        min_samples_leaf=MIN_CHILD,
        early_stopping=False,
        random_state=SEED,
    )
    return Pipeline([("prep", build_preprocessor(options=TREE_OPTIONS)), ("model", model)])


def make_xgboost(spec: BoostSpec) -> Pipeline:
    """XGBoost with the closest equivalent settings (leaf-wise growth, same leaves, rate, rounds and sampling).

    XGBoost limits leaves by the sum of second derivatives (``min_child_weight``), not by a row count, so its
    regularisation is similar but not identical to LightGBM's ``min_child_samples``.
    """
    model = XGBClassifier(
        n_estimators=spec.rounds,
        learning_rate=spec.rate,
        tree_method="hist",
        grow_policy="lossguide",
        max_leaves=spec.leaves,
        max_depth=0,
        subsample=ROW_SAMPLE,
        colsample_bytree=COLUMN_SAMPLE,
        random_state=SEED,
        n_jobs=1,
    )
    return Pipeline([("prep", build_preprocessor(options=TREE_OPTIONS)), ("model", model)])


def fold_curves(train: pd.DataFrame, leaves: int, rate: float, folds: list[tuple[np.ndarray, np.ndarray]] | None = None) -> pd.DataFrame:
    """Per fold and per checkpoint: PR-AUC on held-out rows and on the rows the booster was fitted to.

    One booster with the largest round count is fitted per fold and read at each smaller count, so the curve costs
    one fit, not one per checkpoint.
    """
    rows: list[dict[str, float | int]] = []
    for fold, (fit_idx, val_idx) in enumerate(folds if folds is not None else folds_of(train)):
        fit_df, val_df = train.iloc[fit_idx], train.iloc[val_idx]
        prep = build_preprocessor(options=TREE_OPTIONS).fit(fit_df[COLUMNS])
        x_fit, x_val = prep.transform(fit_df[COLUMNS]), prep.transform(val_df[COLUMNS])
        model = booster(leaves, rate, max(ROUNDS)).fit(x_fit, fit_df[TARGET])
        for k in ROUNDS:
            val_s = np.asarray(model.predict_proba(x_val, num_iteration=k))[:, 1]
            fit_s = np.asarray(model.predict_proba(x_fit, num_iteration=k))[:, 1]
            rows.append(
                {
                    "fold": fold,
                    "rounds": k,
                    "pr_auc": float(average_precision_score(val_df[TARGET], val_s)),
                    "fit_pr_auc": float(average_precision_score(fit_df[TARGET], fit_s)),
                }
            )
    return pd.DataFrame(rows)


def grid(train: pd.DataFrame) -> pd.DataFrame:
    """Validation PR-AUC for every tree size, learning rate and round count."""
    parts = []
    for leaves in LEAVES:
        for rate in RATES:
            curve = fold_curves(train, leaves, rate)
            g = curve.groupby("rounds")
            part = pd.DataFrame(
                {
                    "cv_pr_auc": g["pr_auc"].mean(),
                    "cv_std": g["pr_auc"].std(ddof=1),
                    "fit_pr_auc": g["fit_pr_auc"].mean(),
                }
            ).reset_index()
            parts.append(part.assign(leaves=leaves, rate=rate))
    out = pd.concat(parts, ignore_index=True)
    return out[["leaves", "rate", "rounds", "fit_pr_auc", "cv_pr_auc", "cv_std"]]


def choose(table: pd.DataFrame, tolerance: float = TOLERANCE) -> BoostSpec:
    """Within ``tolerance`` of the best score, the simplest booster: fewest leaves, then fewest rounds, then the larger rate."""
    limit = float(table["cv_pr_auc"].max()) - tolerance
    ok = table[table["cv_pr_auc"] >= limit].sort_values(["leaves", "rounds", "rate"], ascending=[True, True, False])
    best = ok.iloc[0]
    return BoostSpec(int(best["leaves"]), float(best["rate"]), int(best["rounds"]))


def best_per_setting(table: pd.DataFrame) -> pd.DataFrame:
    """For each (leaves, rate), the round count with the best validation score."""
    idx = table.groupby(["leaves", "rate"])["cv_pr_auc"].idxmax()
    return table.loc[idx].reset_index(drop=True)


def curve_for(table: pd.DataFrame, spec: BoostSpec) -> pd.DataFrame:
    """The round-by-round curve for the chosen tree size and learning rate."""
    sel = table[(table["leaves"] == spec.leaves) & (table["rate"] == spec.rate)]
    return sel[["rounds", "fit_pr_auc", "cv_pr_auc", "cv_std"]].reset_index(drop=True)


def nested(train: pd.DataFrame) -> pd.DataFrame:
    """Repeat the whole tuning procedure inside each outer fold, then score the outer fold once.

    For every outer fold the booster is tuned (tree size, rate and rounds, same grid and rule as ``grid`` and
    ``choose``) using only that fold's fitting rows, with an inner split. The tuned booster is then scored on the
    outer fold's held-out rows, which played no part in the choice. The mean over outer folds estimates how well the
    *procedure* does, free of the optimism from picking the best cell on the same folds that are reported.
    """
    rows: list[dict[str, float | int]] = []
    for fold, (fit_idx, val_idx) in enumerate(folds_of(train)):
        fit_df, val_df = train.iloc[fit_idx].reset_index(drop=True), train.iloc[val_idx]
        inner = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED).split(fit_df, fit_df[TARGET]))
        table = grid_on(fit_df, inner)
        spec = choose(table)
        pipe = make_boosted(spec).fit(fit_df[COLUMNS], fit_df[TARGET])
        s = np.asarray(pipe.predict_proba(val_df[COLUMNS]))[:, 1]
        rows.append(
            {
                "fold": fold,
                "leaves": spec.leaves,
                "rate": spec.rate,
                "rounds": spec.rounds,
                "pr_auc": float(average_precision_score(val_df[TARGET], s)),
            }
        )
    return pd.DataFrame(rows)


def grid_on(frame: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """``grid`` on an arbitrary frame and folds (used for the inner loop of ``nested``)."""
    parts = []
    for leaves in LEAVES:
        for rate in RATES:
            curve = fold_curves(frame, leaves, rate, folds)
            g = curve.groupby("rounds")
            part = pd.DataFrame({"cv_pr_auc": g["pr_auc"].mean()}).reset_index().assign(leaves=leaves, rate=rate)
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


@dataclass
class BoostAnalysis:
    table: pd.DataFrame
    chosen: BoostSpec
    best: pd.DataFrame
    curve: pd.DataFrame
    families: pd.DataFrame
    nested: pd.DataFrame


def analyse(train: pd.DataFrame, logistic: Callable[[], Pipeline], forest: ForestSpec) -> BoostAnalysis:
    """Everything in ``reports/gradient_boosting.md``. ``logistic`` and ``forest`` are the earlier chosen models."""
    table = grid(train)
    spec = choose(table)
    named: dict[str, Callable[[], Pipeline]] = {
        "logistic_l2_chosen": logistic,
        "random_forest_chosen": partial(trees.make_forest, ForestSpec(forest.min_samples_leaf, forest.max_features, 400)),
        "lightgbm_default": make_default,
        "lightgbm_chosen": partial(make_boosted, spec),
        "sklearn_hist_same_settings": partial(make_sklearn_hist, spec),
        "xgboost_same_settings": partial(make_xgboost, spec),
    }
    folds = cross_validate(train, named)
    fam = summarize(folds, list(named))
    nest = nested(train)
    pr = folds.pivot(index="fold", columns="model", values="pr_auc")
    pr["lightgbm_nested_tuning"] = nest.set_index("fold")["pr_auc"]
    paired = pr.sub(pr["logistic_l2_chosen"], axis=0)
    fam = fam.reindex([*named, "lightgbm_nested_tuning"])
    fam.loc["lightgbm_nested_tuning", "pr_auc_mean"] = float(nest["pr_auc"].mean())
    fam.loc["lightgbm_nested_tuning", "pr_auc_std"] = float(nest["pr_auc"].std(ddof=1))
    fam["delta_mean"], fam["delta_std"], fam["folds_better"] = paired.mean(), paired.std(ddof=1), (paired > 0).sum()
    return BoostAnalysis(table, spec, best_per_setting(table), curve_for(table, spec), fam, nest)
