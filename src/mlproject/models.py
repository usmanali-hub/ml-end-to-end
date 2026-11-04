"""Linear models for churn: a linear regression used as a classifier, and logistic regression under different penalties.

Every model is a pipeline whose first step is the recommended preprocessing recipe, so scaling, imputation and
encoding are fitted on training rows only. All comparisons use the same 5-fold stratified splits of the
*training* split that the baselines use; the test split has not been used.

Two choices are made here by cross-validation, never on the test rows:

* the regularisation strength ``C`` (smaller C = stronger penalty), for L2 and for L1
* whether to reweight the classes
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .baseline import MAX_ITER, SEED
from .data import CATEGORIES, NUMERIC, TARGET
from .metrics import evaluate
from .preprocess import RECOMMENDED, VARIANTS, Options, build_preprocessor, feature_names

C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)
COLUMNS = [*NUMERIC, *CATEGORIES]


@dataclass(frozen=True)
class Spec:
    """A model configuration. ``family`` is ``linear`` (least squares on the 0/1 outcome) or ``logistic``."""

    name: str
    family: str = "logistic"
    penalty: str = "l2"  # l2 or l1 (logistic only)
    C: float = 1.0
    class_weight: str | None = None


def build(spec: Spec, options: Options = VARIANTS[RECOMMENDED]) -> Pipeline:
    """An unfitted pipeline: preprocessing, then the estimator."""
    model: Any
    if spec.family == "linear":
        model = LinearRegression()
    elif spec.penalty == "l1":
        model = LogisticRegression(
            C=spec.C, l1_ratio=1.0, solver="liblinear", class_weight=spec.class_weight, max_iter=MAX_ITER,
            tol=1e-6, random_state=SEED,  # liblinear visits coordinates in random order; a tight tolerance makes the exact-zero set stable
        )
    else:
        model = LogisticRegression(C=spec.C, class_weight=spec.class_weight, max_iter=MAX_ITER)
    return Pipeline([("prep", build_preprocessor(options=options)), ("model", model)])


def scores(pipe: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Churn scores: probabilities for logistic regression, raw fitted values for the linear model."""
    if isinstance(pipe.named_steps["model"], LinearRegression):
        return np.asarray(pipe.predict(df[COLUMNS]), dtype=float)
    return np.asarray(pipe.predict_proba(df[COLUMNS])[:, 1], dtype=float)


def cross_validate(train: pd.DataFrame, specs: list[Spec], n_splits: int = 5) -> pd.DataFrame:
    """One row per (model, fold): the usual metrics plus the Brier score (mean squared error of the probabilities)."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows: list[dict[str, object]] = []
    for fold, (fit_idx, val_idx) in enumerate(folds.split(train, train[TARGET])):
        fit_df, val_df = train.iloc[fit_idx], train.iloc[val_idx]
        for spec in specs:
            pipe = build(spec).fit(fit_df[COLUMNS], fit_df[TARGET])
            s = scores(pipe, val_df)
            result = evaluate(val_df[TARGET], s, (s >= 0.5).astype(int))
            brier = float(brier_score_loss(val_df[TARGET], np.clip(s, 0.0, 1.0)))
            rows.append({"model": spec.name, "fold": fold, **result, "brier": brier})
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    metrics = ["pr_auc", "roc_auc", "recall", "precision", "brier"]
    grouped = folds.groupby("model")[metrics]
    out = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    return out.loc[order]


def sweep_specs(penalty: str) -> list[Spec]:
    return [Spec(name=f"{penalty}_C={c:g}", penalty=penalty, C=c) for c in C_GRID]


TOLERANCE = 0.001  # PR-AUC; about the fold-to-fold noise on the flat part of the sweep


def simplest_within(table: pd.DataFrame, tolerance: float = TOLERANCE) -> str:
    """The strongest penalty (smallest C) whose mean PR-AUC is within ``tolerance`` of the best.

    Settings that tie are not worth telling apart, and the more heavily penalised one is the safer pick.
    ``table`` must be ordered from the strongest penalty to the weakest.
    """
    mean = table["pr_auc_mean"]
    limit = float(mean.max()) - tolerance
    for name in table.index:
        if float(mean[name]) >= limit:
            return str(name)
    return str(mean.idxmax())


def nonzero_range(spec: Spec, train: pd.DataFrame, seeds: tuple[int, ...] = (0, 1, 2, 3)) -> tuple[int, int]:
    """(fewest, most) non-zero coefficients over several solver orderings, for a fitted L1 model.

    The one-hot columns are exactly redundant (each category's columns sum to one, and the no-internet flags repeat
    each other), so L1 can give a coefficient's weight to any of several identical columns. *How many* are non-zero
    therefore depends on the order the solver visits them, even though the predictions do not. Reporting a range is
    the honest summary.
    """
    counts = []
    for seed in seeds:
        pipe = build(spec)
        pipe.named_steps["model"].set_params(random_state=seed)
        pipe.fit(train[COLUMNS], train[TARGET])
        counts.append(int((pipe.named_steps["model"].coef_ != 0).sum()))
    return min(counts), max(counts)


def odds_ratios(spec: Spec, train: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """The largest coefficients of a fitted logistic model, as odds ratios.

    Numeric inputs are standardised, so their ratio is per one standard deviation; a one-hot input's ratio is
    for having that value versus not having it. Inputs that overlap (the add-on columns, and the no-internet
    flags that repeat each other) share credit, so treat single ratios as descriptive, not causal.
    """
    pipe = build(spec).fit(train[COLUMNS], train[TARGET])
    coef = pipe.named_steps["model"].coef_.ravel()
    names = [n.split("__", 1)[1] for n in feature_names(pipe.named_steps["prep"])]
    table = pd.DataFrame({"input": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
    order = table["coefficient"].abs().sort_values(ascending=False).index[:top]
    return table.loc[order].reset_index(drop=True)


def linear_out_of_range(train: pd.DataFrame) -> float:
    """Share of customers a least-squares 'probability' puts below 0 or above 1 (fitted and scored on training rows)."""
    pipe = build(Spec(name="linear", family="linear")).fit(train[COLUMNS], train[TARGET])
    s = scores(pipe, train)
    return float(((s < 0) | (s > 1)).mean())


@dataclass
class Analysis:
    sweep: pd.DataFrame  # one row per C: l2 and l1 PR-AUC and the number of non-zero L1 coefficients
    families: pd.DataFrame  # the model families side by side, with paired differences from the chosen model
    chosen: Spec
    chosen_l1: Spec
    odds: pd.DataFrame
    linear_out_of_range: float


def analyse(train: pd.DataFrame) -> Analysis:
    """Run every comparison in ``reports/linear_models.md`` on the training split."""
    l2, l1 = sweep_specs("l2"), sweep_specs("l1")
    folds = cross_validate(train, [*l2, *l1])
    summary = summarize(folds, [x.name for x in [*l2, *l1]])
    l2_table, l1_table = summary.loc[[x.name for x in l2]], summary.loc[[x.name for x in l1]]
    chosen = next(x for x in l2 if x.name == simplest_within(l2_table))
    chosen_l1 = next(x for x in l1 if x.name == simplest_within(l1_table))
    sweep = pd.DataFrame(
        {
            "C": C_GRID,
            "l2_pr_auc": l2_table["pr_auc_mean"].to_numpy(),
            "l1_pr_auc": l1_table["pr_auc_mean"].to_numpy(),
        }
    )
    ranges = [nonzero_range(x, train) for x in l1]
    sweep["l1_nonzero_min"], sweep["l1_nonzero_max"] = [r[0] for r in ranges], [r[1] for r in ranges]
    named = [
        Spec("linear_probability", family="linear"),
        Spec("logistic_default", C=1.0),
        replace_name(chosen, "logistic_l2_chosen"),
        replace_name(chosen_l1, "logistic_l1_chosen"),
        Spec("logistic_balanced", C=chosen.C, class_weight="balanced"),
    ]
    fam_folds = cross_validate(train, named)
    fam = summarize(fam_folds, [x.name for x in named])
    pr = fam_folds.pivot(index="fold", columns="model", values="pr_auc")
    paired = pr.sub(pr["logistic_l2_chosen"], axis=0)
    fam["delta_mean"], fam["delta_std"], fam["folds_better"] = paired.mean(), paired.std(ddof=1), (paired > 0).sum()
    return Analysis(sweep, fam, chosen, chosen_l1, odds_ratios(chosen, train), linear_out_of_range(train))


def replace_name(spec: Spec, name: str) -> Spec:
    return Spec(name=name, family=spec.family, penalty=spec.penalty, C=spec.C, class_weight=spec.class_weight)
