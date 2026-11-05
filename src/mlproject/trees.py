"""Decision trees and random forests for churn, tuned on the training split and compared with the logistic model.

A single tree asks a sequence of yes/no questions about the inputs; it is easy to read but memorises noise when
allowed to grow. A random forest averages many trees, each grown on a bootstrap sample and restricted to a random
subset of inputs at every split, which cancels much of that noise. Both are invariant to scaling, so the
preprocessing here only imputes and one-hot encodes.

Every comparison uses the same 5-fold stratified splits of the training split as the baselines and the linear models.
The test split has not been used.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .baseline import SEED
from .data import CATEGORIES, NUMERIC, TARGET
from .metrics import evaluate
from .preprocess import Options, build_preprocessor, feature_names

COLUMNS = [*NUMERIC, *CATEGORIES]
TREE_OPTIONS = Options(scale=False)  # trees do not care about units
DEPTHS: tuple[int | None, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 15, None)
LEAF_SIZES = (5, 10, 20, 40)
MAX_FEATURES: tuple[str | float, ...] = ("sqrt", 0.3, 0.5)
FOREST_SIZES = (10, 25, 50, 100, 200, 400)
GRID_TREES = 150
N_JOBS = 2
TOLERANCE = 0.001


@dataclass(frozen=True)
class ForestSpec:
    min_samples_leaf: int = 1
    max_features: str | float = "sqrt"
    n_estimators: int = GRID_TREES

    @property
    def label(self) -> str:
        return f"leaf={self.min_samples_leaf}, max_features={self.max_features}"


def make_tree(depth: int | None) -> Pipeline:
    return Pipeline(
        [("prep", build_preprocessor(options=TREE_OPTIONS)), ("model", DecisionTreeClassifier(max_depth=depth, random_state=SEED))]
    )


def make_forest(spec: ForestSpec, columns: list[str] | None = None) -> Pipeline:
    numeric = [c for c in NUMERIC if columns is None or c in columns]
    prep = build_preprocessor(numeric + (["random_noise"] if columns and "random_noise" in columns else []), list(CATEGORIES), TREE_OPTIONS)
    forest = RandomForestClassifier(
        n_estimators=spec.n_estimators,
        min_samples_leaf=spec.min_samples_leaf,
        max_features=spec.max_features,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    return Pipeline([("prep", prep), ("model", forest)])


def folds_of(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, train[TARGET]))


def cross_validate(train: pd.DataFrame, builders: Mapping[str, Callable[[], Pipeline]]) -> pd.DataFrame:
    """One row per (model, fold): validation metrics, the Brier score, and the PR-AUC on the fitting rows."""
    rows: list[dict[str, object]] = []
    for fold, (fit_idx, val_idx) in enumerate(folds_of(train)):
        fit_df, val_df = train.iloc[fit_idx], train.iloc[val_idx]
        for name, make in builders.items():
            pipe = make().fit(fit_df[COLUMNS], fit_df[TARGET])
            s = np.asarray(pipe.predict_proba(val_df[COLUMNS])[:, 1], dtype=float)
            result = evaluate(val_df[TARGET].to_numpy(), s, (s >= 0.5).astype(int))
            fit_scores = np.asarray(pipe.predict_proba(fit_df[COLUMNS])[:, 1], dtype=float)
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    **result,
                    "brier": float(brier_score_loss(val_df[TARGET], s)),
                    "fit_pr_auc": float(average_precision_score(fit_df[TARGET], fit_scores)),
                }
            )
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    metrics = ["pr_auc", "roc_auc", "recall", "precision", "brier", "fit_pr_auc"]
    grouped = folds.groupby("model")[metrics]
    return grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std")).loc[order]


def depth_label(depth: int | None) -> str:
    return "none" if depth is None else str(depth)


def tree_sweep(train: pd.DataFrame) -> pd.DataFrame:
    """Single tree at each maximum depth: fitting-rows score versus validation score (the gap is overfitting)."""
    order = [depth_label(d) for d in DEPTHS]
    folds = cross_validate(train, {depth_label(d): partial(make_tree, d) for d in DEPTHS})
    table = summarize(folds, order)
    leaves = []
    for d in DEPTHS:
        pipe = make_tree(d).fit(train[COLUMNS], train[TARGET])
        leaves.append(int(pipe.named_steps["model"].get_n_leaves()))
    out = pd.DataFrame(
        {
            "depth": order,
            "fit_pr_auc": table["fit_pr_auc_mean"].to_numpy(),
            "cv_pr_auc": table["pr_auc_mean"].to_numpy(),
            "cv_std": table["pr_auc_std"].to_numpy(),
            "leaves": leaves,
        }
    )
    return out


def choose_depth(sweep: pd.DataFrame, tolerance: float = TOLERANCE) -> str:
    """The shallowest tree whose validation PR-AUC is within ``tolerance`` of the best."""
    limit = float(sweep["cv_pr_auc"].max()) - tolerance
    for _, row in sweep.iterrows():
        if float(row["cv_pr_auc"]) >= limit:
            return str(row["depth"])
    return str(sweep["depth"].iloc[int(sweep["cv_pr_auc"].to_numpy().argmax())])


def forest_grid(train: pd.DataFrame) -> pd.DataFrame:
    specs = [ForestSpec(leaf, mf) for leaf in LEAF_SIZES for mf in MAX_FEATURES]
    folds = cross_validate(train, {s.label: partial(make_forest, s) for s in specs})
    table = summarize(folds, [s.label for s in specs])
    return pd.DataFrame(
        {
            "min_samples_leaf": [s.min_samples_leaf for s in specs],
            "max_features": [str(s.max_features) for s in specs],
            "fit_pr_auc": table["fit_pr_auc_mean"].to_numpy(),
            "cv_pr_auc": table["pr_auc_mean"].to_numpy(),
        }
    )


def choose_forest(grid: pd.DataFrame, tolerance: float = TOLERANCE) -> ForestSpec:
    """Within ``tolerance`` of the best score, prefer the largest leaf size, then the fewest features per split."""
    limit = float(grid["cv_pr_auc"].max()) - tolerance
    ok = grid[grid["cv_pr_auc"] >= limit].copy()
    ok["mf_rank"] = ok["max_features"].map({str(m): i for i, m in enumerate(MAX_FEATURES)})
    best = ok.sort_values(["min_samples_leaf", "mf_rank"], ascending=[False, True]).iloc[0]
    mf: str | float = best["max_features"] if best["max_features"] == "sqrt" else float(best["max_features"])
    return ForestSpec(int(best["min_samples_leaf"]), mf)


def forest_curve(train: pd.DataFrame, spec: ForestSpec) -> pd.DataFrame:
    """Validation PR-AUC as the forest grows: more trees reduce variance, with diminishing returns."""
    sizes = {str(n): partial(make_forest, ForestSpec(spec.min_samples_leaf, spec.max_features, n)) for n in FOREST_SIZES}
    table = summarize(cross_validate(train, sizes), [str(n) for n in FOREST_SIZES])
    return pd.DataFrame({"trees": list(FOREST_SIZES), "cv_pr_auc": table["pr_auc_mean"].to_numpy()})


def raw_column(feature: str) -> str:
    """The raw input column a transformed feature name came from (``cat__Contract_Two year`` -> ``Contract``)."""
    name = feature.split("__", 1)[1]
    if name in NUMERIC or name == "random_noise":
        return name
    return max((c for c in CATEGORIES if name.startswith(f"{c}_")), key=len)


def importances(train: pd.DataFrame, spec: ForestSpec, seed: int = SEED) -> pd.DataFrame:
    """Two importance measures for the same forest, with a pure-noise column added as a yardstick.

    Fit on 80% of the training split, evaluate on the other 20%. *Impurity* importance is how much a column reduced
    node impurity while the trees were grown, summed over its one-hot columns. *Permutation* importance is how much
    the validation PR-AUC drops when that raw column is shuffled. The noise column carries no information, so any
    importance it receives is an artefact of the measure.
    """
    fit_df, val_df = train_test_split(train, test_size=0.2, random_state=seed, stratify=train[TARGET])
    rng = np.random.default_rng(seed)
    fit_df = fit_df.assign(random_noise=rng.normal(size=len(fit_df)))
    val_df = val_df.assign(random_noise=rng.normal(size=len(val_df)))
    cols = [*COLUMNS, "random_noise"]
    pipe = make_forest(spec, cols).fit(fit_df[cols], fit_df[TARGET])
    names = feature_names(pipe.named_steps["prep"])
    impurity = pd.Series(pipe.named_steps["model"].feature_importances_, index=[raw_column(n) for n in names]).groupby(level=0).sum()
    perm = permutation_importance(pipe, val_df[cols], val_df[TARGET], scoring="average_precision", n_repeats=10, random_state=seed, n_jobs=1)
    permutation = pd.Series(perm.importances_mean, index=cols)
    spread = pd.Series(perm.importances_std, index=cols)
    out = pd.DataFrame({"impurity": impurity, "permutation_drop": permutation, "permutation_std": spread}).fillna(0.0)
    out["impurity_rank"] = out["impurity"].rank(ascending=False).astype(int)
    out["permutation_rank"] = out["permutation_drop"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("permutation_drop", ascending=False)


@dataclass
class TreeAnalysis:
    sweep: pd.DataFrame
    chosen_depth: str
    grid: pd.DataFrame
    chosen_forest: ForestSpec
    curve: pd.DataFrame
    families: pd.DataFrame
    importance: pd.DataFrame


def analyse(train: pd.DataFrame, logistic: Callable[[], Pipeline]) -> TreeAnalysis:
    """Everything in ``reports/trees_and_forests.md``. ``logistic`` builds the chosen logistic pipeline to compare against."""
    sweep = tree_sweep(train)
    depth = choose_depth(sweep)
    grid = forest_grid(train)
    spec = choose_forest(grid)
    curve = forest_curve(train, spec)
    chosen_depth: int | None = None if depth == "none" else int(depth)
    named: dict[str, Callable[[], Pipeline]] = {
        "logistic_l2_chosen": logistic,
        "decision_tree_chosen": lambda: make_tree(chosen_depth),
        "decision_tree_unlimited": lambda: make_tree(None),
        "random_forest_default": lambda: make_forest(ForestSpec(1, "sqrt", 400)),
        "random_forest_chosen": lambda: make_forest(ForestSpec(spec.min_samples_leaf, spec.max_features, 400)),
    }
    folds = cross_validate(train, named)
    fam = summarize(folds, list(named))
    pr = folds.pivot(index="fold", columns="model", values="pr_auc")
    paired = pr.sub(pr["logistic_l2_chosen"], axis=0)
    fam["delta_mean"], fam["delta_std"], fam["folds_better"] = paired.mean(), paired.std(ddof=1), (paired > 0).sum()
    return TreeAnalysis(sweep, depth, grid, spec, curve, fam, importances(train, spec))
