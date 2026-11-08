"""How much to trust cross-validation, and what hyperparameter search does to it.

Two questions, both answered on the training split only (the test split has not been used):

1. **Noise.** A 5-fold score depends on which rows landed in which fold. ``repeated_cv`` re-runs the comparison with
   several different fold assignments to show how much of a gap between two models is just that.
2. **Search optimism.** The best score found by a search is biased upward: with enough tries, some settings look good
   by luck. ``search_optimism`` runs a random search of increasing size *inside* each outer fold and scores the winner on
   the outer fold's untouched rows, so the gap between the inner best and the outer score is measured, not assumed.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .baseline import SEED
from .data import TARGET
from .preprocess import build_preprocessor
from .trees import COLUMNS, TREE_OPTIONS, folds_of

N_CONFIGS = 24
SIZES = (1, 2, 4, 8, 16, 24)
INNER_FOLDS = 3
REPEATS = 4
FOREST_TREES = 150  # fewer trees than the 400 used elsewhere: the repeated comparison refits every model 20 times


@dataclass(frozen=True)
class SearchConfig:
    leaves: int
    rate: float
    rounds: int
    min_child: int
    subsample: float
    colsample: float
    reg_lambda: float


def sample_configs(n: int = N_CONFIGS, seed: int = SEED, space: str = "narrow") -> list[SearchConfig]:
    """Random boosting settings: sizes and rounds from lists, rates and penalties log-uniform, sampling shares uniform.

    ``narrow`` is a moderately regularised space. ``wide`` allows big trees, tiny leaves, fast rates, heavy column
    subsampling and almost no penalty, so settings differ a lot and some are poor.
    """
    rng = np.random.default_rng(seed)
    if space == "wide":
        return [
            SearchConfig(
                leaves=int(rng.choice([8, 16, 32, 64])),
                rate=float(np.exp(rng.uniform(np.log(0.02), np.log(0.3)))),
                rounds=int(rng.choice([50, 100, 150, 200, 300])),
                min_child=int(rng.choice([1, 3, 5, 10])),
                subsample=float(rng.uniform(0.3, 1.0)),
                colsample=float(rng.uniform(0.3, 1.0)),
                reg_lambda=float(np.exp(rng.uniform(np.log(0.001), np.log(0.1)))),
            )
            for _ in range(n)
        ]
    if space != "narrow":
        raise ValueError(f"unknown space {space!r}")
    return [
        SearchConfig(
            leaves=int(rng.choice([4, 6, 8, 12, 16, 24])),
            rate=float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
            rounds=int(rng.choice([50, 100, 150, 200, 300])),
            min_child=int(rng.choice([10, 20, 40, 80])),
            subsample=float(rng.uniform(0.5, 1.0)),
            colsample=float(rng.uniform(0.5, 1.0)),
            reg_lambda=float(np.exp(rng.uniform(np.log(0.1), np.log(10.0)))),
        )
        for _ in range(n)
    ]


def make_search_pipeline(cfg: SearchConfig) -> Pipeline:
    model = LGBMClassifier(
        n_estimators=cfg.rounds,
        learning_rate=cfg.rate,
        num_leaves=cfg.leaves,
        min_child_samples=cfg.min_child,
        subsample=cfg.subsample,
        subsample_freq=1,
        colsample_bytree=cfg.colsample,
        reg_lambda=cfg.reg_lambda,
        random_state=SEED,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    return Pipeline([("prep", build_preprocessor(options=TREE_OPTIONS)), ("model", model)])


def pr_auc(pipe: Pipeline, fit_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    pipe.fit(fit_df[COLUMNS], fit_df[TARGET])
    return float(average_precision_score(val_df[TARGET], np.asarray(pipe.predict_proba(val_df[COLUMNS]))[:, 1]))


def inner_scores(fit_df: pd.DataFrame, configs: list[SearchConfig]) -> np.ndarray:
    """Mean PR-AUC of every configuration over an inner split of ``fit_df``."""
    inner = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED).split(fit_df, fit_df[TARGET]))
    out = np.zeros(len(configs))
    for i, cfg in enumerate(configs):
        out[i] = np.mean([pr_auc(make_search_pipeline(cfg), fit_df.iloc[a], fit_df.iloc[b]) for a, b in inner])
    return out


def search_optimism(train: pd.DataFrame, configs: list[SearchConfig] | None = None, sizes: tuple[int, ...] = SIZES) -> pd.DataFrame:
    """For each search size, the best inner score and the winner's score on the untouched outer fold, per outer fold.

    A search of size ``n`` tries the first ``n`` sampled configurations, so every size is a prefix of the next and the
    comparison across sizes is not confounded by different luck of the draw.
    """
    configs = configs if configs is not None else sample_configs()
    rows: list[dict[str, float | int]] = []
    for fold, (fit_idx, val_idx) in enumerate(folds_of(train)):
        fit_df, val_df = train.iloc[fit_idx].reset_index(drop=True), train.iloc[val_idx]
        inner = inner_scores(fit_df, configs)
        outer_all = np.array([pr_auc(make_search_pipeline(c), fit_df, val_df) for c in configs])
        for n in sizes:
            winner = int(inner[:n].argmax())
            rows.append(
                {
                    "fold": fold,
                    "size": n,
                    "winner": winner,
                    "inner_best": float(inner[:n].max()),
                    "outer": float(outer_all[winner]),
                    "outer_average_config": float(outer_all[:n].mean()),  # what picking at random would score
                    "outer_hindsight_best": float(outer_all[:n].max()),  # what picking with knowledge of the outer fold would score
                }
            )
    return pd.DataFrame(rows)


def summarize_optimism(table: pd.DataFrame) -> pd.DataFrame:
    g = table.assign(gap=table["inner_best"] - table["outer"]).groupby("size")
    return pd.DataFrame(
        {
            "inner_best": g["inner_best"].mean(),
            "outer": g["outer"].mean(),
            "outer_std": g["outer"].std(ddof=1),
            "gap": g["gap"].mean(),
            "gap_std": g["gap"].std(ddof=1),
            "average_config": g["outer_average_config"].mean(),
            "hindsight_best": g["outer_hindsight_best"].mean(),
            "distinct_winners": g["winner"].nunique(),
        }
    ).reset_index()


def repeated_cv(train: pd.DataFrame, builders: Mapping[str, Callable[[], Pipeline]], repeats: int = REPEATS) -> pd.DataFrame:
    """PR-AUC per model, repeat and fold, using a different fold assignment for each repeat."""
    rows: list[dict[str, float | int | str]] = []
    for rep in range(repeats):
        for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + rep).split(train, train[TARGET])):
            fit_df, val_df = train.iloc[a], train.iloc[b]
            for name, build in builders.items():
                rows.append({"model": name, "repeat": rep, "fold": fold, "pr_auc": pr_auc(build(), fit_df, val_df)})
    return pd.DataFrame(rows)


def summarize_repeats(table: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Per model: the mean, the spread of the repeat means (fold-assignment noise), the standard error within a repeat,
    and how a model compares with ``reference`` repeat by repeat."""
    per = table.groupby(["model", "repeat"])["pr_auc"].mean().unstack("repeat")
    fold_sd = table.groupby(["model", "repeat"])["pr_auc"].std(ddof=1).groupby("model").mean()
    diff = per.sub(per.loc[reference], axis=1)
    out = pd.DataFrame(
        {
            "mean": per.mean(axis=1),
            "repeat_sd": per.std(axis=1, ddof=1),
            "fold_se": fold_sd / np.sqrt(5),
            "vs_ref_mean": diff.mean(axis=1),
            "vs_ref_range": diff.max(axis=1) - diff.min(axis=1),
            "repeats_ahead": (diff > 0).sum(axis=1),
        }
    )
    return out.loc[list(table["model"].unique())]


@dataclass
class SearchAnalysis:
    repeats: pd.DataFrame  # summarize_repeats
    optimism: pd.DataFrame  # summarize_optimism, narrow space
    optimism_folds: pd.DataFrame
    configs: list[SearchConfig]
    wide_optimism: pd.DataFrame  # same analysis in the wide space
    wide_folds: pd.DataFrame


def analyse(train: pd.DataFrame, builders: Mapping[str, Callable[[], Pipeline]], reference: str) -> SearchAnalysis:
    configs = sample_configs()
    folds = search_optimism(train, configs)
    wide = search_optimism(train, sample_configs(space="wide"))
    return SearchAnalysis(
        summarize_repeats(repeated_cv(train, builders), reference), summarize_optimism(folds), folds, configs, summarize_optimism(wide), wide
    )
