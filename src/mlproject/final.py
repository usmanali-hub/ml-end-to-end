"""The one-time evaluation on the held-out test split.

Everything before this module chose models and settings with cross-validation on the training split only. This module
fits each chosen model on the *whole* training split and scores the untouched test split once. Rules that keep it honest:

* The candidates and the primary model are fixed by the caller from cross-validation results, before any test score is seen.
* The decision cutoff is chosen from out-of-fold *training* scores, never from test scores.
* Nothing here is used to choose between models. The report states what happened, whatever it is.
* The report refuses to be overwritten, so a second look at the test split cannot quietly replace the first.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from . import evaluation
from .baseline import SEED
from .data import TARGET
from .models import COLUMNS, scores

N_BOOT = 1000


@dataclass
class FinalResult:
    table: pd.DataFrame  # one row per model: PR-AUC with bootstrap interval, ROC-AUC, Brier
    primary: str
    cutoff: float  # chosen on out-of-fold training scores
    at_default: evaluation.Confusion
    at_cutoff: evaluation.Confusion
    n_test: int
    prevalence: float
    cv_pr_auc: dict[str, float]  # the cross-validation estimate each model was selected on


def bootstrap_interval(y: np.ndarray, s: np.ndarray, n: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval for PR-AUC from resampling the test rows with replacement."""
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    while len(vals) < n:
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(float(average_precision_score(y[idx], s[idx])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    builders: Mapping[str, Callable[[], Pipeline]],
    primary: str,
    cv_pr_auc: Mapping[str, float],
) -> FinalResult:
    y = test[TARGET].to_numpy()
    rows = []
    primary_scores = np.zeros(len(test))
    for name, build in builders.items():
        pipe = build().fit(train[COLUMNS], train[TARGET])
        s = scores(pipe, test)
        lo, hi = bootstrap_interval(y, s)
        rows.append(
            {
                "model": name,
                "pr_auc": float(average_precision_score(y, s)),
                "ci_low": lo,
                "ci_high": hi,
                "roc_auc": float(roc_auc_score(y, s)),
                "brier": float(brier_score_loss(y, s)),
            }
        )
        if name == primary:
            primary_scores = s
    table = pd.DataFrame(rows).set_index("model")
    oof = evaluation.out_of_fold_scores(train, builders[primary])
    cutoff = evaluation.best_f1_cutoff(evaluation.sweep(train[TARGET].to_numpy(), oof))
    return FinalResult(
        table=table,
        primary=primary,
        cutoff=cutoff,
        at_default=evaluation.confusion(y, primary_scores, 0.5),
        at_cutoff=evaluation.confusion(y, primary_scores, cutoff),
        n_test=len(test),
        prevalence=float(y.mean()),
        cv_pr_auc=dict(cv_pr_auc),
    )


def markdown(r: FinalResult, n_train: int) -> str:
    lines = [
        "# Final evaluation on the held-out test split",
        "",
        (
            f"Each model was fitted on the whole training split ({n_train:,} rows) and scored once on the untouched test split ({r.n_test:,} rows, "
            f"churn rate {r.prevalence:.1%}). The models and the primary model (`{r.primary}`) were fixed from cross-validation before any test score was seen; "
            "this report did not choose between them. The 95% interval is a bootstrap over test rows."
        ),
        "",
        "| model | cross-validation PR-AUC | test PR-AUC | 95% interval | test ROC-AUC | test Brier |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in r.table.iterrows():
        lines.append(
            f"| `{name}`{' (primary)' if name == r.primary else ''} | {r.cv_pr_auc[str(name)]:.3f} | {row['pr_auc']:.3f} | {row['ci_low']:.3f} to {row['ci_high']:.3f} | {row['roc_auc']:.3f} | {row['brier']:.3f} |"
        )
    lines += [
        f"| (always predict the base rate) | - | {r.prevalence:.3f} | - | 0.500 | - |",
        "",
        "Differences between models smaller than the interval are not evidence that one is better.",
        "",
        f"## Decisions by `{r.primary}`",
        "",
        f"The cutoff {r.cutoff:.2f} maximised F1 on out-of-fold *training* scores; it was not tuned on the test split.",
        "",
        "| cutoff | flagged | true positives | false positives | false negatives | precision | recall | accuracy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, k in ((0.5, r.at_default), (r.cutoff, r.at_cutoff)):
        lines.append(f"| {label:.2f} | {k.flagged:,} | {k.tp} | {k.fp} | {k.fn} | {k.precision:.3f} | {k.recall:.3f} | {k.accuracy:.3f} |")
    lines += ["", "The test split has now been used. Any further change to models or settings must be reported as made after seeing it.", ""]
    return "\n".join(lines)


def write_once(path: Path, text: str, force: bool = False) -> None:
    """Write the report; refuse to replace an existing one unless ``force`` is set."""
    if path.exists() and not force:
        raise FileExistsError(f"{path.name} already exists: the test split has been used once. Pass --force to overwrite.")
    path.write_text(text, encoding="utf-8", newline="\n")
