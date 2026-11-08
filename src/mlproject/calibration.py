"""Evaluation beyond one number: ROC against precision-recall, calibration, and what recalibration does.

Ranking quality and calibration are different properties. A model can order customers perfectly and still report
probabilities that are far too high or too low. This module measures both for the models chosen earlier, on
*out-of-fold* training scores (every customer scored by a model that never saw them, the same 5 folds as everywhere
else). Recalibration (Platt scaling and isotonic regression) is itself fitted out-of-fold over those scores, so the
recalibrated numbers are not graded on the rows the calibrator saw. The test split has not been used.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .baseline import SEED
from .data import TARGET
from .evaluation import out_of_fold_scores

BINS = 10
EPS = 1e-6
VARIANTS = ("raw", "platt", "isotonic")


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1 - EPS)
    return np.asarray(np.log(q / (1 - q)), dtype=float)


def reliability(y: np.ndarray, s: np.ndarray, bins: int = BINS) -> pd.DataFrame:
    """Equal-count bins of the predicted probability: what the model said against what happened."""
    order = np.argsort(s, kind="stable")
    rows = []
    for k, idx in enumerate(np.array_split(order, bins)):
        rows.append({"bin": k + 1, "n": len(idx), "predicted": float(s[idx].mean()), "observed": float(y[idx].mean())})
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Average gap between predicted and observed rate, weighted by bin size."""
    return float((table["n"] * (table["predicted"] - table["observed"]).abs()).sum() / table["n"].sum())


def slope_intercept(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Regress the outcome on the log-odds of the score. Perfect calibration is slope 1, intercept 0;
    slope below 1 means the scores are too extreme (overconfident), intercept above 0 means they are too low."""
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s).reshape(-1, 1), y)
    return float(fit.coef_[0][0]), float(fit.intercept_[0])


def recalibrate(y: np.ndarray, s: np.ndarray, method: str) -> np.ndarray:
    """Out-of-fold recalibration: each row is recalibrated by a calibrator fitted on the other four folds of scores."""
    out = np.zeros(len(s))
    for fit, val in StratifiedKFold(5, shuffle=True, random_state=SEED).split(s.reshape(-1, 1), y):
        if method == "platt":
            m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s[fit]).reshape(-1, 1), y[fit])
            out[val] = m.predict_proba(logit(s[val]).reshape(-1, 1))[:, 1]
        elif method == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s[fit], y[fit])
            out[val] = iso.predict(s[val])
        else:
            raise ValueError(f"unknown method {method!r}")
    return out


def precision_at_recall(y: np.ndarray, s: np.ndarray, recall: float) -> float:
    """Best precision among cutoffs that still catch at least ``recall`` of the churners."""
    p, r, _ = precision_recall_curve(y, s)
    return float(p[r >= recall].max())


def recall_at_false_positive_rate(y: np.ndarray, s: np.ndarray, fpr: float) -> float:
    """Share of churners caught while wrongly flagging at most ``fpr`` of the loyal customers."""
    f, t, _ = roc_curve(y, s)
    return float(t[f <= fpr].max())


def metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    table = reliability(y, s)
    slope, intercept = slope_intercept(y, s)
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "precision_at_recall_50": precision_at_recall(y, s, 0.5),
        "precision_at_recall_80": precision_at_recall(y, s, 0.8),
        "recall_at_fpr_05": recall_at_false_positive_rate(y, s, 0.05),
        "recall_at_fpr_10": recall_at_false_positive_rate(y, s, 0.10),
        "brier": float(brier_score_loss(y, s)),
        "log_loss": float(log_loss(y, np.clip(s, EPS, 1 - EPS))),
        "ece": expected_calibration_error(table),
        "slope": slope,
        "intercept": intercept,
        "mean_score": float(s.mean()),
    }


@dataclass
class CalibrationAnalysis:
    scores: dict[str, np.ndarray]  # raw out-of-fold scores per model
    table: pd.DataFrame  # one row per (model, variant)
    prevalence: float
    n: int


def analyse(train: pd.DataFrame, builders: Mapping[str, Callable[[], Pipeline]]) -> CalibrationAnalysis:
    y = train[TARGET].to_numpy()
    raw = {name: out_of_fold_scores(train, build) for name, build in builders.items()}
    rows = []
    for name, s in raw.items():
        for variant in VARIANTS:
            sv = s if variant == "raw" else recalibrate(y, s, variant)
            rows.append({"model": name, "variant": variant, **metrics(y, sv)})
    return CalibrationAnalysis(raw, pd.DataFrame(rows).set_index(["model", "variant"]), float(y.mean()), len(y))


def figure(a: CalibrationAnalysis, y: np.ndarray, path: Path) -> None:
    """Reliability diagram, ROC curve and precision-recall curve for the raw scores of every model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax_rel, ax_roc, ax_pr = axes
    ax_rel.plot([0, 1], [0, 1], color="0.6", linestyle="--", label="perfect")
    ax_roc.plot([0, 1], [0, 1], color="0.6", linestyle="--", label="no skill")
    ax_pr.axhline(a.prevalence, color="0.6", linestyle="--", label=f"base rate {a.prevalence:.3f}")
    for name, s in a.scores.items():
        rel = reliability(y, s)
        ax_rel.plot(rel["predicted"], rel["observed"], marker="o", label=name)
        f, t, _ = roc_curve(y, s)
        ax_roc.plot(f, t, label=name)
        p, r, _ = precision_recall_curve(y, s)
        ax_pr.plot(r, p, label=name)
    ax_rel.set(xlabel="predicted churn probability (bin mean)", ylabel="observed churn rate", title="Reliability (10 equal-count bins)")
    ax_roc.set(xlabel="false positive rate", ylabel="true positive rate (recall)", title="ROC")
    ax_pr.set(xlabel="recall", ylabel="precision", title="Precision-recall")
    for ax in axes:
        ax.grid(alpha=0.3)
    ax_rel.legend(fontsize=8)
    ax_pr.legend(fontsize=8)
    fig.suptitle("Out-of-fold scores on the training split", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
