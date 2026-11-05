"""Confusion matrices for the churn model: what each cutoff actually does to real customers.

All scores here are *out-of-fold*: every training customer is scored by a model that never saw them, using the same
5 folds as the other comparisons. The test split has not been used. A confusion matrix needs a cutoff (flag a
customer when the predicted churn probability is at or above it), and the right cutoff depends on what a mistake
costs, so several are shown.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .data import TARGET
from .models import COLUMNS, scores
from .trees import folds_of

COST_RATIOS = (1, 2, 5, 10)  # how many times worse missing a churner is than wasting an offer on a loyal customer


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def specificity(self) -> float:
        return self.tn / (self.tn + self.fp) if self.tn + self.fp else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def flagged(self) -> int:
        return self.tp + self.fp


def confusion(y: np.ndarray, s: np.ndarray, cutoff: float) -> Confusion:
    """Counts of true/false positives and negatives when customers scoring at or above ``cutoff`` are flagged."""
    flag = s >= cutoff
    y = np.asarray(y).astype(bool)
    return Confusion(
        tp=int((flag & y).sum()), fp=int((flag & ~y).sum()), fn=int((~flag & y).sum()), tn=int((~flag & ~y).sum())
    )


def out_of_fold_scores(train: pd.DataFrame, build: Callable[[], Pipeline]) -> np.ndarray:
    """Score every training customer with a model fitted on the other four folds."""
    out = np.zeros(len(train))
    for fit_idx, val_idx in folds_of(train):
        pipe = build().fit(train.iloc[fit_idx][COLUMNS], train.iloc[fit_idx][TARGET])
        out[val_idx] = scores(pipe, train.iloc[val_idx])
    return out


def sweep(y: np.ndarray, s: np.ndarray, cutoffs: np.ndarray | None = None) -> pd.DataFrame:
    """Every cutoff's confusion counts and metrics, one row per cutoff."""
    cutoffs = np.round(np.arange(0.02, 0.99, 0.01), 2) if cutoffs is None else cutoffs
    rows = []
    for c in cutoffs:
        k = confusion(y, s, float(c))
        rows.append({"cutoff": float(c), "tp": k.tp, "fp": k.fp, "fn": k.fn, "tn": k.tn, "precision": k.precision, "recall": k.recall, "f1": k.f1})
    return pd.DataFrame(rows)


def best_f1_cutoff(table: pd.DataFrame) -> float:
    return float(table["cutoff"].to_numpy()[int(table["f1"].to_numpy().argmax())])


def cost_optimal_cutoff(table: pd.DataFrame, ratio: float) -> float:
    """The cutoff with the lowest cost when a missed churner costs ``ratio`` times a wasted offer."""
    cost = table["fp"] * 1.0 + table["fn"] * float(ratio)
    return float(table["cutoff"].to_numpy()[int(cost.to_numpy().argmin())])


def top_share(y: np.ndarray, s: np.ndarray, share: float) -> Confusion:
    """Flag exactly the highest-scoring ``share`` of customers (what a limited retention budget allows)."""
    k = round(share * len(s))
    order = np.argsort(-s, kind="stable")
    flag = np.zeros(len(s), dtype=bool)
    flag[order[:k]] = True
    y = np.asarray(y).astype(bool)
    return Confusion(int((flag & y).sum()), int((flag & ~y).sum()), int((~flag & y).sum()), int((~flag & ~y).sum()))
