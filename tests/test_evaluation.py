import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from mlproject import __main__ as cli
from mlproject import evaluation as ev
from mlproject.data import TARGET, load
from mlproject.split import split


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return split(load())[0]


@pytest.fixture(scope="module")
def y(train) -> np.ndarray:
    return train[TARGET].to_numpy()


@pytest.fixture(scope="module")
def oof() -> np.ndarray:
    return cli.cached_oof()


def test_confusion_counts_match_a_hand_worked_example():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.4, 0.6])
    k = ev.confusion(y, s, 0.5)
    assert (k.tp, k.fp, k.fn, k.tn) == (3, 1, 1, 3)  # flagged: 0.9, 0.8, 0.7, 0.6
    assert k.accuracy == 0.75 and k.precision == 0.75 and k.recall == 0.75 and k.specificity == 0.75 and k.f1 == 0.75


def test_a_score_exactly_at_the_cutoff_is_flagged():
    k = ev.confusion(np.array([1, 0]), np.array([0.5, 0.5]), 0.5)
    assert (k.tp, k.fp) == (1, 1)


def test_counts_always_add_up_and_match_scikit_learn(y, oof):
    for cutoff in (0.1, 0.33, 0.5, 0.8):
        k = ev.confusion(y, oof, cutoff)
        assert k.n == len(y) and k.tp + k.fn == int(y.sum())
        tn, fp, fn, tp = confusion_matrix(y, (oof >= cutoff).astype(int)).ravel()
        assert (k.tp, k.fp, k.fn, k.tn) == (tp, fp, fn, tn)
        assert k.precision == pytest.approx(precision_score(y, oof >= cutoff))
        assert k.recall == pytest.approx(recall_score(y, oof >= cutoff))
        assert k.f1 == pytest.approx(f1_score(y, oof >= cutoff))


def test_accuracy_rewards_flagging_nobody(y, oof):
    nobody, everybody = ev.confusion(y, oof, 1.01), ev.confusion(y, oof, 0.0)
    assert nobody.tp == 0 and nobody.recall == 0 and nobody.accuracy == pytest.approx(1 - y.mean())
    assert nobody.accuracy > 0.73
    assert everybody.recall == 1.0 and everybody.precision == pytest.approx(y.mean())


def test_scores_are_out_of_fold_every_customer_scored_once(oof, y):
    assert oof.shape == y.shape and ((oof > 0) & (oof < 1)).all()  # probabilities, none left at the zero placeholder


def test_raising_the_cutoff_flags_fewer_customers_and_catches_fewer(y, oof):
    t = ev.sweep(y, oof)
    flagged = t["tp"] + t["fp"]
    assert flagged.is_monotonic_decreasing and t["recall"].is_monotonic_decreasing
    assert t["fn"].is_monotonic_increasing


def test_best_f1_cutoff_beats_the_default_and_sits_below_it(y, oof):
    t = ev.sweep(y, oof)
    best = ev.best_f1_cutoff(t)
    assert best < 0.5  # with 1 in 4 positives the F1-best cutoff is below one half
    assert ev.confusion(y, oof, best).f1 >= ev.confusion(y, oof, 0.5).f1


def test_costlier_misses_push_the_cutoff_down_and_save_money(y, oof):
    t = ev.sweep(y, oof)
    cutoffs = [ev.cost_optimal_cutoff(t, r) for r in ev.COST_RATIOS]
    assert cutoffs == sorted(cutoffs, reverse=True) and cutoffs[0] > cutoffs[-1]
    for ratio, cut in zip(ev.COST_RATIOS, cutoffs, strict=True):
        at = ev.confusion(y, oof, cut)
        default = ev.confusion(y, oof, 0.5)
        assert at.fp + ratio * at.fn <= default.fp + ratio * default.fn


def test_top_share_flags_exactly_that_share_and_beats_random(y, oof):
    for share in (0.05, 0.1, 0.3):
        k = ev.top_share(y, oof, share)
        assert k.flagged == round(share * len(y))
        assert k.precision > 2 * y.mean() * (1.0 if share < 0.2 else 0.9)  # clear lift over a random list


def test_report_is_about_the_training_split_only(train):
    text = cli.CONFUSION_MD.read_text(encoding="utf-8")
    assert f"{len(train):,}" in text and "has not been used" in text and "other* four folds" in text
