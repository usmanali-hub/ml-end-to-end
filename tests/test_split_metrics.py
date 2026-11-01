import numpy as np
import pytest

from mlproject.data import ID_COLUMN, TARGET, load
from mlproject.metrics import METRIC_NAMES, evaluate
from mlproject.split import split


@pytest.fixture(scope="module")
def parts():
    df = load()
    return df, *split(df)


def test_split_sizes_and_no_overlap(parts):
    df, train, test = parts
    assert len(train) + len(test) == len(df) and len(test) == 1409
    assert set(train[ID_COLUMN]).isdisjoint(test[ID_COLUMN])
    assert set(train[ID_COLUMN]) | set(test[ID_COLUMN]) == set(df[ID_COLUMN])


def test_split_is_stratified(parts):
    df, train, test = parts
    for part in (train, test):
        assert part[TARGET].mean() == pytest.approx(df[TARGET].mean(), abs=0.001)


def test_split_is_deterministic_and_seed_sensitive(parts):
    df, train, _ = parts
    again, _ = split(df)
    assert again[ID_COLUMN].tolist() == train[ID_COLUMN].tolist()
    other, _ = split(df, seed=1)
    assert other[ID_COLUMN].tolist() != train[ID_COLUMN].tolist()


def test_metrics_perfect_and_worst_rankings():
    y = np.array([0, 0, 1, 1])
    perfect = evaluate(y, [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert perfect["pr_auc"] == 1.0 and perfect["roc_auc"] == 1.0 and perfect["f1"] == 1.0
    worst = evaluate(y, [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert worst["roc_auc"] == 0.0 and worst["accuracy"] == 0.0


def test_metrics_hand_computed_confusion_matrix():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])  # tp=2 fn=1 fp=1 tn=4
    m = evaluate(y, np.linspace(0, 1, 8), pred)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_constant_scores_give_chance_level_ranking():
    y = np.array([0, 0, 0, 1])
    m = evaluate(y, [0.25] * 4, [0, 0, 0, 0])
    assert m["roc_auc"] == 0.5
    assert m["pr_auc"] == pytest.approx(0.25)  # equals the positive rate
    assert m["precision"] == 0.0 and m["recall"] == 0.0  # no predicted positives: defined as 0, no crash


def test_metric_names_match_output():
    assert set(evaluate([0, 1], [0.2, 0.8], [0, 1])) == set(METRIC_NAMES)
