import numpy as np
import pandas as pd
import pytest

from mlproject import __main__ as cli
from mlproject import boosting
from mlproject.data import load
from mlproject.split import split


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return split(load())[0]


@pytest.fixture(scope="module")
def a() -> boosting.BoostAnalysis:
    return cli.cached_boosting()


def test_grid_covers_every_setting_and_round_count(a):
    assert len(a.table) == len(boosting.LEAVES) * len(boosting.RATES) * len(boosting.ROUNDS)
    assert set(a.table["rounds"]) == set(boosting.ROUNDS)
    assert a.table["cv_pr_auc"].between(0.4, 0.9).all()


def test_more_rounds_always_fit_the_training_rows_better(a):
    for _, g in a.table.groupby(["leaves", "rate"]):
        fit = g.sort_values("rounds")["fit_pr_auc"].to_numpy()
        assert (np.diff(fit) >= -1e-9).all()


def test_validation_peaks_before_the_last_round_so_rounds_must_be_chosen(a):
    # for the chosen tree size and rate the best round count is not the largest one tried
    curve = a.curve.sort_values("rounds")
    assert curve["cv_pr_auc"].idxmax() != curve.index[-1]
    assert curve["cv_pr_auc"].iloc[-1] < curve["cv_pr_auc"].max()
    assert curve["fit_pr_auc"].iloc[-1] - curve["cv_pr_auc"].iloc[-1] > 0.03  # a visible overfitting gap


def test_chosen_setting_is_the_simplest_within_tolerance(a):
    best = a.table["cv_pr_auc"].max()
    chosen = a.table[
        (a.table["leaves"] == a.chosen.leaves) & (a.table["rate"] == a.chosen.rate) & (a.table["rounds"] == a.chosen.rounds)
    ]["cv_pr_auc"].iloc[0]
    assert chosen >= best - boosting.TOLERANCE
    ok = a.table[a.table["cv_pr_auc"] >= best - boosting.TOLERANCE]
    assert a.chosen.leaves == ok["leaves"].min()


def test_choose_prefers_fewer_leaves_then_fewer_rounds():
    t = pd.DataFrame(
        {
            "leaves": [4, 8, 8, 16],
            "rate": [0.05, 0.05, 0.1, 0.05],
            "rounds": [200, 100, 50, 25],
            "cv_pr_auc": [0.6695, 0.670, 0.6702, 0.670],
        }
    )
    assert boosting.choose(t, 0.001) == boosting.BoostSpec(4, 0.05, 200)
    assert boosting.choose(t, 0.0001) == boosting.BoostSpec(8, 0.1, 50)


def test_best_per_setting_has_one_row_per_setting(a):
    assert len(a.best) == len(boosting.LEAVES) * len(boosting.RATES)
    for _, r in a.best.iterrows():
        sel = a.table[(a.table["leaves"] == r["leaves"]) & (a.table["rate"] == r["rate"])]
        assert r["cv_pr_auc"] == sel["cv_pr_auc"].max()


def test_defaults_are_worse_than_tuned_and_boosting_does_not_beat_logistic_clearly(a):
    pr = a.families["pr_auc_mean"]
    assert pr["lightgbm_default"] < pr["lightgbm_chosen"] - 0.01
    assert a.families["delta_mean"]["lightgbm_chosen"] < 0.01  # no clear win over logistic regression on this data


def test_nested_estimate_is_not_above_the_flattering_one(a):
    pr = a.families["pr_auc_mean"]
    assert pr["lightgbm_nested_tuning"] <= pr["lightgbm_chosen"] + 0.005
    assert len(a.nested) == 5 and set(a.nested["fold"]) == set(range(5))


def test_tuned_boosting_is_deterministic(train):
    spec = boosting.BoostSpec(4, 0.05, 50)
    sub = train.iloc[:1500]
    cols = boosting.COLUMNS
    s1 = boosting.make_boosted(spec).fit(sub[cols], sub["churned"]).predict_proba(sub[cols])[:, 1]
    s2 = boosting.make_boosted(spec).fit(sub[cols], sub["churned"]).predict_proba(sub[cols])[:, 1]
    assert np.array_equal(s1, s2)


def test_test_split_is_not_used(train):
    test = split(load())[1]
    assert set(train["customerID"]).isdisjoint(test["customerID"])


def test_report_is_written_from_the_analysis(a, train):
    md = cli.boosting_markdown(train, a, 0.674)
    assert md.startswith("# Gradient boosting")
    assert "lightgbm_nested_tuning" in md and "What this does not show" in md
