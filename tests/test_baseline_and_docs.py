import json

import numpy as np
import pandas as pd
import pytest

from mlproject import __main__ as cli
from mlproject import baseline, card
from mlproject.data import TARGET, load
from mlproject.split import split


@pytest.fixture(scope="module")
def train():
    return split(load())[0]


@pytest.fixture(scope="module")
def folds(train):
    return baseline.cross_validate(train)


def test_majority_never_predicts_churn_and_cannot_rank(folds):
    m = folds[folds["baseline"] == "majority"]
    assert (m["recall"] == 0).all() and (m["roc_auc"] == 0.5).all()
    assert (m["accuracy"] > 0.73).all()  # high accuracy for a model that is useless: why accuracy is not the target


def test_every_baseline_is_scored_on_every_fold(folds):
    assert len(folds) == 4 * 5
    assert folds.groupby("baseline").size().eq(5).all()


def test_baselines_are_ordered_sensibly(folds):
    pr = folds.groupby("baseline")["pr_auc"].mean()
    assert pr["majority"] < pr["month_to_month"] < pr["short_tenure"]
    assert pr["random_prior"] == pytest.approx(pr["majority"], abs=0.03)  # a weighted coin is no better than the base rate


def test_month_to_month_rule_matches_hand_count(train):
    model = baseline.MonthToMonth().fit(train)
    pred = model.predict(train)
    m2m = train["Contract"] == "Month-to-month"
    assert pred.sum() == m2m.sum()
    expected_recall = train.loc[m2m, TARGET].sum() / train[TARGET].sum()
    assert (pred[train[TARGET] == 1]).mean() == pytest.approx(expected_recall)
    assert model.rates["Two year"] < model.rates["One year"] < model.rates["Month-to-month"]


def test_unknown_contract_falls_back_to_the_base_rate(train):
    model = baseline.MonthToMonth().fit(train)
    odd = train.head(2).assign(Contract="Lifetime")
    assert np.allclose(model.scores(odd), train[TARGET].mean())


def test_fit_uses_training_rows_only(train):
    a = baseline.MonthToMonth().fit(train.head(500))
    b = baseline.MonthToMonth().fit(train.tail(500))
    assert a.rates != b.rates


def test_cross_validation_is_reproducible(train, folds):
    again = baseline.cross_validate(train)
    pd.testing.assert_frame_equal(folds, again)


def test_summary_has_mean_and_std_per_metric(folds):
    s = baseline.summarize(folds)
    assert list(s.index) == ["majority", "random_prior", "month_to_month", "short_tenure"]
    assert "pr_auc_mean" in s.columns and "pr_auc_std" in s.columns


def test_quality_facts_quoted_in_the_card_are_true():
    df = load()
    q = card.quality_facts(df)
    assert q["blank_total_charges"] == 11 and q["blank_all_tenure_zero"] == 1 and q["blank_churned"] == 0
    assert q["no_internet_rows"] == 1526 and q["no_phone_rows"] == 682
    assert q["churned"] == 1869
    assert (df["tenure"] == 0).sum() == 11


def test_committed_docs_and_reports_are_up_to_date():
    card_md, report_md, report_json = cli.build_outputs()
    assert cli.CARD_PATH.read_text(encoding="utf-8") == card_md
    assert cli.REPORT_MD.read_text(encoding="utf-8") == report_md
    assert json.loads(cli.REPORT_JSON.read_text(encoding="utf-8")) == json.loads(report_json)


def test_test_split_is_not_used_by_the_baseline_report():
    df = load()
    train, test = split(df)
    md = cli.REPORT_MD.read_text(encoding="utf-8")
    assert f"{len(train):,} rows" in md and "has not been used" in md
    assert len(test) == 1409
