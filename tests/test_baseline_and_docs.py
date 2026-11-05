import json

import numpy as np
import pandas as pd
import pytest

from mlproject import __main__ as cli
from mlproject import baseline, card
from mlproject.data import TARGET, load
from mlproject.metrics import evaluate
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
    assert len(folds) == 5 * 5
    assert folds.groupby("baseline").size().eq(5).all()


def test_baselines_are_ordered_sensibly(folds):
    pr = folds.groupby("baseline")["pr_auc"].mean()
    assert pr["majority"] < pr["month_to_month"] < pr["short_tenure"] < pr["logistic_reference"]
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
    assert list(s.index) == ["majority", "random_prior", "month_to_month", "short_tenure", "logistic_reference"]
    assert "pr_auc_mean" in s.columns and "pr_auc_std" in s.columns


def test_quality_facts_quoted_in_the_card_are_true():
    df = load()
    q = card.quality_facts(df)
    assert q["blank_total_charges"] == 11 and q["blank_all_tenure_zero"] == 1 and q["blank_churned"] == 0
    assert q["no_internet_rows"] == 1526 and q["no_phone_rows"] == 682
    assert q["churned"] == 1869
    assert (df["tenure"] == 0).sum() == 11


def test_committed_docs_and_reports_are_up_to_date():
    card_md, report_md, report_json, tenure_md, preprocessing_md, linear_md, trees_md, confusion_md = cli.build_outputs()
    assert cli.CARD_PATH.read_text(encoding="utf-8") == card_md
    assert cli.REPORT_MD.read_text(encoding="utf-8") == report_md
    assert json.loads(cli.REPORT_JSON.read_text(encoding="utf-8")) == json.loads(report_json)
    assert cli.TENURE_MD.read_text(encoding="utf-8") == tenure_md
    assert cli.PREPROCESS_MD.read_text(encoding="utf-8") == preprocessing_md
    assert cli.LINEAR_MD.read_text(encoding="utf-8") == linear_md
    assert cli.TREES_MD.read_text(encoding="utf-8") == trees_md
    assert cli.CONFUSION_MD.read_text(encoding="utf-8") == confusion_md


def test_test_split_is_not_used_by_the_baseline_report():
    df = load()
    train, test = split(df)
    md = cli.REPORT_MD.read_text(encoding="utf-8")
    assert f"{len(train):,} rows" in md and "has not been used" in md
    assert len(test) == 1409


def test_short_tenure_pr_auc_does_not_depend_on_the_cutoff(train):
    """PR-AUC ranks customers by score, so the 12-month cutoff only changes precision and recall."""
    results = {}
    for months in (6, 12, 24):
        model = baseline.ShortTenure(months=months).fit(train)
        results[months] = evaluate(train[TARGET], model.scores(train), model.predict(train))
    assert len({round(r["pr_auc"], 12) for r in results.values()}) == 1
    assert results[6]["recall"] < results[12]["recall"] < results[24]["recall"]


def test_logistic_reference_beats_every_rule_and_never_sees_validation_rows(train):
    fit_df, val_df = train.iloc[:4000], train.iloc[4000:]
    model = baseline.LogisticReference().fit(fit_df)
    scaler = model._pipeline.named_steps["prep"].named_transformers_["num"].named_steps["scale"]  # type: ignore[union-attr]
    # the scaler's means come from the fitting rows, not from the validation rows or the full training set
    assert scaler.mean_[1] == pytest.approx(fit_df["tenure"].mean())  # NUMERIC order: SeniorCitizen, tenure, ...
    assert scaler.mean_[1] != pytest.approx(train["tenure"].mean(), abs=1e-6)
    pr = evaluate(val_df[TARGET], model.scores(val_df), model.predict(val_df))["pr_auc"]
    best_rule = max(
        evaluate(val_df[TARGET], m.fit(fit_df).scores(val_df), m.predict(val_df))["pr_auc"]
        for m in (baseline.MonthToMonth(), baseline.ShortTenure())
    )
    assert pr > best_rule + 0.05


def test_dropped_columns_really_are_not_used(train):
    fit_df, val_df = train.iloc[:4000], train.iloc[4000:]
    model = baseline.LogisticReference(drop=("tenure", "TotalCharges")).fit(fit_df)
    changed = val_df.assign(tenure=0, TotalCharges=0.0)
    assert np.allclose(model.scores(val_df), model.scores(changed))  # changing dropped inputs changes nothing
    full = baseline.LogisticReference().fit(fit_df)
    assert not np.allclose(full.scores(val_df), full.scores(changed))


def test_removing_time_columns_costs_some_but_not_all_of_the_signal(train):
    s = baseline.tenure_ablation(train)
    pr = s["pr_auc_mean"]
    assert pr["all_features"] >= pr["without_tenure"] >= pr["without_tenure_and_total_charges"] - 0.01
    assert pr["without_tenure_and_total_charges"] > baseline.summarize(baseline.cross_validate(train)).loc["short_tenure", "pr_auc_mean"]
