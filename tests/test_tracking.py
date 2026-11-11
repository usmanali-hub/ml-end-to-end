import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlproject import tracking
from mlproject.data import CATEGORIES, NUMERIC, TARGET
from mlproject.preprocess import build_preprocessor

mlflow = pytest.importorskip("mlflow")


def frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(["No", "Yes"], n) for c in CATEGORIES})
    df["Contract"] = rng.choice(["Month-to-month", "One year", "Two year"], n)
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 72, n)
    df["MonthlyCharges"] = rng.uniform(20, 110, n)
    df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
    p = 1 / (1 + np.exp(-(1.2 * (df["Contract"] == "Month-to-month") - 0.04 * df["tenure"])))
    df[TARGET] = (rng.random(n) < p).astype(int)
    return df


def build(c: float) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(NUMERIC, CATEGORIES)), ("model", LogisticRegression(C=c, max_iter=500))])


@pytest.fixture()
def logged(tmp_path):
    uri = tracking.store_uri(tmp_path / "db" / "mlflow.db")
    train = frame()
    cands = [
        tracking.Candidate("strong_penalty", {"family": "logistic", "C": 0.01}, lambda: build(0.01)),
        tracking.Candidate("weak_penalty", {"family": "logistic", "C": 10.0}, lambda: build(10.0)),
    ]
    ids = tracking.log_runs(train, cands, uri, experiment="test-churn")
    return uri, ids, train


def test_one_run_per_candidate_with_params_tags_and_fold_metrics(logged):
    uri, ids, train = logged
    assert len(ids) == 2 and len(set(ids)) == 2
    run = tracking.lookup(uri, "weak_penalty", experiment="test-churn")
    assert run["params.C"] == "10.0" and run["params.seed"] == "42" and run["params.n_train"] == str(len(train))
    assert run["tags.primary_metric"] == "pr_auc" and "test split untouched" in run["tags.split"]
    assert run["tags.git_commit"] and run["tags.data_sha256"]
    assert 0.0 < run["metrics.cv_pr_auc_mean"] <= 1.0 and run["metrics.cv_pr_auc_std"] >= 0.0


def test_every_fold_is_logged_as_a_step(logged):
    uri, ids, _ = logged
    mlflow.set_tracking_uri(uri)
    history = mlflow.tracking.MlflowClient().get_metric_history(ids[0], "pr_auc")
    assert sorted(m.step for m in history) == [0, 1, 2, 3, 4]
    mean = float(np.mean([m.value for m in history]))
    assert tracking.lookup(uri, "strong_penalty", experiment="test-churn")["metrics.cv_pr_auc_mean"] == pytest.approx(mean)


def test_comparison_is_sorted_best_first_and_has_one_row_per_model(logged):
    uri, _, _ = logged
    table = tracking.comparison(uri, experiment="test-churn")
    assert list(table["model"].sort_values()) == ["strong_penalty", "weak_penalty"]
    assert table["pr_auc"].is_monotonic_decreasing
    assert "| model | PR-AUC |" in tracking.markdown_table(table)


def test_rerunning_keeps_history_but_the_comparison_shows_only_the_latest_run(logged):
    uri, _, train = logged
    tracking.log_runs(train, [tracking.Candidate("weak_penalty", {"family": "logistic", "C": 10.0}, lambda: build(10.0))], uri, experiment="test-churn")
    mlflow.set_tracking_uri(uri)
    assert len(mlflow.search_runs(experiment_names=["test-churn"])) == 3
    assert len(tracking.comparison(uri, experiment="test-churn")) == 2


def test_lookup_of_an_unknown_run_raises(logged):
    uri, _, _ = logged
    with pytest.raises(KeyError):
        tracking.lookup(uri, "no_such_model", experiment="test-churn")


def test_empty_store_gives_an_empty_comparison(tmp_path):
    mlflow.set_tracking_uri(tracking.store_uri(tmp_path / "e.db"))
    mlflow.set_experiment("empty")
    assert tracking.comparison(tracking.store_uri(tmp_path / "e.db"), experiment="empty").empty


def test_git_commit_is_unknown_outside_a_repository_and_fingerprint_is_stable(tmp_path):
    assert tracking.git_commit(tmp_path) == "unknown"
    assert tracking.data_fingerprint() == tracking.data_fingerprint() and len(tracking.data_fingerprint()) == 12
    assert tracking.data_fingerprint(tmp_path / "nope.csv") == "missing"
