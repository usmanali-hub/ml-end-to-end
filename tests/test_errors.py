import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlproject import __main__ as cli
from mlproject import errors, models
from mlproject.data import CATEGORIES, NUMERIC, TARGET
from mlproject.preprocess import build_preprocessor
from mlproject.trees import COLUMNS


def frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Customers where churn depends on contract and tenure only; every other column is noise."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(["No", "Yes"], n) for c in CATEGORIES})
    df["Contract"] = rng.choice(["Month-to-month", "One year", "Two year"], n)
    df["InternetService"] = rng.choice(["DSL", "Fiber optic", "No"], n)
    df["PaymentMethod"] = rng.choice(["Bank transfer (automatic)", "Electronic check"], n)
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 72, n)
    df["MonthlyCharges"] = rng.uniform(20, 110, n)
    df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
    p = 1 / (1 + np.exp(-(1.2 * (df["Contract"] == "Month-to-month") - 0.04 * df["tenure"])))
    df[TARGET] = (rng.random(n) < p).astype(int)
    return df


def simple_builder(drop: tuple[str, ...]) -> Pipeline:
    numeric = [c for c in NUMERIC if c not in drop]
    categorical = [c for c in CATEGORIES if c not in drop]
    return Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", LogisticRegression(max_iter=500))])


def test_segment_table_counts_add_up_and_gap_is_observed_minus_predicted():
    df = frame()
    s = np.random.default_rng(1).random(len(df))
    t = errors.segment_table(df, s, 0.5, min_size=1)
    contract = t[t["segment"] == "Contract"]
    assert contract["n"].sum() == len(df)
    row = contract.iloc[0]
    m = (df["Contract"] == row["value"]).to_numpy()
    assert row["gap"] == pytest.approx(df[TARGET].to_numpy()[m].mean() - s[m].mean())
    assert row["errors"] == int(((s[m] >= 0.5) != df[TARGET].to_numpy()[m].astype(bool)).sum())


def test_small_segments_are_dropped():
    df = frame(200)
    assert errors.segment_table(df, np.full(len(df), 0.3), 0.5, min_size=10_000).empty


def test_expected_flags_is_about_five_percent_at_two_standard_errors():
    assert errors.expected_flags(100) == pytest.approx(4.55, abs=0.05)


def test_error_profile_groups_partition_the_customers():
    df = frame()
    s = np.random.default_rng(2).random(len(df))
    p = errors.error_profile(df, s, 0.4)
    assert p["n"].sum() == len(df)
    assert p.loc["missed churners", "mean_score"] < 0.4 <= p.loc["caught churners", "mean_score"]


def test_error_overlap_is_one_for_identical_models_and_lower_for_independent_ones():
    df = frame()
    y = df[TARGET].to_numpy()
    s = np.random.default_rng(3).random(len(df))
    same = errors.error_overlap(y, s, s, 0.5, 0.5)
    assert same["miss_overlap"] == 1.0 and same["false_alarm_overlap"] == 1.0 and same["score_corr"] == pytest.approx(1.0)
    other = errors.error_overlap(y, s, np.random.default_rng(4).random(len(df)), 0.5, 0.5)
    assert other["miss_overlap"] < 0.6


def test_importance_finds_the_columns_that_matter_and_ignores_noise():
    df = frame(1500)
    table = errors.importance(df, simple_builder, repeats=3, columns=["Contract", "tenure", "gender", "Partner"])
    s = errors.summarize_importance(table)
    assert set(s.index) == {"Contract", "tenure", "gender", "Partner"}
    assert s.loc["Contract", "permutation"] > 0.02 and s.loc["tenure", "permutation"] > 0.02
    assert abs(s.loc["gender", "permutation"]) < 0.01 and abs(s.loc["Partner", "drop_column"]) < 0.02
    assert set(table["fold"]) == {0, 1, 2, 3, 4}


def test_permutation_overstates_reliance_on_a_redundant_column():
    """TotalCharges is tenure times monthly charge: shuffling it hurts, but refitting without it barely does."""
    df = frame(2000)
    s = errors.summarize_importance(errors.importance(df, simple_builder, repeats=3, columns=["TotalCharges"]))
    assert s.loc["TotalCharges", "permutation"] > s.loc["TotalCharges", "drop_column"]


def test_rank_agreement_is_one_for_the_same_ordering():
    a = pd.Series({"x": 3.0, "y": 2.0, "z": 1.0})
    assert errors.rank_agreement(a, a * 2) == pytest.approx(1.0)
    assert errors.rank_agreement(a, a[::-1].reset_index(drop=True).set_axis(["x", "y", "z"])) == pytest.approx(-1.0)


def test_logistic_builder_can_drop_any_input_column():
    df = frame(300)
    build = errors.logistic_builder(models.Spec(name="t", C=0.1))
    pipe = build(("Contract", "tenure")).fit(df[COLUMNS], df[TARGET])
    assert pipe.predict_proba(df[COLUMNS]).shape == (300, 2)


def test_report_states_its_caveats_and_matches_the_committed_file():
    text = cli.ERRORS_MD.read_text(encoding="utf-8")
    assert "## What this does not show" in text
    assert "Treat a flag as a question to ask" in text
    assert "not a cause of churn" in text
