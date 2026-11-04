import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from mlproject import __main__ as cli
from mlproject import baseline, models
from mlproject.data import TARGET, load
from mlproject.split import split


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return split(load())[0]


@pytest.fixture(scope="module")
def analysis(train: pd.DataFrame) -> models.Analysis:
    return models.analyse(train)


def test_every_model_is_a_pipeline_that_starts_with_the_recommended_preprocessing():
    for spec in (models.Spec("a", family="linear"), models.Spec("b"), models.Spec("c", penalty="l1", C=0.1)):
        pipe = models.build(spec)
        assert isinstance(pipe, Pipeline) and list(pipe.named_steps) == ["prep", "model"]
    assert isinstance(models.build(models.Spec("a", family="linear")).named_steps["model"], LinearRegression)
    assert isinstance(models.build(models.Spec("b")).named_steps["model"], LogisticRegression)


def test_folds_match_the_baselines_so_comparisons_are_paired(train):
    ours = models.cross_validate(train, [models.Spec("x")])
    folds = StratifiedKFold(5, shuffle=True, random_state=baseline.SEED)
    sizes = [len(v) for _, v in folds.split(train, train[TARGET])]
    assert sorted(ours.groupby("fold").size().index) == [0, 1, 2, 3, 4] and len(sizes) == 5
    assert models.cross_validate(train, [models.Spec("x")]).equals(ours)  # deterministic


def test_l2_logistic_with_c_one_reproduces_the_recommended_recipe_score(train):
    mine = models.summarize(models.cross_validate(train, [models.Spec("m", C=1.0)]), ["m"]).loc["m", "pr_auc_mean"]
    recipe = baseline.compare_variants(train).loc[models.RECOMMENDED, "pr_auc_mean"]
    assert mine == pytest.approx(recipe, abs=1e-9)


def test_simplest_within_picks_the_strongest_penalty_inside_the_tolerance():
    table = pd.DataFrame({"pr_auc_mean": [0.60, 0.669, 0.670, 0.6705]}, index=["a", "b", "c", "d"])
    assert models.simplest_within(table, tolerance=0.001) == "c"  # the limit is 0.6695: "b" at 0.669 just misses it
    assert models.simplest_within(table, tolerance=0.002) == "b"
    assert models.simplest_within(table, tolerance=0.0) == "d"


def test_sweep_has_the_shape_the_report_describes(analysis):
    sw = analysis.sweep.set_index("C")
    flat = sw.loc[0.03:, "l2_pr_auc"]
    assert flat.between(0.670, 0.675).all()  # "from C = 0.03 upward the PR-AUC is flat"
    assert sw.loc[0.001, "l2_pr_auc"] < sw.loc[0.03, "l2_pr_auc"] - 0.005  # a strong penalty hurts
    assert sw.loc[0.001, "l1_nonzero_max"] == 0 and sw.loc[0.001, "l1_pr_auc"] == pytest.approx(0.265, abs=0.005)  # L1 removes everything


def test_l1_matches_l2_with_far_fewer_inputs(analysis):
    row = analysis.sweep.set_index("C").loc[analysis.chosen_l1.C]
    assert row["l1_nonzero_max"] <= 30  # of 45 inputs
    assert row["l1_pr_auc"] == pytest.approx(analysis.sweep.set_index("C").loc[analysis.chosen.C, "l2_pr_auc"], abs=0.003)


def test_chosen_penalty_is_in_the_flat_region_not_the_edge(analysis):
    assert 0.01 <= analysis.chosen.C <= 1.0 and analysis.chosen.penalty == "l2"


def test_linear_probability_is_close_in_ranking_but_not_a_probability(analysis):
    fam = analysis.families
    assert -0.02 < fam.loc["linear_probability", "delta_mean"] < 0  # slightly behind, not hopeless
    assert analysis.linear_out_of_range > 0.1
    assert fam.loc["linear_probability", "brier_mean"] > fam.loc["logistic_l2_chosen", "brier_mean"]


def test_class_weights_change_the_cutoff_trade_off_not_the_ranking(analysis):
    fam = analysis.families
    assert abs(fam.loc["logistic_balanced", "delta_mean"]) < 0.003  # ranking quality unchanged
    assert fam.loc["logistic_balanced", "recall_mean"] > fam.loc["logistic_l2_chosen", "recall_mean"] + 0.15
    assert fam.loc["logistic_balanced", "precision_mean"] < fam.loc["logistic_l2_chosen", "precision_mean"] - 0.1
    assert fam.loc["logistic_balanced", "brier_mean"] > fam.loc["logistic_l2_chosen", "brier_mean"] + 0.02


def test_every_logistic_model_beats_the_best_rule_baseline(analysis, train):
    best_rule = baseline.summarize(baseline.cross_validate(train)).loc["short_tenure", "pr_auc_mean"]
    assert (analysis.families["pr_auc_mean"] > best_rule + 0.1).all()


def test_odds_ratios_are_exp_of_coefficients_and_sorted_by_size(analysis):
    odds = analysis.odds
    assert np.allclose(odds["odds_ratio"], np.exp(odds["coefficient"]))
    assert odds["coefficient"].abs().is_monotonic_decreasing and len(odds) == 10
    top = set(odds["input"])
    assert "Contract_Month-to-month" in top and "Contract_Two year" in top  # what the EDA found


def test_contract_coefficients_point_the_expected_way(train):
    table = models.odds_ratios(models.Spec("m", C=0.03), train, top=45).set_index("input")
    assert table.loc["Contract_Month-to-month", "coefficient"] > 0 > table.loc["Contract_Two year", "coefficient"]


def test_linear_models_report_is_about_the_training_split_only(train):
    text = cli.LINEAR_MD.read_text(encoding="utf-8")
    assert f"{len(train):,} rows" in text and "has not been used" in text


def test_one_hot_columns_are_exactly_redundant_which_is_why_l1_counts_vary(train):
    """Each category's dummies sum to 1, so the design matrix is rank-deficient: L1 can pick among identical copies."""
    prep = models.build(models.Spec("x")).named_steps["prep"].fit(train[models.COLUMNS])
    design = np.column_stack([np.ones(len(train)), prep.transform(train[models.COLUMNS])])
    assert np.linalg.matrix_rank(design) < design.shape[1] - 10


def test_l1_non_zero_range_is_ordered_and_stays_in_a_narrow_band(analysis):
    sw = analysis.sweep
    assert (sw["l1_nonzero_min"] <= sw["l1_nonzero_max"]).all()
    assert (sw["l1_nonzero_max"] - sw["l1_nonzero_min"]).max() <= 8  # the count wobbles a little, not wildly
