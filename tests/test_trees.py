import numpy as np
import pandas as pd
import pytest

from mlproject import __main__ as cli
from mlproject import baseline, trees
from mlproject.data import CATEGORIES, NUMERIC, TARGET, load
from mlproject.split import split


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return split(load())[0]


@pytest.fixture(scope="module")
def a() -> trees.TreeAnalysis:
    return cli.cached_trees()


def test_a_deeper_tree_always_fits_its_own_rows_better(a):
    fit = a.sweep["fit_pr_auc"].to_numpy()
    assert (np.diff(fit) >= -1e-9).all()
    assert fit[-1] > 0.999  # an unlimited tree memorises the training rows (a few identical customers with different outcomes cannot be separated)


def test_validation_score_peaks_at_a_moderate_depth_and_collapses_when_unlimited(a):
    s = a.sweep.set_index("depth")["cv_pr_auc"]
    assert s.idxmax() in {"4", "5", "6"}
    assert s["none"] < s.max() - 0.15
    assert s["1"] < s["3"] < s["5"]  # more depth helps until it starts to memorise


def test_an_unlimited_tree_is_worse_than_the_one_line_contract_rule(a, train):
    rule = baseline.summarize(baseline.cross_validate(train)).loc["month_to_month", "pr_auc_mean"]
    assert a.sweep.set_index("depth").loc["none", "cv_pr_auc"] < rule


def test_chosen_depth_is_the_shallowest_within_tolerance(a):
    best = a.sweep["cv_pr_auc"].max()
    chosen = a.sweep[a.sweep["depth"] == a.chosen_depth].iloc[0]
    assert chosen["cv_pr_auc"] >= best - trees.TOLERANCE
    shallower = a.sweep.iloc[: a.sweep.index[a.sweep["depth"] == a.chosen_depth][0]]
    assert (shallower["cv_pr_auc"] < best - trees.TOLERANCE).all()


def test_choose_depth_and_choose_forest_on_toy_tables():
    sweep = pd.DataFrame({"depth": ["1", "2", "3"], "cv_pr_auc": [0.5, 0.6995, 0.7]})
    assert trees.choose_depth(sweep, tolerance=0.001) == "2"
    assert trees.choose_depth(sweep, tolerance=0.0) == "3"
    grid = pd.DataFrame(
        {
            "min_samples_leaf": [5, 20, 40, 40],
            "max_features": ["sqrt", "0.5", "0.5", "sqrt"],
            "cv_pr_auc": [0.70, 0.699, 0.60, 0.5],
        }
    )
    assert trees.choose_forest(grid, tolerance=0.002) == trees.ForestSpec(20, 0.5)  # 40-leaf cells are too far below the best


def test_forest_beats_a_single_tree_and_tuning_matters(a):
    f = a.families["pr_auc_mean"]
    assert f["random_forest_chosen"] > f["decision_tree_chosen"] + 0.03
    assert f["random_forest_chosen"] > f["random_forest_default"] + 0.03
    assert a.families.loc["random_forest_default", "fit_pr_auc_mean"] > 0.99  # the default forest memorises its training rows


def test_forest_does_not_beat_logistic_regression_here(a):
    row = a.families.loc["random_forest_chosen"]
    assert row["delta_mean"] < 0.002
    assert row["delta_mean"] > -0.03  # close, but not better


def test_more_trees_help_then_stop_helping(a):
    c = a.curve.set_index("trees")["cv_pr_auc"]
    assert c[10] < c[50] - 0.003
    assert abs(c[400] - c[100]) < 0.003


def test_grid_shows_leaf_size_matters_more_than_max_features(a):
    g = a.grid
    by_leaf = g.groupby("min_samples_leaf")["cv_pr_auc"].mean()
    assert by_leaf[5] < by_leaf[20] - 0.003
    spread_features = g.groupby("max_features")["cv_pr_auc"].mean()
    assert spread_features.max() - spread_features.min() < 0.006


def test_impurity_importance_gives_a_pure_noise_column_real_weight_but_permutation_does_not(a):
    imp = a.importance
    real = imp.drop(index="random_noise")
    assert imp.loc["random_noise", "impurity"] > real["impurity"].min()  # noise outranks at least some real columns
    assert (real["impurity"] < imp.loc["random_noise", "impurity"]).sum() >= 3
    assert abs(imp.loc["random_noise", "permutation_drop"]) < 0.005  # shuffling it changes nothing


def test_top_columns_match_what_exploration_found(a):
    top3 = set(a.importance.sort_values("permutation_drop", ascending=False).index[:3])
    assert {"Contract", "tenure"} <= top3


def test_importance_covers_every_raw_column_once(a):
    assert set(a.importance.index) == {*NUMERIC, *CATEGORIES, "random_noise"}
    assert np.isclose(a.importance["impurity"].sum(), 1.0)


def test_raw_column_maps_one_hot_names_back_to_inputs():
    assert trees.raw_column("cat__Contract_Two year") == "Contract"
    assert trees.raw_column("cat__PaymentMethod_Electronic check") == "PaymentMethod"
    assert trees.raw_column("num__tenure") == "tenure"
    assert trees.raw_column("num__random_noise") == "random_noise"


def test_forest_is_reproducible_for_a_seed(train):
    spec = trees.ForestSpec(20, 0.3, 50)
    cols = trees.COLUMNS
    a1 = trees.make_forest(spec).fit(train[cols], train[TARGET]).predict_proba(train[cols])
    a2 = trees.make_forest(spec).fit(train[cols], train[TARGET]).predict_proba(train[cols])
    assert np.allclose(a1, a2)


def test_trees_report_uses_the_training_split_only(train):
    text = cli.TREES_MD.read_text(encoding="utf-8")
    assert f"{len(train):,} rows" in text and "has not been used" in text
