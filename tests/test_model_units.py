"""Unit tests for the selection rules and model builders that the long analyses depend on.

The analyses themselves (`analyse`, `grid`, `nested`) take minutes and are checked through the committed reports.
These tests pin the small rules inside them on synthetic tables and frames, so a wrong tie-break or a mis-built
pipeline fails in seconds and points at one function.
"""

import numpy as np
import pandas as pd
import pytest

from mlproject import boosting, models, search, trees
from mlproject.data import CATEGORIES, TARGET


def frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(allowed, n) for c, allowed in CATEGORIES.items()})
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 72, n)
    df["MonthlyCharges"] = rng.uniform(20, 110, n)
    df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
    p = 1 / (1 + np.exp(-(1.2 * (df["Contract"] == "Month-to-month") - 0.04 * df["tenure"])))
    df[TARGET] = (rng.random(n) < p).astype(int)
    return df


# ---- models ---------------------------------------------------------------------------------------------------------


def table(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"pr_auc_mean": list(values.values())}, index=list(values))


def test_simplest_within_takes_the_strongest_penalty_inside_the_tolerance():
    t = table({"C=0.01": 0.660, "C=0.03": 0.6695, "C=0.1": 0.670, "C=1": 0.668})
    assert models.simplest_within(t, 0.001) == "C=0.03"  # within 0.001 of the best, and a stronger penalty than C=0.1
    assert models.simplest_within(t, 0.0) == "C=0.1"  # zero tolerance means the best
    assert models.simplest_within(t, 0.02) == "C=0.01"


def test_simplest_within_includes_a_score_exactly_at_the_tolerance_limit():
    t = table({"C=0.01": 0.5, "C=0.1": 0.75})  # binary fractions, so the limit 0.75 - 0.25 = 0.5 is exact
    assert models.simplest_within(t, 0.25) == "C=0.01"


def test_tolerance_table_reports_the_c_each_tolerance_picks():
    t = table({"l2_C=0.01": 0.660, "l2_C=0.03": 0.6695, "l2_C=0.1": 0.670})
    out = models.tolerance_table(t)
    assert list(out["tolerance"]) == list(models.TOLERANCES)
    assert out.loc[out["tolerance"] == 0.0, "C"].item() == 0.1
    assert out["C"].is_monotonic_decreasing  # a larger tolerance never picks a weaker penalty


def test_sweep_specs_cover_the_grid_in_order_with_readable_names():
    specs = models.sweep_specs("l1")
    assert [s.C for s in specs] == list(models.C_GRID)
    assert specs[0].name == "l1_C=0.001" and all(s.penalty == "l1" for s in specs)


def test_summarize_orders_models_and_reports_mean_and_sample_std():
    folds = pd.DataFrame(
        {
            "model": ["b"] * 3 + ["a"] * 3,
            "pr_auc": [0.6, 0.7, 0.8, 0.5, 0.5, 0.5],
            "roc_auc": 0.8, "recall": 0.5, "precision": 0.5, "brier": 0.2,
        }
    )
    s = models.summarize(folds, ["a", "b"])
    assert list(s.index) == ["a", "b"]
    assert s.loc["b", "pr_auc_mean"] == pytest.approx(0.7) and s.loc["b", "pr_auc_std"] == pytest.approx(0.1)
    assert s.loc["a", "pr_auc_std"] == 0


def test_build_respects_family_penalty_and_dropped_columns():
    df = frame()
    for spec, kind in (
        (models.Spec(name="lin", family="linear"), "LinearRegression"),
        (models.Spec(name="l1", penalty="l1", C=0.1), "LogisticRegression"),
        (models.Spec(name="l2", C=0.1, class_weight="balanced", drop=("gender", "tenure")), "LogisticRegression"),
    ):
        pipe = models.build(spec).fit(df[models.COLUMNS], df[TARGET])
        assert type(pipe.named_steps["model"]).__name__ == kind
    dropped = models.build(models.Spec(name="d", drop=("gender", "tenure")))
    names = " ".join(dropped.named_steps["prep"].fit(df[models.COLUMNS]).get_feature_names_out())
    assert "gender" not in names and "tenure" not in names and "Contract" in names


def test_scores_are_probabilities_for_logistic_and_raw_values_for_the_linear_model():
    df = frame(400)
    logistic = models.build(models.Spec(name="l")).fit(df[models.COLUMNS], df[TARGET])
    s = models.scores(logistic, df)
    assert s.min() >= 0 and s.max() <= 1
    linear = models.build(models.Spec(name="lin", family="linear")).fit(df[models.COLUMNS], df[TARGET])
    np.testing.assert_allclose(models.scores(linear, df), linear.predict(df[models.COLUMNS]))


def test_odds_ratios_are_exp_of_coefficients_sorted_by_size():
    t = models.odds_ratios(models.Spec(name="l", C=1.0), frame(600), top=5)
    assert len(t) == 5
    np.testing.assert_allclose(t["odds_ratio"], np.exp(t["coefficient"]))
    assert t["coefficient"].abs().is_monotonic_decreasing


def test_replace_name_changes_only_the_name():
    spec = models.Spec(name="a", penalty="l1", C=0.3, class_weight="balanced", drop=("gender",))
    new = models.replace_name(spec, "b")
    assert new.name == "b" and new.penalty == "l1" and new.C == 0.3 and new.class_weight == "balanced" and new.drop == ("gender",)


# ---- trees ----------------------------------------------------------------------------------------------------------


def test_choose_depth_picks_the_shallowest_tree_within_tolerance():
    sweep = pd.DataFrame({"depth": ["2", "3", "4", "none"], "cv_pr_auc": [0.60, 0.649, 0.650, 0.55]})
    assert trees.choose_depth(sweep, 0.002) == "3"
    assert trees.choose_depth(sweep, 0.0) == "4"


def test_depth_label():
    assert trees.depth_label(None) == "none" and trees.depth_label(5) == "5"


def test_choose_forest_prefers_larger_leaves_then_fewer_features():
    grid = pd.DataFrame(
        {
            "min_samples_leaf": [10, 40, 40, 80],
            "max_features": ["sqrt", "0.5", "0.3", "sqrt"],
            "cv_pr_auc": [0.670, 0.669, 0.6695, 0.640],
        }
    )
    spec = trees.choose_forest(grid, 0.002)
    assert spec == trees.ForestSpec(40, 0.3)  # leaf 80 is too weak; of the leaf-40 pair, fewer features wins (0.3 before 0.5)
    assert trees.choose_forest(grid, 0.0) == trees.ForestSpec(10, "sqrt")  # zero tolerance means the best


def test_raw_column_maps_transformed_names_back_to_inputs():
    assert trees.raw_column("num__tenure") == "tenure"
    assert trees.raw_column("cat__Contract_Two year") == "Contract"
    assert trees.raw_column("cat__PaymentMethod_Mailed check") == "PaymentMethod"
    assert trees.raw_column("cat__OnlineSecurity_No internet service") == "OnlineSecurity"
    assert trees.raw_column("num__random_noise") == "random_noise"


def test_tree_depth_and_forest_columns_are_applied():
    df = frame()
    tree = trees.make_tree(2).fit(df[models.COLUMNS], df[TARGET])
    assert tree.named_steps["model"].get_depth() <= 2
    small = trees.make_forest(trees.ForestSpec(5, 0.5, 10), columns=["tenure", "Contract"] + [c for c in CATEGORIES if c != "Contract"])
    small.fit(df[models.COLUMNS], df[TARGET])
    names = list(small.named_steps["prep"].get_feature_names_out())
    assert "num__tenure" in names and "num__MonthlyCharges" not in names
    assert trees.ForestSpec(5, 0.5).label == "leaf=5, max_features=0.5"


# ---- boosting -------------------------------------------------------------------------------------------------------


def boost_table() -> pd.DataFrame:
    rows = []
    for leaves, rate, best in ((4, 0.05, 0.672), (4, 0.1, 0.671), (8, 0.05, 0.6725), (16, 0.05, 0.660)):
        for rounds, drop in ((50, 0.01), (150, 0.0)):
            rows.append({"leaves": leaves, "rate": rate, "rounds": rounds, "cv_pr_auc": best - drop, "fit_pr_auc": 0.7, "cv_std": 0.01})
    return pd.DataFrame(rows)


def test_boost_choose_prefers_fewest_leaves_then_fewest_rounds_then_larger_rate():
    t = boost_table()
    assert boosting.choose(t, 0.002) == boosting.BoostSpec(4, 0.1, 150)  # leaves=4 qualifies; rate 0.1 beats 0.05 on the rate rule
    assert boosting.choose(t, 0.0) == boosting.BoostSpec(8, 0.05, 150)  # the single best row
    assert boosting.BoostSpec(4, 0.05, 150).label == "leaves=4, rate=0.05, rounds=150"


def test_best_per_setting_keeps_the_best_round_count_for_each_leaves_and_rate():
    out = boosting.best_per_setting(boost_table())
    assert len(out) == 4 and (out["rounds"] == 150).all()


def test_curve_for_selects_the_chosen_leaves_and_rate_only():
    c = boosting.curve_for(boost_table(), boosting.BoostSpec(4, 0.05, 150))
    assert list(c["rounds"]) == [50, 150] and list(c.columns) == ["rounds", "fit_pr_auc", "cv_pr_auc", "cv_std"]


@pytest.mark.parametrize("make", ["boosted", "hist", "xgboost"])
def test_every_boosting_pipeline_fits_and_gives_probabilities(make):
    df = frame()
    spec = boosting.BoostSpec(4, 0.1, 20)
    pipe = {"boosted": boosting.make_boosted, "hist": boosting.make_sklearn_hist, "xgboost": boosting.make_xgboost}[make](spec)
    p = pipe.fit(df[models.COLUMNS], df[TARGET]).predict_proba(df[models.COLUMNS])
    assert p.shape == (len(df), 2) and np.allclose(p.sum(axis=1), 1)


def test_booster_settings_and_library_default_differ_as_documented():
    b, d = boosting.booster(4, 0.05, 150), boosting.library_default()
    assert (b.num_leaves, b.learning_rate, b.n_estimators, b.subsample) == (4, 0.05, 150, boosting.ROW_SAMPLE)
    assert (d.num_leaves, d.learning_rate, d.n_estimators) == (31, 0.1, 100)
    assert boosting.make_default().fit(frame()[models.COLUMNS], frame()[TARGET]).predict_proba(frame()[models.COLUMNS]).shape[1] == 2


# ---- search ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("space", ["narrow", "wide"])
def test_sample_configs_are_reproducible_and_inside_their_ranges(space):
    a, b = search.sample_configs(30, seed=1, space=space), search.sample_configs(30, seed=1, space=space)
    assert a == b and a != search.sample_configs(30, seed=2, space=space)
    lo_rate, hi_rate = (0.01, 0.2) if space == "narrow" else (0.02, 0.3)
    for c in a:
        assert lo_rate <= c.rate <= hi_rate and 0 < c.subsample <= 1 and 0 < c.colsample <= 1 and c.reg_lambda > 0


def test_unknown_search_space_is_rejected():
    with pytest.raises(ValueError, match="unknown space"):
        search.sample_configs(1, space="huge")


def test_search_pipeline_uses_the_config():
    cfg = search.sample_configs(1, seed=3)[0]
    model = search.make_search_pipeline(cfg).named_steps["model"]
    assert (model.num_leaves, model.n_estimators, model.min_child_samples) == (cfg.leaves, cfg.rounds, cfg.min_child)


def test_summarize_optimism_gap_is_inner_best_minus_outer():
    t = pd.DataFrame(
        {
            "size": [10, 10, 20, 20],
            "inner_best": [0.70, 0.72, 0.68, 0.70],
            "outer": [0.66, 0.68, 0.67, 0.67],
            "outer_average_config": [0.65, 0.65, 0.65, 0.65],
            "outer_hindsight_best": [0.70, 0.71, 0.69, 0.70],
            "winner": ["a", "b", "a", "a"],
        }
    )
    s = search.summarize_optimism(t).set_index("size")
    assert s.loc[10, "gap"] == pytest.approx(0.04) and s.loc[20, "gap"] == pytest.approx(0.02)
    assert s.loc[10, "distinct_winners"] == 2 and s.loc[20, "distinct_winners"] == 1


def test_summarize_repeats_compares_each_model_with_the_reference_repeat_by_repeat():
    rows = []
    for repeat, (ref, other) in enumerate([(0.66, 0.67), (0.67, 0.66), (0.65, 0.66)]):
        for fold in range(5):
            rows += [{"model": "ref", "repeat": repeat, "fold": fold, "pr_auc": ref + 0.001 * fold},
                     {"model": "other", "repeat": repeat, "fold": fold, "pr_auc": other + 0.001 * fold}]
    s = search.summarize_repeats(pd.DataFrame(rows), "ref")
    assert list(s.index) == ["ref", "other"]
    assert s.loc["ref", "vs_ref_mean"] == 0 and s.loc["other", "repeats_ahead"] == 2
    assert s.loc["other", "vs_ref_mean"] == pytest.approx((0.01 - 0.01 + 0.01) / 3)
    assert s.loc["other", "vs_ref_range"] == pytest.approx(0.02)
