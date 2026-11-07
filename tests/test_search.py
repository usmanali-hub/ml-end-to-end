import numpy as np
import pandas as pd
import pytest

from mlproject import __main__ as cli
from mlproject import search
from mlproject.data import load
from mlproject.split import split


@pytest.fixture(scope="module")
def a() -> search.SearchAnalysis:
    return cli.cached_search()


def test_sampled_settings_are_reproducible_in_range_and_nested():
    cfgs = search.sample_configs(24)
    assert cfgs == search.sample_configs(24)
    assert search.sample_configs(10) == cfgs[:10]  # a smaller search is a prefix of a larger one
    assert all(4 <= c.leaves <= 24 and 0.01 <= c.rate <= 0.2 and 0.5 <= c.subsample <= 1 and 0.1 <= c.reg_lambda <= 10 for c in cfgs)
    assert len({c.rate for c in cfgs}) > 20


def test_a_bigger_search_never_has_a_worse_inner_best_or_hindsight_best(a):
    o = a.optimism.sort_values("size")
    assert (np.diff(o["inner_best"].to_numpy()) >= -1e-12).all()
    assert (np.diff(o["hindsight_best"].to_numpy()) >= -1e-12).all()
    assert list(o["size"]) == list(search.SIZES)


def test_winner_is_never_better_than_hindsight_best_on_any_fold(a):
    f = a.optimism_folds
    assert (f["outer"] <= f["outer_hindsight_best"] + 1e-12).all()
    assert (f["outer_average_config"] <= f["outer_hindsight_best"] + 1e-12).all()
    assert len(f) == 5 * len(search.SIZES)


def test_a_search_of_one_has_no_selection(a):
    one = a.optimism.set_index("size").loc[1]
    assert one["distinct_winners"] == 1
    assert one["outer"] == pytest.approx(one["average_config"])
    assert one["outer"] == pytest.approx(one["hindsight_best"])


def test_searching_beats_an_average_setting(a):
    last = a.optimism.set_index("size").loc[max(search.SIZES)]
    assert last["outer"] > last["average_config"]


def test_repeated_cv_shapes_and_reference_row(a):
    r = a.repeats
    assert set(r.index) == {"logistic_l2_chosen", "random_forest_chosen", "lightgbm_chosen", "lightgbm_default"}
    assert r.loc["logistic_l2_chosen", "vs_ref_mean"] == 0 and r.loc["logistic_l2_chosen", "repeats_ahead"] == 0
    assert (r["repeat_sd"] < 0.02).all() and (r["fold_se"] > r["repeat_sd"]).all()
    assert r.loc["lightgbm_default", "mean"] < r.loc["lightgbm_chosen", "mean"]


def test_summarize_repeats_on_a_toy_table():
    rows = []
    for rep, (ref, other) in enumerate([(0.60, 0.62), (0.62, 0.61), (0.64, 0.66)]):
        for fold in range(5):
            rows.append({"model": "ref", "repeat": rep, "fold": fold, "pr_auc": ref + 0.01 * (fold - 2)})
            rows.append({"model": "other", "repeat": rep, "fold": fold, "pr_auc": other + 0.01 * (fold - 2)})
    s = search.summarize_repeats(pd.DataFrame(rows), "ref")
    assert s.loc["ref", "mean"] == pytest.approx(0.62)
    assert s.loc["other", "vs_ref_mean"] == pytest.approx((0.02 - 0.01 + 0.02) / 3)
    assert s.loc["other", "vs_ref_range"] == pytest.approx(0.03)
    assert s.loc["other", "repeats_ahead"] == 2
    assert s.loc["ref", "fold_se"] == pytest.approx(np.std([-0.02, -0.01, 0, 0.01, 0.02], ddof=1) / np.sqrt(5))


def test_summarize_optimism_on_a_toy_table():
    t = pd.DataFrame(
        {
            "fold": [0, 1, 0, 1],
            "size": [1, 1, 2, 2],
            "winner": [0, 0, 1, 0],
            "inner_best": [0.60, 0.62, 0.65, 0.64],
            "outer": [0.58, 0.60, 0.60, 0.62],
            "outer_average_config": [0.58, 0.60, 0.59, 0.61],
            "outer_hindsight_best": [0.58, 0.60, 0.61, 0.63],
        }
    )
    s = search.summarize_optimism(t).set_index("size")
    assert s.loc[1, "gap"] == pytest.approx(0.02)
    assert s.loc[2, "gap"] == pytest.approx((0.05 + 0.02) / 2)
    assert s.loc[2, "distinct_winners"] == 2


def test_report_mentions_what_it_cannot_show(a):
    md = cli.search_markdown(split(load())[0], a)
    assert md.startswith("# Cross-validation noise and hyperparameter search")
    assert "What this does not show" in md and "test split is still untouched" in md
