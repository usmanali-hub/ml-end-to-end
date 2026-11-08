from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlproject import __main__ as cli
from mlproject import calibration
from mlproject.data import TARGET, load
from mlproject.split import split


@pytest.fixture(scope="module")
def a() -> calibration.CalibrationAnalysis:
    return cli.cached_calibration()


@pytest.fixture(scope="module")
def y() -> np.ndarray:
    return split(load())[0][TARGET].to_numpy()


def toy(n: int = 4000, seed: int = 0, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Scores that are perfectly calibrated when ``shift`` is 0: the outcome is drawn with probability equal to the score."""
    rng = np.random.default_rng(seed)
    true_p = rng.beta(2, 5, n)
    y = (rng.random(n) < true_p).astype(int)
    s = 1 / (1 + np.exp(-(np.log(true_p / (1 - true_p)) + shift)))
    return y, s


def test_reliability_bins_partition_the_rows_and_are_ordered():
    y, s = toy()
    r = calibration.reliability(y, s)
    assert r["n"].sum() == len(y) and len(r) == calibration.BINS
    assert (np.diff(r["predicted"].to_numpy()) > 0).all()


def test_calibrated_scores_have_small_ece_and_slope_near_one_and_shifted_scores_do_not():
    y, s = toy()
    good = calibration.metrics(y, s)
    assert good["ece"] < 0.03 and 0.85 < good["slope"] < 1.15 and abs(good["intercept"]) < 0.3
    bad = calibration.metrics(y, np.asarray(toy(shift=1.5)[1]))
    assert bad["ece"] > 0.1 and bad["intercept"] < -1.0  # scores too high: negative intercept
    assert bad["roc_auc"] == pytest.approx(good["roc_auc"])  # a monotone shift does not change the ranking


def test_overconfident_scores_have_slope_below_one():
    y, s = toy()
    stretched = 1 / (1 + np.exp(-2.5 * np.log(s / (1 - s))))
    assert calibration.slope_intercept(y, stretched)[0] < 0.6


def test_platt_scaling_repairs_a_shifted_model_without_changing_its_ranking():
    y, _ = toy()
    shifted = toy(shift=1.5)[1]
    fixed = calibration.recalibrate(y, shifted, "platt")
    assert calibration.expected_calibration_error(calibration.reliability(y, fixed)) < 0.03
    assert abs(float(fixed.mean()) - float(y.mean())) < 0.02
    assert calibration.metrics(y, fixed)["roc_auc"] == pytest.approx(calibration.metrics(y, shifted)["roc_auc"], abs=0.002)


def test_isotonic_creates_ties_and_unknown_method_raises():
    y, s = toy(1500)
    iso = calibration.recalibrate(y, s, "isotonic")
    assert len(np.unique(iso)) < len(np.unique(s)) / 5
    with pytest.raises(ValueError):
        calibration.recalibrate(y, s, "magic")


def test_operating_points_on_a_hand_made_example():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    assert calibration.precision_at_recall(y, s, 2 / 3) == pytest.approx(1.0)  # top two are both churners
    assert calibration.precision_at_recall(y, s, 1.0) == pytest.approx(0.75)  # catching all three needs the top four
    assert calibration.recall_at_false_positive_rate(y, s, 0.0) == pytest.approx(2 / 3)
    assert calibration.recall_at_false_positive_rate(y, s, 0.2) == pytest.approx(1.0)


def test_class_weights_change_calibration_but_not_ranking(a):
    t = a.table
    chosen, bal = t.loc[("logistic_l2_chosen", "raw")], t.loc[("logistic_balanced", "raw")]
    assert bal["roc_auc"] == pytest.approx(chosen["roc_auc"], abs=0.003)
    assert bal["ece"] > 5 * chosen["ece"] and bal["brier"] > chosen["brier"] + 0.01
    assert bal["mean_score"] > a.prevalence + 0.1 and abs(chosen["mean_score"] - a.prevalence) < 0.02


def test_platt_repairs_the_class_weighted_model(a):
    t = a.table
    raw, fixed = t.loc[("logistic_balanced", "raw")], t.loc[("logistic_balanced", "platt")]
    assert fixed["brier"] < raw["brier"] - 0.01 and fixed["ece"] < 0.03
    assert fixed["pr_auc"] == pytest.approx(raw["pr_auc"], abs=0.005)


def test_isotonic_costs_ranking_on_this_data_size(a):
    t = a.table
    assert t.loc[("logistic_l2_chosen", "isotonic"), "pr_auc"] < t.loc[("logistic_l2_chosen", "raw"), "pr_auc"]


def test_every_model_and_variant_is_present_and_pr_auc_is_harder_than_roc(a):
    assert set(a.table.index.get_level_values("variant")) == set(calibration.VARIANTS)
    assert len(a.table) == len(a.scores) * 3
    raw = a.table.xs("raw", level="variant")
    assert (raw["pr_auc"] < raw["roc_auc"]).all()
    assert (raw["precision_at_recall_80"] < raw["precision_at_recall_50"]).all()
    assert (raw["recall_at_fpr_10"] >= raw["recall_at_fpr_05"]).all()


def test_scores_are_out_of_fold_probabilities(a, y):
    for s in a.scores.values():
        assert len(s) == len(y) and ((s >= 0) & (s <= 1)).all()


def test_figure_is_written(a, y, tmp_path: Path):
    out = tmp_path / "figs" / "calibration.png"
    calibration.figure(a, y, out)
    assert out.exists() and out.stat().st_size > 20_000


def test_report_states_the_main_findings(a, y):
    md = cli.calibration_markdown(a, y)
    assert md.startswith("# Ranking and calibration")
    assert "Class weights wreck calibration" in md and "Isotonic regression lowers PR-AUC" in md and "What this does not show" in md
    assert "figures/calibration.png" in md


def test_table_is_a_dataframe_with_expected_columns(a):
    assert isinstance(a.table, pd.DataFrame)
    assert {"roc_auc", "pr_auc", "brier", "log_loss", "ece", "slope", "intercept", "mean_score"} <= set(a.table.columns)
