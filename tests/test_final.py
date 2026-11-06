"""Tests for the final-evaluation machinery. They use only slices of the *training* split: the real test split is never scored here."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlproject import __main__ as cli
from mlproject import final, models
from mlproject.data import TARGET, load
from mlproject.split import split


@pytest.fixture(scope="module")
def pseudo() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = split(load())[0]
    fit, held = train.iloc[:3500].reset_index(drop=True), train.iloc[3500:].reset_index(drop=True)
    return fit, held


def builder():
    return models.build(models.Spec(name="l2_c0.03", C=0.03))


def test_bootstrap_interval_brackets_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.25).astype(int)
    s = np.clip(0.25 * y + rng.random(400) * 0.75, 0, 1)
    from sklearn.metrics import average_precision_score

    lo, hi = final.bootstrap_interval(y, s, n=200)
    assert lo < average_precision_score(y, s) < hi
    assert final.bootstrap_interval(y, s, n=200) == (lo, hi)


def test_final_scores_models_and_chooses_cutoff_from_training_only(pseudo):
    fit, held = pseudo
    r = final.evaluate_final(fit, held, {"logistic": builder}, "logistic", {"logistic": 0.67})
    assert 0.0 < r.cutoff < 1.0
    assert r.table.loc["logistic", "ci_low"] < r.table.loc["logistic", "pr_auc"] < r.table.loc["logistic", "ci_high"]
    assert r.at_default.tp + r.at_default.fp + r.at_default.fn + r.at_default.tn == len(held)
    # the cutoff must not depend on the held-out rows: scrambling their labels leaves it unchanged
    scrambled = held.assign(**{TARGET: np.random.default_rng(0).permutation(held[TARGET].to_numpy())})
    r2 = final.evaluate_final(fit, scrambled, {"logistic": builder}, "logistic", {"logistic": 0.67})
    assert r2.cutoff == r.cutoff


def test_markdown_states_the_rules(pseudo):
    fit, held = pseudo
    r = final.evaluate_final(fit, held, {"logistic": builder}, "logistic", {"logistic": 0.67})
    md = final.markdown(r, len(fit))
    assert "(primary)" in md and "did not choose between them" in md and "not tuned on the test split" in md


def test_report_cannot_be_overwritten_without_force(tmp_path: Path):
    path = tmp_path / "final_test.md"
    final.write_once(path, "first")
    with pytest.raises(FileExistsError):
        final.write_once(path, "second")
    assert path.read_text() == "first"
    final.write_once(path, "third", force=True)
    assert path.read_text() == "third"


def test_command_refuses_before_scoring_when_a_report_exists(tmp_path: Path, monkeypatch, capsys):
    existing = tmp_path / "final_test.md"
    existing.write_text("already used")
    monkeypatch.setattr(cli, "FINAL_MD", existing)

    def boom(*a, **k):
        raise AssertionError("must not touch models or data when the report exists")

    monkeypatch.setattr(cli, "cached_analysis", boom)
    monkeypatch.setattr(cli, "load", boom)
    assert cli.main(["final"]) == 1
    assert "already exists" in capsys.readouterr().out
    assert existing.read_text() == "already used"
