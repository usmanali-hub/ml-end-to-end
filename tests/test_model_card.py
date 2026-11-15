import numpy as np
import pandas as pd
import pytest

from mlproject import __main__ as cli
from mlproject import modelcard


def inputs(**over) -> modelcard.CardInputs:
    results = pd.DataFrame(
        {
            "pr_auc": [0.265, 0.674, 0.672],
            "pr_auc_sd": [0.0, 0.018, 0.022],
            "roc_auc": [0.5, 0.850, 0.850],
            "brier": [np.nan, 0.133, 0.134],
        },
        index=["majority", "logistic", "boosting"],
    )
    base = {
        "results": results, "primary": "logistic", "penalty": "l2", "C": 0.03,
        "calibration": {"ece": 0.013, "slope": 1.06, "intercept": 0.03, "brier": 0.133},
        "cutoff": 0.33, "at_cutoff": {"tp": 1085.0, "fp": 812.0, "fn": 410.0, "tn": 3327.0, "precision": 0.572, "recall": 0.726},
        "month_to_month_error_share": 0.85, "month_to_month_customer_share": 0.55, "n_train": 5634, "n_test": 1409, "prevalence": 0.265,
        "n_features": 19, "final_report_exists": False,
    }
    return modelcard.CardInputs(**{**base, **over})


def test_the_card_bolds_the_primary_model_and_marks_missing_numbers():
    text = modelcard.render(inputs())
    assert "| **logistic** | 0.674 ± 0.018 | 0.850 | 0.133 |" in text
    assert "| majority | 0.265 ± 0.000 | 0.500 | n/a |" in text


def test_the_margin_over_the_next_model_is_stated_and_called_a_tie_when_small():
    text = modelcard.render(inputs())
    assert "only by 0.002 over the next best" in text and "ties the more complex models" in text


def test_cutoff_counts_are_reported_from_the_confusion_numbers():
    text = modelcard.render(inputs())
    assert "flags 1,897 of 5,634 customers (34%)" in text
    assert "precision 0.57" in text and "410 churners are missed, 812 non-churners are flagged" in text


def test_test_split_status_follows_whether_the_final_report_exists():
    assert "has **not been used**" in modelcard.render(inputs())
    assert "evaluated once" in modelcard.render(inputs(final_report_exists=True))
    assert "has **not been used**" not in modelcard.render(inputs(final_report_exists=True))


def test_the_card_states_limits_that_must_not_be_dropped():
    text = modelcard.render(inputs())
    for needle in ("Not intended for", "No fairness audit", "Importance is not cause", "cross-validation only", "Do not edit by hand"):
        assert needle in text


def test_committed_model_card_is_up_to_date_and_consistent_with_the_reports():
    card = cli.build_model_card()
    assert cli.MODEL_CARD_PATH.read_text(encoding="utf-8") == card
    assert "| **logistic regression (L2)** | 0.674" in card
    errors_report = cli.ERRORS_MD.read_text(encoding="utf-8")
    assert "0.33" in errors_report and "default cutoff is 0.33" in card
    assert cli.FINAL_REPORT_PATH.exists() == ("has **not been used**" not in card)


def test_render_requires_the_primary_model_to_be_in_the_results():
    with pytest.raises(KeyError):
        modelcard.render(inputs(primary="missing"))
