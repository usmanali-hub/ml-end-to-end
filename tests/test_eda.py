import json
import math
import re
from pathlib import Path

import pandas as pd
import pytest

from mlproject import eda
from mlproject.data import FEATURES, TARGET, load
from mlproject.split import split


@pytest.fixture(scope="module")
def train():
    return split(load())[0]


def test_wilson_interval_known_values():
    lo, hi = eda.wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = eda.wilson_interval(5, 10)
    assert (lo, hi) == (pytest.approx(0.2366, abs=1e-3), pytest.approx(0.7634, abs=1e-3))
    assert math.isnan(eda.wilson_interval(0, 0)[0])


@pytest.mark.parametrize("k,n", [(1, 3), (50, 100), (99, 100), (0, 1), (1, 1), (400, 1000)])
def test_wilson_interval_contains_the_rate_and_stays_in_bounds(k, n):
    lo, hi = eda.wilson_interval(k, n)
    assert 0 <= lo <= k / n <= hi <= 1


def test_wider_interval_for_fewer_customers():
    narrow = eda.wilson_interval(250, 1000)
    wide = eda.wilson_interval(25, 100)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_rate_table_matches_manual_counts(train):
    table = eda.rate_table(train, "Contract")
    for contract, row in table.iterrows():
        subset = train[train["Contract"] == contract]
        assert row["n"] == len(subset) and row["churned"] == subset[TARGET].sum()
        assert row["rate"] == pytest.approx(subset[TARGET].mean())
        assert row["lo"] <= row["rate"] <= row["hi"]
    assert table["n"].sum() == len(train)


def test_binned_rate_table_is_right_closed_and_complete(train):
    table = eda.rate_table(train, "tenure", eda.TENURE_BINS, eda.TENURE_LABELS)
    assert list(table.index) == eda.TENURE_LABELS
    assert table["n"].sum() == len(train)  # tenure 0 lands in the first bin, nobody is dropped
    assert table.loc["0-6", "n"] == (train["tenure"] <= 6).sum()
    assert table.loc["7-12", "n"] == ((train["tenure"] > 6) & (train["tenure"] <= 12)).sum()


def test_stratified_rates_blank_small_cells():
    df = pd.DataFrame({"a": ["x"] * 40 + ["y"] * 5, "b": ["p"] * 45, TARGET: [1, 0] * 20 + [1] * 5})
    rate, size = eda.stratified_rates(df, "a", "b", min_n=30)
    assert rate.loc["x", "p"] == 0.5 and size.loc["y", "p"] == 5
    assert math.isnan(rate.loc["y", "p"])


def test_cramers_v_hand_computed_table():
    # [[10, 20], [30, 40]]: chi-square = n (ad - bc)^2 / (r1 r2 c1 c2) = 0.7937..., V = sqrt(chi2 / n)
    a = pd.Series(["u"] * 30 + ["v"] * 70)
    b = pd.Series(["p"] * 10 + ["q"] * 20 + ["p"] * 30 + ["q"] * 40)
    assert eda.cramers_v(a, b) == pytest.approx(math.sqrt(100 * 200**2 / (30 * 70 * 40 * 60) / 100), abs=1e-9)
    assert eda.cramers_v(a, b) == pytest.approx(eda.cramers_v(b, a))  # symmetric


def test_cramers_v_extremes():
    s = pd.Series(["a", "b", "a", "b", "a", "b"] * 10)
    assert eda.cramers_v(s, s) == pytest.approx(1.0)
    assert eda.cramers_v(pd.Series(["a"] * 20), pd.Series(["x", "y"] * 10)) == 0.0  # a constant has no association


def test_association_ranking_covers_every_feature_and_is_sorted(train):
    r = eda.association_ranking(train)
    assert set(r["feature"]) == set(FEATURES) and len(r) == 19
    assert r["cramers_v"].is_monotonic_decreasing
    assert r["cramers_v"].between(0, 1).all()
    assert r.loc[0, "feature"] == "Contract"
    assert "gender" in set(r.tail(3)["feature"])  # gender carries no signal about churn


def test_missing_values_are_only_the_new_customers(train):
    m = eda.missing_summary(train)
    assert list(m.index) == ["TotalCharges"]
    assert m.loc["TotalCharges", "missing"] == (train["tenure"] == 0).sum()


def test_findings_quoted_in_the_notebook_are_true(train):
    """The notebook states these numbers in prose; if the data or split changes, this test fails first."""
    c = eda.rate_table(train, "Contract")["rate"]
    assert (round(c["Month-to-month"], 3), round(c["One year"], 3), round(c["Two year"], 3)) == (0.427, 0.111, 0.029)
    i = eda.rate_table(train, "InternetService")["rate"]
    assert (round(i["Fiber optic"], 3), round(i["DSL"], 3), round(i["No"], 3)) == (0.421, 0.187, 0.072)
    t = eda.rate_table(train, "tenure", eda.TENURE_BINS, eda.TENURE_LABELS)["rate"]
    assert round(t["0-6"], 3) == 0.522 and round(t["49-72"], 3) == 0.099
    g = eda.rate_table(train, "gender")["rate"]
    assert abs(g["Female"] - g["Male"]) < 0.01
    rate, _ = eda.stratified_rates(train, "Contract", "InternetService")
    assert round(rate.loc["Month-to-month", "Fiber optic"], 3) == 0.551 and round(rate.loc["Month-to-month", "DSL"], 3) == 0.318
    m2m = train[train["Contract"] == "Month-to-month"]
    p = eda.rate_table(m2m, "PaymentMethod")["rate"]
    assert round(p["Electronic check"], 3) == 0.542
    assert max(p["Bank transfer (automatic)"], p["Credit card (automatic)"], p["Mailed check"]) < 0.34
    dsl = eda.rate_table(m2m[m2m["InternetService"] == "DSL"], "MonthlyCharges", eda.CHARGE_BINS, eda.CHARGE_LABELS)["rate"]
    assert dsl["<=35"] > dsl["36-70"] > dsl["71-90"]  # within DSL, higher bills do not mean more churn


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_committed_with_outputs_and_no_errors():
    cells = [c for c in _notebook()["cells"] if c["cell_type"] == "code"]
    assert cells and all(c["outputs"] is not None for c in cells)
    outputs = [o for c in cells for o in c["outputs"]]
    assert not [o for o in outputs if o.get("output_type") == "error"]
    assert sum("image/png" in o.get("data", {}) for o in outputs) == 5  # every chart was rendered


def test_notebook_only_ever_touches_the_training_split():
    source = "\n".join("".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "code")
    assert "train, _ = split(load())" in source  # the test half is thrown away immediately
    assert not re.search(r"\btest\b", source.replace("# the second value is the held-out test set; it is deliberately never used here", ""))


def test_notebook_prose_numbers_match_the_data_it_prints(train):
    text = "\n".join("".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "markdown")
    missing = int(train["TotalCharges"].isna().sum())
    assert f"only missing values are the {missing} brand-new customers" in text
    assert f"{len(train):,} *training* rows" in text
