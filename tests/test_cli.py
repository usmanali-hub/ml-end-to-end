"""The command-line entry points, end to end on the real training data (the test split is never read)."""

import json

import pandas as pd
import pytest

from mlproject import __main__ as cli
from mlproject import registry
from mlproject.data import ID_COLUMN, load, load_raw
from mlproject.split import split


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def customers(tmp_path_factory):
    raw = load_raw()
    train_ids = set(split(load())[0][ID_COLUMN])
    rows = raw[raw[ID_COLUMN].isin(train_ids)].head(120).drop(columns=["Churn"])
    path = tmp_path_factory.mktemp("csv") / "customers.csv"
    rows.to_csv(path, index=False)
    return path


def test_check_reports_the_data(capsys):
    assert cli.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "data is valid: 7,043 rows" in out and "churn rate 26.5%" in out


def test_train_export_and_predict_agree_across_runtimes(store, customers, tmp_path, capsys):
    assert cli.main(["train"]) == 0
    assert cli.main(["export"]) == 0
    assert "decision agreement 1.0000" in capsys.readouterr().out
    pickled, onnx = tmp_path / "p.csv", tmp_path / "o.csv"
    assert cli.main(["predict", "--input", str(customers), "--output", str(pickled)]) == 0
    assert cli.main(["predict", "--input", str(customers), "--output", str(onnx), "--runtime", "onnx"]) == 0
    a, b = pd.read_csv(pickled), pd.read_csv(onnx)
    assert (a["churn_probability"] - b["churn_probability"]).abs().max() < 1e-4
    assert list(a["flagged"]) == list(b["flagged"]) and list(a[ID_COLUMN]) == list(b[ID_COLUMN])
    meta = json.loads(next(store.glob("churn-logistic/*/metadata.json")).read_text())
    assert meta["sequence"] == 1 and "out-of-fold" in meta["threshold_source"]


def test_predict_prints_to_stdout_when_no_output_is_given(store, customers, capsys):
    assert cli.main(["train", "--threshold", "0.4"]) == 0
    capsys.readouterr()
    assert cli.main(["predict", "--input", str(customers)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "customerID,churn_probability,flagged,model_version" and len(lines) == 121


def test_commands_fail_cleanly_before_a_model_exists(store, customers, capsys):
    assert cli.main(["predict", "--input", str(customers)]) == 1
    assert "no saved model" in capsys.readouterr().out
    assert cli.main(["export"]) == 1
    assert "no saved model" in capsys.readouterr().out
    assert cli.main(["predict", "--input", str(customers), "--runtime", "onnx"]) == 1
    assert cli.main(["predict"]) == 2


def test_runs_with_an_empty_store_says_how_to_start(tmp_path, capsys):
    pytest.importorskip("mlflow")
    assert cli.main(["runs", "--store", str(tmp_path / "none.db")]) == 0
    assert "no runs logged yet" in capsys.readouterr().out


def test_unknown_command_is_an_argument_error():
    with pytest.raises(SystemExit):
        cli.main(["frobnicate"])
