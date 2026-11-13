import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlproject import __main__ as cli
from mlproject import registry
from mlproject.data import CATEGORIES, NUMERIC, TARGET
from mlproject.models import COLUMNS
from mlproject.preprocess import build_preprocessor


def frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(allowed, n) for c, allowed in CATEGORIES.items()})
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 72, n)
    df["MonthlyCharges"] = rng.uniform(20, 110, n).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
    p = 1 / (1 + np.exp(-(1.2 * (df["Contract"] == "Month-to-month") - 0.04 * df["tenure"])))
    df[TARGET] = (rng.random(n) < p).astype(int)
    df.insert(0, "customerID", [f"id-{i}" for i in range(n)])
    return df


@pytest.fixture
def fitted() -> Pipeline:
    df = frame()
    pipe = Pipeline([("prep", build_preprocessor(list(NUMERIC), list(CATEGORIES))), ("model", LogisticRegression(max_iter=300))])
    return pipe.fit(df[COLUMNS], df[TARGET])


def save(pipe: Pipeline, root, threshold: float = 0.5):
    return registry.save(pipe, "m", train_rows=300, data_fingerprint="abc", commit="deadbeef", threshold=threshold, root=root)


def load_saved(pipe: Pipeline, root):
    save(pipe, root)
    return registry.load_model("m", root=root)


def as_csv_text(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(str)


def test_round_trip_gives_identical_scores_and_records_the_facts(tmp_path, fitted):
    path = save(fitted, tmp_path)
    meta = json.loads((path / registry.METADATA_FILE).read_text())
    assert path.name == meta["version"] == meta["sha256"][:12]
    assert meta["train_rows"] == 300 and meta["git_commit"] == "deadbeef" and meta["input_columns"] == COLUMNS
    model = registry.load_model("m", root=tmp_path)
    df = frame(50, seed=1)
    np.testing.assert_array_equal(model.pipeline.predict_proba(df[COLUMNS]), fitted.predict_proba(df[COLUMNS]))


def test_saving_the_same_model_twice_keeps_one_version(tmp_path, fitted):
    assert save(fitted, tmp_path) == save(fitted, tmp_path)
    assert len(registry.versions("m", tmp_path)) == 1
    assert [p.name for p in (tmp_path / "m").iterdir() if p.name.startswith(".")] == []


def test_a_retrained_model_gets_a_new_version_and_either_can_be_loaded(tmp_path, fitted):
    first = save(fitted, tmp_path)
    df = frame(seed=5)
    other = Pipeline([("prep", build_preprocessor(list(NUMERIC), list(CATEGORIES))), ("model", LogisticRegression(C=0.1, max_iter=300))])
    second = save(other.fit(df[COLUMNS], df[TARGET]), tmp_path)
    assert first != second and len(registry.versions("m", tmp_path)) == 2
    assert registry.load_model("m", root=tmp_path).version in {first.name, second.name}
    assert registry.load_model("m", first.name, root=tmp_path).version == first.name


def test_loading_refuses_a_tampered_file(tmp_path, fitted):
    path = save(fitted, tmp_path)
    with (path / registry.MODEL_FILE).open("ab") as f:
        f.write(b"x")
    with pytest.raises(registry.ModelError, match="does not match the hash"):
        registry.load_model("m", root=tmp_path)


def test_loading_refuses_other_library_versions(tmp_path, fitted):
    path = save(fitted, tmp_path)
    meta_path = path / registry.METADATA_FILE
    meta = json.loads(meta_path.read_text())
    meta["libraries"]["scikit-learn"] = "0.1.0"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(registry.ModelError, match="scikit-learn 0.1.0"):
        registry.load_model("m", root=tmp_path)


def test_missing_model_or_version_is_a_clear_error(tmp_path, fitted):
    with pytest.raises(registry.ModelError, match="no saved model"):
        registry.load_model("m", root=tmp_path)
    save(fitted, tmp_path)
    with pytest.raises(registry.ModelError, match="no version 'nope'"):
        registry.load_model("m", "nope", root=tmp_path)


def test_save_rejects_bad_names_and_thresholds(tmp_path, fitted):
    for bad in ("", "../x", "a/b", "a.b"):
        with pytest.raises(ValueError):
            registry.save(fitted, bad, train_rows=1, data_fingerprint="x", commit="x", root=tmp_path)
    with pytest.raises(ValueError):
        save(fitted, tmp_path, threshold=1.0)


def test_scores_keep_input_order_ids_and_apply_the_saved_threshold(tmp_path, fitted):
    model = load_saved(fitted, tmp_path)
    df = frame(40, seed=2)
    out = registry.score_frame(model, as_csv_text(df))
    assert list(out["customerID"]) == list(df["customerID"])
    assert (out["flagged"] == (out["churn_probability"] >= 0.5)).all()
    assert set(out["model_version"]) == {model.version}
    np.testing.assert_allclose(out["churn_probability"], fitted.predict_proba(df[COLUMNS])[:, 1], atol=1e-6)


def test_a_blank_total_charges_is_allowed_and_scored(tmp_path, fitted):
    model = load_saved(fitted, tmp_path)
    df = as_csv_text(frame(5, seed=3))
    df.loc[0, "TotalCharges"] = " "
    df.loc[0, "tenure"] = "0"
    out = registry.score_frame(model, df)
    assert out["churn_probability"].between(0, 1).all()


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("Contract", "Weekly", "Contract must be one of"),
        ("tenure", "-1", "tenure must be"),
        ("tenure", "abc", "tenure is missing or not a number"),
        ("tenure", "", "tenure is missing or not a number"),
        ("MonthlyCharges", "0", "MonthlyCharges must be positive"),
        ("TotalCharges", "-5", "TotalCharges cannot be negative"),
        ("SeniorCitizen", "2", "SeniorCitizen must be 0 or 1"),
        ("gender", "", "gender must be one of"),
    ],
)
def test_bad_rows_are_refused_and_named(column, value, message):
    df = as_csv_text(frame(6, seed=4))
    df.loc[2, column] = value
    with pytest.raises(registry.InputError, match=message) as err:
        registry.prepare(df)
    assert "data rows 4" in str(err.value)  # index 2 is the 3rd data row, line 4 of the file


def test_missing_columns_and_empty_input_are_refused():
    df = as_csv_text(frame(3))
    with pytest.raises(registry.InputError, match="missing columns"):
        registry.prepare(df.drop(columns=["tenure"]))
    with pytest.raises(registry.InputError, match="no rows"):
        registry.prepare(df.iloc[0:0])


def test_extra_columns_are_ignored():
    df = as_csv_text(frame(3))
    assert list(registry.prepare(df.assign(notes="x", Churn="Yes")).columns) == COLUMNS


def test_predict_command_reports_errors_without_a_traceback(tmp_path, fitted, monkeypatch, capsys):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    save(fitted, tmp_path)
    good = tmp_path / "good.csv"
    as_csv_text(frame(4)).drop(columns=[TARGET]).to_csv(good, index=False)
    out = tmp_path / "out.csv"
    assert cli.run_predict("m", None, str(good), str(out)) == 0
    assert len(pd.read_csv(out)) == 4
    bad = tmp_path / "bad.csv"
    as_csv_text(frame(4)).assign(tenure="x").to_csv(bad, index=False)
    assert cli.run_predict("m", None, str(bad), None) == 1
    assert "tenure is missing or not a number" in capsys.readouterr().out
    assert cli.run_predict("m", None, str(tmp_path / "absent.csv"), None) == 1
    assert cli.run_predict("m", None, None, None) == 2



def test_train_command_fits_on_the_training_split_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    assert cli.run_train(0.4) == 0
    model = registry.load_model(cli.MODEL_NAME, root=tmp_path)
    assert model.metadata["train_rows"] == 5634 and model.metadata["threshold"] == 0.4
    assert "saved" in capsys.readouterr().out
