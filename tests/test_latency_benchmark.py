import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from mlproject import registry

SPEC = importlib.util.spec_from_file_location("latency_benchmark", Path(__file__).resolve().parents[1] / "scripts" / "latency_benchmark.py")
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_time_calls_reports_ordered_percentiles_and_runs_warmup_plus_repeats():
    calls = []
    p50, p95 = bench.time_calls(lambda: calls.append(1), repeats=50, warmup=5)
    assert len(calls) == 55 and 0 <= p50 <= p95


def test_markdown_has_one_row_per_case():
    table = pd.DataFrame([{"runtime": "a", "batch": "1 customer", "p50_ms": 1.234, "p95_ms": 2.0}, {"runtime": "b", "batch": "9 customers", "p50_ms": 3.0, "p95_ms": 4.5}])
    lines = bench.markdown(table).splitlines()
    assert len(lines) == 4 and lines[2] == "| a | 1 customer | 1.23 | 2.00 |"


def test_main_fails_cleanly_without_a_saved_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["latency_benchmark.py"])
    assert bench.main() == 1
    assert "no saved model" in capsys.readouterr().out


def test_benchmark_covers_validation_and_both_runtimes(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    from mlproject import __main__ as cli

    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    assert cli.run_train(0.5) == 0 and cli.run_export(cli.MODEL_NAME, None) == 0
    table = bench.benchmark(rows=50, repeats=10)
    assert len(table) == 7 and {"input validation only", "ONNX model only"} <= set(table["runtime"])
    one = table[table["batch"] == "1 customer"].set_index("runtime")["p50_ms"].to_dict()
    assert one["ONNX model only"] < one["ONNX runtime + validation"]  # the model is not the whole cost
