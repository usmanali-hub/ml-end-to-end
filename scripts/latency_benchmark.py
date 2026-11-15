"""Measure scoring latency of the saved model: pickled pipeline vs ONNX, one customer at a time and in batches.

Uses real *training* rows (never the test split). Run `python -m mlproject train && python -m mlproject export` first, or `make demo`.
Prints a Markdown table of milliseconds per call (p50 and p95) so the numbers in the README can be reproduced on any machine.
"""

import argparse
import statistics
import time
from collections.abc import Callable

import pandas as pd

from mlproject import registry
from mlproject.data import TARGET, load
from mlproject.split import split

MODEL = "churn-logistic"


def time_calls(fn: Callable[[], object], repeats: int, warmup: int = 10) -> tuple[float, float]:
    """p50 and p95 of ``repeats`` calls, in milliseconds, after ``warmup`` untimed calls."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return statistics.median(samples), samples[min(len(samples) - 1, int(0.95 * len(samples)))]


def benchmark(rows: int = 1000, repeats: int = 200) -> pd.DataFrame:
    train, _ = split(load())
    customers = train.drop(columns=[TARGET]).head(rows).astype(str)
    one, many = customers.iloc[:1], customers
    pickled = registry.load_model(MODEL)
    onnx = registry.load_onnx(MODEL)
    ready = registry.prepare(one)  # validated once, so the next two rows separate checking from predicting
    cases: list[tuple[str, str, Callable[[], object]]] = [
        ("input validation only", "1 customer", lambda: registry.prepare(one)),
        ("ONNX model only", "1 customer", lambda: onnx.proba(ready)),
        ("pickled pipeline model only", "1 customer", lambda: pickled.pipeline.predict_proba(ready)),
        ("pickled pipeline + validation", "1 customer", lambda: registry.score_frame(pickled, one)),
        ("ONNX runtime + validation", "1 customer", lambda: registry.score_frame_onnx(onnx, one)),
        ("pickled pipeline + validation", f"{rows} customers", lambda: registry.score_frame(pickled, many)),
        ("ONNX runtime + validation", f"{rows} customers", lambda: registry.score_frame_onnx(onnx, many)),
    ]
    out = []
    for runtime, size, fn in cases:
        p50, p95 = time_calls(fn, repeats if size == "1 customer" else max(10, repeats // 10))
        out.append({"runtime": runtime, "batch": size, "p50_ms": p50, "p95_ms": p95})
    return pd.DataFrame(out)


def markdown(table: pd.DataFrame) -> str:
    lines = ["| runtime | batch | p50 (ms) | p95 (ms) |", "|---|---|---|---|"]
    for row in table.to_dict("index").values():
        lines.append(f"| {row['runtime']} | {row['batch']} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000, help="batch size for the batch case")
    parser.add_argument("--repeats", type=int, default=200, help="timed calls for the single-customer case")
    args = parser.parse_args()
    try:
        table = benchmark(args.rows, args.repeats)
    except registry.ModelError as err:
        print(f"error: {err}")
        return 1
    print("Scoring latency in milliseconds per call (in-process; each row says whether input validation is included)\n")
    print(markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
