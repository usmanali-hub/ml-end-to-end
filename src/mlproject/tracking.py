"""Experiment tracking with MLflow: one logged run per candidate model, so any result can be traced to its inputs.

The reports in ``reports/`` are the *conclusions*. Tracking is the *ledger*: for every candidate it records the
hyperparameters, the data fingerprint, the code version (git commit), the split and the seed, and the cross-validated
metrics (the mean, the spread, and every fold as a step). Months later, a number can be matched to exactly what
produced it, and two runs can be compared without re-reading code.

Only cross-validation on the training split is logged. The test split is not touched here.

Runs go to a local SQLite store (``.mlflow/mlflow.db``, ignored by git) so nothing needs a server. MLflow is an
optional dependency (``pip install -e ".[tracking]"``); this module is only imported by the ``track`` and ``runs``
commands.
"""

import hashlib
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from .baseline import SEED
from .data import RAW_PATH, ROOT, TARGET
from .trees import cross_validate

EXPERIMENT = "churn-models"
DEFAULT_STORE = ROOT / ".mlflow" / "mlflow.db"
METRICS = ("pr_auc", "roc_auc", "brier", "recall", "precision")


@dataclass(frozen=True)
class Candidate:
    name: str
    params: Mapping[str, str | int | float]
    build: Callable[[], Pipeline]


def git_commit(repo: Path = ROOT) -> str:
    """The current commit (with ``-dirty`` when the working tree has uncommitted changes), or ``unknown``."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{sha[:12]}-dirty" if dirty else sha[:12]


def data_fingerprint(path: Path = RAW_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else "missing"


def store_uri(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def log_runs(train: pd.DataFrame, candidates: list[Candidate], uri: str, experiment: str = EXPERIMENT) -> list[str]:
    """Cross-validate every candidate on the same 5 folds and log one MLflow run each. Returns the run ids."""
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    folds = cross_validate(train, {c.name: c.build for c in candidates})
    ids: list[str] = []
    shared: dict[str, str | int | float] = {"seed": SEED, "cv_folds": 5, "n_train": len(train), "positive_rate": round(float(train[TARGET].mean()), 4)}
    tags = {"git_commit": git_commit(), "data_sha256": data_fingerprint(), "primary_metric": "pr_auc", "split": "training split only; test split untouched"}
    for c in candidates:
        rows = folds[folds["model"] == c.name].sort_values("fold")
        with mlflow.start_run(run_name=c.name) as run:
            mlflow.log_params({**shared, **c.params})
            mlflow.set_tags(tags)
            for metric in METRICS:
                for fold, value in zip(rows["fold"], rows[metric], strict=True):
                    mlflow.log_metric(metric, float(value), step=int(fold))
                mlflow.log_metric(f"cv_{metric}_mean", float(rows[metric].mean()))
                mlflow.log_metric(f"cv_{metric}_std", float(rows[metric].std(ddof=1)))
            mlflow.log_dict({**shared, **c.params}, "params.json")
            ids.append(run.info.run_id)
    return ids


def comparison(uri: str, experiment: str = EXPERIMENT) -> pd.DataFrame:
    """The latest run of each candidate, as a table: its cross-validated PR-AUC, ROC-AUC and Brier score and its commit."""
    import mlflow

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=["attributes.start_time DESC"])
    if runs.empty:
        return pd.DataFrame()
    latest = runs.drop_duplicates("tags.mlflow.runName")
    out = pd.DataFrame(
        {
            "model": latest["tags.mlflow.runName"],
            "pr_auc": latest["metrics.cv_pr_auc_mean"],
            "pr_auc_std": latest["metrics.cv_pr_auc_std"],
            "roc_auc": latest["metrics.cv_roc_auc_mean"],
            "brier": latest["metrics.cv_brier_mean"],
            "git_commit": latest["tags.git_commit"],
        }
    )
    return out.sort_values("pr_auc", ascending=False).reset_index(drop=True)


def markdown_table(table: pd.DataFrame) -> str:
    lines = ["| model | PR-AUC | fold sd | ROC-AUC | Brier |", "|---|---|---|---|---|"]
    lines += [f"| `{r.model}` | {r.pr_auc:.3f} | {r.pr_auc_std:.3f} | {r.roc_auc:.3f} | {r.brier:.3f} |" for r in table.itertuples()]
    return "\n".join(lines)


def lookup(uri: str, run_name: str, experiment: str = EXPERIMENT) -> dict[str, Any]:
    """Params, tags and metrics of the latest run with this name (used to trace a number back to its inputs)."""
    import mlflow

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[experiment], filter_string=f"tags.mlflow.runName = '{run_name}'", order_by=["attributes.start_time DESC"], max_results=1)
    if runs.empty:
        raise KeyError(run_name)
    return {k: v for k, v in runs.iloc[0].to_dict().items() if isinstance(k, str) and k.startswith(("params.", "tags.", "metrics."))}
