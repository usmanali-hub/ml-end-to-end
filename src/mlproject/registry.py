"""Save a fitted model with the facts needed to trust it later, and score new customers with it.

A saved model is a directory ``models/<name>/<version>/`` holding two files:

* ``model.joblib``  the fitted scikit-learn pipeline (preprocessing and estimator together)
* ``metadata.json`` what the model is and where it came from: input columns, training rows, data fingerprint, git
  commit, library versions, the decision threshold, and the SHA-256 of ``model.joblib``

The version is the first 12 hex characters of that SHA-256, so identical bytes always get the same version and a
changed model can never reuse an old version number.

Loading checks the hash and the library versions. A joblib file is a pickle: loading one **runs code** from the file,
so the hash only detects accidental corruption or a mismatch with ``metadata.json``. It does not make an untrusted
file safe, because someone who can replace the model can replace the metadata too. Only load models you made.

Scoring validates the input before the model sees it and refuses the whole batch on any bad row, rather than
guessing values or silently dropping customers.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.pipeline import Pipeline

from .data import CATEGORIES, FEATURES, ID_COLUMN, NUMERIC, ROOT
from .models import COLUMNS

MODELS_DIR = ROOT / "models"
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
SCHEMA_VERSION = 1
LIBRARIES = {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}
MAX_BAD_ROWS_SHOWN = 5


class ModelError(ValueError):
    """A saved model is missing, damaged, or was built with incompatible libraries."""


class InputError(ValueError):
    """The rows to score do not match the schema the model was trained on."""


@dataclass(frozen=True)
class SavedModel:
    pipeline: Pipeline
    metadata: dict[str, Any]
    path: Path

    @property
    def version(self) -> str:
        return str(self.metadata["version"])


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(
    pipeline: Pipeline,
    name: str,
    *,
    train_rows: int,
    data_fingerprint: str,
    commit: str,
    threshold: float = 0.5,
    metrics: dict[str, float] | None = None,
    root: Path | None = None,
) -> Path:
    """Write the model and its metadata; return the version directory. Saving the same bytes twice is a no-op."""
    root = root or MODELS_DIR
    if not name or any(c in name for c in "/\\.") or name.startswith("-"):
        raise ValueError("model name must be non-empty and contain no slashes or dots")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    staging = root / name / f".staging-{datetime.now(timezone.utc).timestamp():.0f}-{id(pipeline)}"
    staging.mkdir(parents=True)
    try:
        joblib.dump(pipeline, staging / MODEL_FILE)
        digest = sha256_of(staging / MODEL_FILE)
        version = digest[:12]
        final = root / name / version
        if final.exists():
            return final
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "version": version,
            "sha256": digest,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_columns": COLUMNS,
            "target": "churned (1 = customer left)",
            "train_rows": train_rows,
            "data_fingerprint": data_fingerprint,
            "git_commit": commit,
            "threshold": threshold,
            "metrics": metrics or {},
            "libraries": LIBRARIES,
        }
        (staging / METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.rename(final)
        return final
    finally:
        if staging.exists():
            for f in staging.iterdir():
                f.unlink()
            staging.rmdir()


def versions(name: str, root: Path | None = None) -> list[str]:
    """Saved versions of a model, oldest first by creation time."""
    root = root or MODELS_DIR
    base = root / name
    if not base.is_dir():
        return []
    found = []
    for d in base.iterdir():
        meta = d / METADATA_FILE
        if d.is_dir() and not d.name.startswith(".") and meta.exists():
            found.append((json.loads(meta.read_text(encoding="utf-8"))["created_utc"], d.name))
    return [v for _, v in sorted(found)]


def _major_minor(version: str) -> tuple[str, ...]:
    return tuple(version.split(".")[:2])


def load_model(name: str, version: str | None = None, root: Path | None = None) -> SavedModel:
    """Load a saved model (the newest when no version is given) after checking its hash and library versions."""
    root = root or MODELS_DIR
    available = versions(name, root)
    if not available:
        raise ModelError(f"no saved model named {name!r} under {root}; run `python -m mlproject train` first")
    version = version or available[-1]
    if version not in available:
        raise ModelError(f"model {name!r} has no version {version!r}; available: {', '.join(available)}")
    path = root / name / version
    metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
    model_file = path / MODEL_FILE
    if not model_file.exists():
        raise ModelError(f"{model_file} is missing")
    if sha256_of(model_file) != metadata["sha256"]:
        raise ModelError(f"{model_file} does not match the hash in its metadata; it is damaged or was replaced")
    for lib, saved in metadata["libraries"].items():
        current = LIBRARIES.get(lib)
        if current is not None and _major_minor(current) != _major_minor(saved):
            raise ModelError(
                f"model was saved with {lib} {saved} but {current} is installed; "
                "retrain with `python -m mlproject train` (pickled models are not portable across versions)"
            )
    pipeline = joblib.load(model_file)
    if not isinstance(pipeline, Pipeline):
        raise ModelError(f"{model_file} does not contain a scikit-learn Pipeline")
    return SavedModel(pipeline, metadata, path)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Check raw customer rows against the training schema and return the model inputs.

    Numeric columns may arrive as text (a CSV); blank ``TotalCharges`` is allowed (a brand-new customer) because the
    model imputes it. Anything else missing, unparseable, out of range or an unseen category raises
    :class:`InputError` naming the rows. Extra columns are ignored.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise InputError(f"missing columns: {missing}")
    if df.empty:
        raise InputError("no rows to score")
    out = df[FEATURES].copy()
    problems: list[str] = []

    def flag(mask: pd.Series, message: str) -> None:
        rows = [int(i) + 2 for i in np.flatnonzero(mask.to_numpy())]  # +2: header line, then 1-based
        if rows:
            shown = ", ".join(map(str, rows[:MAX_BAD_ROWS_SHOWN])) + (" ..." if len(rows) > MAX_BAD_ROWS_SHOWN else "")
            problems.append(f"{message} (data rows {shown}; {len(rows)} in total)")

    for col in NUMERIC:
        text = out[col].astype("string").str.strip()
        blank_ok = col == "TotalCharges"
        parsed = pd.to_numeric(text.where(text != ""), errors="coerce").astype(float)
        absent = text.isna() | (text == "")
        flag(parsed.isna() & ~(absent & blank_ok), f"{col} is missing or not a number")
        out[col] = parsed
    flag(~out["SeniorCitizen"].isin([0, 1]) & out["SeniorCitizen"].notna(), "SeniorCitizen must be 0 or 1")
    flag((out["tenure"] < 0) | (out["tenure"] % 1 != 0), "tenure must be a whole number of months, 0 or more")
    flag(out["MonthlyCharges"] <= 0, "MonthlyCharges must be positive")
    flag(out["TotalCharges"] < 0, "TotalCharges cannot be negative")
    for col, allowed in CATEGORIES.items():
        values = out[col].astype("string").str.strip()
        flag(~values.isin(allowed), f"{col} must be one of {allowed}")
        out[col] = values
    if problems:
        raise InputError("; ".join(problems))
    return out[COLUMNS]


def score_frame(model: SavedModel, df: pd.DataFrame) -> pd.DataFrame:
    """Churn probability and a flag at the model's saved threshold, one row per input row, in input order."""
    x = prepare(df)
    proba = np.asarray(model.pipeline.predict_proba(x)[:, 1], dtype=float)
    threshold = float(model.metadata["threshold"])
    result = pd.DataFrame({"churn_probability": proba.round(6), "flagged": proba >= threshold})
    if ID_COLUMN in df.columns:
        result.insert(0, ID_COLUMN, df[ID_COLUMN].to_numpy())
    result["model_version"] = model.version
    return result
