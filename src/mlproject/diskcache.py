"""A small on-disk cache for the slow analyses, keyed on everything that could change their result.

The analyses behind the reports take minutes. Their results are a pure function of the source code, the data and the
library versions, so they are stored under a fingerprint of exactly those inputs: change any source file, the data
file, or a library version and the fingerprint changes, the old entry is ignored and the analysis runs again.
A corrupt or unreadable entry is ignored too. Set ``MLPROJECT_NO_CACHE=1`` to bypass the cache entirely.

This saves time when nothing relevant changed (a documentation edit, a CI rerun, running the report command and then
the tests). It cannot make a real code change cheaper: that still recomputes.
"""

import hashlib
import importlib.metadata
import os
import pickle
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TypeVar

from .data import RAW_PATH, ROOT

T = TypeVar("T")
CACHE_DIR = ROOT / ".cache" / "analyses"
LIBRARIES = ("numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "xgboost", "torch")


def enabled() -> bool:
    return os.environ.get("MLPROJECT_NO_CACHE", "") not in {"1", "true", "yes"}


def fingerprint(source_dir: Path | None = None, data_path: Path | None = None) -> str:
    """Hash of every source file, the data file and the versions of the libraries the analyses use."""
    src = source_dir if source_dir is not None else Path(__file__).resolve().parent
    data = data_path if data_path is not None else RAW_PATH
    h = hashlib.sha256()
    for path in sorted(src.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update(data.read_bytes() if data.exists() else b"no data file")
    for lib in LIBRARIES:
        try:
            h.update(f"{lib}=={importlib.metadata.version(lib)}".encode())
        except importlib.metadata.PackageNotFoundError:
            h.update(f"{lib} missing".encode())
    return h.hexdigest()[:16]


def disk_cached(name: str, cache_dir: Path | None = None) -> Callable[[Callable[[], T]], Callable[[], T]]:
    """Cache a zero-argument analysis in memory and on disk under ``name`` and the current fingerprint."""

    def decorate(fn: Callable[[], T]) -> Callable[[], T]:
        @cache
        def wrapper() -> T:
            if not enabled():
                return fn()
            directory = cache_dir if cache_dir is not None else CACHE_DIR
            path = directory / f"{name}-{fingerprint()}.pkl"
            if path.exists():
                try:
                    with path.open("rb") as f:
                        loaded: T = pickle.load(f)  # our own cache file, keyed on a fingerprint of our own code
                        return loaded
                except Exception:  # noqa: BLE001 - unpickling can raise many types; any failure means recompute
                    path.unlink(missing_ok=True)
            result = fn()
            directory.mkdir(parents=True, exist_ok=True)
            for old in directory.glob(f"{name}-*.pkl"):
                old.unlink(missing_ok=True)  # entries for other fingerprints are stale
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump(result, f)
            tmp.replace(path)
            return result

        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorate
