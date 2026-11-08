from pathlib import Path

import pytest

from mlproject import diskcache


def make_tree(tmp_path: Path, code: str = "x = 1\n", data: str = "a,b\n1,2\n") -> tuple[Path, Path]:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "mod.py").write_text(code)
    d = tmp_path / "data.csv"
    d.write_text(data)
    return src, d


def test_fingerprint_changes_with_source_and_data_and_is_stable_otherwise(tmp_path: Path):
    src, d = make_tree(tmp_path)
    base = diskcache.fingerprint(src, d)
    assert base == diskcache.fingerprint(src, d)
    (src / "mod.py").write_text("x = 2\n")
    assert diskcache.fingerprint(src, d) != base
    (src / "mod.py").write_text("x = 1\n")
    assert diskcache.fingerprint(src, d) == base
    d.write_text("a,b\n1,3\n")
    assert diskcache.fingerprint(src, d) != base
    (src / "extra.py").write_text("y = 1\n")
    assert diskcache.fingerprint(src, d) != base


def test_result_is_computed_once_then_read_from_disk(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MLPROJECT_NO_CACHE", raising=False)
    calls = []

    def compute() -> dict[str, int]:
        calls.append(1)
        return {"answer": 42}

    first = diskcache.disk_cached("demo", tmp_path)(compute)
    assert first() == {"answer": 42} and first() == {"answer": 42}
    assert len(calls) == 1
    again = diskcache.disk_cached("demo", tmp_path)(compute)  # a fresh process: new wrapper, same files
    assert again() == {"answer": 42}
    assert len(calls) == 1  # served from disk
    assert len(list(tmp_path.glob("demo-*.pkl"))) == 1


def test_a_corrupt_entry_is_ignored_and_replaced(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MLPROJECT_NO_CACHE", raising=False)
    first = diskcache.disk_cached("demo", tmp_path)(lambda: [1, 2, 3])
    first()
    entry = next(tmp_path.glob("demo-*.pkl"))
    entry.write_bytes(b"not a pickle")
    calls = []

    def compute() -> list[int]:
        calls.append(1)
        return [1, 2, 3]

    assert diskcache.disk_cached("demo", tmp_path)(compute)() == [1, 2, 3]
    assert len(calls) == 1
    assert entry.read_bytes() != b"not a pickle"


def test_stale_entries_for_other_fingerprints_are_removed(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MLPROJECT_NO_CACHE", raising=False)
    (tmp_path / "demo-oldfingerprint.pkl").write_bytes(b"stale")
    diskcache.disk_cached("demo", tmp_path)(lambda: 7)()
    names = [p.name for p in tmp_path.glob("demo-*.pkl")]
    assert len(names) == 1 and "oldfingerprint" not in names[0]


def test_the_cache_can_be_bypassed(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MLPROJECT_NO_CACHE", "1")
    calls = []

    def compute() -> int:
        calls.append(1)
        return 1

    diskcache.disk_cached("demo", tmp_path)(compute)()
    assert len(calls) == 1 and not list(tmp_path.glob("*.pkl"))


@pytest.mark.parametrize("flag,expected", [("1", False), ("true", False), ("", True), ("0", True)])
def test_enabled_flag(monkeypatch, flag, expected):
    monkeypatch.setenv("MLPROJECT_NO_CACHE", flag)
    assert diskcache.enabled() is expected
