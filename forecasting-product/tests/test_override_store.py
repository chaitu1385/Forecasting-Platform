"""
Tests for OverrideStore — src/overrides/store.py.

Covers:
  - add / retrieve / delete overrides (DuckDB path)
  - override_id format
  - SKU filtering (old_sku, new_sku, both)
  - approval logic (auto-approved vs pending)
  - empty store returns empty DataFrame
  - close() without error
"""

import errno
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from src.overrides import store as storage

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("arrow_method", ["to_arrow_table", "fetch_arrow_table"])
@pytest.mark.parametrize("empty", [False, True])
def test_arrow_result_preserves_rows_and_schema_across_duckdb_apis(arrow_method, empty):
    from polars.testing import assert_frame_equal
    from src.overrides.store import OverrideStore

    expected = pl.DataFrame({
        "old_sku": ["B", "A"],
        "effective_date": [date(2024, 1, 1), None],
        "created_at": [datetime(2024, 1, 1, 12, 30), None],
        "proportion": [0.6, None],
    })
    if empty:
        expected = expected.head(0)
    result = SimpleNamespace(**{arrow_method: lambda: expected.to_arrow()})
    store = OverrideStore.__new__(OverrideStore)
    store._use_duckdb = True
    store._conn = SimpleNamespace(execute=lambda query, params: result)
    assert_frame_equal(store.get_all(), expected)


def test_arrow_conversion_error_is_not_hidden_by_legacy_fallback():
    from src.overrides.store import OverrideStore

    def fail():
        raise RuntimeError("Arrow conversion failed")

    result = SimpleNamespace(to_arrow_table=fail)
    store = OverrideStore.__new__(OverrideStore)
    store._use_duckdb = True
    store._conn = SimpleNamespace(execute=lambda query, params: result)
    with pytest.raises(RuntimeError, match="Arrow conversion failed"):
        store.get_all()


class TestOverrideStoreDuckDB(unittest.TestCase):
    """Tests using DuckDB backend (default when duckdb is installed)."""

    def _make_store(self, tmpdir: str):
        from src.overrides.store import OverrideStore

        return OverrideStore(db_path=f"{tmpdir}/overrides.duckdb")

    def test_add_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            store.add_override("OLD-A", "NEW-A", 0.6)
            df = store.get_all()
            self.assertEqual(df.shape[0], 1)
            self.assertEqual(df["old_sku"][0], "OLD-A")
            self.assertEqual(df["new_sku"][0], "NEW-A")
            self.assertAlmostEqual(df["proportion"][0], 0.6)
            store.close()

    def test_override_id_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            oid = store.add_override("X", "Y", 0.5)
            self.assertRegex(oid, r"^OVR-[A-F0-9]{8}$")
            store.close()

    def test_filter_by_old_sku(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            store.add_override("A", "X", 0.3)
            store.add_override("B", "Y", 0.4)
            df = store.get_overrides(old_sku="A")
            self.assertEqual(df.shape[0], 1)
            self.assertEqual(df["old_sku"][0], "A")
            store.close()

    def test_filter_by_new_sku(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            store.add_override("A", "X", 0.3)
            store.add_override("B", "Y", 0.4)
            df = store.get_overrides(new_sku="Y")
            self.assertEqual(df.shape[0], 1)
            self.assertEqual(df["new_sku"][0], "Y")
            store.close()

    def test_filter_by_both_skus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            store.add_override("A", "X", 0.3)
            store.add_override("A", "Y", 0.4)
            store.add_override("B", "X", 0.5)
            df = store.get_overrides(old_sku="A", new_sku="X")
            self.assertEqual(df.shape[0], 1)
            store.close()

    def test_delete_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            oid = store.add_override("A", "X", 0.3)
            ok = store.delete_override(oid)
            self.assertTrue(ok)
            df = store.get_all()
            self.assertEqual(df.shape[0], 0)
            store.close()

    def test_approval_auto_approved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            store.add_override("A", "X", 0.8, approval_threshold=0.0)
            df = store.get_all()
            self.assertEqual(df["status"][0], "approved")
            store.close()

    def test_approval_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            store.add_override("A", "X", 0.8, approval_threshold=0.5)
            df = store.get_all()
            self.assertEqual(df["status"][0], "pending_approval")
            store.close()

    def test_approval_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            store.add_override("A", "X", 0.3, approval_threshold=0.5)
            df = store.get_all()
            self.assertEqual(df["status"][0], "approved")
            store.close()

    def test_get_all_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            df = store.get_all()
            self.assertEqual(df.shape[0], 0)
            store.close()

    def test_close_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            store.close()  # should not raise
            # Verify store can still be inspected after close
            self.assertIsNotNone(store)

    def test_multiple_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            store.add_override("A", "X", 0.3)
            store.add_override("B", "Y", 0.4)
            store.add_override("C", "Z", 0.5)
            df = store.get_all()
            self.assertEqual(df.shape[0], 3)
            store.close()

    def test_scenario_and_ramp_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            store.add_override(
                "A", "X", 0.5,
                scenario="scenario_a",
                ramp_shape="sigmoid",
            )
            df = store.get_all()
            self.assertEqual(df["scenario"][0], "scenario_a")
            self.assertEqual(df["ramp_shape"][0], "sigmoid")
            store.close()

    def test_created_by_and_notes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            store.add_override(
                "A", "X", 0.5,
                created_by="alice",
                notes="test override",
            )
            df = store.get_all()
            self.assertEqual(df["created_by"][0], "alice")
            self.assertEqual(df["notes"][0], "test override")
            store.close()


def _open_store(backend: str, path: Path):
    if backend == "csv":
        with patch.object(storage, "_HAS_DUCKDB", False):
            return storage.OverrideStore(str(path.with_suffix(".duckdb")))
    if backend == "parquet":
        return storage.ParquetOverrideStore(str(path))
    assert storage._HAS_DUCKDB, "The required native profile must include real DuckDB"
    return storage.OverrideStore(str(path))


@pytest.fixture(params=["duckdb", "parquet", "csv"])
def backend(request):
    return request.param


@pytest.fixture
def stored_override(tmp_path, backend):
    path = tmp_path / f"overrides.{backend}"
    store = _open_store(backend, path)
    try:
        yield store, path
    finally:
        store.close()


def _write_fixture(frame, path):
    if path.suffix == ".parquet":
        frame.write_parquet(path)
    else:
        frame.write_csv(path)


@pytest.mark.parametrize("pending", [False, True])
def test_patch_preserves_identity_metadata_other_rows_and_restart(stored_override, backend, pending):
    store, path = stored_override
    override_id = store.add_override(
        "001", "002", 0.75, scenario="custom-scenario", ramp_shape="custom-ramp",
        effective_date="2026-02-28", created_by="original-planner", notes="001",
        approval_threshold=0.5 if pending else 0.0,
    )
    other_id = store.add_override("003", "004", 0.5, notes="untouched")
    approved_at = datetime(2026, 1, 2, 3, 4, 5, 123456)
    if backend == "duckdb":
        store._conn.execute(
            "UPDATE transition_overrides SET approved_by = ?, approved_at = ? WHERE override_id = ?",
            ["original-manager", approved_at, override_id],
        )
    else:
        _write_fixture(store.get_all().with_columns(
            pl.when(pl.col("override_id") == override_id).then(pl.lit("original-manager"))
            .otherwise(pl.col("approved_by")).alias("approved_by"),
            pl.when(pl.col("override_id") == override_id).then(pl.lit(approved_at.isoformat()))
            .otherwise(pl.col("approved_at")).alias("approved_at"),
        ), path)
    before = {row["override_id"]: row for row in store.get_all().to_dicts()}
    assert store.update_override(override_id, {"proportion": 0.25, "notes": "updated"})
    store.close()
    reopened = _open_store(backend, path)
    try:
        after = {row["override_id"]: row for row in reopened.get_all().to_dicts()}
        assert set(after) == {override_id, other_id}
        assert after[override_id] == {**before[override_id], "proportion": 0.25, "notes": "updated"}
        assert after[other_id] == before[other_id]
        assert after[override_id]["status"] == ("pending_approval" if pending else "approved")
        assert after[override_id]["old_sku"] == "001"
    finally:
        reopened.close()


@pytest.mark.parametrize("changes", [
    {"proportion": 0}, {"proportion": 1},
    {"notes": None}, {"effective_date": None}, {"notes": ""},
    {"scenario": ""}, {"ramp_shape": ""},
    {"effective_date": "2028-02-29"},
    {"notes": None, "effective_date": None, "proportion": 0.0},
    {"scenario": "custom'); DELETE FROM transition_overrides; --", "ramp_shape": "custom-shape"},
])
def test_patch_omitted_null_zero_and_parameterized_text(stored_override, changes):
    store, _ = stored_override
    override_id = store.add_override(
        "old", "new", 0.75, notes="before", effective_date="2026-01-02",
    )
    before = store.get_all().row(0, named=True)
    assert store.update_override(override_id, changes)
    observed = store.get_all().row(0, named=True)
    expected = {**before, **changes}
    if isinstance(before["effective_date"], date) and changes.get("effective_date"):
        expected["effective_date"] = date.fromisoformat(changes["effective_date"])
    assert observed == expected


@pytest.mark.parametrize("changes", [
    {}, {"override_id": "replacement"}, {"old_sku": "replacement"}, {"new_sku": "replacement"},
    {"created_by": "replacement"}, {"created_at": "2026-01-01"},
    {"status": "approved"}, {"approved_by": "replacement"}, {"approved_at": None},
    {"unknown_column": 1}, {"notes = NULL --": "injected"},
    {"proportion": None}, {"proportion": True}, {"proportion": "0.5"},
    {"proportion": float("nan")}, {"proportion": float("inf")}, {"proportion": -float("inf")},
    {"proportion": -0.1}, {"proportion": 1.1},
    {"scenario": None}, {"ramp_shape": None}, {"scenario": 3}, {"notes": 7},
    {"effective_date": ""}, {"effective_date": "2026-02-29"},
    {"effective_date": "2026-01-01T12:00:00"}, {"effective_date": 20260101},
])
def test_invalid_patch_does_not_change_records(stored_override, changes):
    store, _ = stored_override
    override_id = store.add_override("old", "new", 0.75, notes="before")
    before = store.get_all()
    with pytest.raises(storage.OverrideValidationError):
        store.update_override(override_id, changes)
    assert_frame_equal(store.get_all(), before)


@pytest.mark.parametrize("populated", [False, True])
def test_missing_mutations_return_false_without_creating_data(stored_override, populated):
    store, path = stored_override
    if populated:
        store.add_override("old", "new", 0.75)
    before = store.get_all()
    assert store.update_override("OVR-MISSING", {"notes": "missing"}) is False
    assert store.delete_override("OVR-MISSING") is False
    assert_frame_equal(store.get_all(), before)
    if not populated:
        assert not path.exists()


def test_duckdb_native_update_failure_rolls_back_all_fields(tmp_path):
    path = tmp_path / "checked.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute(storage._CREATE_TABLE_SQL.replace(
        "notes           VARCHAR,", "notes           VARCHAR CHECK (notes <> 'write-failure'),",
    ))
    connection.close()
    store = _open_store("duckdb", path)
    try:
        override_id = store.add_override("old", "new", 0.75, notes="before")
        before = store.get_all()
        with pytest.raises(duckdb.Error):
            store.update_override(override_id, {"proportion": 0.25, "notes": "write-failure"})
        assert_frame_equal(store.get_all(), before)
        assert store.update_override(override_id, {"notes": "after"})
    finally:
        store.close()
    reopened = _open_store("duckdb", path)
    try:
        row = reopened.get_all().row(0, named=True)
        assert row["override_id"] == override_id
        assert row["proportion"] == 0.75 and row["notes"] == "after"
    finally:
        reopened.close()


def test_duckdb_failure_after_update_before_commit_rolls_back(tmp_path):
    store = _open_store("duckdb", tmp_path / "commit.duckdb")
    override_id = store.add_override("old", "new", 0.75, notes="before")
    connection = store._conn
    before = store.get_all()

    class CommitFailure:
        def execute(self, sql, *args):
            if sql == "COMMIT":
                raise OSError("injected pre-commit failure")
            return connection.execute(sql, *args)

    store._conn = CommitFailure()
    try:
        with pytest.raises(OSError, match="pre-commit"):
            store.update_override(override_id, {"proportion": 0.25})
    finally:
        store._conn = connection
    try:
        assert_frame_equal(store.get_all(), before)
        assert store.update_override(override_id, {"notes": "after"})
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_duckdb_ambiguous_legacy_ids_are_not_mutated(tmp_path, operation):
    path = tmp_path / "duplicates.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute(storage._CREATE_TABLE_SQL.replace("PRIMARY KEY", ""))
    connection.close()
    store = _open_store("duckdb", path)
    try:
        override_id = store.add_override("old", "new", 0.75)
        store._conn.execute("INSERT INTO transition_overrides SELECT * FROM transition_overrides")
        before = store.get_all()
        with pytest.raises(storage.OverrideDataError, match="duplicated"):
            if operation == "update":
                store.update_override(override_id, {"notes": "invalid"})
            else:
                store.delete_override(override_id)
        assert_frame_equal(store.get_all(), before)
    finally:
        store.close()


@pytest.mark.parametrize("backend", ["parquet", "csv"])
@pytest.mark.parametrize("operation", ["add", "update", "delete"])
@pytest.mark.parametrize("failure", ["serialize", "flush", "replace"])
def test_file_failure_preserves_original_bytes_and_only_cleans_owned_temporary(
    stored_override, backend, operation, failure, monkeypatch,
):
    store, path = stored_override
    override_id = store.add_override("old", "new", 0.75, notes="before")
    before = path.read_bytes()
    foreign = path.with_name(f".{path.name}.foreign.tmp")
    foreign.write_bytes(b"not-owned-by-this-mutation")

    def fail_serialization(self, handle):
        handle.write(b"partially-serialized")
        raise OSError("injected serialization failure")

    def fail_flush(descriptor):
        raise OSError("injected flush failure")

    def fail_replace(source, destination):
        assert Path(source).parent == path.parent
        assert Path(destination) == path
        assert Path(source).read_bytes()
        raise OSError("injected replacement failure")

    with monkeypatch.context() as fault:
        if failure == "serialize":
            fault.setattr(pl.DataFrame, f"write_{backend}", fail_serialization)
        elif failure == "flush":
            fault.setattr(storage.os, "fsync", fail_flush)
        else:
            fault.setattr(storage.os, "replace", fail_replace)
        with pytest.raises(OSError, match="injected"):
            if operation == "add":
                store.add_override("another", "newer", 0.4)
            elif operation == "update":
                store.update_override(override_id, {"notes": "after"})
            else:
                store.delete_override(override_id)
    assert path.read_bytes() == before
    assert set(path.parent.glob(f".{path.name}.*.tmp")) == {foreign}
    assert foreign.read_bytes() == b"not-owned-by-this-mutation"
    assert store.get_all().row(0, named=True)["notes"] == "before"
    assert store.update_override(override_id, {"notes": "retry"})


@pytest.mark.parametrize("backend", ["parquet", "csv"])
@pytest.mark.parametrize("malformation", ["duplicate", "missing-column", "null-id", "bad-date", "nan"])
@pytest.mark.parametrize("operation", ["add", "update", "delete"])
def test_malformed_files_fail_explicitly_without_mutation(
    stored_override, malformation, operation,
):
    store, path = stored_override
    override_id = store.add_override("old", "new", 0.75)
    frame = store.get_all()
    if malformation == "duplicate":
        frame = pl.concat([frame, frame])
    elif malformation == "missing-column":
        frame = frame.drop("status")
    elif malformation == "null-id":
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("override_id"))
    elif malformation == "bad-date":
        frame = frame.with_columns(pl.lit("2026-02-30").alias("effective_date"))
    else:
        frame = frame.with_columns(pl.lit(float("nan")).alias("proportion"))
    _write_fixture(frame, path)
    before = path.read_bytes()
    with pytest.raises(storage.OverrideDataError):
        if operation == "add":
            store.add_override("another", "newer", 0.5)
        elif operation == "update":
            store.update_override(override_id, {"notes": "after"})
        else:
            store.delete_override(override_id)
    assert path.read_bytes() == before


@pytest.mark.parametrize("backend", ["parquet", "csv"])
def test_same_canonical_path_mutex_is_nonblocking_and_other_paths_are_independent(
    stored_override, backend,
):
    store, path = stored_override
    override_id = store.add_override("old", "new", 0.75)
    alias = _open_store(backend, path.parent / "unused" / ".." / path.name)
    other_path = path.with_name(f"independent.{backend}")
    other = _open_store(backend, other_path)
    lock_path = path.with_name(path.name + ".lock")
    inode = lock_path.stat().st_ino
    try:
        with ThreadPoolExecutor(max_workers=1) as pool, storage._override_file_lock(path):
            blocked = pool.submit(alias.update_override, override_id, {"notes": "blocked"})
            with pytest.raises(storage.OverrideStoreBusy):
                blocked.result(timeout=5)
            assert pool.submit(other.add_override, "a", "b", 0.5).result(timeout=5)
            with pytest.raises(storage.OverrideStoreBusy):
                store.get_all()
        assert alias.update_override(override_id, {"notes": "explicit retry"})
        assert lock_path.stat().st_ino == inode
    finally:
        alias.close()
        other.close()


def test_unexpected_lock_open_error_is_not_disguised_as_busy(tmp_path, monkeypatch):
    def fail_open(*args):
        raise OSError(errno.EIO, "injected disk error")

    with monkeypatch.context() as fault:
        fault.setattr(storage.os, "open", fail_open)
        with pytest.raises(OSError, match="disk error"):
            with storage._override_file_lock(tmp_path / "overrides.parquet"):
                pytest.fail("The failed lock was acquired")
    with storage._override_file_lock(tmp_path / "overrides.parquet"):
        pass


if __name__ == "__main__":
    unittest.main()
