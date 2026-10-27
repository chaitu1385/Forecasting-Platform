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

import re
import tempfile
import unittest
from datetime import date, datetime
from types import SimpleNamespace

import polars as pl
import pytest

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
    store._conn = SimpleNamespace(execute=lambda query, params: result)
    assert_frame_equal(store.get_all(), expected)


def test_arrow_conversion_error_is_not_hidden_by_legacy_fallback():
    from src.overrides.store import OverrideStore

    def fail():
        raise RuntimeError("Arrow conversion failed")

    result = SimpleNamespace(to_arrow_table=fail)
    store = OverrideStore.__new__(OverrideStore)
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
            oid = store.add_override("OLD-A", "NEW-A", 0.6)
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


if __name__ == "__main__":
    unittest.main()
