"""
Tests for Azure Fabric portability features.

Covers:
  - requirements-fabric.txt validity
  - ParquetOverrideStore CRUD operations
  - get_override_store factory function
  - FabricNotebookAdapter import safety (no Spark required)
"""

import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]


# ── requirements-fabric.txt ──────────────────────────────────────────────────

class TestRequirementsFabric:
    """Verify the managed-runtime candidate excludes locally supplied engines."""

    def _parse_requirements(self, path: Path):
        """Parse a requirements file, returning package names (lowercase)."""
        names = set()
        if not path.exists():
            return names
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                names.update(self._parse_requirements(path.parent / line[3:].strip()))
                continue
            if line.startswith("--"):
                continue
            # Extract package name before version specifier
            for sep in (">=", "<=", "==", "~=", "!=", ">", "<"):
                if sep in line:
                    line = line[:line.index(sep)]
                    break
            names.add(line.strip().lower().replace("-", "_"))
        return names

    def test_fabric_requirements_exist(self):
        path = ROOT / "requirements-fabric.txt"
        assert path.exists(), "requirements-fabric.txt not found"

    def test_fabric_requirements_non_empty(self):
        path = ROOT / "requirements-fabric.txt"
        packages = self._parse_requirements(path)
        assert len(packages) >= 5, f"Expected at least 5 packages, got {len(packages)}"

    @pytest.mark.parametrize("pkg", ["pyspark", "duckdb", "neuralforecast"])
    def test_excludes_heavy_dependencies(self, pkg):
        path = ROOT / "requirements-fabric.txt"
        packages = self._parse_requirements(path)
        assert pkg not in packages, f"{pkg} should not be in Fabric requirements"

    def test_includes_core_packages(self):
        path = ROOT / "requirements-fabric.txt"
        packages = self._parse_requirements(path)
        for pkg in ["polars", "numpy", "pandas", "pyyaml", "statsforecast"]:
            assert pkg in packages, f"{pkg} should be in Fabric requirements"


# ── ParquetOverrideStore ─────────────────────────────────────────────────────

class TestParquetOverrideStore:
    """Test the Parquet-based override store fallback."""

    @pytest.fixture
    def store(self, tmp_path):
        from src.overrides.store import ParquetOverrideStore
        return ParquetOverrideStore(str(tmp_path / "overrides.parquet"))

    def test_add_and_retrieve(self, store):
        oid = store.add_override("SKU_A", "SKU_B", 0.5)
        assert oid.startswith("OVR-")

        result = store.get_all()
        assert len(result) == 1
        assert result["old_sku"][0] == "SKU_A"
        assert result["new_sku"][0] == "SKU_B"
        assert result["proportion"][0] == 0.5

    def test_filter_by_sku(self, store):
        store.add_override("SKU_A", "SKU_B", 0.5)
        store.add_override("SKU_C", "SKU_D", 0.3)

        result = store.get_overrides(old_sku="SKU_A")
        assert len(result) == 1
        assert result["old_sku"][0] == "SKU_A"

        result = store.get_overrides(new_sku="SKU_D")
        assert len(result) == 1
        assert result["new_sku"][0] == "SKU_D"

    def test_delete_override(self, store):
        oid = store.add_override("SKU_A", "SKU_B", 0.5)
        assert store.delete_override(oid)
        assert len(store.get_all()) == 0

    def test_delete_nonexistent(self, store):
        store.add_override("SKU_A", "SKU_B", 0.5)
        assert not store.delete_override("OVR-NOTEXIST")
        assert len(store.get_all()) == 1

    def test_approval_threshold(self, store):
        oid = store.add_override("A", "B", 0.8, approval_threshold=0.5)
        result = store.get_all()
        assert result["status"][0] == "pending_approval"

    def test_auto_approved(self, store):
        oid = store.add_override("A", "B", 0.3, approval_threshold=0.5)
        result = store.get_all()
        assert result["status"][0] == "approved"

    def test_empty_store(self, store):
        result = store.get_all()
        assert result.is_empty()

    def test_close_noop(self, store):
        store.close()  # should not raise
        assert store is not None  # verify store object still valid after close


# ── get_override_store factory ───────────────────────────────────────────────

class TestGetOverrideStore:
    """Test the auto-detecting factory function."""

    def test_parquet_backend(self, tmp_path):
        from src.overrides.store import ParquetOverrideStore, get_override_store
        store = get_override_store(str(tmp_path / "ovr"), backend="parquet")
        assert isinstance(store, ParquetOverrideStore)

    def test_auto_backend_returns_store(self, tmp_path):
        from src.overrides.store import get_override_store
        store = get_override_store(str(tmp_path / "ovr"), backend="auto")
        # Should succeed regardless of whether DuckDB is installed
        assert hasattr(store, "add_override")
        assert hasattr(store, "get_overrides")

    def test_round_trip_with_factory(self, tmp_path):
        from src.overrides.store import get_override_store
        store = get_override_store(str(tmp_path / "ovr"), backend="parquet")
        oid = store.add_override("X", "Y", 0.7)
        result = store.get_all()
        assert len(result) == 1
        assert result.filter(pl.col("override_id") == oid).height == 1


# ── FabricNotebookAdapter import safety ──────────────────────────────────────

class TestFabricAdapterImport:
    """Verify the adapter module can be imported without Spark."""

    def test_module_importable(self):
        """The module itself should import without error."""
        # The adapter wraps Spark but doesn't require it at import time
        from src.fabric import notebook_adapter
        assert hasattr(notebook_adapter, "FabricNotebookAdapter")

    def test_class_exists(self):
        from src.fabric.notebook_adapter import FabricNotebookAdapter

        assert callable(FabricNotebookAdapter)
