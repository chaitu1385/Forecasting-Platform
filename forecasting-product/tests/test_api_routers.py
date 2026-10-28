"""Integration tests for new API router endpoints (series, hierarchy, overrides, pipeline, analytics, governance)."""

import io
import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from src.api.app import create_app

import pytest

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
#  Factory helpers
# --------------------------------------------------------------------------- #

def _make_actuals(data_dir: Path, lob="retail", n_series=5, n_weeks=52):
    """Write synthetic actuals Parquet to data_dir/history/{lob}/."""
    rng = np.random.RandomState(42)
    rows = []
    base = date(2023, 1, 2)
    for i in range(n_series):
        for w in range(n_weeks):
            rows.append({
                "series_id": f"sku_{i}",
                "week": base + timedelta(weeks=w),
                "quantity": float(max(0, rng.normal(100, 30))),
                "category": f"cat_{i % 3}",
                "region": f"region_{i % 2}",
            })
    df = pl.DataFrame(rows)
    out_dir = data_dir / "history" / lob
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "actuals.parquet")
    return df


def _make_forecast(data_dir: Path, lob="retail", n_series=5, n_weeks=13):
    """Write synthetic forecast Parquet."""
    rng = np.random.RandomState(42)
    rows = []
    base = date(2024, 1, 1)
    for i in range(n_series):
        for w in range(n_weeks):
            rows.append({
                "series_id": f"sku_{i}",
                "week": base + timedelta(weeks=w),
                "forecast": float(rng.normal(100, 15)),
                "model": "auto_arima",
            })
    df = pl.DataFrame(rows)
    fc_dir = data_dir / "forecasts" / lob
    fc_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(fc_dir / f"forecast_{lob}_2024-01-01.parquet")
    return df


def _make_metrics(metrics_dir: Path, lob="retail", n_series=5, n_weeks=30):
    """Write synthetic metrics Parquet."""
    rng = np.random.RandomState(42)
    rows = []
    base = date(2023, 6, 1)
    actuals = {
        (series, fold, step): float(max(1, rng.normal(100, 20)))
        for series in range(n_series) for fold in range(2) for step in range(n_weeks)
    }
    for model in ("auto_arima", "lgbm_direct", "seasonal_naive"):
        for i in range(n_series):
            for fold in range(2):
                for w in range(n_weeks):
                    actual = actuals[i, fold, w]
                    forecast = actual * (1 + rng.normal(0, 0.15))
                    rows.append({
                        "series_id": f"sku_{i}",
                        "model_id": model,
                        "fold": fold,
                        "run_id": f"metrics-{lob}",
                        "forecast_step": w + 1,
                        "target_week": base + timedelta(weeks=fold * n_weeks + w),
                        "actual": actual,
                        "forecast": forecast,
                        "wmape": abs(actual - forecast) / max(abs(actual), 1e-9),
                        "normalized_bias": (forecast - actual) / max(abs(actual), 1e-9),
                        "lob": lob,
                        "run_type": "backtest",
                    })
    df = pl.DataFrame(rows)
    out_dir = metrics_dir / "backtest" / f"lob={lob}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "metrics.parquet")
    return df


def _make_csv_bytes(df: pl.DataFrame) -> bytes:
    """Convert a Polars DataFrame to CSV bytes for upload."""
    buf = io.BytesIO()
    df.write_csv(buf)
    return buf.getvalue()


def _actuals_csv_bytes(n_series=3, n_weeks=26):
    """Generate actuals CSV bytes for file upload."""
    rng = np.random.RandomState(42)
    rows = []
    base = date(2023, 1, 2)
    for i in range(n_series):
        for w in range(n_weeks):
            rows.append({
                "series_id": f"sku_{i}",
                "week": str(base + timedelta(weeks=w)),
                "quantity": float(max(0, rng.normal(100, 30))),
                "category": f"cat_{i % 2}",
                "region": f"region_{i % 2}",
            })
    return _make_csv_bytes(pl.DataFrame(rows))


# --------------------------------------------------------------------------- #
#  Test cases
# --------------------------------------------------------------------------- #


class TestSeriesRouter(unittest.TestCase):
    """Tests for /series/* endpoints."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = Path(self.tmpdir) / "data"
        self.metrics_dir = Path(self.tmpdir) / "metrics"
        self.data_dir.mkdir(parents=True)
        self.metrics_dir.mkdir(parents=True)
        _make_actuals(self.data_dir, n_series=3, n_weeks=30)

        app = create_app(
            data_dir=str(self.data_dir),
            metrics_dir=str(self.metrics_dir),
            auth_enabled=False,
            dev_mode=True,
            jwt_secret="",
        )
        self.client = TestClient(app)

    def test_list_series(self):
        resp = self.client.get("/series/retail")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["lob"], "retail")
        self.assertEqual(data["series_count"], 3)
        self.assertEqual(len(data["series"]), 3)
        # Each series should have SBC fields
        item = data["series"][0]
        self.assertIn("adi", item)
        self.assertIn("cv2", item)
        self.assertIn("demand_class", item)

    def test_list_series_404(self):
        resp = self.client.get("/series/nonexistent")
        self.assertEqual(resp.status_code, 404)

    def test_detect_breaks_from_lob(self):
        resp = self.client.post("/series/breaks?lob=retail&method=cusum&penalty=3.0")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_series", data)
        self.assertIn("series_with_breaks", data)
        self.assertIn("total_breaks", data)

    def test_detect_breaks_with_upload(self):
        csv = _actuals_csv_bytes(n_series=2, n_weeks=30)
        resp = self.client.post(
            "/series/breaks?method=cusum",
            files={"file": ("test.csv", csv, "text/csv")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_series", data)

    def test_cleansing_audit_from_lob(self):
        resp = self.client.post("/series/cleansing-audit?lob=retail")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_series", data)
        self.assertIn("total_outliers", data)
        self.assertIn("outlier_pct", data)
        self.assertIn("rows_modified", data)

    def test_regressor_screen_from_lob(self):
        resp = self.client.post("/series/regressor-screen?lob=retail&target_col=quantity")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("screened_columns", data)
        self.assertIn("dropped_columns", data)

    def test_regressor_screen_no_data(self):
        resp = self.client.post("/series/regressor-screen")
        self.assertEqual(resp.status_code, 400)


class TestOverridesRouter(unittest.TestCase):
    """Tests for /overrides/* CRUD endpoints."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = Path(self.tmpdir) / "data"
        self.data_dir.mkdir(parents=True)

        app = create_app(
            data_dir=str(self.data_dir),
            metrics_dir=str(self.data_dir / "metrics"),
            auth_enabled=False,
            dev_mode=True,
            jwt_secret="",
        )
        self.client = TestClient(app)

    def test_list_overrides_empty(self):
        resp = self.client.get("/overrides")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["count"], 0)

    def test_create_and_list_override(self):
        body = {
            "old_sku": "SKU_OLD_1",
            "new_sku": "SKU_NEW_1",
            "proportion": 0.75,
            "scenario": "manual",
            "ramp_shape": "linear",
        }
        resp = self.client.post("/overrides", json=body)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "created")
        override_id = data["override_id"]

        # List
        resp = self.client.get("/overrides")
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(resp.json()["count"], 1)

        # Delete
        resp = self.client.delete(f"/overrides/{override_id}")
        self.assertEqual(resp.status_code, 200)


class TestHierarchyRouter(unittest.TestCase):
    """Tests for /hierarchy/* endpoints."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        app = create_app(
            data_dir=str(Path(self.tmpdir) / "data"),
            metrics_dir=str(Path(self.tmpdir) / "metrics"),
            auth_enabled=False,
            dev_mode=True,
            jwt_secret="",
        )
        self.client = TestClient(app)

    def _make_hierarchy_csv(self):
        rows = []
        for cat in ("A", "B"):
            for sub in ("X", "Y"):
                for i in range(3):
                    sid = f"{cat}_{sub}_{i}"
                    rows.append({
                        "series_id": sid,
                        "category": cat,
                        "subcategory": f"{cat}_{sub}",
                        "week": "2024-01-01",
                        "quantity": 100.0,
                        "forecast": 95.0,
                    })
        return _make_csv_bytes(pl.DataFrame(rows))

    def test_build_hierarchy(self):
        csv = self._make_hierarchy_csv()
        resp = self.client.post(
            "/hierarchy/build?levels=category,subcategory,series_id&id_column=series_id&name=test",
            files={"file": ("hier.csv", csv, "text/csv")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "test")
        self.assertIn("level_stats", data)
        self.assertGreater(data["total_nodes"], 0)

    def test_aggregate_hierarchy(self):
        csv = self._make_hierarchy_csv()
        resp = self.client.post(
            "/hierarchy/aggregate?levels=category,subcategory,series_id&target_level=category&value_columns=quantity",
            files={"file": ("hier.csv", csv, "text/csv")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["target_level"], "category")
        self.assertGreater(data["total_rows"], 0)

    def test_build_missing_levels(self):
        csv = _make_csv_bytes(pl.DataFrame({"series_id": ["a"], "quantity": [1.0]}))
        resp = self.client.post(
            "/hierarchy/build?levels=missing_col&id_column=series_id",
            files={"file": ("test.csv", csv, "text/csv")},
        )
        self.assertEqual(resp.status_code, 400)


class TestPipelineRouter(unittest.TestCase):
    """Tests for /pipeline/* endpoints."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = Path(self.tmpdir) / "data"
        self.metrics_dir = Path(self.tmpdir) / "metrics"
        self.data_dir.mkdir(parents=True)
        self.metrics_dir.mkdir(parents=True)

        app = create_app(
            data_dir=str(self.data_dir),
            metrics_dir=str(self.metrics_dir),
            auth_enabled=False,
            dev_mode=True,
            jwt_secret="",
        )
        self.client = TestClient(app)

    def test_manifests_empty(self):
        resp = self.client.get("/pipeline/manifests")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["count"], 0)

    def test_costs_empty(self):
        resp = self.client.get("/pipeline/costs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["count"], 0)

    def test_analyze_multi_file(self):
        # Create two CSV files
        ts_df = pl.DataFrame({
            "series_id": ["a", "a", "b", "b"],
            "week": ["2024-01-01", "2024-01-08", "2024-01-01", "2024-01-08"],
            "quantity": [100.0, 110.0, 200.0, 190.0],
        })
        dim_df = pl.DataFrame({
            "series_id": ["a", "b"],
            "category": ["cat1", "cat2"],
            "region": ["east", "west"],
        })

        ts_csv = _make_csv_bytes(ts_df)
        dim_csv = _make_csv_bytes(dim_df)

        resp = self.client.post(
            "/pipeline/analyze-multi-file?lob_name=test",
            files=[
                ("files", ("timeseries.csv", ts_csv, "text/csv")),
                ("files", ("dimensions.csv", dim_csv, "text/csv")),
            ],
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("profiles", data)
        self.assertEqual(len(data["profiles"]), 2)


class TestAnalyticsRouter(unittest.TestCase):
    """Tests for /metrics/*/fva, /metrics/*/calibration, /forecast/compare, /forecast/constrain."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = Path(self.tmpdir) / "data"
        self.metrics_dir = Path(self.tmpdir) / "metrics"
        self.data_dir.mkdir(parents=True)
        self.metrics_dir.mkdir(parents=True)
        _make_actuals(self.data_dir, n_series=3, n_weeks=52)
        _make_forecast(self.data_dir, n_series=3)
        _make_metrics(self.metrics_dir, n_series=3)

        app = create_app(
            data_dir=str(self.data_dir),
            metrics_dir=str(self.metrics_dir),
            auth_enabled=False,
            dev_mode=True,
            jwt_secret="",
        )
        self.client = TestClient(app)

    def test_fva(self):
        resp = self.client.get("/metrics/retail/fva")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["lob"], "retail")

    def test_fva_404(self):
        resp = self.client.get("/metrics/nonexistent/fva")
        self.assertEqual(resp.status_code, 404)

    def test_constrain_forecast(self):
        forecast_df = pl.DataFrame({
            "series_id": ["a", "a", "b", "b"],
            "week": ["2024-01-01", "2024-01-08", "2024-01-01", "2024-01-08"],
            "forecast": [100.0, -5.0, 200.0, 300.0],
        })
        csv = _make_csv_bytes(forecast_df)

        resp = self.client.post(
            "/forecast/constrain?min_demand=0&max_capacity=250",
            files={"file": ("forecast.csv", csv, "text/csv")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["rows_modified"], 0)
        self.assertEqual(data["constraints_applied"]["min_demand"], 0.0)
        self.assertEqual(data["constraints_applied"]["max_capacity"], 250.0)

    def test_compare_forecasts(self):
        model_df = pl.DataFrame({
            "series_id": ["a", "a"],
            "week": ["2024-01-01", "2024-01-08"],
            "forecast": [100.0, 110.0],
        })
        ext_df = pl.DataFrame({
            "series_id": ["a", "a"],
            "week": ["2024-01-01", "2024-01-08"],
            "forecast": [95.0, 115.0],
        })

        resp = self.client.post(
            "/forecast/compare?external_name=external",
            files=[
                ("model_file", ("model.csv", _make_csv_bytes(model_df), "text/csv")),
                ("external_file", ("ext.csv", _make_csv_bytes(ext_df), "text/csv")),
            ],
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("comparison", data)
        self.assertIn("summary", data)


class TestGovernanceRouter(unittest.TestCase):
    """Tests for /governance/* endpoints."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = Path(self.tmpdir) / "data"
        self.data_dir.mkdir(parents=True)

        app = create_app(
            data_dir=str(self.data_dir),
            metrics_dir=str(self.data_dir / "metrics"),
            auth_enabled=False,
            dev_mode=True,
            jwt_secret="",
        )
        self.client = TestClient(app)

    def test_list_model_cards_empty(self):
        resp = self.client.get("/governance/model-cards")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["count"], 0)

    def test_get_model_card_404(self):
        resp = self.client.get("/governance/model-cards/nonexistent")
        self.assertEqual(resp.status_code, 404)

    def test_lineage_empty(self):
        resp = self.client.get("/governance/lineage")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["count"], 0)

    def test_export_invalid_type(self):
        resp = self.client.post("/governance/export/invalid?lob=retail")
        self.assertEqual(resp.status_code, 400)


class TestRouterRegistration(unittest.TestCase):
    """Verify all routers are properly registered."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        app = create_app(
            data_dir=str(Path(self.tmpdir) / "data"),
            metrics_dir=str(Path(self.tmpdir) / "metrics"),
            auth_enabled=False,
            dev_mode=True,
            jwt_secret="",
        )
        self.client = TestClient(app)
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        self.routes = response.json()["paths"]

    def test_series_routes_registered(self):
        self.assertIn("/series/{lob}", self.routes)
        self.assertIn("/series/breaks", self.routes)
        self.assertIn("/series/cleansing-audit", self.routes)
        self.assertIn("/series/regressor-screen", self.routes)

    def test_hierarchy_routes_registered(self):
        self.assertIn("/hierarchy/build", self.routes)
        self.assertIn("/hierarchy/aggregate", self.routes)
        self.assertIn("/hierarchy/reconcile", self.routes)

    def test_sku_mapping_routes_registered(self):
        self.assertIn("/sku-mapping/phase1", self.routes)
        self.assertIn("/sku-mapping/phase2", self.routes)

    def test_override_routes_registered(self):
        self.assertIn("/overrides", self.routes)
        self.assertIn("/overrides/{override_id}", self.routes)

    def test_pipeline_routes_registered(self):
        self.assertIn("/pipeline/backtest", self.routes)
        self.assertIn("/pipeline/forecast", self.routes)
        self.assertIn("/pipeline/manifests", self.routes)
        self.assertIn("/pipeline/costs", self.routes)
        self.assertIn("/pipeline/analyze-multi-file", self.routes)

    def test_analytics_routes_registered(self):
        self.assertIn("/metrics/{lob}/fva", self.routes)
        self.assertIn("/metrics/{lob}/calibration", self.routes)
        self.assertIn("/metrics/{lob}/shap", self.routes)
        self.assertIn("/forecast/decompose", self.routes)
        self.assertIn("/forecast/compare", self.routes)
        self.assertIn("/forecast/constrain", self.routes)

    def test_governance_routes_registered(self):
        self.assertIn("/governance/model-cards", self.routes)
        self.assertIn("/governance/model-cards/{model_name}", self.routes)
        self.assertIn("/governance/lineage", self.routes)
        self.assertIn("/governance/export/{report_type}", self.routes)

    def test_health_still_works(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
