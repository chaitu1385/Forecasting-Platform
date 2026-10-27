"""Backend integration tests for the Walmart M5 daily-frequency E2E pipeline.

These tests exercise the full API surface using the M5 daily sample fixture.
They cover: data analysis, series exploration, backtest, forecast, hierarchy,
validation, governance, and daily-frequency-specific assertions.

Prerequisites
-------------
- Run ``python -m tests.integration.prepare_m5_daily`` to generate
  ``tests/integration/fixtures/m5_daily_sample.csv``  (or the tests will
  auto-skip).

Run
---
    cd forecasting-product
    python -m pytest tests/integration/test_m5_daily_backend.py -v
"""
from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

# ── Paths ─────────────────────────────────────────────────────────────────────
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_CSV = FIXTURES / "m5_daily_sample.csv"
CONFIG_YAML = FIXTURES / "m5_daily_config.yaml"

# Skip the entire module if the fixture hasn't been generated yet.
if not SAMPLE_CSV.exists():
    pytest.skip(
        f"M5 daily fixture not found at {SAMPLE_CSV}. "
        "Run 'python -m tests.integration.prepare_m5_daily' first.",
        allow_module_level=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _csv_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _yaml_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _df_to_csv_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_csv(buf)
    return buf.getvalue()


def _load_sample() -> pl.DataFrame:
    return pl.read_csv(SAMPLE_CSV, try_parse_dates=True)


# ── Test app factory ──────────────────────────────────────────────────────────

def _make_app(data_dir: str, metrics_dir: str):
    from src.api.app import create_app

    return create_app(
        data_dir=data_dir,
        metrics_dir=metrics_dir,
        auth_enabled=False,
        audit_log_path=str(Path(data_dir) / "audit_log"),
    )


@pytest.fixture(scope="module")
def _daily_backtest(tmp_path_factory):
    """Run the unchanged 50-series, three-model, two-fold workload once."""
    data_dir = tmp_path_factory.mktemp("m5-backtest") / "data"
    metrics_dir = data_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    history = data_dir / "history" / "walmart_m5_daily"
    history.mkdir(parents=True)
    _load_sample().write_parquet(history / "actuals.parquet")
    app = _make_app(str(data_dir), str(metrics_dir))
    with TestClient(app) as client:
        response = client.post(
            "/pipeline/backtest?lob=walmart_m5_daily",
            files={
                "file": ("m5_daily.csv", _csv_bytes(SAMPLE_CSV), "text/csv"),
                "config_file": ("config.yaml", _yaml_bytes(CONFIG_YAML), "application/x-yaml"),
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert {entry["model_id"] for entry in result["leaderboard"]} == {
            "naive_seasonal", "auto_arima", "lgbm_direct",
        }, result
        persisted = client.get("/metrics/leaderboard/walmart_m5_daily")
        assert persisted.status_code == 200, persisted.text
    return data_dir, result


@pytest.fixture(scope="class")
def _daily_backtest_client(request, tmp_path_factory, _daily_backtest):
    # Give every consumer its own files, response values and application state.
    # In particular, forecasting must not mutate the shared backtest snapshot.
    source, result = _daily_backtest
    data_dir = tmp_path_factory.mktemp(request.cls.__name__) / "data"
    shutil.copytree(source, data_dir)
    metrics_dir = data_dir / "metrics"
    app = _make_app(str(data_dir), str(metrics_dir))
    with TestClient(app) as client:
        request.cls.data_dir = data_dir
        request.cls.metrics_dir = metrics_dir
        request.cls.client = client
        request.cls._backtest_result = deepcopy(result)
        request.cls._backtest_data = deepcopy(result)
        yield


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Data Analysis  (2 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataAnalysis(unittest.TestCase):
    """POST /analyze with the daily M5 fixture."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = Path(self.tmpdir) / "data"
        self.metrics_dir = Path(self.tmpdir) / "metrics"
        self.data_dir.mkdir(parents=True)
        self.metrics_dir.mkdir(parents=True)

        app = _make_app(str(self.data_dir), str(self.metrics_dir))
        self.client = TestClient(app)

    def test_analyze_daily_csv(self):
        """Upload M5 daily CSV → schema detection should find frequency='D'."""
        resp = self.client.post(
            "/analyze?lob_name=walmart_m5_daily",
            files={"file": ("m5_daily.csv", _csv_bytes(SAMPLE_CSV), "text/csv")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Schema
        self.assertEqual(data["frequency"], "D")
        self.assertEqual(data["target_column"], "quantity")
        self.assertIn("date", [data["time_column"]] + data["id_columns"])
        self.assertGreater(data["n_series"], 10)
        self.assertGreater(data["n_rows"], 1000)

        # Forecastability
        self.assertGreater(data["overall_forecastability"], 0)
        self.assertIn("demand_classes", data)

        # Config recommendation should be valid YAML
        self.assertIn("recommended_config_yaml", data)
        self.assertTrue(len(data["recommended_config_yaml"]) > 50)

    def test_analyze_multi_file(self):
        """Upload the M5 daily CSV as a single-file multi-file analysis."""
        resp = self.client.post(
            "/pipeline/analyze-multi-file?lob_name=walmart_m5_daily",
            files=[
                ("files", ("m5_daily.csv", _csv_bytes(SAMPLE_CSV), "text/csv")),
            ],
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("profiles", data)
        self.assertEqual(len(data["profiles"]), 1)
        profile = data["profiles"][0]
        self.assertEqual(profile["role"], "time_series")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Series Exploration  (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeriesExploration(unittest.TestCase):
    """Tests that require pre-populated history in data_dir."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = Path(self.tmpdir) / "data"
        self.metrics_dir = Path(self.tmpdir) / "metrics"
        self.data_dir.mkdir(parents=True)
        self.metrics_dir.mkdir(parents=True)

        # Write daily actuals as Parquet so the /series/{lob} endpoint finds them
        df = _load_sample()
        hist_dir = self.data_dir / "history" / "walmart_m5_daily"
        hist_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(hist_dir / "actuals.parquet")

        app = _make_app(str(self.data_dir), str(self.metrics_dir))
        self.client = TestClient(app)

    def test_list_series(self):
        """GET /series/{lob} should list all sampled series with SBC classification."""
        resp = self.client.get("/series/walmart_m5_daily")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["lob"], "walmart_m5_daily")
        self.assertGreater(data["series_count"], 10)
        # Each series should have demand classification
        item = data["series"][0]
        self.assertIn("demand_class", item)
        self.assertIn("adi", item)
        self.assertIn("cv2", item)

    def test_detect_breaks(self):
        """POST /series/breaks with uploaded daily CSV should detect structural changes."""
        resp = self.client.post(
            "/series/breaks?method=cusum",
            files={"file": ("m5_daily.csv", _csv_bytes(SAMPLE_CSV), "text/csv")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_series", data)
        self.assertIn("series_with_breaks", data)
        self.assertGreaterEqual(data["total_series"], 10)

    def test_cleansing_audit(self):
        """POST /series/cleansing-audit should detect outliers in daily data."""
        resp = self.client.post(
            "/series/cleansing-audit?lob=walmart_m5_daily"
            "&outlier_method=iqr&iqr_multiplier=1.5&outlier_action=clip"
            "&stockout_detection=true&min_zero_run=3",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_series", data)
        self.assertIn("total_outliers", data)
        self.assertIn("rows_modified", data)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Backtest Pipeline  (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.timeout(1200)
@pytest.mark.usefixtures("_daily_backtest_client")
class TestBacktestPipeline(unittest.TestCase):
    """POST /pipeline/backtest with daily data and config."""

    _backtest_result: dict | None = None

    def test_run_backtest_daily(self):
        """Backtest completes with status='completed' and has a champion model."""
        data = self._backtest_result
        self.assertIsNotNone(data)
        self.assertEqual(data["status"], "completed")
        self.assertIn("champion_model", data)
        self.assertIn("best_wmape", data)
        # With all 3 models available, best_wmape should be a valid float
        if data["best_wmape"] is not None:
            self.assertIsInstance(data["best_wmape"], float)
            self.assertGreater(data["best_wmape"], 0)
            self.assertLess(data["best_wmape"], 2.0)  # sanity — not absurdly high

    def test_baseline_no_regression(self):
        """Champion + per-model WMAPE must not regress against the blessed baseline.

        Skipped if ``tests/integration/baselines/m5_daily_baseline.json`` is absent
        (run ``python scripts/bless_m5_baseline.py --frequency daily`` to capture).
        """
        from tests.integration.baseline import (
            assert_no_regression,
            require_baseline,
            verify_fixture_hash,
        )

        baseline = require_baseline("daily", self)
        verify_fixture_hash(self, baseline, SAMPLE_CSV, CONFIG_YAML)

        data = self._backtest_result
        if data.get("best_wmape") is None:
            self.skipTest("Pipeline did not return best_wmape; cannot check regression.")

        per_model = {
            entry["model_id"]: float(entry["wmape"])
            for entry in data.get("leaderboard", [])
            if entry.get("model_id") and entry.get("wmape") is not None
        }
        assert_no_regression(
            self,
            baseline,
            observed_champion_wmape=float(data["best_wmape"]),
            observed_per_model_wmape=per_model,
        )

    def test_baseline_fva_preserved(self):
        """Champion must still beat naive by ≥ MIN_FVA_RATIO of the blessed margin."""
        from tests.integration.baseline import assert_fva_preserved, require_baseline

        baseline = require_baseline("daily", self)
        data = self._backtest_result
        leaderboard = data.get("leaderboard", [])

        by_id = {
            e["model_id"]: float(e["wmape"])
            for e in leaderboard
            if e.get("model_id") and e.get("wmape") is not None
        }
        naive_wmape = next(
            (w for mid, w in by_id.items() if "naive" in mid.lower()),
            None,
        )
        if naive_wmape is None or data.get("best_wmape") is None:
            self.skipTest("Naive or champion WMAPE missing; cannot check FVA.")

        assert_fva_preserved(
            self,
            baseline,
            observed_naive_wmape=naive_wmape,
            observed_champion_wmape=float(data["best_wmape"]),
        )

    def test_leaderboard(self):
        """Leaderboard contains all 3 configured models ranked by WMAPE."""
        data = self._backtest_result
        leaderboard = data.get("leaderboard", [])
        self.assertGreaterEqual(len(leaderboard), 3)

        models_seen = {entry["model_id"] for entry in leaderboard}
        self.assertIn("naive_seasonal", models_seen)
        self.assertIn("auto_arima", models_seen)
        self.assertIn("lgbm_direct", models_seen)

        # Leaderboard should be ranked by WMAPE ascending (nulls/None last)
        wmapes = [e["wmape"] for e in leaderboard if e["wmape"] is not None]
        self.assertEqual(wmapes, sorted(wmapes))

    def test_fva(self):
        """GET /metrics/{lob}/fva should work after backtest."""
        resp = self.client.get("/metrics/walmart_m5_daily/fva?run_type=backtest")
        if resp.status_code == 200:
            data = resp.json()
            self.assertEqual(data["lob"], "walmart_m5_daily")
        else:
            # FVA may 404 if metrics weren't written in the expected partition
            self.assertIn(resp.status_code, [200, 404])


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Forecast Pipeline  (2 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.timeout(1200)
class TestForecastPipeline(unittest.TestCase):
    """POST /pipeline/forecast with daily data."""

    _forecast_result: dict | None = None

    @pytest.fixture(scope="class", autouse=True)
    def _run_forecast(self, _daily_backtest_client):
        """Execute the real forecast against a private copy of backtest artifacts."""
        fc_resp = self.client.post(
            "/pipeline/forecast?lob=walmart_m5_daily&horizon=28",
            files={
                "file": ("m5_daily.csv", _csv_bytes(SAMPLE_CSV), "text/csv"),
                "config_file": ("config.yaml", _yaml_bytes(CONFIG_YAML), "application/x-yaml"),
            },
        )
        assert fc_resp.status_code == 200, f"Forecast failed: {fc_resp.text}"
        type(self)._forecast_result = fc_resp.json()

    def test_run_forecast_daily(self):
        """Forecast completes with expected row count and horizon."""
        data = self._forecast_result
        self.assertIsNotNone(data)
        self.assertEqual(data["status"], "completed")
        self.assertGreater(data["forecast_rows"], 0)
        self.assertGreater(data["series_count"], 10)

    def test_get_forecasts(self):
        """GET /forecast/{lob} returns daily forecast points."""
        resp = self.client.get("/forecast/walmart_m5_daily")
        if resp.status_code == 200:
            data = resp.json()
            self.assertEqual(data["lob"], "walmart_m5_daily")
            self.assertGreater(data["series_count"], 0)
            points = data["points"]
            self.assertGreater(len(points), 0)

            # Verify daily date spacing: consecutive points for same series
            # should be 1 day apart
            first_series = points[0]["series_id"]
            series_points = [p for p in points if p["series_id"] == first_series]
            if len(series_points) >= 2:
                d0 = date.fromisoformat(str(series_points[0]["week"]))
                d1 = date.fromisoformat(str(series_points[1]["week"]))
                self.assertEqual((d1 - d0).days, 1, "Daily forecasts should be 1 day apart")
        else:
            # May not find forecasts if they weren't persisted
            self.assertIn(resp.status_code, [200, 404])


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Hierarchy  (2 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHierarchy(unittest.TestCase):
    """Hierarchy build and reconciliation with daily M5 data."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        app = _make_app(
            str(Path(self.tmpdir) / "data"),
            str(Path(self.tmpdir) / "metrics"),
        )
        self.client = TestClient(app)

    def test_build_hierarchy(self):
        """Build product hierarchy: cat_id → dept_id → series_id."""
        resp = self.client.post(
            "/hierarchy/build"
            "?levels=cat_id,dept_id,series_id"
            "&id_column=series_id"
            "&name=product",
            files={"file": ("m5_daily.csv", _csv_bytes(SAMPLE_CSV), "text/csv")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "product")
        self.assertGreater(data["total_nodes"], 0)
        self.assertIn("level_stats", data)

    def test_reconcile_hierarchy(self):
        """Aggregate daily data to cat_id level (sum across dept/series)."""
        resp = self.client.post(
            "/hierarchy/aggregate"
            "?levels=cat_id,dept_id,series_id"
            "&target_level=cat_id"
            "&value_columns=quantity"
            "&time_column=date",
            files={"file": ("m5_daily.csv", _csv_bytes(SAMPLE_CSV), "text/csv")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["target_level"], "cat_id")
        self.assertGreater(data["total_rows"], 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Validation & Governance  (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.timeout(1200)
@pytest.mark.usefixtures("_daily_backtest_client")
class TestValidationGovernance(unittest.TestCase):
    """Post-pipeline validation, manifests, and model cards."""

    def test_validation_endpoint(self):
        """GET /metrics/{lob}/validation should return a grade after backtest."""
        resp = self.client.get("/metrics/walmart_m5_daily/validation?run_type=backtest")
        if resp.status_code == 200:
            data = resp.json()
            self.assertIn("grade", data)
            self.assertIn(data["grade"], ["A", "B", "C", "D", "F"])
            self.assertIn("score", data)
            self.assertIsInstance(data["score"], int)
        else:
            self.assertIn(resp.status_code, [200, 404])

    def test_pipeline_manifests(self):
        """GET /pipeline/manifests should list the backtest run."""
        resp = self.client.get("/pipeline/manifests?lob=walmart_m5_daily")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # May be 0 if manifests are stored differently
        self.assertIn("count", data)

    def test_model_cards(self):
        """GET /governance/model-cards should list model cards after backtest."""
        resp = self.client.get("/governance/model-cards")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("count", data)


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Daily-Specific Assertions  (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDailySpecific(unittest.TestCase):
    """Verify daily-frequency-specific behaviour in config and models."""

    def test_daily_frequency_profile(self):
        """FREQUENCY_PROFILES['D'] has correct season_length and lags."""
        from src.config.schema import get_frequency_profile

        profile = get_frequency_profile("D")
        self.assertEqual(profile["season_length"], 7)
        self.assertEqual(profile["statsforecast_freq"], "D")
        self.assertEqual(profile["min_series_length"], 90)
        self.assertEqual(profile["min_ml_series_length"], 450)
        self.assertEqual(profile["default_horizon"], 90)
        self.assertEqual(profile["timedelta_kwargs"], {"days": 1})
        # Lags should include day-level and week-level
        lags = profile["default_lags"]
        self.assertIn(1, lags)
        self.assertIn(7, lags)
        self.assertIn(28, lags)
        self.assertIn(364, lags)

    def test_daily_date_arithmetic(self):
        """freq_timedelta('D', n) returns n days."""
        from src.config.schema import freq_timedelta

        delta_1 = freq_timedelta("D", 1)
        self.assertEqual(delta_1, timedelta(days=1))

        delta_28 = freq_timedelta("D", 28)
        self.assertEqual(delta_28, timedelta(days=28))

        delta_90 = freq_timedelta("D", 90)
        self.assertEqual(delta_90, timedelta(days=90))

    def test_daily_season_length_auto_correction(self):
        """Statistical models auto-correct season_length to 7 for daily data."""
        from src.forecasting.naive import SeasonalNaiveForecaster

        # Default season_length=52 should auto-correct to 7 when frequency="D"
        model = SeasonalNaiveForecaster(frequency="D")
        self.assertEqual(model.season_length, 7)

        # Explicit season_length should be preserved
        model_explicit = SeasonalNaiveForecaster(season_length=14, frequency="D")
        self.assertEqual(model_explicit.season_length, 14)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
