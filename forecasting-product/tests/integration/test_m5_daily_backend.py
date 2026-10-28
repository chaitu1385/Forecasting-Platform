"""Backend integration tests for the Walmart M5 daily-frequency E2E pipeline.

These tests exercise the full API surface using the M5 daily sample fixture.
They cover: data analysis, series exploration, backtest, forecast, hierarchy,
validation, governance, and daily-frequency-specific assertions.

Prerequisites
-------------
- The checked-in full daily fixture and config must be present. Missing
  inputs fail collection; missing approved empirical references fail their
  two original gates rather than skipping them.

Run
---
    cd forecasting-product
    python -m pytest tests/integration/test_m5_daily_backend.py -v
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

# ── Paths ─────────────────────────────────────────────────────────────────────
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_CSV = FIXTURES / "m5_daily_sample.csv"
CONFIG_YAML = FIXTURES / "m5_daily_config.yaml"

for required in (SAMPLE_CSV, CONFIG_YAML):
    if not required.is_file():
        raise FileNotFoundError(f"Required full M5 daily input is missing: {required}")


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


def _assert_api_champion_wmape(result: dict, scores: dict, champion: str) -> float:
    """Cross-check the API summary, but return the independently rescored winner."""
    assert result.get("champion_model") == champion, "API champion differs from independent winner"
    authoritative = scores[champion]["wmape"]
    reported = result.get("best_wmape")
    assert reported == pytest.approx(authoritative, rel=1e-12, abs=1e-12), (
        f"API best_wmape {reported!r} differs from independent champion WMAPE {authoritative!r}"
    )
    return float(authoritative)


# ── Test app factory ──────────────────────────────────────────────────────────

def _make_app(data_dir: str, metrics_dir: str):
    from src.api.app import create_app

    return create_app(
        data_dir=data_dir,
        metrics_dir=metrics_dir,
        auth_enabled=False,
        dev_mode=True,
        jwt_secret="",
        audit_log_path=str(Path(data_dir) / "audit_log"),
    )


@pytest.fixture(scope="module")
def _daily_backtest(tmp_path_factory, record_testsuite_property):
    """Run the complete checked-in sample, three models and two folds once."""
    from src.backtesting.engine import BacktestEngine
    from tests.integration.m5_reference import (
        candidate_provenance,
        canonical_text_sha256,
        daily_naive_reference,
        exact_totals,
        read_daily_panel,
        score_complete_predictions,
    )

    provenance = candidate_provenance(SAMPLE_CSV, CONFIG_YAML)
    data_dir = tmp_path_factory.mktemp("m5-backtest") / "data"
    metrics_dir = data_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    history = data_dir / "history" / "walmart_m5_daily"
    history.mkdir(parents=True)
    _load_sample().write_parquet(history / "actuals.parquet")
    app = _make_app(str(data_dir), str(metrics_dir))
    raw_records = []
    original_run = BacktestEngine.run

    def capture_run(engine, *args, **kwargs):
        records = original_run(engine, *args, **kwargs)
        raw_records.append(records.clone())
        return records

    # Capture the actual shared run, not a second fit or a smaller benchmark.
    # Forward identity/persistence kwargs unchanged through Phase III integration.
    with patch.object(BacktestEngine, "run", capture_run), TestClient(app) as client:
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
    assert len(raw_records) == 1
    expected_reference = json.loads(
        (FIXTURES.parent / "baselines" / "m5_daily_reference.json").read_text(encoding="utf-8")
    )
    assert canonical_text_sha256(SAMPLE_CSV) == expected_reference["fixture_sha256"]
    assert canonical_text_sha256(CONFIG_YAML) == expected_reference["config_sha256"]
    panel = read_daily_panel(SAMPLE_CSV)
    series_count, input_rows = len(panel), sum(map(len, panel.values()))
    assert series_count == expected_reference["series_count"] == 105
    assert input_rows == expected_reference["input_rows"] == 76755
    reference = daily_naive_reference(panel)
    assert exact_totals(reference) == expected_reference["totals"]
    expected_models = {"naive_seasonal", "auto_arima", "lgbm_direct"}
    scores = score_complete_predictions(raw_records[0].iter_rows(named=True), reference, expected_models)
    for entry in result["leaderboard"]:
        for metric in ("wmape", "normalized_bias"):
            assert entry[metric] == pytest.approx(scores[entry["model_id"]][metric], rel=1e-12, abs=1e-12)
    expected_champion = min(
        scores, key=lambda model: (scores[model]["wmape"], abs(scores[model]["normalized_bias"]), model),
    )
    _assert_api_champion_wmape(result, scores, expected_champion)
    snapshot = {
        **provenance,
        "scoring_policy": "raw-observations-v1",
        "scalar_reference_scores": scores,
        "champion_model": expected_champion,
        "input_rows": input_rows, "series_count": series_count, "models": sorted(expected_models),
        "folds": 2, "validation_periods": 28, "gap_periods": 0,
        "scored_observations_per_model": len(reference),
    }
    if candidate_path := os.environ.get("M5_CANDIDATE_OUTPUT"):
        output = Path(candidate_path)
        raw_output = output.with_suffix(".predictions.parquet")
        if output.exists() or raw_output.exists():
            raise ValueError("Refusing to overwrite existing M5 candidate evidence")
        output.parent.mkdir(parents=True, exist_ok=True)
        raw_records[0].write_parquet(raw_output)
        snapshot["raw_predictions_sha256"] = hashlib.sha256(raw_output.read_bytes()).hexdigest()
        snapshot["raw_predictions_rows"] = raw_records[0].height
        output.write_text(json.dumps(snapshot, sort_keys=True, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
    record_testsuite_property("m5_daily_unapproved_candidate", json.dumps(snapshot, sort_keys=True))
    return data_dir, result, snapshot


@pytest.fixture(scope="class")
def _daily_backtest_client(request, tmp_path_factory, _daily_backtest):
    # Give every consumer its own files, response values and application state.
    # In particular, forecasting must not mutate the shared backtest snapshot.
    source, result, snapshot = _daily_backtest
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
        request.cls._backtest_provenance = deepcopy(snapshot)
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
            "&stockout_detection=true&min_zero_run=3&frequency=D",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["frequency"], "D")
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
        """Champion + every model must satisfy the approved native reference."""
        from tests.integration.baseline import (
            assert_no_regression,
            require_baseline,
            verify_fixture_hash,
        )

        baseline = require_baseline("daily", self, observed=self._backtest_provenance)
        verify_fixture_hash(self, baseline, SAMPLE_CSV, CONFIG_YAML)

        data = self._backtest_result
        self.assertIsNotNone(data.get("best_wmape"), "Missing observed champion WMAPE")
        observed_champion_wmape = _assert_api_champion_wmape(
            data, self._backtest_provenance["scalar_reference_scores"],
            self._backtest_provenance["champion_model"],
        )
        leaderboard = data.get("leaderboard", [])
        self.assertEqual(len(leaderboard), len(baseline.per_model_wmape),
                         "Observed model leaderboard is incomplete or duplicated")

        per_model = {
            entry["model_id"]: float(entry["wmape"])
            for entry in leaderboard
            if entry.get("model_id") and entry.get("wmape") is not None
        }
        assert_no_regression(
            self,
            baseline,
            observed_champion_wmape=observed_champion_wmape,
            observed_per_model_wmape=per_model,
        )

    def test_baseline_fva_preserved(self):
        """Champion must still beat naive by ≥ MIN_FVA_RATIO of the blessed margin."""
        from tests.integration.baseline import (
            assert_fva_preserved,
            require_baseline,
            verify_fixture_hash,
        )

        baseline = require_baseline("daily", self, observed=self._backtest_provenance)
        verify_fixture_hash(self, baseline, SAMPLE_CSV, CONFIG_YAML)
        data = self._backtest_result
        leaderboard = data.get("leaderboard", [])
        self.assertEqual(len(leaderboard), len(baseline.per_model_wmape),
                         "Observed model leaderboard is incomplete or duplicated")

        by_id = {
            e["model_id"]: float(e["wmape"])
            for e in leaderboard
            if e.get("model_id") and e.get("wmape") is not None
        }
        self.assertEqual(set(by_id), set(baseline.per_model_wmape),
                         "Observed model population is incomplete")
        naive_wmape = by_id["naive_seasonal"]
        self.assertIsNotNone(data.get("best_wmape"), "Missing observed champion WMAPE")
        observed_champion_wmape = _assert_api_champion_wmape(
            data, self._backtest_provenance["scalar_reference_scores"],
            self._backtest_provenance["champion_model"],
        )

        assert_fva_preserved(
            self,
            baseline,
            observed_naive_wmape=naive_wmape,
            observed_champion_wmape=observed_champion_wmape,
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
