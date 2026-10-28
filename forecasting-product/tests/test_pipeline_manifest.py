"""Tests for pipeline manifest / provenance."""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import polars as pl
import pytest

from src.config.schema import (
    CleansingConfig,
    DataQualityConfig,
    ForecastConfig,
    PlatformConfig,
    ValidationConfig,
)
from src.pipeline.manifest import (
    PipelineManifest,
    _hash_config,
    _hash_dataframe,
    build_manifest,
    read_manifest,
    write_manifest,
)

from conftest import make_actuals as _make_actuals

pytestmark = pytest.mark.unit


def _make_forecast(n_series: int = 2, horizon: int = 4) -> pl.DataFrame:
    rows = []
    base = date(2024, 12, 30)
    for s in range(n_series):
        for w in range(horizon):
            rows.append({
                "series_id": f"S{s:03d}",
                "week": base + timedelta(weeks=w),
                "forecast": float(100 + s * 10),
            })
    return pl.DataFrame(rows)


class _MockSeriesBuilder:
    """Minimal mock that mimics SeriesBuilder report attributes."""

    def __init__(
        self,
        validation_report=None,
        cleansing_report=None,
        regressor_screen_report=None,
    ):
        self._last_validation_report = validation_report
        self._last_cleansing_report = cleansing_report
        self._last_regressor_screen_report = regressor_screen_report


@dataclass
class _MockValidationReport:
    passed: bool = True
    warnings: list = None
    errors: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


@dataclass
class _MockCleansingReport:
    total_outliers: int = 0
    total_stockout_periods: int = 0
    rows_modified: int = 0


@dataclass
class _MockRegressorScreenReport:
    dropped_columns: list = None
    warnings: list = None

    def __post_init__(self):
        if self.dropped_columns is None:
            self.dropped_columns = []
        if self.warnings is None:
            self.warnings = []


class TestBuildManifest:
    def test_build_manifest_minimal(self):
        config = PlatformConfig(lob="retail")
        actuals = _make_actuals()
        forecast = _make_forecast()
        builder = _MockSeriesBuilder()

        manifest = build_manifest(
            run_id="test123",
            config=config,
            actuals=actuals,
            series_builder=builder,
            champion_model_id="lgbm_direct",
            forecast=forecast,
            forecast_file="forecast_retail_2024-12-30.parquet",
        )

        assert manifest.run_id == "test123"
        assert manifest.lob == "retail"
        assert manifest.input_row_count == len(actuals)
        assert manifest.input_series_count == 2
        assert manifest.champion_model_id == "lgbm_direct"
        assert manifest.forecast_row_count == len(forecast)
        assert manifest.forecast_file == "forecast_retail_2024-12-30.parquet"
        assert manifest.cleansing_applied is False
        assert manifest.validation_applied is False
        assert manifest.regressor_screen_applied is False

    def test_build_manifest_with_cleansing(self):
        config = PlatformConfig()
        actuals = _make_actuals()
        forecast = _make_forecast()
        cleansing = _MockCleansingReport(
            total_outliers=47, total_stockout_periods=12, rows_modified=59
        )
        builder = _MockSeriesBuilder(cleansing_report=cleansing)

        manifest = build_manifest(
            "run1", config, actuals, builder, "ets", forecast, "f.parquet"
        )

        assert manifest.cleansing_applied is True
        assert manifest.outliers_clipped == 47
        assert manifest.stockout_periods_imputed == 12
        assert manifest.rows_modified == 59

    def test_build_manifest_with_validation(self):
        config = PlatformConfig()
        actuals = _make_actuals()
        forecast = _make_forecast()
        validation = _MockValidationReport(
            passed=True,
            warnings=["warn1", "warn2"],
            errors=[],
        )
        builder = _MockSeriesBuilder(validation_report=validation)

        manifest = build_manifest(
            "run2", config, actuals, builder, "arima", forecast, "f.parquet"
        )

        assert manifest.validation_applied is True
        assert manifest.validation_passed is True
        assert manifest.validation_warnings == 2
        assert manifest.validation_errors == 0

    def test_build_manifest_with_regressor_screen(self):
        config = PlatformConfig()
        actuals = _make_actuals()
        forecast = _make_forecast()
        screen = _MockRegressorScreenReport(
            dropped_columns=["const_col"],
            warnings=["dropped const_col (zero variance)"],
        )
        builder = _MockSeriesBuilder(regressor_screen_report=screen)

        manifest = build_manifest(
            "run3", config, actuals, builder, "lgbm", forecast, "f.parquet"
        )

        assert manifest.regressor_screen_applied is True
        assert manifest.regressors_dropped == ["const_col"]
        assert len(manifest.regressor_warnings) == 1

    def test_backtest_wmape_passed_through(self):
        config = PlatformConfig()
        actuals = _make_actuals()
        forecast = _make_forecast()
        builder = _MockSeriesBuilder()

        manifest = build_manifest(
            "run4", config, actuals, builder, "lgbm", forecast, "f.parquet",
            backtest_wmape=0.098,
        )

        assert manifest.backtest_wmape == 0.098

    def test_date_range_populated(self):
        config = PlatformConfig()
        actuals = _make_actuals(n_weeks=52)
        forecast = _make_forecast()
        builder = _MockSeriesBuilder()

        manifest = build_manifest(
            "run5", config, actuals, builder, "lgbm", forecast, "f.parquet"
        )

        assert isinstance(manifest.date_range_start, str)
        assert isinstance(manifest.date_range_end, str)


class TestConfigHash:
    def test_config_hash_deterministic(self):
        config = PlatformConfig(lob="retail")
        h1 = _hash_config(config)
        h2 = _hash_config(config)
        assert h1 == h2
        assert len(h1) == 64

    def test_config_hash_changes_with_different_config(self):
        c1 = PlatformConfig(lob="retail")
        c2 = PlatformConfig(lob="wholesale")
        assert _hash_config(c1) != _hash_config(c2)


class TestDataHash:
    def test_input_data_hash_deterministic(self):
        df = _make_actuals()
        h1 = _hash_dataframe(df)
        h2 = _hash_dataframe(df)
        assert h1 == h2
        assert len(h1) == 64

    def test_input_data_hash_changes_with_different_data(self):
        df1 = _make_actuals(n_weeks=52)
        df2 = _make_actuals(n_weeks=26)
        assert _hash_dataframe(df1) != _hash_dataframe(df2)


class TestWriteReadManifest:
    def test_write_manifest_creates_json(self, tmp_path):
        forecast_path = tmp_path / "forecast_retail_2024-01-01.parquet"
        forecast_path.touch()  # create dummy file

        manifest = PipelineManifest(
            run_id="abc",
            lob="retail",
            champion_model_id="lgbm",
            forecast_row_count=100,
        )

        result_path = write_manifest(manifest, str(forecast_path))
        assert Path(result_path).exists()
        assert result_path.endswith("_manifest.json")

        # Verify valid JSON
        data = json.loads(Path(result_path).read_text())
        assert data["run_id"] == "abc"
        assert data["lob"] == "retail"

    def test_read_manifest_roundtrip(self, tmp_path):
        forecast_path = tmp_path / "forecast.parquet"
        forecast_path.touch()

        original = PipelineManifest(
            run_id="xyz",
            timestamp="2024-06-01T12:00:00",
            lob="retail",
            input_data_hash="aabbcc",
            input_row_count=5000,
            input_series_count=50,
            config_hash="12345678",
            champion_model_id="lgbm_direct",
            backtest_wmape=0.098,
            forecast_horizon=39,
            forecast_row_count=1950,
            forecast_file="forecast.parquet",
            cleansing_applied=True,
            outliers_clipped=47,
            stockout_periods_imputed=12,
            rows_modified=59,
            regressors_dropped=["const_col"],
        )

        manifest_path = write_manifest(original, str(forecast_path))
        loaded = read_manifest(manifest_path)

        assert loaded.run_id == original.run_id
        assert loaded.lob == original.lob
        assert loaded.input_data_hash == original.input_data_hash
        assert loaded.input_row_count == original.input_row_count
        assert loaded.config_hash == original.config_hash
        assert loaded.champion_model_id == original.champion_model_id
        assert loaded.backtest_wmape == original.backtest_wmape
        assert loaded.cleansing_applied is True
        assert loaded.outliers_clipped == 47
        assert loaded.regressors_dropped == ["const_col"]

    def test_manifest_filename_convention(self, tmp_path):
        forecast_path = tmp_path / "forecast_retail_2024-06-01.parquet"
        forecast_path.touch()

        manifest = PipelineManifest(run_id="test")
        result_path = write_manifest(manifest, str(forecast_path))
        assert Path(result_path).name == "forecast_retail_2024-06-01_manifest.json"

    def test_manifest_dates_iso_format(self, tmp_path):
        forecast_path = tmp_path / "f.parquet"
        forecast_path.touch()

        manifest = PipelineManifest(
            run_id="test",
            date_range_start="2024-01-01",
            date_range_end="2024-12-30",
        )
        result_path = write_manifest(manifest, str(forecast_path))
        data = json.loads(Path(result_path).read_text())
        assert data["date_range_start"] == "2024-01-01"
        assert data["date_range_end"] == "2024-12-30"


class TestForecastPipelineIntegration:
    def test_integration_manifest_written(self, tmp_path):
        """End-to-end: the completed bundle has one authoritative manifest."""
        from src.pipeline.forecast import ForecastPipeline

        config = PlatformConfig(
            lob="test_lob",
            forecast=ForecastConfig(horizon_weeks=4),
        )
        config.output.forecast_path = str(tmp_path)

        pipeline = ForecastPipeline(config)
        actuals = _make_actuals(n_weeks=52, n_series=2)

        forecast = pipeline.run(actuals, champion_model="naive_seasonal")

        # Check manifest was written
        manifest_files = list(tmp_path.glob("runs/*/*/manifest.json"))
        assert len(manifest_files) == 1

        data = json.loads(manifest_files[0].read_text())
        assert data["schema_version"] == 1
        assert data["run_id"] == pipeline.context.run_id == pipeline.last_run.run_id
        assert data["lob"] == "test_lob"
        assert data["champion_model_id"] == "naive_seasonal"
        assert data["forecast_row_count"] == len(forecast)
        assert data["input_row_count"] == len(actuals)
        assert data["config_hash"] != ""
        assert data["input_data_hash"] != ""
