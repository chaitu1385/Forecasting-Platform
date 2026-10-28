"""
Tests for prediction interval calibration.

Covers:
  - Interval coverage computation
  - Calibration report generation (per-model, per-series)
  - Conformal residual computation
  - Conformal correction (widens narrow intervals, preserves good ones)
  - Integration with BacktestEngine and pipelines
"""

import math
import random
import unittest
from datetime import date, timedelta

import polars as pl

from src.config.schema import (
    BacktestConfig,
    CalibrationConfig,
    ForecastConfig,
    PlatformConfig,
)
from src.evaluation.calibration import (
    CalibrationReport,
    IntervalCoverage,
    apply_conformal_correction,
    compute_calibration_report,
    compute_conformal_residuals,
    compute_interval_coverage,
)
import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _make_backtest_with_quantiles(
    n_series: int = 3,
    n_weeks: int = 52,
    n_folds: int = 2,
    model_ids: list = None,
    miscalibrated: bool = False,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Synthetic backtest results with actual, forecast, forecast_p10, forecast_p90.

    If miscalibrated=True, intervals are too narrow (~50% coverage).
    Otherwise, intervals are well-calibrated (~80% coverage for P10-P90).
    """
    random.seed(seed)
    if model_ids is None:
        model_ids = ["naive_seasonal"]

    rows = []
    for model_id in model_ids:
        for fold in range(n_folds):
            for s in range(1, n_series + 1):
                base = 100.0 * s
                for w in range(n_weeks):
                    actual = base + random.gauss(0, base * 0.10)
                    forecast = base + random.gauss(0, base * 0.05)

                    if miscalibrated:
                        # Narrow intervals: ±2% of forecast (will miss many actuals)
                        half_width = abs(forecast) * 0.02
                    else:
                        # Wide intervals: ±15% of forecast (should cover ~80%+)
                        half_width = abs(forecast) * 0.15

                    rows.append({
                        "run_id": f"bt-{fold}",
                        "run_type": "backtest",
                        "run_date": date(2024, 1, 1),
                        "lob": "test",
                        "model_id": model_id,
                        "fold": fold,
                        "grain_level": "series",
                        "series_id": f"S{s}",
                        "channel": "",
                        "target_week": date(2024, 1, 7) + timedelta(weeks=w),
                        "actual": actual,
                        "forecast": forecast,
                        "forecast_p10": forecast - half_width,
                        "forecast_p90": forecast + half_width,
                        "wmape": 0.05,
                        "normalized_bias": 0.01,
                    })

    return pl.DataFrame(rows)


def _make_forecast_with_quantiles(
    n_series: int = 3,
    n_weeks: int = 13,
) -> pl.DataFrame:
    """Synthetic production forecast with quantile columns."""
    rows = []
    for s in range(1, n_series + 1):
        base = 100.0 * s
        for w in range(n_weeks):
            forecast = base
            rows.append({
                "series_id": f"S{s}",
                "week": date(2025, 1, 6) + timedelta(weeks=w),
                "forecast": forecast,
                "forecast_p10": forecast - 10.0,
                "forecast_p90": forecast + 10.0,
            })
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Interval Coverage Tests
# --------------------------------------------------------------------------- #

class TestIntervalCoverage(unittest.TestCase):

    def test_perfect_coverage(self):
        """All actuals inside [lower, upper] → 100%."""
        actuals = pl.Series([10, 20, 30, 40, 50], dtype=pl.Float64)
        lower = pl.Series([5, 15, 25, 35, 45], dtype=pl.Float64)
        upper = pl.Series([15, 25, 35, 45, 55], dtype=pl.Float64)
        self.assertAlmostEqual(compute_interval_coverage(actuals, lower, upper), 1.0)

    def test_zero_coverage(self):
        """All actuals outside → 0%."""
        actuals = pl.Series([100, 200, 300], dtype=pl.Float64)
        lower = pl.Series([0, 0, 0], dtype=pl.Float64)
        upper = pl.Series([10, 10, 10], dtype=pl.Float64)
        self.assertAlmostEqual(compute_interval_coverage(actuals, lower, upper), 0.0)

    def test_partial_coverage(self):
        """2 out of 4 actuals inside → 50%."""
        actuals = pl.Series([5, 15, 25, 35], dtype=pl.Float64)
        lower = pl.Series([10, 10, 20, 40], dtype=pl.Float64)
        upper = pl.Series([20, 20, 30, 50], dtype=pl.Float64)
        self.assertAlmostEqual(compute_interval_coverage(actuals, lower, upper), 0.5)

    def test_empty_input(self):
        """Empty series → 0%."""
        actuals = pl.Series([], dtype=pl.Float64)
        lower = pl.Series([], dtype=pl.Float64)
        upper = pl.Series([], dtype=pl.Float64)
        self.assertAlmostEqual(compute_interval_coverage(actuals, lower, upper), 0.0)


# --------------------------------------------------------------------------- #
#  Calibration Report Tests
# --------------------------------------------------------------------------- #

class TestCalibrationReport(unittest.TestCase):

    def test_well_calibrated_report(self):
        """Wide intervals should give empirical coverage close to 80%."""
        bt = _make_backtest_with_quantiles(miscalibrated=False)
        report = compute_calibration_report(
            bt, quantiles=[0.1, 0.9], coverage_targets={"80": 0.80},
        )
        cov = report.model_reports["naive_seasonal"][0]
        self.assertGreater(cov.empirical, 0.70, "Well-calibrated intervals should cover >70%")
        self.assertEqual(cov.nominal, 0.80)

    def test_miscalibrated_report_detected(self):
        """Narrow intervals should show large positive miscalibration."""
        bt = _make_backtest_with_quantiles(miscalibrated=True)
        report = compute_calibration_report(
            bt, quantiles=[0.1, 0.9], coverage_targets={"80": 0.80},
        )
        cov = report.model_reports["naive_seasonal"][0]
        self.assertGreater(cov.miscalibration, 0.10, "Should detect miscalibration")
        self.assertLess(cov.empirical, 0.70, "Narrow intervals should have low coverage")

    def test_sharpness_computed(self):
        """Sharpness should be positive (interval width > 0)."""
        bt = _make_backtest_with_quantiles(miscalibrated=False)
        report = compute_calibration_report(
            bt, quantiles=[0.1, 0.9], coverage_targets={"80": 0.80},
        )
        cov = report.model_reports["naive_seasonal"][0]
        self.assertGreater(cov.sharpness, 0, "Interval width should be positive")

    def test_per_series_breakdown(self):
        """Per-series report should have entries for each series."""
        bt = _make_backtest_with_quantiles(n_series=3)
        report = compute_calibration_report(
            bt, quantiles=[0.1, 0.9], coverage_targets={"80": 0.80},
        )
        series_ids = report.per_series["series_id"].unique().to_list()
        self.assertEqual(sorted(series_ids), ["S1", "S2", "S3"])

    def test_multi_model_report(self):
        """Each model should get independent coverage stats."""
        bt = _make_backtest_with_quantiles(model_ids=["model_a", "model_b"])
        report = compute_calibration_report(
            bt, quantiles=[0.1, 0.9], coverage_targets={"80": 0.80},
        )
        self.assertIn("model_a", report.model_reports)
        self.assertIn("model_b", report.model_reports)
        self.assertEqual(len(report.model_reports), 2)


# --------------------------------------------------------------------------- #
#  Conformal Residuals Tests
# --------------------------------------------------------------------------- #

class TestConformalResiduals(unittest.TestCase):

    def test_residuals_computed_correctly(self):
        """Nonconformity scores should be max(lower - actual, actual - upper)."""
        bt = pl.DataFrame({
            "series_id": ["S1", "S1", "S1"],
            "model_id": ["m", "m", "m"],
            "actual": [50.0, 150.0, 100.0],       # inside, above, inside
            "forecast_p10": [40.0, 90.0, 80.0],
            "forecast_p90": [120.0, 110.0, 120.0],
        })
        resids = compute_conformal_residuals(
            bt, quantiles=[0.1, 0.9],
            coverage_targets={"80": 0.80},
        )
        scores = resids["residual_80"].to_list()
        # Row 0: max(40-50, 50-120) = max(-10, -70) = -10 (inside)
        # Row 1: max(90-150, 150-110) = max(-60, 40) = 40 (above upper)
        # Row 2: max(80-100, 100-120) = max(-20, -20) = -20 (inside)
        self.assertAlmostEqual(scores[0], -10.0)
        self.assertAlmostEqual(scores[1], 40.0)
        self.assertAlmostEqual(scores[2], -20.0)

    def test_residuals_per_series(self):
        """Residuals should be computed independently per series."""
        bt = _make_backtest_with_quantiles(n_series=2)
        resids = compute_conformal_residuals(
            bt, quantiles=[0.1, 0.9],
            coverage_targets={"80": 0.80},
        )
        self.assertIn("residual_80", resids.columns)
        self.assertIn("series_id", resids.columns)

    def test_residuals_shape(self):
        """Residuals should have same row count as input."""
        bt = _make_backtest_with_quantiles(n_series=2, n_weeks=10)
        resids = compute_conformal_residuals(
            bt, quantiles=[0.1, 0.9],
            coverage_targets={"80": 0.80},
        )
        self.assertEqual(resids.height, bt.height)


# --------------------------------------------------------------------------- #
#  Conformal Correction Tests
# --------------------------------------------------------------------------- #

class TestConformalCorrection(unittest.TestCase):

    def test_correction_widens_narrow_intervals(self):
        """After correction, coverage on calibration data should improve."""
        # Miscalibrated backtest data
        bt = _make_backtest_with_quantiles(miscalibrated=True, n_weeks=100)
        resids = compute_conformal_residuals(
            bt, quantiles=[0.1, 0.9],
            coverage_targets={"80": 0.80},
        )

        # Apply correction to a forecast
        forecast = _make_forecast_with_quantiles()
        corrected = apply_conformal_correction(
            forecast, resids, quantiles=[0.1, 0.9],
            coverage_targets={"80": 0.80},
        )

        # Corrected intervals should be wider than original
        orig_width = (forecast["forecast_p90"] - forecast["forecast_p10"]).mean()
        corr_width = (corrected["forecast_p90"] - corrected["forecast_p10"]).mean()
        self.assertGreater(corr_width, orig_width, "Correction should widen narrow intervals")

    def test_correction_preserves_well_calibrated(self):
        """Well-calibrated intervals should see minimal correction."""
        bt = _make_backtest_with_quantiles(miscalibrated=False, n_weeks=100)
        resids = compute_conformal_residuals(
            bt, quantiles=[0.1, 0.9],
            coverage_targets={"80": 0.80},
        )

        forecast = _make_forecast_with_quantiles()
        corrected = apply_conformal_correction(
            forecast, resids, quantiles=[0.1, 0.9],
            coverage_targets={"80": 0.80},
        )

        # Width change should be modest (residuals are mostly negative when calibrated)
        orig_width = float((forecast["forecast_p90"] - forecast["forecast_p10"]).mean())
        corr_width = float((corrected["forecast_p90"] - corrected["forecast_p10"]).mean())
        # Allow ±50% change for well-calibrated (some adjustment is normal)
        self.assertLess(
            abs(corr_width - orig_width) / orig_width, 0.5,
            "Well-calibrated intervals should not change drastically",
        )

    def test_correction_column_names_preserved(self):
        """Output should have same quantile columns as input."""
        bt = _make_backtest_with_quantiles()
        resids = compute_conformal_residuals(
            bt, quantiles=[0.1, 0.9],
            coverage_targets={"80": 0.80},
        )

        forecast = _make_forecast_with_quantiles()
        corrected = apply_conformal_correction(
            forecast, resids, quantiles=[0.1, 0.9],
            coverage_targets={"80": 0.80},
        )

        self.assertIn("forecast_p10", corrected.columns)
        self.assertIn("forecast_p90", corrected.columns)
        self.assertEqual(corrected.height, forecast.height)

    def test_correction_with_empty_residuals(self):
        """Empty residuals → no-op."""
        forecast = _make_forecast_with_quantiles()
        empty_resids = pl.DataFrame(schema={
            "series_id": pl.Utf8, "model_id": pl.Utf8, "residual_80": pl.Float64,
        })
        corrected = apply_conformal_correction(
            forecast, empty_resids, quantiles=[0.1, 0.9],
            coverage_targets={"80": 0.80},
        )
        # Should be unchanged
        self.assertEqual(
            forecast["forecast_p10"].to_list(),
            corrected["forecast_p10"].to_list(),
        )

    def test_correction_filters_by_model(self):
        """When model_id is specified, only that model's residuals are used."""
        bt = _make_backtest_with_quantiles(model_ids=["model_a", "model_b"])
        resids = compute_conformal_residuals(
            bt, quantiles=[0.1, 0.9],
            coverage_targets={"80": 0.80},
        )

        forecast = _make_forecast_with_quantiles()
        corrected = apply_conformal_correction(
            forecast, resids, quantiles=[0.1, 0.9],
            coverage_targets={"80": 0.80},
            model_id="model_a",
        )
        # Should succeed and produce adjusted intervals
        self.assertIn("forecast_p10", corrected.columns)

    def test_negative_correction_preserves_ordered_median_and_point(self):
        forecast = pl.DataFrame({
            "forecast": [100.0], "forecast_p10": [80.0],
            "forecast_p50": [100.0], "forecast_p90": [120.0],
        })
        corrected = apply_conformal_correction(
            forecast, pl.DataFrame({"residual_80": [-5.0]}),
            quantiles=[0.1, 0.5, 0.9], coverage_targets={"80": 0.80},
        )
        self.assertEqual(corrected.to_dicts(), [{
            "forecast": 100.0, "forecast_p10": 85.0,
            "forecast_p50": 100.0, "forecast_p90": 115.0,
        }])

    def test_correction_rejects_median_crossing_even_when_outer_bounds_are_ordered(self):
        forecast = pl.DataFrame({
            "forecast": [100.0], "forecast_p10": [99.0],
            "forecast_p50": [100.0], "forecast_p90": [120.0],
        })
        with self.assertRaisesRegex(ValueError, "ordered quantiles.*forecast_p10.*forecast_p50"):
            apply_conformal_correction(
                forecast, pl.DataFrame({"residual_80": [-2.0]}),
                quantiles=[0.1, 0.5, 0.9], coverage_targets={"80": 0.80},
            )
        self.assertEqual(forecast["forecast_p10"].item(), 99.0)

    def test_correction_rejects_crossed_outer_bounds_without_a_requested_median(self):
        forecast = _make_forecast_with_quantiles()
        with self.assertRaisesRegex(ValueError, "ordered quantiles"):
            apply_conformal_correction(
                forecast, pl.DataFrame({"residual_80": [-20.0]}),
                quantiles=[0.1, 0.9], coverage_targets={"80": 0.80},
            )

    def test_nested_corrections_cannot_cross_other_requested_quantiles(self):
        forecast = pl.DataFrame({
            "forecast_p5": [70.0], "forecast_p10": [80.0],
            "forecast_p50": [100.0], "forecast_p90": [120.0], "forecast_p95": [130.0],
        })
        with self.assertRaisesRegex(ValueError, "ordered quantiles"):
            apply_conformal_correction(
                forecast, pl.DataFrame({"residual_80": [0.0], "residual_90": [-15.0]}),
                quantiles=[0.05, 0.1, 0.5, 0.9, 0.95],
                coverage_targets={"80": 0.80, "90": 0.90},
            )

    def test_nonfinite_residuals_fail_instead_of_being_dropped_or_propagated(self):
        for value in [None, float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite numeric"):
                apply_conformal_correction(
                    _make_forecast_with_quantiles(),
                    pl.DataFrame({"residual_80": [value]}, schema={"residual_80": pl.Float64}),
                    quantiles=[0.1, 0.9], coverage_targets={"80": 0.80},
                )


# --------------------------------------------------------------------------- #
#  Integration Tests
# --------------------------------------------------------------------------- #

class TestCalibrationIntegration(unittest.TestCase):

    def _make_config(self, calibration_enabled: bool = True) -> PlatformConfig:
        return PlatformConfig(
            forecast=ForecastConfig(
                quantiles=[0.1, 0.9],
                calibration=CalibrationConfig(
                    enabled=calibration_enabled,
                    conformal_correction=True,
                    coverage_targets={"80": 0.80},
                ),
            ),
            backtest=BacktestConfig(n_folds=2, val_weeks=13),
        )

    def test_backtest_engine_captures_quantiles(self):
        """BacktestEngine should include forecast_p10/p90 when calibration enabled."""
        from src.backtesting.engine import BacktestEngine

        config = self._make_config(calibration_enabled=True)
        engine = BacktestEngine(config, metric_store=None)

        # Build a simple synthetic dataset
        random.seed(42)
        rows = []
        for s in range(1, 3):
            for w in range(104):
                rows.append({
                    "series_id": f"S{s}",
                    "week": date(2021, 1, 4) + timedelta(weeks=w),
                    "quantity": 100.0 * s + random.gauss(0, 5),
                })
        data = pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))

        from src.forecasting.naive import SeasonalNaiveForecaster
        results = engine.run(data, [SeasonalNaiveForecaster()])

        if not results.is_empty():
            # Quantile columns should be present
            self.assertIn("forecast_p10", results.columns)
            self.assertIn("forecast_p90", results.columns)

    def test_backtest_pipeline_returns_calibration_report(self):
        """BacktestPipeline.run() should include calibration_report when enabled."""
        from src.pipeline.backtest import BacktestPipeline

        config = self._make_config(calibration_enabled=True)
        config.data_quality.fill_gaps = False
        config.data_quality.min_series_length_weeks = 0

        pipeline = BacktestPipeline(config)

        random.seed(42)
        rows = []
        for s in range(1, 3):
            for w in range(104):
                rows.append({
                    "series_id": f"S{s}",
                    "week": date(2021, 1, 4) + timedelta(weeks=w),
                    "quantity": 100.0 * s + random.gauss(0, 5),
                })
        data = pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))

        result = pipeline.run(data)

        self.assertIn("calibration_report", result)
        self.assertIn("conformal_residuals", result)
        if result["calibration_report"] is not None:
            self.assertIsInstance(result["calibration_report"], CalibrationReport)

    def test_disabled_returns_none(self):
        """Calibration disabled → None in result dict."""
        from src.pipeline.backtest import BacktestPipeline

        config = self._make_config(calibration_enabled=False)
        config.data_quality.fill_gaps = False
        config.data_quality.min_series_length_weeks = 0

        pipeline = BacktestPipeline(config)

        random.seed(42)
        rows = []
        for s in range(1, 3):
            for w in range(104):
                rows.append({
                    "series_id": f"S{s}",
                    "week": date(2021, 1, 4) + timedelta(weeks=w),
                    "quantity": 100.0 * s + random.gauss(0, 5),
                })
        data = pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))

        result = pipeline.run(data)
        self.assertIsNone(result.get("calibration_report"))


if __name__ == "__main__":
    unittest.main()
