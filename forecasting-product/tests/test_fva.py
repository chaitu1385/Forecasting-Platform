"""Tests for Forecast Value Add (FVA) analysis."""

import polars as pl
import pytest

from src.metrics.fva import (
    classify_fva,
    compute_fva_between_layers,
    compute_fva_cascade,
    compute_layer_metrics,
    compute_total_fva,
)
from src.analytics.fva_analyzer import FVAAnalyzer

pytestmark = pytest.mark.unit


class TestFVAClassification:
    def test_adds_value(self):
        assert classify_fva(0.05) == "ADDS_VALUE"

    def test_destroys_value(self):
        assert classify_fva(-0.05) == "DESTROYS_VALUE"

    def test_neutral(self):
        assert classify_fva(0.01) == "NEUTRAL"
        assert classify_fva(-0.01) == "NEUTRAL"
        assert classify_fva(0.0) == "NEUTRAL"


class TestLayerMetrics:
    def test_compute_layer_metrics(self):
        actual = pl.Series([100.0, 200.0, 150.0])
        forecast = pl.Series([110.0, 190.0, 160.0])
        metrics = compute_layer_metrics(actual, forecast)

        assert "wmape" in metrics
        assert "bias" in metrics
        assert "mae" in metrics
        assert metrics["wmape"] >= 0
        assert metrics["mae"] >= 0


class TestFVABetweenLayers:
    def test_child_better_than_parent(self):
        actual = pl.Series([100.0, 200.0, 150.0])
        parent = pl.Series([150.0, 250.0, 200.0])  # bad forecast
        child = pl.Series([105.0, 195.0, 155.0])    # good forecast

        result = compute_fva_between_layers(actual, parent, child)

        assert result["fva_wmape"] > 0  # positive = child is better
        assert result["fva_class"] == "ADDS_VALUE"

    def test_child_worse_than_parent(self):
        actual = pl.Series([100.0, 200.0, 150.0])
        parent = pl.Series([105.0, 195.0, 155.0])   # good forecast
        child = pl.Series([150.0, 250.0, 200.0])     # bad forecast

        result = compute_fva_between_layers(actual, parent, child)

        assert result["fva_wmape"] < 0  # negative = child is worse
        assert result["fva_class"] == "DESTROYS_VALUE"


class TestFVACascade:
    def test_full_cascade(self):
        actual = pl.Series([100.0, 200.0, 150.0, 120.0])
        forecasts = {
            "naive": pl.Series([130.0, 230.0, 180.0, 150.0]),       # worst
            "statistical": pl.Series([115.0, 210.0, 160.0, 130.0]), # middle
            "ml": pl.Series([105.0, 195.0, 155.0, 122.0]),          # best
        }

        results = compute_fva_cascade(actual, forecasts)

        assert len(results) == 3
        assert results[0]["layer"] == "naive"
        assert results[0]["fva_class"] == "BASELINE"
        assert results[1]["layer"] == "statistical"
        assert results[1]["fva_wmape"] > 0  # stat better than naive
        assert results[2]["layer"] == "ml"
        assert results[2]["fva_wmape"] > 0  # ml better than stat

    def test_total_fva(self):
        actual = pl.Series([100.0, 200.0])
        forecasts = {
            "naive": pl.Series([150.0, 250.0]),
            "ml": pl.Series([105.0, 195.0]),
        }
        total = compute_total_fva(actual, forecasts)
        assert total > 0


class TestFVAAnalyzer:
    def _make_backtest_results(self) -> pl.DataFrame:
        """Create synthetic backtest results with multiple models."""
        rows = []
        for fold in range(2):
            for sid in ["S001", "S002", "S003"]:
                # Naive model (worst)
                rows.append({
                    "model_id": "naive_seasonal",
                    "run_id": "fva-test", "lob": "test", "forecast_step": 1,
                    "fold": fold,
                    "series_id": sid,
                    "target_week": "2024-06-01",
                    "actual": 100.0,
                    "forecast": 130.0,
                    "wmape": 0.30,
                    "normalized_bias": 0.30,
                })
                # Statistical model (middle)
                rows.append({
                    "model_id": "auto_arima",
                    "run_id": "fva-test", "lob": "test", "forecast_step": 1,
                    "fold": fold,
                    "series_id": sid,
                    "target_week": "2024-06-01",
                    "actual": 100.0,
                    "forecast": 115.0,
                    "wmape": 0.15,
                    "normalized_bias": 0.15,
                })
                # ML model (best)
                rows.append({
                    "model_id": "lgbm_direct",
                    "run_id": "fva-test", "lob": "test", "forecast_step": 1,
                    "fold": fold,
                    "series_id": sid,
                    "target_week": "2024-06-01",
                    "actual": 100.0,
                    "forecast": 105.0,
                    "wmape": 0.05,
                    "normalized_bias": 0.05,
                })
        return pl.DataFrame(rows)

    def test_compute_fva_detail(self):
        analyzer = FVAAnalyzer()
        results = self._make_backtest_results()
        detail = analyzer.compute_fva_detail(results)

        assert not detail.is_empty()
        assert "forecast_layer" in detail.columns
        assert "fva_wmape" in detail.columns
        assert "fva_class" in detail.columns

        # Check naive is baseline
        naive_rows = detail.filter(pl.col("forecast_layer") == "naive")
        assert all(c == "BASELINE" for c in naive_rows["fva_class"].to_list())

    def test_summarize(self):
        analyzer = FVAAnalyzer()
        results = self._make_backtest_results()
        detail = analyzer.compute_fva_detail(results)
        summary = analyzer.summarize(detail)

        assert not summary.is_empty()
        assert "mean_wmape" in summary.columns
        assert "pct_adds_value" in summary.columns

    def test_layer_leaderboard(self):
        analyzer = FVAAnalyzer()
        results = self._make_backtest_results()
        detail = analyzer.compute_fva_detail(results)
        board = analyzer.layer_leaderboard(detail)

        assert not board.is_empty()
        assert "rank" in board.columns
        assert "recommendation" in board.columns
        assert "robustness_score" in board.columns

    def test_model_layer_classification(self):
        analyzer = FVAAnalyzer()
        assert analyzer.classify_model_layer("naive_seasonal") == "naive"
        assert analyzer.classify_model_layer("seasonal_naive") == "naive"
        assert analyzer.classify_model_layer("auto_arima") == "statistical"
        assert analyzer.classify_model_layer("lgbm_direct") == "ml"
        assert analyzer.classify_model_layer("unknown_model") == "ml"
