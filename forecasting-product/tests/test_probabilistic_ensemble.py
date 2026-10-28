"""
Tests for probabilistic forecasting (predict_quantiles) and weighted ensemble.

Covers:
  - BaseForecaster default predict_quantiles (degenerate P10=P50=P90)
  - SeasonalNaiveForecaster: proper intervals from YoY residuals
  - WeightedEnsembleForecaster: weighted average + weight normalisation
  - ChampionSelector.compute_ensemble_weights: inverse-WMAPE weights, sums to 1
  - BacktestPipeline: selection_strategy="weighted_ensemble" smoke test
  - ForecastPipeline: quantile columns emitted when config.forecast.quantiles set
"""

from datetime import date, timedelta
from typing import List

import polars as pl
import pytest

from src.forecasting.naive import SeasonalNaiveForecaster
from src.forecasting.ensemble import WeightedEnsembleForecaster
from src.backtesting.champion import ChampionSelector
from src.config.schema import (
    BacktestConfig,
    ForecastConfig,
    PlatformConfig,
)

from conftest import make_weekly_series as _make_weekly_series

pytestmark = pytest.mark.unit


def _naive(season_length: int = 52) -> SeasonalNaiveForecaster:
    f = SeasonalNaiveForecaster(season_length=season_length)
    return f


# ── predict_quantiles: SeasonalNaive ──────────────────────────────────────────


class TestNaivePredictQuantiles:

    def test_returns_correct_columns(self):
        f = _naive()
        data = _make_weekly_series(n_series=1, n_weeks=104)
        f.fit(data, target_col="quantity", time_col="week", id_col="series_id")
        qdf = f.predict_quantiles(horizon=13, quantiles=[0.1, 0.5, 0.9])
        assert "forecast_p10" in qdf.columns
        assert "forecast_p50" in qdf.columns
        assert "forecast_p90" in qdf.columns

    def test_returns_correct_row_count(self):
        f = _naive()
        data = _make_weekly_series(n_series=2, n_weeks=104)
        f.fit(data, target_col="quantity", time_col="week", id_col="series_id")
        qdf = f.predict_quantiles(horizon=13, quantiles=[0.1, 0.5, 0.9])
        assert len(qdf) == 2 * 13  # 2 series × 13 weeks

    def test_p10_le_p50_le_p90(self):
        """Every requested percentile is ordered, including the point/P50."""
        f = _naive()
        data = _make_weekly_series(n_series=2, n_weeks=104)
        f.fit(data, target_col="quantity", time_col="week", id_col="series_id")
        qdf = f.predict_quantiles(horizon=13, quantiles=[0.1, 0.5, 0.9])
        violations = (qdf["forecast_p10"] > qdf["forecast_p90"] + 1e-6).sum()
        assert violations == 0, f"{violations} rows where P10 > P90"
        assert (qdf["forecast_p10"] <= qdf["forecast_p50"]).all()
        assert (qdf["forecast_p50"] <= qdf["forecast_p90"]).all()

    def test_base_class_degenerate_intervals(self):
        """When a forecaster doesn't override, all quantiles equal point forecast."""
        f = _naive()
        data = _make_weekly_series(n_series=1, n_weeks=104)
        f.fit(data)
        # Call base-class default by temporarily removing override
        from src.forecasting.base import BaseForecaster
        qdf = BaseForecaster.predict_quantiles(
            f, horizon=4, quantiles=[0.1, 0.5, 0.9]
        )
        # All three quantile columns should equal each other (degenerate intervals)
        diff = (qdf["forecast_p10"] - qdf["forecast_p90"]).abs().max()
        assert diff == pytest.approx(0.0, abs=1e-9)

    def test_single_season_fallback(self):
        """Short series (< 2 seasons) should still return without error."""
        f = _naive(season_length=52)
        data = _make_weekly_series(n_series=1, n_weeks=40)  # < 52 weeks
        f.fit(data)
        qdf = f.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])
        assert len(qdf) == 4
        assert qdf["forecast_p10"].is_null().sum() == 0

    def test_p50_equals_point_forecast(self):
        """forecast_p50 must equal the plain predict() output."""
        f = _naive()
        data = _make_weekly_series(n_series=1, n_weeks=104)
        f.fit(data)
        point = f.predict(horizon=8)
        qdf = f.predict_quantiles(horizon=8, quantiles=[0.5])
        # Join and compare
        merged = point.join(qdf, on=["series_id", "week"], how="inner")
        diff = (merged["forecast"] - merged["forecast_p50"]).abs().max()
        assert diff == pytest.approx(0.0, abs=1e-9)


# ── WeightedEnsembleForecaster ─────────────────────────────────────────────────


class TestWeightedEnsembleForecaster:

    def _make_ensemble(self, weights=None):
        f1 = _naive(season_length=52)
        f2 = _naive(season_length=52)
        f2.name = "naive_alt"  # give it a different name
        w = weights or {"naive_seasonal": 0.6, "naive_alt": 0.4}
        return WeightedEnsembleForecaster(forecasters=[f1, f2], weights=w)

    def test_weights_normalised_to_one(self):
        ens = self._make_ensemble(weights={"naive_seasonal": 3.0, "naive_alt": 1.0})
        assert sum(ens._weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_uniform_fallback_when_all_zero_weights(self):
        ens = self._make_ensemble(weights={"naive_seasonal": 0.0, "naive_alt": 0.0})
        assert sum(ens._weights.values()) == pytest.approx(1.0, abs=1e-9)
        # Both models get equal weight
        assert ens._weights["naive_seasonal"] == pytest.approx(0.5, abs=1e-9)

    def test_empty_forecasters_raises(self):
        with pytest.raises(ValueError):
            WeightedEnsembleForecaster(forecasters=[], weights={})

    def test_predict_returns_correct_shape(self):
        ens = self._make_ensemble()
        data = _make_weekly_series(n_series=2, n_weeks=104)
        ens.fit(data)
        result = ens.predict(horizon=13)
        assert len(result) == 2 * 13

    def test_predict_weighted_average(self):
        """
        Ensemble prediction equals the analytical weighted average of two models
        with different names.
        """
        f1 = _naive(season_length=52)   # name = "naive_seasonal"
        f2 = _naive(season_length=52)
        f2.name = "naive_alt"

        data = _make_weekly_series(n_series=1, n_weeks=104)
        f1.fit(data)
        f2.fit(data)

        ens = WeightedEnsembleForecaster(
            forecasters=[f1, f2],
            weights={"naive_seasonal": 0.6, "naive_alt": 0.4},
        )
        ens.fit(data)

        p_ens = ens.predict(horizon=8)
        p_f1 = f1.predict(horizon=8)
        p_f2 = f2.predict(horizon=8)

        # Manual weighted average (both models produce the same values since
        # they're identical naive forecasters, so the blend equals either one)
        expected = (
            p_f1.with_columns((pl.col("forecast") * 0.6).alias("forecast"))
            .join(
                p_f2.with_columns((pl.col("forecast") * 0.4).alias("f2")),
                on=["series_id", "week"],
            )
            .with_columns((pl.col("forecast") + pl.col("f2")).alias("forecast"))
            .select(["series_id", "week", "forecast"])
        )

        merged = p_ens.join(expected.rename({"forecast": "expected"}), on=["series_id", "week"])
        diff = (merged["forecast"] - merged["expected"]).abs().max()
        assert diff == pytest.approx(0.0, abs=1e-6)

    def test_predict_quantiles_returns_correct_columns(self):
        ens = self._make_ensemble()
        data = _make_weekly_series(n_series=2, n_weeks=104)
        ens.fit(data)
        qdf = ens.predict_quantiles(horizon=8, quantiles=[0.1, 0.5, 0.9])
        assert "forecast_p10" in qdf.columns
        assert "forecast_p50" in qdf.columns
        assert "forecast_p90" in qdf.columns
        assert len(qdf) == 2 * 8

    def test_predict_quantiles_monotone(self):
        ens = self._make_ensemble()
        data = _make_weekly_series(n_series=2, n_weeks=104)
        ens.fit(data)
        qdf = ens.predict_quantiles(horizon=8, quantiles=[0.1, 0.5, 0.9])
        assert (qdf["forecast_p10"] > qdf["forecast_p90"] + 1e-6).sum() == 0
        assert (qdf["forecast_p10"] <= qdf["forecast_p50"]).all()
        assert (qdf["forecast_p50"] <= qdf["forecast_p90"]).all()

    def test_get_params_contains_weights(self):
        ens = self._make_ensemble()
        params = ens.get_params()
        assert "weights" in params
        assert "models" in params


# ── ChampionSelector.compute_ensemble_weights ─────────────────────────────────


class TestComputeEnsembleWeights:

    def _make_backtest_results(self) -> pl.DataFrame:
        rows = []
        models = [("lgbm_direct", 0.08), ("auto_arima", 0.15), ("naive_seasonal", 0.20)]
        for model, wmape in models:
            for fold in range(3):
                for sid in ["SKU-000", "SKU-001"]:
                    rows.append({
                        "model_id": model,
                        "fold": fold,
                        "series_id": sid,
                        "wmape": wmape + 0.01 * fold,
                        "normalized_bias": 0.0,
                        "run_id": "ensemble-oracle",
                        "lob": "test",
                        "target_week": date(2024, 1, 1) + timedelta(weeks=fold),
                        "forecast_step": 1,
                        "actual": 100.0,
                        "forecast": 100.0 * (1 + wmape + 0.01 * fold),
                    })
        return pl.DataFrame(rows)

    def test_weights_sum_to_one(self):
        cfg = BacktestConfig()
        sel = ChampionSelector(cfg)
        results = self._make_backtest_results()
        weights = sel.compute_ensemble_weights(results)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_best_model_gets_highest_weight(self):
        cfg = BacktestConfig()
        sel = ChampionSelector(cfg)
        results = self._make_backtest_results()
        weights = sel.compute_ensemble_weights(results)
        # lgbm_direct has lowest WMAPE (0.08) → should get highest weight
        assert weights["lgbm_direct"] > weights["auto_arima"]
        assert weights["auto_arima"] > weights["naive_seasonal"]

    def test_empty_results_returns_empty(self):
        cfg = BacktestConfig()
        sel = ChampionSelector(cfg)
        weights = sel.compute_ensemble_weights(pl.DataFrame())
        assert weights == {}

    def test_single_model(self):
        cfg = BacktestConfig()
        sel = ChampionSelector(cfg)
        results = pl.DataFrame({
            "model_id": ["lgbm_direct"] * 4,
            "fold": [0, 0, 1, 1],
            "series_id": ["A", "B"] * 2,
            "wmape": [0.10] * 4,
            "normalized_bias": [0.0] * 4,
            "run_id": ["single-model"] * 4,
            "lob": ["test"] * 4,
            "target_week": [date(2024, 1, 1)] * 2 + [date(2024, 1, 8)] * 2,
            "forecast_step": [1] * 4,
            "actual": [100.0] * 4,
            "forecast": [110.0] * 4,
        })
        weights = sel.compute_ensemble_weights(results)
        assert weights["lgbm_direct"] == pytest.approx(1.0, abs=1e-9)


# ── BacktestConfig: selection_strategy field ──────────────────────────────────


class TestSelectionStrategyConfig:

    def test_default_is_champion(self):
        cfg = BacktestConfig()
        assert cfg.selection_strategy == "champion"

    def test_can_set_weighted_ensemble(self):
        cfg = BacktestConfig(selection_strategy="weighted_ensemble")
        assert cfg.selection_strategy == "weighted_ensemble"


# ── ForecastConfig: quantiles field ───────────────────────────────────────────


class TestForecastConfigQuantiles:

    def test_default_empty(self):
        cfg = ForecastConfig()
        assert cfg.quantiles == []

    def test_can_configure_quantiles(self):
        cfg = ForecastConfig(quantiles=[0.1, 0.5, 0.9])
        assert cfg.quantiles == [0.1, 0.5, 0.9]


# ── BacktestPipeline ensemble integration smoke test ─────────────────────────


class TestBacktestPipelineEnsemble:

    def test_ensemble_returned_when_strategy_weighted(self, tmp_path):
        """BacktestPipeline.run() returns an ensemble object when strategy=weighted_ensemble."""
        import sys
        import os
        sys.path.insert(0, str(tmp_path))

        from src.config.schema import (
            PlatformConfig, ForecastConfig, BacktestConfig, OutputConfig
        )
        from src.pipeline.backtest import BacktestPipeline

        config = PlatformConfig(
            lob="test",
            forecast=ForecastConfig(
                horizon_weeks=4,
                forecasters=["naive_seasonal"],
                target_column="quantity",
                time_column="week",
                series_id_column="series_id",
            ),
            backtest=BacktestConfig(
                n_folds=2,
                val_weeks=4,
                gap_weeks=0,
                selection_strategy="weighted_ensemble",
            ),
            output=OutputConfig(
                forecast_path=str(tmp_path / "forecasts"),
                metrics_path=str(tmp_path / "metrics"),
            ),
            metrics=["wmape", "normalized_bias"],
        )

        pipeline = BacktestPipeline(config)
        data = _make_weekly_series(n_series=2, n_weeks=52)
        results = pipeline.run(data)

        assert "ensemble" in results
        assert isinstance(results["ensemble"], WeightedEnsembleForecaster)

    def test_no_ensemble_when_strategy_champion(self, tmp_path):
        from src.config.schema import (
            PlatformConfig, ForecastConfig, BacktestConfig, OutputConfig
        )
        from src.pipeline.backtest import BacktestPipeline

        config = PlatformConfig(
            lob="test",
            forecast=ForecastConfig(
                horizon_weeks=4,
                forecasters=["naive_seasonal"],
                target_column="quantity",
                time_column="week",
                series_id_column="series_id",
            ),
            backtest=BacktestConfig(
                n_folds=2,
                val_weeks=4,
                gap_weeks=0,
                selection_strategy="champion",
            ),
            output=OutputConfig(
                forecast_path=str(tmp_path / "forecasts"),
                metrics_path=str(tmp_path / "metrics"),
            ),
            metrics=["wmape", "normalized_bias"],
        )

        pipeline = BacktestPipeline(config)
        data = _make_weekly_series(n_series=2, n_weeks=52)
        results = pipeline.run(data)

        assert results["ensemble"] is None
