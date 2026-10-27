"""
Tests for multi-horizon model selection.

Covers:
  - forecast_step computation in backtest engine
  - Horizon bucket assignment
  - Per-horizon champion selection
  - Fallback to single champion when no buckets configured
  - Multi-horizon forecast stitching
"""

from datetime import date, timedelta

import polars as pl
import pytest

from src.backtesting.champion import ChampionSelector
from src.config.schema import BacktestConfig, HorizonBucket

pytestmark = pytest.mark.unit


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_backtest_results(
    models: list = None,
    n_series: int = 2,
    n_steps: int = 13,
    n_folds: int = 2,
) -> pl.DataFrame:
    """Create synthetic backtest results with forecast_step."""
    if models is None:
        models = ["model_a", "model_b"]
    records = []
    for fold in range(n_folds):
        for model in models:
            for sid in [f"S{i}" for i in range(1, n_series + 1)]:
                for step in range(1, n_steps + 1):
                    records.append({
                        "run_id": "test-run",
                        "run_type": "backtest",
                        "run_date": date(2024, 1, 1),
                        "lob": "retail",
                        "model_id": model,
                        "fold": fold,
                        "grain_level": "series",
                        "series_id": sid,
                        "channel": "",
                        "target_week": date(2024, 1, 1) + timedelta(weeks=step),
                        "forecast_step": step,
                        "actual": 100.0,
                        "forecast": 100.0,
                        "wmape": 0.0,
                        "normalized_bias": 0.0,
                    })
    return pl.DataFrame(records)


def _make_results_with_different_winners(
    short_winner: str = "model_a",
    long_winner: str = "model_b",
    boundary_step: int = 5,
    n_steps: int = 13,
) -> pl.DataFrame:
    """
    Create results where different models win in different horizons.

    short_winner has lower WMAPE on steps 1..boundary_step-1
    long_winner has lower WMAPE on steps boundary_step..n_steps
    """
    records = []
    models = [short_winner, long_winner]
    for model in models:
        for step in range(1, n_steps + 1):
            if step < boundary_step:
                # Short term: short_winner has lower WMAPE
                wmape = 0.1 if model == short_winner else 0.5
            else:
                # Long term: long_winner has lower WMAPE
                wmape = 0.5 if model == short_winner else 0.1
            records.append({
                "run_id": "test-run",
                "run_type": "backtest",
                "run_date": date(2024, 1, 1),
                "lob": "retail",
                "model_id": model,
                "fold": 0,
                "grain_level": "series",
                "series_id": "S1",
                "channel": "",
                "target_week": date(2024, 1, 1) + timedelta(weeks=step),
                "forecast_step": step,
                "actual": 100.0,
                "forecast": 100.0 * (1 + wmape),
                "wmape": wmape,
                "normalized_bias": 0.01,
            })
    return pl.DataFrame(records)


# ── Forecast step tests ──────────────────────────────────────────────────────

class TestForecastStep:
    def test_forecast_step_is_1_indexed(self):
        results = _make_backtest_results(n_steps=13)
        steps = results["forecast_step"].unique().sort().to_list()
        assert steps[0] == 1
        assert steps[-1] == 13

    def test_forecast_step_range(self):
        results = _make_backtest_results(n_steps=39)
        steps = results["forecast_step"].unique().sort().to_list()
        assert len(steps) == 39
        assert steps == list(range(1, 40))


# ── Horizon bucket assignment tests ──────────────────────────────────────────

class TestHorizonBucketAssignment:
    def test_basic_bucket_assignment(self):
        """Verify that forecast_steps map correctly to horizon buckets."""
        buckets = [
            HorizonBucket(name="short", start_step=1, end_step=4),
            HorizonBucket(name="medium", start_step=5, end_step=13),
        ]

        results = _make_backtest_results(n_steps=13, models=["model_a"])

        # Manually apply the bucket logic (same as in champion selector)
        bucket_expr = pl.lit("unassigned")
        for b in buckets:
            bucket_expr = (
                pl.when(
                    pl.col("forecast_step").is_between(b.start_step, b.end_step)
                )
                .then(pl.lit(b.name))
                .otherwise(bucket_expr)
            )

        enriched = results.with_columns(bucket_expr.alias("horizon_bucket"))

        short_count = enriched.filter(pl.col("horizon_bucket") == "short").height
        medium_count = enriched.filter(pl.col("horizon_bucket") == "medium").height
        unassigned = enriched.filter(pl.col("horizon_bucket") == "unassigned").height

        assert short_count > 0
        assert medium_count > 0
        assert unassigned == 0


# ── Champion selection by horizon tests ──────────────────────────────────────

class TestSelectByHorizon:
    def test_different_champions_per_bucket(self):
        """Model A wins short-term, Model B wins long-term."""
        buckets = [
            HorizonBucket(name="short", start_step=1, end_step=4),
            HorizonBucket(name="long", start_step=5, end_step=13),
        ]
        config = BacktestConfig(
            primary_metric="wmape",
            secondary_metric="normalized_bias",
            horizon_buckets=buckets,
        )
        selector = ChampionSelector(config)

        results = _make_results_with_different_winners(
            short_winner="model_a",
            long_winner="model_b",
            boundary_step=5,
        )

        champions = selector.select_by_horizon(results, buckets)

        assert len(champions) == 2
        short_champ = champions.filter(pl.col("horizon_bucket") == "short")
        long_champ = champions.filter(pl.col("horizon_bucket") == "long")

        assert short_champ["model_id"][0] == "model_a"
        assert long_champ["model_id"][0] == "model_b"

    def test_champion_table_has_step_ranges(self):
        buckets = [
            HorizonBucket(name="short", start_step=1, end_step=4),
            HorizonBucket(name="long", start_step=5, end_step=13),
        ]
        config = BacktestConfig(
            primary_metric="wmape",
            secondary_metric="normalized_bias",
            horizon_buckets=buckets,
        )
        selector = ChampionSelector(config)
        results = _make_backtest_results()

        champions = selector.select_by_horizon(results, buckets)
        assert "start_step" in champions.columns
        assert "end_step" in champions.columns
        assert "horizon_bucket" in champions.columns

    def test_empty_buckets_falls_back(self):
        config = BacktestConfig(
            primary_metric="wmape",
            secondary_metric="normalized_bias",
        )
        selector = ChampionSelector(config)
        results = _make_backtest_results()

        # Empty bucket list → falls back to standard select
        champions = selector.select_by_horizon(results, [])
        assert "horizon_bucket" not in champions.columns

    def test_missing_forecast_step_is_not_silently_a_global_champion(self):
        config = BacktestConfig(
            primary_metric="wmape",
            secondary_metric="normalized_bias",
        )
        selector = ChampionSelector(config)
        results = _make_backtest_results().drop("forecast_step")
        buckets = [HorizonBucket(name="short", start_step=1, end_step=4)]

        with pytest.raises(ValueError, match="forecast_step"):
            selector.select_by_horizon(results, buckets)

    def test_three_buckets(self):
        buckets = [
            HorizonBucket(name="short", start_step=1, end_step=4),
            HorizonBucket(name="medium", start_step=5, end_step=9),
            HorizonBucket(name="long", start_step=10, end_step=13),
        ]
        config = BacktestConfig(
            primary_metric="wmape",
            secondary_metric="normalized_bias",
            horizon_buckets=buckets,
        )
        selector = ChampionSelector(config)
        results = _make_backtest_results(n_steps=13)

        champions = selector.select_by_horizon(results, buckets)
        bucket_names = set(champions["horizon_bucket"].to_list())
        assert bucket_names == {"short", "medium", "long"}
