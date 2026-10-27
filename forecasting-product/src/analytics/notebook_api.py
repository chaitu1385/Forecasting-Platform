"""
Data-scientist-friendly analytics API.

Provides pre-built queries over the metric store for common analysis
patterns.  All methods return Polars DataFrames — ready for inline
exploration in notebooks.

Usage
-----
>>> from src.analytics import ForecastAnalytics
>>> fa = ForecastAnalytics(metric_store_path="data/metrics/")
>>> fa.model_leaderboard(lob="surface")
>>> fa.accuracy_over_time(model="lgbm_direct", channel="consumer")
>>> fa.compare_reconciliation_strategies(metric="wmape")
"""

from typing import Optional

import polars as pl

from ..metrics.store import MetricStore


class ForecastAnalytics:
    """
    Notebook-ready analytics over the unified metric store.

    Designed for interactive exploration: each method does one query and
    returns a DataFrame.  Chain with ``.head()``, ``.filter()``, etc.
    """

    def __init__(self, metric_store_path: str = "data/metrics/", runs_path: Optional[str] = None):
        self._store = MetricStore(metric_store_path, runs_path=runs_path)

    # ── Model comparison ──────────────────────────────────────────────────

    def model_leaderboard(
        self,
        lob: Optional[str] = None,
        run_type: str = "backtest",
        grain_level: Optional[str] = None,
        primary_metric: str = "wmape",
        secondary_metric: str = "normalized_bias",
        run_id: Optional[str] = None,
        latest: bool = False,
    ) -> pl.DataFrame:
        """
        Rank models by primary metric (lower WMAPE = better).

        Returns one row per model with aggregated metrics.
        """
        return self._store.leaderboard(
            run_type=run_type,
            lob=lob,
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
            grain_level=grain_level,
            run_id=run_id,
            latest=latest,
        )

    def model_comparison_by_fold(
        self,
        lob: Optional[str] = None,
        metric: str = "wmape",
        run_id: Optional[str] = None,
    ) -> pl.DataFrame:
        """Per-fold model performance — shows stability across time."""
        df = self._store.read(run_type="backtest", lob=lob, run_id=run_id)
        if df.is_empty():
            return df

        return (
            df.group_by(["model_id", "fold"])
            .agg(pl.col(metric).mean().alias(metric))
            .sort(["model_id", "fold"])
        )

    # ── Accuracy over time ────────────────────────────────────────────────

    def accuracy_over_time(
        self,
        model: Optional[str] = None,
        lob: Optional[str] = None,
        run_type: str = "live",
        metric: str = "wmape",
        run_id: Optional[str] = None,
    ) -> pl.DataFrame:
        """Weekly accuracy trend for a model — identifies drift."""
        return self._store.accuracy_over_time(
            model_id=model or "",
            run_type=run_type,
            lob=lob,
            metric=metric,
            run_id=run_id,
        )

    # ── Grain drill-down ──────────────────────────────────────────────────

    def accuracy_by_grain(
        self,
        grain_column: str,
        lob: Optional[str] = None,
        model: Optional[str] = None,
        run_type: str = "backtest",
        metric: str = "wmape",
        run_id: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Accuracy broken down by a hierarchy grain.

        Parameters
        ----------
        grain_column:
            Column to group by (e.g. "series_id", "channel").
        """
        df = self._store.read(run_type=run_type, lob=lob, model_id=model, run_id=run_id)
        if df.is_empty() or grain_column not in df.columns:
            return df

        return (
            df.group_by(grain_column)
            .agg([
                pl.col(metric).mean().alias(f"avg_{metric}"),
                pl.col(metric).std().alias(f"std_{metric}"),
                pl.col("series_id").n_unique().alias("n_series"),
            ])
            .sort(f"avg_{metric}")
        )

    # ── Bias analysis ─────────────────────────────────────────────────────

    def bias_distribution(
        self,
        model: Optional[str] = None,
        lob: Optional[str] = None,
        run_type: str = "backtest",
        run_id: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Bias distribution across series.

        Returns per-series average normalized bias — helps identify
        which series are systematically over/under-forecasted.
        """
        df = self._store.read(run_type=run_type, lob=lob, model_id=model, run_id=run_id)
        if df.is_empty():
            return df

        return (
            df.group_by("series_id")
            .agg([
                pl.col("normalized_bias").mean().alias("avg_bias"),
                pl.col("normalized_bias").std().alias("std_bias"),
                pl.col("wmape").mean().alias("avg_wmape"),
            ])
            .sort("avg_bias")
        )

    # ── Transition impact ─────────────────────────────────────────────────

    def transition_impact(
        self,
        old_sku: str,
        new_sku: str,
        run_type: str = "backtest",
        metric: str = "wmape",
        lob: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Compare accuracy before and after a product transition.

        Shows how well the stitched series forecasts compared to the
        original series.
        """
        df = self._store.read(run_type=run_type, lob=lob, run_id=run_id)
        if df.is_empty():
            return df

        return df.filter(
            pl.col("series_id").is_in([old_sku, new_sku])
        ).group_by(["series_id", "model_id"]).agg(
            pl.col(metric).mean().alias(f"avg_{metric}")
        ).sort(["series_id", f"avg_{metric}"])

    # ── Backtest vs Live comparison ───────────────────────────────────────

    def backtest_vs_live(
        self,
        lob: Optional[str] = None,
        metric: str = "wmape",
        backtest_run_id: Optional[str] = None,
        live_run_id: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Compare backtest performance to live performance per model.

        Answers: "does our backtesting champion actually win in production?"
        """
        bt = self._store.read(run_type="backtest", lob=lob, run_id=backtest_run_id)
        live = self._store.read(run_type="live", lob=lob, run_id=live_run_id)

        if bt.is_empty() or live.is_empty():
            return pl.DataFrame(schema={
                "model_id": pl.Utf8,
                f"backtest_{metric}": pl.Float64,
                f"live_{metric}": pl.Float64,
            })

        bt_agg = (
            bt.group_by("model_id")
            .agg(pl.col(metric).mean().alias(f"backtest_{metric}"))
        )
        live_agg = (
            live.group_by("model_id")
            .agg(pl.col(metric).mean().alias(f"live_{metric}"))
        )

        return bt_agg.join(live_agg, on="model_id", how="outer_coalesce").sort(
            f"backtest_{metric}"
        )
