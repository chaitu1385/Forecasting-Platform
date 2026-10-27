"""
Unified metric store — Parquet-backed.

Both backtesting and live performance write to the same schema, enabling
cross-comparison ("does our backtesting champion actually win in production?").
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

from .aggregation import aggregate_metrics, rank_metrics

# The canonical schema for all metric records
METRIC_SCHEMA = {
    "run_id": pl.Utf8,
    "run_type": pl.Utf8,       # "backtest" | "live"
    "run_date": pl.Date,
    "lob": pl.Utf8,
    "model_id": pl.Utf8,
    "fold": pl.Int32,          # backtest fold index (-1 for live)
    "grain_level": pl.Utf8,    # "country", "subregion", etc.
    "series_id": pl.Utf8,
    "channel": pl.Utf8,
    "target_week": pl.Date,
    "forecast_step": pl.Int32,
    "actual": pl.Float64,
    "forecast": pl.Float64,
    "wmape": pl.Float64,
    "normalized_bias": pl.Float64,
    "mape": pl.Float64,
    "mae": pl.Float64,
    "rmse": pl.Float64,
    "mase": pl.Float64,
    "mase_scale": pl.Float64,
}


class MetricStore:
    """
    Append-only Parquet store for forecast accuracy metrics.

    Each write appends a new Parquet file (partitioned by run_type and lob).
    Reads scan across all partitions.  This avoids the need for a database
    while supporting incremental writes from both backtest and live engines.
    """

    def __init__(self, base_path: str = "data/metrics/"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write(self, records: pl.DataFrame, run_type: str, lob: str) -> Path:
        """
        Write metric records to the store.

        Parameters
        ----------
        records:
            DataFrame conforming to METRIC_SCHEMA.
        run_type:
            "backtest" or "live".
        lob:
            Line of business identifier.

        Returns
        -------
        Path to the written Parquet file.
        """
        partition_dir = self.base_path / f"run_type={run_type}" / f"lob={lob}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{timestamp}.parquet"
        path = partition_dir / filename

        records.write_parquet(str(path))
        return path

    def read(
        self,
        run_type: Optional[str] = None,
        lob: Optional[str] = None,
        model_id: Optional[str] = None,
        grain_level: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Read metrics with optional filters.

        Returns an empty DataFrame with the correct schema if no data exists.
        """
        pattern = str(self.base_path / "**" / "*.parquet")
        try:
            df = pl.read_parquet(pattern)
        except Exception:
            return pl.DataFrame(schema=METRIC_SCHEMA)

        if run_type is not None:
            df = df.filter(pl.col("run_type") == run_type)
        if lob is not None:
            df = df.filter(pl.col("lob") == lob)
        if model_id is not None:
            df = df.filter(pl.col("model_id") == model_id)
        if grain_level is not None:
            df = df.filter(pl.col("grain_level") == grain_level)

        return df

    def leaderboard(
        self,
        run_type: str = "backtest",
        lob: Optional[str] = None,
        primary_metric: str = "wmape",
        secondary_metric: str = "normalized_bias",
        grain_level: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Compute a model leaderboard ranked by primary metric.

        Returns one row per model with aggregated metric values.
        """
        df = self.read(run_type=run_type, lob=lob, grain_level=grain_level)
        return self.leaderboard_from_records(df, primary_metric, secondary_metric)

    @staticmethod
    def leaderboard_from_records(
        df: pl.DataFrame,
        primary_metric: str = "wmape",
        secondary_metric: str = "normalized_bias",
    ) -> pl.DataFrame:
        """Compute a run-scoped leaderboard from validated raw observations."""

        if df.is_empty():
            return pl.DataFrame(schema={
                "model_id": pl.Utf8,
                primary_metric: pl.Float64,
                secondary_metric: pl.Float64,
                "n_series": pl.UInt32,
            })

        board = aggregate_metrics(df, [primary_metric, secondary_metric])
        return rank_metrics(board, primary_metric, secondary_metric)

    def accuracy_over_time(
        self,
        model_id: str,
        run_type: str = "live",
        lob: Optional[str] = None,
        metric: str = "wmape",
        time_column: str = "target_week",
    ) -> pl.DataFrame:
        """
        Metric trend over time for a specific model.

        Returns one row per week with the aggregated metric.
        """
        df = self.read(run_type=run_type, lob=lob, model_id=model_id)

        if df.is_empty():
            return pl.DataFrame(schema={time_column: pl.Date, metric: pl.Float64})

        return aggregate_metrics(df, [metric], [time_column]).select(time_column, metric).sort(time_column)
