"""
Unified metric store — Parquet-backed.

Both backtesting and live performance write to the same schema, enabling
cross-comparison ("does our backtesting champion actually win in production?").
"""

from pathlib import Path
from typing import Optional

import polars as pl

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
}


class MetricStore:
    """
    Append-only Parquet store for forecast accuracy metrics.

    Each write appends a new Parquet file (partitioned by run_type and lob).
    Reads scan across all partitions.  This avoids the need for a database
    while supporting incremental writes from both backtest and live engines.
    """

    def __init__(self, base_path: str = "data/metrics/", runs_path: Optional[str] = None):
        self.base_path = Path(base_path).resolve()
        self.runs_path = Path(runs_path).resolve() if runs_path is not None else self.base_path / "runs"

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
        from ..pipeline.run_store import atomic_frame_append, checked_child, validate_lob

        validate_lob(lob)
        if run_type not in {"backtest", "live"}:
            raise ValueError("Unsupported metric run type")
        for column, value in (("run_type", run_type), ("lob", lob)):
            if column not in records.columns:
                records = records.with_columns(pl.lit(value).alias(column))
            elif records.filter(pl.col(column).is_null() | (pl.col(column) != value)).height:
                raise ValueError("Metric partition does not match record identity")
        partition_dir = checked_child(self.base_path, f"run_type={run_type}", f"lob={lob}")
        return atomic_frame_append(records, partition_dir)

    def read(
        self,
        run_type: Optional[str] = None,
        lob: Optional[str] = None,
        model_id: Optional[str] = None,
        grain_level: Optional[str] = None,
        run_id: Optional[str] = None,
        latest: bool = False,
    ) -> pl.DataFrame:
        """
        Read metrics with optional filters.

        Returns an empty DataFrame with the correct schema if no data exists.
        """
        from ..pipeline.run_store import RunStore, checked_child, validate_lob

        if lob is not None:
            validate_lob(lob)
        store = RunStore(self.runs_path)
        committed = store.list(lob=lob, run_type="backtest") if run_type in (None, "backtest") else []
        if run_id is not None:
            committed = [item for item in committed if item.run_id == run_id]
        if latest:
            selected = {}
            for item in committed:
                selected.setdefault(item.lob, item)
            committed = list(selected.values())
        frames = [store.frame(item.lob, item.run_id, "metrics") for item in committed]
        # Explicit legacy partitions only: never scan a run's uncommitted files.
        if not committed or (not latest and run_id is None):
            files = set(self.base_path.glob("run_type=*/lob=*/*.parquet"))
            files.update(self.base_path.glob("backtest/lob=*/*.parquet"))
            files.update(self.base_path.glob("live/lob=*/*.parquet"))
            files.update(self.base_path.glob("*.parquet"))
            for filename in sorted(files):
                path = checked_child(self.base_path, *filename.relative_to(self.base_path).parts)
                try:
                    frame = pl.read_parquet(path, hive_partitioning=False)
                except Exception as exc:
                    from ..pipeline.run_store import RunIntegrityError
                    raise RunIntegrityError("unreadable_legacy_metrics") from exc
                partition_type = path.parent.parent.name.removeprefix("run_type=")
                partition_lob = path.parent.name.partition("=")[2]
                for column, value in (("run_type", partition_type), ("lob", partition_lob)):
                    if column not in frame.columns and path.parent != self.base_path and value:
                        frame = frame.with_columns(pl.lit(value).alias(column))
                frames.append(frame)
        if not frames:
            return pl.DataFrame(schema=METRIC_SCHEMA)
        df = pl.concat(frames, how="diagonal_relaxed")

        if run_type is not None:
            df = df.filter(pl.col("run_type") == run_type)
        if lob is not None:
            df = df.filter(pl.col("lob") == lob)
        if model_id is not None:
            df = df.filter(pl.col("model_id") == model_id)
        if grain_level is not None:
            df = df.filter(pl.col("grain_level") == grain_level)
        if run_id is not None:
            df = df.filter(pl.col("run_id") == run_id)

        return df

    def leaderboard(
        self,
        run_type: str = "backtest",
        lob: Optional[str] = None,
        primary_metric: str = "wmape",
        secondary_metric: str = "normalized_bias",
        grain_level: Optional[str] = None,
        run_id: Optional[str] = None,
        latest: bool = False,
    ) -> pl.DataFrame:
        """
        Compute a model leaderboard ranked by primary metric.

        Returns one row per model with aggregated metric values.
        """
        df = self.read(run_type=run_type, lob=lob, grain_level=grain_level, run_id=run_id, latest=latest)
        return self.leaderboard_from_records(df, primary_metric, secondary_metric)

    @staticmethod
    def leaderboard_from_records(
        df: pl.DataFrame,
        primary_metric: str = "wmape",
        secondary_metric: str = "normalized_bias",
    ) -> pl.DataFrame:
        """Apply the existing aggregation formula to one explicit record set."""

        if df.is_empty():
            return pl.DataFrame(schema={
                "model_id": pl.Utf8,
                primary_metric: pl.Float64,
                secondary_metric: pl.Float64,
                "n_series": pl.UInt32,
            })

        board = (
            df.group_by("model_id")
            .agg([
                pl.col(primary_metric).mean().alias(primary_metric),
                pl.col(secondary_metric).mean().alias(secondary_metric),
                pl.col("series_id").n_unique().alias("n_series"),
            ])
            .sort(primary_metric)
        )
        return board

    def accuracy_over_time(
        self,
        model_id: str,
        run_type: str = "live",
        lob: Optional[str] = None,
        metric: str = "wmape",
        time_column: str = "target_week",
        run_id: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Metric trend over time for a specific model.

        Returns one row per week with the aggregated metric.
        """
        df = self.read(run_type=run_type, lob=lob, model_id=model_id, run_id=run_id)

        if df.is_empty():
            return pl.DataFrame(schema={time_column: pl.Date, metric: pl.Float64})

        return (
            df.group_by(time_column)
            .agg(pl.col(metric).mean().alias(metric))
            .sort(time_column)
        )
