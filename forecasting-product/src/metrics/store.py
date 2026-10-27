"""
Unified metric store — Parquet-backed.

Both backtesting and live performance write to the same schema, enabling
cross-comparison ("does our backtesting champion actually win in production?").
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import polars as pl

from .aggregation import aggregate_metrics, observation_metrics, rank_metrics

if TYPE_CHECKING:
    from ..pipeline.manifest import PipelineManifest
    from ..pipeline.run_store import RunStore

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

    def __init__(self, base_path: str = "data/metrics/", runs_path: Optional[str] = None):
        self.base_path = Path(base_path).resolve()
        self.runs_path = Path(runs_path).resolve() if runs_path is not None else self.base_path / "runs"

    def _committed_runs(
        self, run_type: Optional[str], lob: Optional[str], run_id: Optional[str], latest: bool,
    ) -> tuple[RunStore, list[PipelineManifest]]:
        from ..pipeline.run_store import RunStore, validate_lob

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
        return store, committed

    def _legacy_frames(self) -> list[pl.DataFrame]:
        from ..pipeline.run_store import RunIntegrityError, checked_child

        frames = []
        files = set(self.base_path.glob("run_type=*/lob=*/*.parquet"))
        files.update(self.base_path.glob("backtest/lob=*/*.parquet"))
        files.update(self.base_path.glob("live/lob=*/*.parquet"))
        files.update(self.base_path.glob("*.parquet"))
        for filename in sorted(files):
            path = checked_child(self.base_path, *filename.relative_to(self.base_path).parts)
            try:
                frame = pl.read_parquet(path, hive_partitioning=False)
            except Exception as exc:
                raise RunIntegrityError("unreadable_legacy_metrics") from exc
            partition_type = path.parent.parent.name.removeprefix("run_type=")
            partition_lob = path.parent.name.partition("=")[2]
            for column, value in (("run_type", partition_type), ("lob", partition_lob)):
                if column not in frame.columns and path.parent != self.base_path and value:
                    frame = frame.with_columns(pl.lit(value).alias(column))
            frames.append(frame)
        return frames

    @staticmethod
    def _filter_records(
        df: pl.DataFrame, run_type: Optional[str], lob: Optional[str],
        model_id: Optional[str], grain_level: Optional[str], run_id: Optional[str],
    ) -> pl.DataFrame:
        for column, value in (
            ("run_type", run_type), ("lob", lob), ("model_id", model_id),
            ("grain_level", grain_level), ("run_id", run_id),
        ):
            if value is not None:
                df = df.filter(pl.col(column) == value)
        return df

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
        store, committed = self._committed_runs(run_type, lob, run_id, latest)
        frames = [store.frame(item.lob, item.run_id, "metrics") for item in committed]
        # Explicit legacy partitions only: never scan a run's uncommitted files.
        if not committed or (not latest and run_id is None):
            frames.extend(self._legacy_frames())
        if not frames:
            return pl.DataFrame(schema=METRIC_SCHEMA)
        df = pl.concat(frames, how="diagonal_relaxed")
        return self._filter_records(df, run_type, lob, model_id, grain_level, run_id)

    def leaderboard(
        self,
        run_type: str = "backtest",
        lob: Optional[str] = None,
        primary_metric: Optional[str] = None,
        secondary_metric: Optional[str] = None,
        grain_level: Optional[str] = None,
        run_id: Optional[str] = None,
        latest: bool = False,
    ) -> pl.DataFrame:
        """Return the selected stored ranking, or score explicitly unbundled rows."""
        return self.leaderboard_result(
            run_type, lob, primary_metric, secondary_metric, grain_level, run_id, latest,
        )[0]

    def leaderboard_result(
        self,
        run_type: str = "backtest",
        lob: Optional[str] = None,
        primary_metric: Optional[str] = None,
        secondary_metric: Optional[str] = None,
        grain_level: Optional[str] = None,
        run_id: Optional[str] = None,
        latest: bool = False,
    ) -> tuple[pl.DataFrame, Optional[PipelineManifest]]:
        """
        Resolve ranking and identity together, without a second latest lookup.

        Select ``run_id`` or ``latest=True`` within a LOB when history contains
        multiple runs. Criteria cannot override a committed result. Only
        unbundled raw metrics use the current scorer and its explicit policy.
        """
        from ..pipeline.manifest import read_scoring_contract

        store, committed = self._committed_runs(run_type, lob, run_id, latest)
        if committed:
            if len(committed) != 1:
                raise ValueError("Select one lob and run_id, or latest=True within one lob, for a stored leaderboard")
            selected = committed[0]
            contract = read_scoring_contract(selected.execution)
            for key, requested in (("primary_metric", primary_metric), ("secondary_metric", secondary_metric)):
                if requested is not None and (contract is None or requested != contract[key]):
                    raise ValueError("Stored scoring criteria cannot be overridden; retrieve without metric overrides")
            if grain_level is not None:
                metrics = store.frame(selected.lob, selected.run_id, "metrics")
                if "grain_level" not in metrics.columns or metrics["grain_level"].unique().to_list() != [grain_level]:
                    raise ValueError("The stored leaderboard cannot be rebuilt at a different grain")
            return store.frame(selected.lob, selected.run_id, "leaderboard"), selected

        frames = self._legacy_frames()
        df = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(schema=METRIC_SCHEMA)
        df = self._filter_records(df, run_type, lob, None, grain_level, run_id)
        if latest and run_id is None and not df.is_empty() and "run_id" in df.columns and df["run_id"].n_unique() > 1:
            if "run_date" not in df.columns or df["run_date"].null_count():
                raise ValueError("Legacy metrics have no reliable latest ordering; select an explicit run_id")
            latest_id = df.select("run_id", "run_date").sort(["run_date", "run_id"])["run_id"][-1]
            df = df.filter(pl.col("run_id") == latest_id)
        return self.leaderboard_from_records(
            df, primary_metric or "wmape", secondary_metric or "normalized_bias",
        ), None

    @staticmethod
    def leaderboard_from_records(
        df: pl.DataFrame,
        primary_metric: str = "wmape",
        secondary_metric: str = "normalized_bias",
        group_col: Optional[str] = None,
    ) -> pl.DataFrame:
        """Compute a run-scoped leaderboard from validated raw observations."""

        if df.is_empty():
            return pl.DataFrame(schema={
                "model_id": pl.Utf8,
                primary_metric: pl.Float64,
                secondary_metric: pl.Float64,
                "n_series": pl.UInt32,
            })

        if group_col is None and "competition_scope" in df.columns:
            declared = df["competition_scope"].unique().to_list()
            if len(declared) != 1 or declared[0] is None:
                raise ValueError("Ambiguous competition scope; select a run and explicit grouping")
            group_col = None if declared[0] == "global" else declared[0]
        if group_col is not None and group_col not in df.columns:
            raise ValueError(f"Declared competition column {group_col!r} is missing")
        groups = [group_col] if group_col else []
        board = aggregate_metrics(df, [primary_metric, secondary_metric], groups)
        board = rank_metrics(board, primary_metric, secondary_metric, groups).with_columns(
            (pl.col(group_col).cast(pl.Utf8) if group_col else pl.lit("global")).alias("group_key"),
            pl.lit(group_col or "global").alias("competition_scope"),
        )
        return board.with_columns(
            pl.col("model_id").cum_count().over(["competition_scope", "group_key"]).alias("rank"),
        )

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

        return observation_metrics(df, [metric], [time_column]).sort(time_column)
