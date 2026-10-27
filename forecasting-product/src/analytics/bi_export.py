"""
Power BI / BI tool export layer.

Writes Parquet files in a Hive-partitioned layout that Power BI (and
similar tools) can directly consume.  No API needed — the BI tool
reads the Parquet files directly from disk or a shared mount.
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

from ..metrics.store import MetricStore

logger = logging.getLogger(__name__)


class BIExporter:
    """
    Export forecast and accuracy data for BI consumption.

    Writes to a partitioned directory structure::

        bi_exports/
        ├── forecast_vs_actual/
        │   └── lob=surface/channel=consumer/
        │       └── <run_date>_<export_id>.parquet
        ├── model_leaderboard/
        │   └── lob=surface/
        │       └── <run_date>_<export_id>.parquet
        └── bias_report/
            └── lob=surface/
                └── <run_date>_<export_id>.parquet
    """

    def __init__(self, base_path: str = "data/bi_exports/"):
        self.base_path = Path(base_path).resolve()

    def _write_export(
        self, frame: pl.DataFrame, report: str, lob: str, run_date: Optional[str],
        run_id: Optional[str] = None, actuals_run_id: Optional[str] = None,
    ) -> Path:
        from datetime import date

        from ..pipeline.run_store import (
            atomic_frame_append,
            checked_child,
            validate_lob,
            validate_run_id,
        )

        validate_lob(lob)
        if run_id is not None:
            validate_run_id(run_id)
        if actuals_run_id is not None:
            validate_run_id(actuals_run_id)
        day = date.fromisoformat(run_date).isoformat() if run_date else date.today().isoformat()
        frame = frame.with_columns(
            pl.lit(run_id, dtype=pl.Utf8).alias("source_run_id"),
            pl.lit(actuals_run_id, dtype=pl.Utf8).alias("actuals_run_id"),
        )
        directory = checked_child(self.base_path, report, f"lob={lob}")
        return atomic_frame_append(frame, directory, prefix=day)

    def export_forecast_vs_actual(
        self,
        forecasts: pl.DataFrame,
        actuals: pl.DataFrame,
        lob: str,
        time_col: str = "week",
        id_col: str = "series_id",
        run_date: Optional[str] = None,
        run_id: Optional[str] = None,
        actuals_run_id: Optional[str] = None,
    ) -> Path:
        """
        Export a forecast-vs-actual comparison table.

        Joins forecasts and actuals, adds error columns, and writes
        as Parquet partitioned by LOB.
        """
        merged = forecasts.join(
            actuals, on=[id_col, time_col], how="inner", suffix="_actual"
        )

        path = self._write_export(merged, "forecast_vs_actual", lob, run_date, run_id, actuals_run_id)

        logger.info("BI export: forecast_vs_actual → %s", path)
        return path

    def export_leaderboard(
        self,
        metric_store: MetricStore,
        lob: str,
        run_type: str = "backtest",
        run_date: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Path:
        """Export model leaderboard for BI."""
        board = metric_store.leaderboard(run_type=run_type, lob=lob, run_id=run_id, latest=True)
        path = self._write_export(board, "model_leaderboard", lob, run_date, run_id)

        logger.info("BI export: model_leaderboard → %s", path)
        return path

    def export_bias_report(
        self,
        metric_store: MetricStore,
        lob: str,
        model_id: Optional[str] = None,
        run_type: str = "backtest",
        run_date: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Path:
        """Export per-series bias report for BI."""
        df = metric_store.read(run_type=run_type, lob=lob, model_id=model_id, run_id=run_id, latest=True)
        if not df.is_empty():
            bias = (
                df.group_by("series_id")
                .agg([
                    pl.col("normalized_bias").mean().alias("avg_bias"),
                    pl.col("wmape").mean().alias("avg_wmape"),
                    pl.col("actual").sum().alias("total_volume"),
                ])
                .sort("avg_bias")
            )
        else:
            bias = pl.DataFrame(schema={
                "series_id": pl.Utf8,
                "avg_bias": pl.Float64,
                "avg_wmape": pl.Float64,
                "total_volume": pl.Float64,
            })

        path = self._write_export(bias, "bias_report", lob, run_date, run_id)

        logger.info("BI export: bias_report → %s", path)
        return path

    def export_fva(
        self,
        fva_summary: pl.DataFrame,
        fva_detail: pl.DataFrame,
        lob: str,
        run_date: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> dict:
        """
        Export FVA analysis tables for BI consumption.

        Parameters
        ----------
        fva_summary:
            Aggregated FVA summary (from FVAAnalyzer.summarize).
        fva_detail:
            Per-series FVA detail (from FVAAnalyzer.compute_fva_detail).
        lob:
            Line of business identifier.
        run_date:
            ISO date string. Defaults to today.

        Returns
        -------
        Dict with paths to written Parquet files.
        """
        paths = {}

        # FVA summary
        if not fva_summary.is_empty():
            path = self._write_export(fva_summary, "fva_summary", lob, run_date, run_id)
            paths["fva_summary"] = path
            logger.info("BI export: fva_summary → %s", path)

        # FVA detail
        if not fva_detail.is_empty():
            path = self._write_export(fva_detail, "fva_detail", lob, run_date, run_id)
            paths["fva_detail"] = path
            logger.info("BI export: fva_detail → %s", path)

        return paths
