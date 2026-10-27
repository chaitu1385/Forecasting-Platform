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
        │       └── <run_date>.parquet
        ├── model_leaderboard/
        │   └── lob=surface/
        │       └── <run_date>.parquet
        └── bias_report/
            └── lob=surface/
                └── <run_date>.parquet
    """

    def __init__(self, base_path: str = "data/bi_exports/"):
        self.base_path = Path(base_path)

    def export_forecast_vs_actual(
        self,
        forecasts: pl.DataFrame,
        actuals: pl.DataFrame,
        lob: str,
        time_col: str = "week",
        id_col: str = "series_id",
        run_date: Optional[str] = None,
    ) -> Path:
        """
        Export a forecast-vs-actual comparison table.

        Joins forecasts and actuals, adds error columns, and writes
        as Parquet partitioned by LOB.
        """
        from datetime import date as dt_date
        run_date = run_date or dt_date.today().isoformat()

        merged = forecasts.join(
            actuals, on=[id_col, time_col], how="inner", suffix="_actual"
        )

        out_dir = self.base_path / "forecast_vs_actual" / f"lob={lob}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{run_date}.parquet"
        merged.write_parquet(str(path))

        logger.info("BI export: forecast_vs_actual → %s", path)
        return path

    def export_leaderboard(
        self,
        metric_store: MetricStore,
        lob: str,
        run_type: str = "backtest",
        run_date: Optional[str] = None,
    ) -> Path:
        """Export model leaderboard for BI."""
        from datetime import date as dt_date
        run_date = run_date or dt_date.today().isoformat()

        board = metric_store.leaderboard(run_type=run_type, lob=lob)

        out_dir = self.base_path / "model_leaderboard" / f"lob={lob}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{run_date}.parquet"
        board.write_parquet(str(path))

        logger.info("BI export: model_leaderboard → %s", path)
        return path

    def export_bias_report(
        self,
        metric_store: MetricStore,
        lob: str,
        model_id: Optional[str] = None,
        run_type: str = "backtest",
        run_date: Optional[str] = None,
    ) -> Path:
        """Export per-series bias report for BI."""
        from datetime import date as dt_date
        run_date = run_date or dt_date.today().isoformat()

        df = metric_store.read(run_type=run_type, lob=lob, model_id=model_id)
        if not df.is_empty():
            from ..metrics.aggregation import observation_metrics

            bias = (
                observation_metrics(df, ["wmape", "normalized_bias"], ["series_id"])
                .with_columns(
                    pl.col("normalized_bias").alias("avg_bias"),
                    pl.col("wmape").alias("avg_wmape"),
                )
                .sort("avg_bias")
            )
        else:
            bias = pl.DataFrame(schema={
                "series_id": pl.Utf8,
                "avg_bias": pl.Float64,
                "avg_wmape": pl.Float64,
                "total_volume": pl.Float64,
            })

        out_dir = self.base_path / "bias_report" / f"lob={lob}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{run_date}.parquet"
        bias.write_parquet(str(path))

        logger.info("BI export: bias_report → %s", path)
        return path

    def export_fva(
        self,
        fva_summary: pl.DataFrame,
        fva_detail: pl.DataFrame,
        lob: str,
        run_date: Optional[str] = None,
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
        from datetime import date as dt_date
        run_date = run_date or dt_date.today().isoformat()
        paths = {}

        # FVA summary
        if not fva_summary.is_empty():
            out_dir = self.base_path / "fva_summary" / f"lob={lob}"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{run_date}.parquet"
            fva_summary.write_parquet(str(path))
            paths["fva_summary"] = path
            logger.info("BI export: fva_summary → %s", path)

        # FVA detail
        if not fva_detail.is_empty():
            out_dir = self.base_path / "fva_detail" / f"lob={lob}"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{run_date}.parquet"
            fva_detail.write_parquet(str(path))
            paths["fva_detail"] = path
            logger.info("BI export: fva_detail → %s", path)

        return paths
