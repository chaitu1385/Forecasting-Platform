"""
FabricNotebookAdapter — one-liner setup for Fabric notebook cells.

Translates platform YAML config + FabricConfig into a ready-to-run pipeline
context, eliminating the boilerplate in ``notebooks/01-03``.

Usage
-----
In a Fabric notebook cell:

>>> from src.fabric.notebook_adapter import FabricNotebookAdapter
>>> adapter = FabricNotebookAdapter("configs/platform_config.yaml")
>>> adapter.summary()
>>> actuals = adapter.load_actuals()
>>> forecasts = adapter.run_forecast(champion="lgbm_direct")
>>> adapter.write_results(forecasts, "forecasts")
"""

import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from pyspark.sql import DataFrame as SparkDataFrame  # noqa: F401
    _HAS_SPARK = True
except ImportError:
    _HAS_SPARK = False


class FabricNotebookAdapter:
    """
    One-liner setup for Microsoft Fabric Lakehouse notebooks.

    Initialises Spark, loads config, connects to the Lakehouse, and exposes
    high-level methods for backtest / forecast / write cycles.

    Parameters
    ----------
    config_path:
        Path to the platform YAML config (e.g. ``configs/platform_config.yaml``).
    fabric_config_path:
        Path to the Fabric-specific YAML config (default:
        ``configs/fabric_config.yaml``).
    lob:
        Line-of-business identifier (overrides ``FORECAST_LOB`` env var).
    workspace:
        Fabric workspace name (overrides ``FABRIC_WORKSPACE`` env var).
    lakehouse:
        Fabric lakehouse name (overrides ``FABRIC_LAKEHOUSE`` env var).
    """

    def __init__(
        self,
        config_path: str = "configs/platform_config.yaml",
        fabric_config_path: str = "configs/fabric_config.yaml",
        lob: Optional[str] = None,
        workspace: Optional[str] = None,
        lakehouse: Optional[str] = None,
    ):
        if not _HAS_SPARK:
            raise ImportError(
                "PySpark is required for FabricNotebookAdapter.  "
                "This class is designed for use inside Fabric Spark notebooks."
            )

        self._lob = lob or os.environ.get("FORECAST_LOB", "default")
        self._workspace = workspace or os.environ.get("FABRIC_WORKSPACE", "")
        self._lakehouse = lakehouse or os.environ.get("FABRIC_LAKEHOUSE", "")

        # ── Load configs ──────────────────────────────────────────────────
        from ..config.loader import load_config
        self._config = load_config(config_path)
        self._config.lob = self._lob

        import yaml
        with open(fabric_config_path) as f:
            self._fabric_yaml = yaml.safe_load(f)

        # ── Spark session ─────────────────────────────────────────────────
        from ..spark.session import get_or_create_spark
        self._spark = get_or_create_spark(
            app_name=f"ForecastingPlatform-{self._lob}"
        )
        self._spark.sparkContext.setLogLevel("WARN")

        # ── Fabric Lakehouse client ───────────────────────────────────────
        from .config import FabricConfig
        from .lakehouse import FabricLakehouse

        self._fabric_cfg = FabricConfig(
            workspace=self._workspace,
            lakehouse=self._lakehouse,
            environment=os.environ.get("FABRIC_ENVIRONMENT", "development"),
        )
        self._lakehouse_client = FabricLakehouse(self._spark, self._fabric_cfg)

        # ── Delta writer ──────────────────────────────────────────────────
        from .delta_writer import DeltaWriter
        self._writer = DeltaWriter(self._spark, self._fabric_cfg)

        # ── Spark pipeline ────────────────────────────────────────────────
        from ..spark.pipeline import SparkForecastPipeline
        self._pipeline = SparkForecastPipeline(self._spark, self._config)

        logger.info(
            "FabricNotebookAdapter ready: workspace=%s, lakehouse=%s, lob=%s",
            self._workspace, self._lakehouse, self._lob,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def spark(self):
        """Active SparkSession."""
        return self._spark

    @property
    def lakehouse(self):
        """FabricLakehouse client for direct table reads/writes."""
        return self._lakehouse_client

    @property
    def config(self):
        """Loaded PlatformConfig."""
        return self._config

    @property
    def fabric_config(self):
        """FabricConfig with workspace/lakehouse identifiers."""
        return self._fabric_cfg

    @property
    def lob(self) -> str:
        """Line-of-business identifier."""
        return self._lob

    # ── High-level operations ─────────────────────────────────────────────────

    def load_actuals(
        self,
        table_name: str = "actuals",
        date_col: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs,
    ):
        """
        Load actuals from a Lakehouse Delta table.

        Parameters
        ----------
        table_name:
            Delta table name (default: ``"actuals"``).
        date_col:
            Date column for partition pruning (optional).
        start_date:
            Inclusive lower bound date (ISO-8601).
        end_date:
            Exclusive upper bound date (ISO-8601).

        Returns
        -------
        pyspark.sql.DataFrame
        """
        df = self._lakehouse_client.read_table(
            table_name,
            date_col=date_col,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
        logger.info("Loaded %s from Lakehouse", table_name)
        return df

    def build_series(self, actuals_sdf):
        """
        Build canonical series panel from raw actuals using SparkSeriesBuilder.

        Parameters
        ----------
        actuals_sdf:
            Raw actuals Spark DataFrame.

        Returns
        -------
        pyspark.sql.DataFrame with standardised columns.
        """
        from ..spark.series_builder import SparkSeriesBuilder

        sb_cfg = self._fabric_yaml.get("series_builder", {})
        builder = SparkSeriesBuilder.from_config(sb_cfg)
        return builder.build(actuals_sdf)

    def run_backtest(
        self,
        actuals_sdf,
        model_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Run distributed walk-forward backtest.

        Parameters
        ----------
        actuals_sdf:
            Canonical actuals Spark DataFrame.
        model_names:
            Models to evaluate (defaults to config.forecast.forecasters).

        Returns
        -------
        Spark DataFrame with validated raw observations and a ``model`` column.
        Project with ``backtest_metrics_for_legacy_storage`` when writing a
        pre-existing per-series/model/fold Delta metrics table.
        """
        return self._pipeline.run_backtest(
            actuals_sdf,
            model_names=model_names or self._config.forecast.forecasters,
            **kwargs,
        )

    def select_champion(
        self,
        backtest_results_sdf,
        primary_metric: Optional[str] = None,
        secondary_metric: Optional[str] = None,
    ) -> str:
        """
        Select the best model from backtest results.

        Returns the champion model name.
        """
        from pyspark.sql import functions as F

        leaderboard = self._pipeline.select_champion(
            backtest_results_sdf,
            primary_metric=primary_metric or self._config.backtest.primary_metric,
            secondary_metric=secondary_metric or self._config.backtest.secondary_metric,
        )
        model_column = "model" if "model" in leaderboard.columns else "model_id"
        champion = (
            leaderboard
            .filter(F.col("rank") == 1)
            .select(model_column)
            .first()[0]
        )
        logger.info("Champion model: %s", champion)
        return champion

    def run_forecast(
        self,
        actuals_sdf,
        champion: str,
        horizon: Optional[int] = None,
        **kwargs,
    ):
        """
        Generate forecasts using the champion model.

        Parameters
        ----------
        actuals_sdf:
            Canonical actuals Spark DataFrame.
        champion:
            Model name to use for forecasting.
        horizon:
            Forecast horizon (defaults to config value).

        Returns
        -------
        Spark DataFrame: [series_id, week, forecast].
        """
        return self._pipeline.run_forecast(
            actuals_sdf,
            champion_model=champion,
            horizon=horizon,
            **kwargs,
        )

    def write_results(
        self,
        df,
        table_name: str,
        mode: str = "upsert",
        partition_by: Optional[List[str]] = None,
        merge_keys: Optional[List[str]] = None,
    ) -> str:
        """
        Write results to a Lakehouse Delta table.

        Parameters
        ----------
        df:
            Spark DataFrame to write.
        table_name:
            Target table name.
        mode:
            ``"upsert"`` | ``"overwrite"`` | ``"append"``.
        partition_by:
            Partition columns (for overwrite/append).
        merge_keys:
            Merge keys (for upsert mode).

        Returns
        -------
        str — ABFSS path where data was written.
        """
        if mode == "upsert" and merge_keys:
            self._writer.upsert(df, table_name, merge_keys=merge_keys,
                                partition_by=partition_by)
            return self._fabric_cfg.table_path(table_name)
        elif mode == "append":
            return self._writer.append(df, table_name, partition_by=partition_by)
        else:
            return self._lakehouse_client.write_table(
                df, table_name, mode=mode, partition_by=partition_by
            )

    def write_forecasts(
        self,
        forecasts_sdf,
        mode: str = "upsert",
        forecast_origin: Optional[str] = None,
    ) -> None:
        """
        Write forecast results with standard partitioning.

        Uses ``DeltaWriter.write_forecasts`` for idempotent writes.
        """
        origin = forecast_origin or date.today().isoformat()
        self._writer.write_forecasts(
            df=forecasts_sdf,
            lob=self._lob,
            forecast_origin=origin,
            mode=mode,
        )
        logger.info("Forecasts written: lob=%s, origin=%s", self._lob, origin)

    def optimize_table(
        self,
        table_name: str,
        z_order_by: Optional[List[str]] = None,
    ) -> None:
        """Run OPTIMIZE (and optional Z-ORDER) on a Delta table."""
        self._lakehouse_client.optimize(table_name, z_order_by=z_order_by)

    # ── Utility ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Print a summary of the adapter configuration."""
        info = {
            "workspace": self._workspace,
            "lakehouse": self._lakehouse,
            "lob": self._lob,
            "spark_version": self._spark.version,
            "horizon": self._config.forecast.horizon_weeks,
            "frequency": self._config.forecast.frequency,
            "forecasters": self._config.forecast.forecasters,
            "n_folds": self._config.backtest.n_folds,
        }
        for k, v in info.items():
            print(f"  {k:16s}: {v}")
        return info
