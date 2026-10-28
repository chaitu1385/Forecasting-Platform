"""
Production Fabric Deployment Orchestrator
(Design document §11 / Phase 2)

Chains the full forecast cycle — pre-flight validation, optional re-train,
champion selection, and forecast — as a single atomic run with audit logging.

Workflow
--------
1. Pre-flight checks
   a. Data freshness: actuals table has rows within the last ``max_staleness_days``.
   b. Series count: number of distinct series IDs ≥ ``min_series_count``.
   c. Forecast schema: expected columns present on the forecasts table.

2. Backtest (optional)
   Run distributed walk-forward CV and elect a champion model.
   Skipped when ``force_retrain=False`` and a champion already exists in the
   leaderboard table.

3. Forecast
   Fit the champion model on all actuals and write 39-week-ahead forecasts to
   the forecasts Delta table.

4. Post-run checks
   Verify that the forecasts table for the current run contains at least
   ``min_forecast_rows`` rows (configurable; default 1).

5. Audit log
   Append a single row to the ``deploy_log`` Delta table for observability:
   run_id, lob, status, champion_model, n_forecast_rows, error (if any).

Usage
-----
>>> from src.fabric.deployment import DeploymentOrchestrator, DeploymentConfig
>>> cfg = DeploymentConfig(lob="rossmann", workspace="my-ws", lakehouse="my-lh")
>>> orch = DeploymentOrchestrator(spark, config=platform_config, deploy_config=cfg)
>>> result = orch.run(actuals_sdf=actuals_sdf)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Deployment config ──────────────────────────────────────────────────────────

@dataclass
class DeploymentConfig:
    """
    Parameters that control the production deployment cycle.

    Parameters
    ----------
    lob:
        Line-of-business identifier (must match leaderboard table).
    workspace:
        Fabric workspace name (or ``""`` for local mode).
    lakehouse:
        Fabric lakehouse name (or ``""`` for local mode).
    force_retrain:
        When True, always run backtest even if a champion already exists.
    models:
        Models to backtest.  Defaults to config.forecast.forecasters.
    horizon_weeks:
        Forecast horizon.  0 = take from platform config.
    max_staleness_days:
        Preflight check — actuals must have a row within this many days.
        Set to 0 to skip the freshness check.
    min_series_count:
        Preflight check — actuals must contain at least this many distinct
        series IDs.  Set to 0 to skip.
    min_forecast_rows:
        Post-run check — forecast output must have at least this many rows.
        Set to 0 to skip.
    write_mode:
        Delta write mode for forecasts table:
        ``"upsert"``, ``"overwrite_partition"``, or ``"append"``.
    deploy_log_table:
        Name of the audit log Delta table.
    """

    lob: str = "rossmann"
    workspace: str = ""
    lakehouse: str = ""
    force_retrain: bool = False
    models: Optional[List[str]] = None
    horizon_weeks: int = 0
    max_staleness_days: int = 14
    min_series_count: int = 1
    min_forecast_rows: int = 1
    write_mode: str = "upsert"
    deploy_log_table: str = "deploy_log"


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class DeploymentResult:
    """Outcome of a single ``DeploymentOrchestrator.run()`` call."""

    run_id: str
    lob: str
    run_date: str
    status: str                    # "success" | "failed"
    champion_model: str = ""
    n_forecast_rows: int = 0
    retrained: bool = False
    error: str = ""
    preflight_warnings: List[str] = field(default_factory=list)
    storage_contract: str = "external_delta_or_legacy_file"


# ── Orchestrator ───────────────────────────────────────────────────────────────

class DeploymentOrchestrator:
    """
    Orchestrates the end-to-end production forecast deployment cycle.

    Parameters
    ----------
    spark:
        Active SparkSession.
    config:
        ``PlatformConfig`` from ``load_config()``.
    deploy_config:
        ``DeploymentConfig`` instance.
    """

    def __init__(self, spark, config, deploy_config: DeploymentConfig):
        self.spark = spark
        self.config = config
        self.dc = deploy_config
        self._correlation_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, actuals_sdf) -> DeploymentResult:
        """
        Execute the full deployment cycle for one LOB.

        Parameters
        ----------
        actuals_sdf:
            Canonical actuals Spark DataFrame with columns
            ``[series_id, week, quantity]``.

        Returns
        -------
        ``DeploymentResult`` with status, champion model, and row counts.
        """
        run_id = uuid.uuid4().hex
        self._correlation_id = run_id
        run_date = date.today().isoformat()
        result = DeploymentResult(
            run_id=run_id,
            lob=self.dc.lob,
            run_date=run_date,
            status="failed",
            storage_contract="external_delta" if self.dc.workspace else "legacy_atomic_forecast_file",
        )

        try:
            # ── Step 1: Pre-flight ─────────────────────────────────────────
            warnings = self._preflight(actuals_sdf)
            result.preflight_warnings = warnings
            for w in warnings:
                logger.warning("[%s] Pre-flight warning: %s", run_id, w)

            # ── Step 2: Backtest / champion resolution ─────────────────────
            champion, retrained = self._resolve_champion(actuals_sdf)
            result.champion_model = champion
            result.retrained = retrained
            logger.info("[%s] Champion: %s (retrained=%s)", run_id, champion, retrained)

            # ── Step 3: Forecast ───────────────────────────────────────────
            from src.spark.pipeline import SparkForecastPipeline
            horizon = self.dc.horizon_weeks or self.config.forecast.horizon_weeks
            pipeline = SparkForecastPipeline(self.spark, self.config, correlation_id=run_id)
            forecasts_sdf = pipeline.run_forecast(
                actuals_sdf=actuals_sdf,
                champion_model=champion,
                horizon=horizon,
            )
            forecasts_sdf.cache()

            # ── Step 4: Post-run check ─────────────────────────────────────
            n_rows = self._postrun_check(forecasts_sdf)
            result.n_forecast_rows = n_rows

            # ── Step 5: Write forecasts ────────────────────────────────────
            self._write_forecasts(forecasts_sdf, run_date, run_id)

            result.status = "success"
            logger.info("[%s] Deployment succeeded. Forecast rows: %d", run_id, n_rows)

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            logger.exception("[%s] Deployment FAILED: %s", run_id, exc)

        finally:
            # ── Step 6: Audit log ──────────────────────────────────────────
            try:
                self._write_audit_log(result)
            except Exception as log_exc:
                logger.warning("[%s] Failed to write audit log: %s", run_id, log_exc)

        if result.status == "failed":
            raise RuntimeError(
                f"Deployment [{run_id}] failed: {result.error}"
            )

        return result

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def _preflight(self, actuals_sdf) -> List[str]:
        """
        Run pre-flight validation checks.  Returns a list of warning strings.
        Does NOT raise on soft failures — warnings are surfaced but the run
        continues (giving planners visibility without aborting).
        """
        warnings: List[str] = []

        # ── Series count check ─────────────────────────────────────────────
        if self.dc.min_series_count > 0:
            n_series = actuals_sdf.select("series_id").distinct().count()
            if n_series < self.dc.min_series_count:
                warnings.append(
                    f"Only {n_series} distinct series found "
                    f"(minimum expected: {self.dc.min_series_count})."
                )
            else:
                logger.info("Pre-flight: %d series found (OK).", n_series)

        # ── Data freshness check ───────────────────────────────────────────
        if self.dc.max_staleness_days > 0:
            try:
                from pyspark.sql import functions as F
                max_week = actuals_sdf.agg(F.max("week").alias("max_week")).collect()[0]["max_week"]
                if max_week is not None:
                    staleness = (date.today() - max_week).days
                    if staleness > self.dc.max_staleness_days:
                        warnings.append(
                            f"Actuals are stale: latest week = {max_week} "
                            f"({staleness} days ago; limit = {self.dc.max_staleness_days})."
                        )
                    else:
                        logger.info(
                            "Pre-flight: actuals freshness OK (latest week=%s, %d days ago).",
                            max_week, staleness,
                        )
                else:
                    warnings.append("Could not determine latest actuals date (empty series?).")
            except Exception as exc:
                warnings.append(f"Freshness check skipped (error: {exc}).")

        return warnings

    # ------------------------------------------------------------------
    # Champion resolution
    # ------------------------------------------------------------------

    def _resolve_champion(self, actuals_sdf) -> Tuple[str, bool]:
        """
        Return ``(champion_model_name, retrained: bool)``.

        Runs backtest when ``force_retrain=True`` or no champion found.
        """
        from pyspark.sql import functions as F

        from src.spark.pipeline import SparkForecastPipeline

        pipeline = SparkForecastPipeline(self.spark, self.config, correlation_id=self._correlation_id)
        models = self.dc.models or self.config.forecast.forecasters

        # Try to read existing champion from leaderboard (Fabric mode only)
        if not self.dc.force_retrain and self.dc.workspace:
            try:
                champion = self._read_existing_champion()
                if champion:
                    logger.info("Reusing existing champion: %s", champion)
                    return champion, False
            except Exception as exc:
                logger.warning("Could not read leaderboard: %s — running backtest.", exc)

        # Run backtest
        logger.info("Running backtest for models: %s", models)
        backtest_sdf = pipeline.run_backtest(actuals_sdf, model_names=models)
        leaderboard_sdf = pipeline.select_champion(
            backtest_sdf,
            primary_metric=self.config.backtest.primary_metric,
            secondary_metric=self.config.backtest.secondary_metric,
        )

        # Write leaderboard if in Fabric mode
        if self.dc.workspace:
            try:
                self._write_leaderboard(backtest_sdf, leaderboard_sdf)
            except Exception as exc:
                logger.warning("Could not write leaderboard to Lakehouse: %s", exc)

        champion_row = (
            leaderboard_sdf
            .filter(F.col("rank") == 1)
            .select("model")
            .limit(1)
            .collect()
        )
        if not champion_row:
            raise RuntimeError(
                "Champion selection returned no rows. "
                "Check that backtest results are non-empty."
            )
        champion = champion_row[0]["model"]
        return champion, True

    def _read_existing_champion(self) -> Optional[str]:
        """Read champion model from leaderboard Delta table. Returns None if absent."""
        from pyspark.sql import functions as F

        from src.fabric.config import FabricConfig
        from src.fabric.lakehouse import FabricLakehouse

        fabric_cfg = FabricConfig(workspace=self.dc.workspace, lakehouse=self.dc.lakehouse)
        lh = FabricLakehouse(self.spark, fabric_cfg)
        row = (
            lh.read_table("leaderboard")
            .filter(F.col("lob") == self.dc.lob)
            .orderBy(F.col("run_date").desc(), F.col("rank").asc())
            .select("champion_model")
            .limit(1)
            .first()
        )
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Post-run checks
    # ------------------------------------------------------------------

    def _postrun_check(self, forecasts_sdf) -> int:
        """
        Validate forecast output.  Returns row count.
        Raises ``RuntimeError`` if ``min_forecast_rows`` not met.
        """
        n_rows = forecasts_sdf.count()
        if self.dc.min_forecast_rows > 0 and n_rows < self.dc.min_forecast_rows:
            raise RuntimeError(
                f"Post-run check failed: forecast has {n_rows} rows "
                f"(minimum required: {self.dc.min_forecast_rows})."
            )
        logger.info("Post-run check: %d forecast rows (OK).", n_rows)
        return n_rows

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def _write_forecasts(self, forecasts_sdf, forecast_origin: str, run_id: str) -> None:
        """Write forecasts to Lakehouse (if configured) or local Parquet."""
        if self.dc.workspace:
            from src.fabric.config import FabricConfig
            from src.fabric.delta_writer import DeltaWriter
            from src.fabric.lakehouse import FabricLakehouse

            fabric_cfg = FabricConfig(
                workspace=self.dc.workspace, lakehouse=self.dc.lakehouse
            )
            writer = DeltaWriter(self.spark, fabric_cfg)
            writer.write_forecasts(
                df=forecasts_sdf,
                lob=self.dc.lob,
                forecast_origin=forecast_origin,
                mode=self.dc.write_mode,
            )
            lh = FabricLakehouse(self.spark, fabric_cfg)
            lh.optimize("forecasts", z_order_by=["series_id", "week"])
            logger.info("Forecasts written to Lakehouse.")
        else:
            from pathlib import Path

            import polars as pl

            from ..pipeline.run_store import (
                atomic_frame_append,
                checked_child,
                validate_lob,
                validate_run_id,
            )

            validate_lob(self.dc.lob)
            validate_run_id(run_id)
            output_path = checked_child(Path(self.config.output.forecast_path), self.dc.lob)
            out_file = atomic_frame_append(
                pl.from_pandas(forecasts_sdf.toPandas()), output_path,
                prefix=f"forecast_{self.dc.lob}_{forecast_origin}",
            )
            logger.info("Legacy Spark adapter forecast saved: run_id=%s file=%s; no local run-bundle guarantee",
                        run_id, out_file.name)

    def _write_leaderboard(self, backtest_sdf, leaderboard_sdf) -> None:
        """Write backtest results and leaderboard to Lakehouse."""
        from pyspark.sql import functions as F

        from src.fabric.config import FabricConfig
        from src.fabric.delta_writer import DeltaWriter
        from src.spark.pipeline import (
            backtest_metrics_for_legacy_storage,
            leaderboard_for_legacy_storage,
        )

        run_date = date.today().isoformat()
        fabric_cfg = FabricConfig(workspace=self.dc.workspace, lakehouse=self.dc.lakehouse)
        writer = DeltaWriter(self.spark, fabric_cfg)

        out_backtest = (
            backtest_metrics_for_legacy_storage(backtest_sdf, self.config.forecast.series_id_column)
            .withColumn("lob", F.lit(self.dc.lob))
            .withColumn("run_date", F.lit(run_date))
        )
        writer.append(out_backtest, "backtest_results", partition_by=["lob", "run_date"])

        # Add champion_model column (first-ranked model for this LOB)
        champion = (
            leaderboard_sdf
            .filter(F.col("rank") == 1)
            .select("model")
            .limit(1)
            .collect()[0]["model"]
        )
        lb_out = (
            leaderboard_for_legacy_storage(leaderboard_sdf)
            .withColumn("lob", F.lit(self.dc.lob))
            .withColumn("run_date", F.lit(run_date))
            .withColumn("champion_model", F.lit(champion))
        )
        writer.upsert(lb_out, "leaderboard", merge_keys=["lob", "run_date", "model"])

    def _write_audit_log(self, result: DeploymentResult) -> None:
        """Append a single audit log row to the deploy_log Delta table."""
        import pyspark.sql.types as T

        schema = T.StructType([
            T.StructField("run_id",           T.StringType(), False),
            T.StructField("lob",              T.StringType(), False),
            T.StructField("run_date",         T.StringType(), False),
            T.StructField("status",           T.StringType(), False),
            T.StructField("champion_model",   T.StringType(), True),
            T.StructField("n_forecast_rows",  T.LongType(),   True),
            T.StructField("retrained",        T.BooleanType(), True),
            T.StructField("error",            T.StringType(), True),
            T.StructField("preflight_warnings", T.StringType(), True),
        ])

        row_data = [{
            "run_id":              result.run_id,
            "lob":                 result.lob,
            "run_date":            result.run_date,
            "status":              result.status,
            "champion_model":      result.champion_model or None,
            "n_forecast_rows":     result.n_forecast_rows,
            "retrained":           result.retrained,
            "error":               result.error or None,
            "preflight_warnings":  "; ".join(result.preflight_warnings) or None,
        }]

        log_sdf = self.spark.createDataFrame(row_data, schema=schema)

        if self.dc.workspace:
            from src.fabric.config import FabricConfig
            from src.fabric.delta_writer import DeltaWriter
            fabric_cfg = FabricConfig(
                workspace=self.dc.workspace, lakehouse=self.dc.lakehouse
            )
            writer = DeltaWriter(self.spark, fabric_cfg)
            writer.append(log_sdf, self.dc.deploy_log_table, partition_by=["lob", "run_date"])
        else:
            from pathlib import Path
            out_dir = Path("data/deploy_logs") / self.dc.lob
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{result.run_id}.parquet"
            log_sdf.toPandas().to_parquet(out_file, index=False)

        logger.info(
            "Audit log written: run_id=%s status=%s champion=%s rows=%d",
            result.run_id, result.status, result.champion_model, result.n_forecast_rows,
        )
