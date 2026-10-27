"""
SparkForecastPipeline — distributed forecast and backtest via pandas_udf.

Design
------
The existing forecasters (``src.forecasting.*``) are Polars/pandas-based
single-node models.  This pipeline fans them out across the Spark cluster
using ``applyInPandas`` (GROUPED_MAP pattern):

  1. Partition the actuals DataFrame by ``series_id``.
  2. Each Spark task picks up one partition (one series) and calls the
     forecaster's ``fit → predict`` cycle locally.
  3. Results are collected back into a single Spark DataFrame.

Pickle safety
-------------
Only plain Python scalars (strings, ints) are broadcast to executors —
never the full PlatformConfig object.  This prevents serialization failures
caused by live resources (DuckDB connections, file handles) that may be
attached to config objects on the driver.

Usage
-----
>>> from src.spark.pipeline import SparkForecastPipeline
>>> from src.config import load_config
>>> config   = load_config("configs/platform_config.yaml")
>>> pipeline = SparkForecastPipeline(spark, config)
>>> forecasts_sdf = pipeline.run_forecast(actuals_sdf, champion_model="lgbm_direct")
"""

import logging
from datetime import date
from typing import List, Optional

from ..observability.context import new_run_id

logger = logging.getLogger(__name__)


# ML model names that train a single model across all series in a group.
# These benefit from larger partitions (by LOB) rather than per-series.
_BATCH_ML_MODELS = {"lgbm_direct", "xgboost_direct"}


def public_spark_leaderboard(frame):
    """Keep the established public ``model`` identifier at the Spark boundary."""
    if "model_id" not in frame.columns:
        raise ValueError("Canonical Spark leaderboard requires model_id")
    return frame.withColumnRenamed("model_id", "model")


def leaderboard_for_legacy_storage(frame):
    """Project new-run scores to the existing Delta leaderboard column schema.

    The historical ``mean_*`` names are deprecated wire aliases for the current
    metric definitions, not instructions to average observation metric columns.
    This function never reads or recomputes an existing stored leaderboard.
    """
    columns = ["model", "wmape", "normalized_bias", "mae", "n_evals", "rank"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Leaderboard persistence requires current score columns: {sorted(missing)}")
    return frame.select(columns).toDF(
        "model", "mean_wmape", "mean_normalized_bias", "mean_mae", "n_evals", "rank",
    )


def backtest_metrics_for_legacy_storage(frame, id_col: str = "series_id"):
    """Preserve the existing series/model/fold Delta grain for newly scored runs."""
    from pyspark.sql import functions as F

    required = {id_col, "model", "fold", "actual", "forecast"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Backtest persistence requires raw observations: {sorted(required)}")
    error = F.col("forecast") - F.col("actual")
    volume = F.sum(F.abs(F.col("actual")))
    return frame.groupby(id_col, "model", "fold").agg(
        F.when(volume > 0, F.sum(F.abs(error)) / volume).alias("wmape"),
        F.when(volume > 0, F.sum(error) / volume).alias("normalized_bias"),
        F.avg(F.abs(error)).alias("mae"),
    )


def _extract_forecast_params(config, champion_model: str, horizon: int) -> dict:
    """
    Pull only the primitive values needed by executor tasks from a PlatformConfig.

    Returns a plain dict that is guaranteed to pickle cleanly — no live
    connections, file handles, or complex objects.
    """
    fc = config.forecast
    bt = config.backtest
    return {
        "model_name":  champion_model,
        "horizon":     horizon,
        "id_col":      fc.series_id_column,
        "time_col":    fc.time_column,
        "target_col":  fc.target_column,
        "frequency":   fc.frequency,
        # backtest params (also needed for run_backtest path)
        "n_folds":     bt.n_folds,
        "val_weeks":   bt.val_weeks,
        "gap_weeks":   bt.gap_weeks,
    }


def forecast_partition(pdf, params: dict):
    """Pure executor entry point; no Spark service or filesystem is required."""
    import pandas as pd
    import polars as pl

    from ..backtesting.scoring import _validated_frame, require_exact_coverage
    from ..forecasting.registry import registry
    from ..utils.calendar import PeriodCalendar

    id_col, time_col, target_col = params["id_col"], params["time_col"], params["target_col"]
    if pdf.empty:
        return pd.DataFrame(columns=[id_col, time_col, "forecast"])
    series = _validated_frame(pl.from_pandas(pdf), id_col, time_col, [target_col], "Training data")
    forecaster = registry.build(params["model_name"], frequency=params["frequency"])
    prepared = forecaster.validate_and_prepare(series, target_col=target_col, time_col=time_col, id_col=id_col)
    prepared = _validated_frame(prepared, id_col, time_col, [target_col], "Prepared training data")
    if set(prepared[id_col].unique().to_list()) != set(series[id_col].unique().to_list()):
        raise ValueError("Training preparation changed series coverage")
    series = prepared
    calendar = PeriodCalendar.from_dates(series[time_col].unique().to_list(), params["frequency"])
    expected = []
    for group in series.partition_by(id_col):
        expected.extend(
            {id_col: group[id_col][0], time_col: label}
            for label in calendar.future_dates(group[time_col].max(), params["horizon"])
        )
    forecaster.fit(series, target_col=target_col, time_col=time_col, id_col=id_col)
    result = require_exact_coverage(
        pl.DataFrame(expected), forecaster.predict(params["horizon"], id_col=id_col, time_col=time_col),
        id_col, time_col, ["forecast"], "Predictions",
    )
    return result.select(id_col, time_col, "forecast").to_pandas()


def backtest_partition(pdf, params: dict):
    """Pure keyed backtest worker; a failed model/fold fails the Spark task.

    Never truncate positional arrays or omit a failed hard case. Every emitted
    observation has passed the same coverage contract as the local engine.
    """
    import pandas as pd
    import polars as pl

    from ..backtesting.cross_validator import WalkForwardCV
    from ..backtesting.scoring import _validated_frame, evaluate_model_fold
    from ..forecasting.registry import registry
    from ..metrics.store import METRIC_SCHEMA

    if pdf.empty:
        return pd.DataFrame(columns=list(METRIC_SCHEMA))
    id_col, time_col, target_col = params["id_col"], params["time_col"], params["target_col"]
    series = _validated_frame(pl.from_pandas(pdf), id_col, time_col, [target_col], "Training data")
    cv = WalkForwardCV(
        n_folds=params["n_folds"], val_weeks=params["val_weeks"],
        gap_weeks=params["gap_weeks"], frequency=params["frequency"],
    )
    results = []
    for fold, train, val in cv.split_data(series, time_col):
        for model_name in params["model_names"]:
            model = registry.build(model_name, frequency=params["frequency"])
            results.append(evaluate_model_fold(
                train, val, model, fold_index=fold.fold_index,
                run_id=params["run_id"], run_date=date.fromisoformat(params["run_date"]),
                lob=params["lob"], target_col=target_col, time_col=time_col, id_col=id_col,
                metrics_list=params["metrics"], val_periods=params["val_weeks"],
                gap_periods=params["gap_weeks"], frequency=params["frequency"],
                quantiles=[], calibration_enabled=False,
            ))
    if not results:
        raise ValueError("Insufficient data: no complete Spark backtest folds")
    result = pl.concat(results)
    for column, dtype in METRIC_SCHEMA.items():
        if column not in result.columns:
            result = result.with_columns(pl.lit(None).cast(dtype).alias(column))
    return result.select(list(METRIC_SCHEMA)).to_pandas()


class SparkForecastPipeline:
    """
    Distributed forecast / backtest pipeline built on PySpark.

    Parameters
    ----------
    spark:
        Active SparkSession.
    config:
        ``PlatformConfig`` instance (loaded from YAML).
    """

    def __init__(self, spark, config, correlation_id: Optional[str] = None):
        self.spark = spark
        self.config = config
        self.correlation_id = correlation_id
        self.run_id: Optional[str] = None

    # ── forecast ──────────────────────────────────────────────────────────────

    def run_forecast(
        self,
        actuals_sdf,
        champion_model: str = "naive_seasonal",
        horizon: Optional[int] = None,
        num_partitions: Optional[int] = None,
    ):
        """
        Fit a champion model per series and generate forecasts.

        Parameters
        ----------
        actuals_sdf:
            Spark DataFrame with columns
            [series_id_col, time_col, target_col, …].
        champion_model:
            Registered forecaster name (see ``src.forecasting.registry``).
        horizon:
            Forecast horizon in weeks.  Defaults to ``config.forecast.horizon_weeks``.
        num_partitions:
            Number of Spark partitions (≈ parallelism).  Defaults to
            number of distinct series (capped at 2 000).

        Returns
        -------
        Spark DataFrame: [series_id, week, forecast].
        """
        self.run_id = new_run_id()
        from pyspark.sql import types as T

        from .utils import repartition_by_series

        horizon = horizon or self.config.forecast.horizon_weeks
        params = _extract_forecast_params(self.config, champion_model, horizon)

        # Broadcast only the plain-scalar dict — safe to pickle on any executor.
        params_bc = self.spark.sparkContext.broadcast(params)

        id_col   = params["id_col"]
        time_col = params["time_col"]

        output_schema = T.StructType([
            T.StructField(id_col,     T.StringType(), nullable=False),
            T.StructField(time_col,   T.DateType(),   nullable=False),
            T.StructField("forecast", T.DoubleType(), nullable=True),
        ])

        def _forecast_partition(pdf):
            """Called once per series partition.  Runs entirely on the executor.

            Note: ``applyInPandas`` requires pandas DataFrames at the boundary.
            All internal processing uses Polars.
            """
            return forecast_partition(pdf, params_bc.value)

        # Batch-aware partitioning: ML models (LightGBM, XGBoost) train one
        # model across all series in a group, so partition by LOB for fewer,
        # larger partitions.  Statistical/neural models are per-series.
        if champion_model in _BATCH_ML_MODELS and "lob" in actuals_sdf.columns:
            logger.info(
                "Using LOB-based partitioning for batch ML model %s",
                champion_model,
            )
            forecasts_sdf = actuals_sdf.groupby("lob").applyInPandas(
                _forecast_partition, schema=output_schema
            )
        else:
            actuals_sdf = repartition_by_series(actuals_sdf, id_col, num_partitions)
            forecasts_sdf = actuals_sdf.groupby(id_col).applyInPandas(
                _forecast_partition, schema=output_schema
            )
        logger.info(
            "Spark forecast plan launched: run_id=%s parent=%s model=%s horizon=%d storage_contract=external",
            self.run_id, self.correlation_id, champion_model, horizon,
        )
        return forecasts_sdf

    # ── backtest ──────────────────────────────────────────────────────────────

    def run_backtest(
        self,
        actuals_sdf,
        model_names: Optional[List[str]] = None,
        num_partitions: Optional[int] = None,
    ):
        """
        Run walk-forward backtest for multiple models across all series.

        Each (series × model × fold) is evaluated independently on an
        executor, producing per-series metrics.

        Parameters
        ----------
        actuals_sdf:
            Spark DataFrame with actuals.
        model_names:
            List of model names to evaluate.  Defaults to
            ``config.forecast.forecasters``.
        num_partitions:
            Spark parallelism.

        Returns
        -------
        Spark DataFrame of raw observations with the public ``model`` identifier.
        Use ``backtest_metrics_for_legacy_storage`` at legacy Delta write sites.
        """
        import polars as pl

        self.run_id = new_run_id()
        logger.info("Spark backtest plan: run_id=%s parent=%s storage_contract=external",
                    self.run_id, self.correlation_id)
        from pyspark.sql import types as T

        from ..backtesting.scoring import require_fold_local_preprocessing
        from ..metrics.store import METRIC_SCHEMA
        from .utils import repartition_by_series

        require_fold_local_preprocessing(self.config)
        model_names = model_names or self.config.forecast.forecasters
        if len(model_names) != len(set(model_names)):
            raise ValueError("Backtest models must have distinct model_id values")

        # Plain-scalar params dict — no config object on the wire.
        params = {
            "model_names": list(model_names),
            "id_col":      self.config.forecast.series_id_column,
            "time_col":    self.config.forecast.time_column,
            "target_col":  self.config.forecast.target_column,
            "n_folds":     self.config.backtest.n_folds,
            "val_weeks":   self.config.backtest.val_weeks,
            "gap_weeks":   self.config.backtest.gap_weeks,
            "frequency":   self.config.forecast.frequency,
            "metrics":     list(self.config.metrics),
            "lob":         self.config.lob,
            "run_id":      self.run_id,
            "run_date":    date.today().isoformat(),
        }
        params_bc = self.spark.sparkContext.broadcast(params)

        id_col   = params["id_col"]

        types = {pl.String: T.StringType(), pl.Date: T.DateType(), pl.Int32: T.IntegerType(), pl.Float64: T.DoubleType()}
        output_schema = T.StructType([
            T.StructField(column, types[dtype], nullable=dtype == pl.Float64)
            for column, dtype in METRIC_SCHEMA.items()
        ])

        def _backtest_partition(pdf):
            """Per-series backtest.  Runs entirely on the executor.

            Note: ``applyInPandas`` requires pandas DataFrames at the boundary.
            All internal processing uses Polars.
            """
            return backtest_partition(pdf, params_bc.value)

        actuals_sdf = repartition_by_series(actuals_sdf, id_col, num_partitions)
        results_sdf = actuals_sdf.groupby(id_col).applyInPandas(
            _backtest_partition, schema=output_schema
        )
        logger.info(
            "SparkForecastPipeline.run_backtest launched (models=%s, folds=%d)",
            model_names, self.config.backtest.n_folds,
        )
        results_sdf = results_sdf.withColumnRenamed("model_id", "model")
        if id_col != "series_id":
            results_sdf = results_sdf.withColumnRenamed("series_id", id_col)
        return results_sdf

    # ── champion selection ────────────────────────────────────────────────────

    def select_champion(
        self, backtest_results_sdf, primary_metric: str = "wmape",
        secondary_metric: str = "normalized_bias",
    ):
        """
        Aggregate backtest results and select the champion model.

        Fully distributed — all aggregation and ranking runs on Spark
        executors.  No ``collect()`` call here.

        Parameters
        ----------
        backtest_results_sdf:
            Output of ``run_backtest``.
        primary_metric:
            Metric column to minimise for champion selection
            (``"wmape"`` | ``"mae"`` | ``"normalized_bias"``).

        Returns
        -------
        Spark DataFrame with model, canonical raw-error aggregate metrics and rank.
        Legacy Delta writes use ``leaderboard_for_legacy_storage`` explicitly.
        """
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        from ..metrics.definitions import METRIC_DIRECTIONS

        supported = {"wmape", "macro_wmape", "normalized_bias", "mae", "mape", "rmse", "mase"}
        if primary_metric not in supported or secondary_metric not in supported:
            raise ValueError("Unsupported Spark selection metric")
        df = backtest_results_sdf
        if "model_id" not in df.columns and "model" in df.columns:
            df = df.withColumnRenamed("model", "model_id")
        id_col = self.config.forecast.series_id_column
        if "series_id" not in df.columns and id_col in df.columns:
            df = df.withColumnRenamed(id_col, "series_id")
        keys = ["run_id", "lob", "fold", "series_id", "target_week", "forecast_step"]
        missing = set(keys + ["model_id", "actual", "forecast"]) - set(df.columns)
        if missing:
            raise ValueError(f"Comparable Spark scoring requires raw observation columns: {sorted(missing)}")
        n_models = df.select("model_id").distinct().agg(F.count("*").alias("_models"))
        key_counts = df.groupby(*keys).agg(
            F.count("*").alias("_rows"), F.countDistinct("model_id").alias("_unique_models"),
            F.countDistinct("actual").alias("_actual_values"),
        ).crossJoin(n_models)
        validity = key_counts.agg(F.min(F.when(
            (F.col("_rows") == F.col("_models"))
            & (F.col("_unique_models") == F.col("_models"))
            & (F.col("_actual_values") == 1), 1
        ).otherwise(0)).alias("_keys_valid"))
        row_valid = F.lit(True)
        for column in keys + ["model_id", "actual", "forecast"]:
            row_valid = row_valid & F.col(column).isNotNull()
        for column in ("actual", "forecast"):
            row_valid = row_valid & ~F.isnan(column) & (F.abs(F.col(column)) != float("inf"))
        if "model" in df.columns:
            row_valid = row_valid & (F.col("model") == F.col("model_id"))
        validity = validity.crossJoin(df.agg(
            F.countDistinct("run_id").alias("_runs"),
            F.min(F.when(row_valid, 1).otherwise(0)).alias("_values_valid"),
        ))

        error = F.col("forecast") - F.col("actual")
        volume = F.sum(F.abs(F.col("actual")))
        leaderboard = df.groupby("model_id").agg(
            F.when(volume > 0, F.sum(F.abs(error)) / volume).alias("wmape"),
            F.when(volume > 0, F.sum(error) / volume).alias("normalized_bias"),
            F.mean(F.abs(error)).alias("mae"),
            F.mean(F.when(F.col("actual") != 0, F.abs(error) / F.abs(F.col("actual")))).alias("mape"),
            F.sqrt(F.mean(error * error)).alias("rmse"),
            F.countDistinct(F.struct("lob", "series_id", "fold")).alias("n_evals"),
            F.count("*").alias("n_observations"),
        )
        if "mase" in (primary_metric, secondary_metric):
            valid_scale = F.col("mase_scale").isNotNull() & (F.col("mase_scale") > 0)
            scaled = df.groupby("model_id").agg(
                F.when(F.min(F.when(valid_scale, 1).otherwise(0)) == 1,
                       F.mean(F.abs(error) / F.col("mase_scale"))).alias("mase")
            )
            leaderboard = leaderboard.join(scaled, "model_id")
        if "macro_wmape" in (primary_metric, secondary_metric):
            units = df.groupby("model_id", "run_id", "lob", "series_id", "fold").agg(
                F.when(volume > 0, F.sum(F.abs(error)) / volume).alias("_unit_wmape")
            )
            macro = units.groupby("model_id").agg(
                F.when(F.count("_unit_wmape") == F.count("*"), F.mean("_unit_wmape")).alias("macro_wmape")
            )
            leaderboard = leaderboard.join(macro, "model_id")
        leaderboard = leaderboard.crossJoin(validity).withColumn(
            primary_metric,
            F.when(
                (F.col("_keys_valid") == 1) & (F.col("_values_valid") == 1)
                & (F.col("_runs") == 1) & F.col(primary_metric).isNotNull(),
                F.col(primary_metric),
            ).otherwise(F.raise_error("Invalid scoring coverage or undefined primary metric").cast("double")),
        ).drop("_keys_valid", "_values_valid", "_runs")
        ordering = [
            (F.abs(F.col(name)) if METRIC_DIRECTIONS[name] == "minimize_absolute" else F.col(name)).asc_nulls_last()
            for name in (primary_metric, secondary_metric)
        ] + [F.col("model_id").asc()]
        return public_spark_leaderboard(
            leaderboard.withColumn("rank", F.row_number().over(Window.orderBy(*ordering))).orderBy("rank")
        )
