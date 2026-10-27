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
from typing import List, Optional

from ..observability.context import new_run_id

logger = logging.getLogger(__name__)


# ML model names that train a single model across all series in a group.
# These benefit from larger partitions (by LOB) rather than per-series.
_BATCH_ML_MODELS = {"lgbm_direct", "xgboost_direct"}


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
        # backtest params (also needed for run_backtest path)
        "n_folds":     bt.n_folds,
        "val_weeks":   bt.val_weeks,
        "gap_weeks":   bt.gap_weeks,
    }


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
            import pandas as pd  # Spark applyInPandas requires pandas DataFrame
            import polars as pl

            from src.forecasting.registry import registry

            p = params_bc.value
            if pdf.empty:
                return pd.DataFrame(columns=[p["id_col"], p["time_col"], "forecast"])

            series = pl.from_pandas(pdf)
            forecaster = registry.build(p["model_name"])
            forecaster.fit(series, target_col=p["target_col"],
                           time_col=p["time_col"], id_col=p["id_col"])
            result = forecaster.predict(horizon=p["horizon"],
                                        id_col=p["id_col"], time_col=p["time_col"])
            return result.to_pandas()

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
        Spark DataFrame: [series_id, model, fold, wmape, normalized_bias, mae].
        """
        self.run_id = new_run_id()
        logger.info("Spark backtest plan: run_id=%s parent=%s storage_contract=external",
                    self.run_id, self.correlation_id)
        from pyspark.sql import types as T

        from .utils import repartition_by_series

        model_names = model_names or self.config.forecast.forecasters

        # Plain-scalar params dict — no config object on the wire.
        params = {
            "model_names": list(model_names),
            "id_col":      self.config.forecast.series_id_column,
            "time_col":    self.config.forecast.time_column,
            "target_col":  self.config.forecast.target_column,
            "n_folds":     self.config.backtest.n_folds,
            "val_weeks":   self.config.backtest.val_weeks,
            "gap_weeks":   self.config.backtest.gap_weeks,
        }
        params_bc = self.spark.sparkContext.broadcast(params)

        id_col   = params["id_col"]

        output_schema = T.StructType([
            T.StructField(id_col,            T.StringType(),  nullable=False),
            T.StructField("model",           T.StringType(),  nullable=False),
            T.StructField("fold",            T.IntegerType(), nullable=False),
            T.StructField("wmape",           T.DoubleType(),  nullable=True),
            T.StructField("normalized_bias", T.DoubleType(),  nullable=True),
            T.StructField("mae",             T.DoubleType(),  nullable=True),
        ])

        def _backtest_partition(pdf):
            """Per-series backtest.  Runs entirely on the executor.

            Note: ``applyInPandas`` requires pandas DataFrames at the boundary.
            All internal processing uses Polars.
            """
            import numpy as np
            import pandas as pd  # Spark applyInPandas requires pandas DataFrame
            import polars as pl

            from src.forecasting.registry import registry

            p = params_bc.value
            id_c, time_c, tgt_c = p["id_col"], p["time_col"], p["target_col"]

            if pdf.empty:
                return pd.DataFrame(
                    columns=[id_c, "model", "fold", "wmape", "normalized_bias", "mae"]
                )

            series = pl.from_pandas(pdf).sort(time_c)
            n = len(series)
            rows = []

            for model_name in p["model_names"]:
                for fold in range(p["n_folds"]):
                    val_end   = n - fold * p["val_weeks"]
                    val_start = val_end - p["val_weeks"]
                    train_end = val_start - p["gap_weeks"]

                    if train_end < 10 or val_start < 0:
                        continue

                    train = series[:train_end]
                    val   = series[val_start:val_end]

                    try:
                        forecaster = registry.build(model_name)
                        forecaster.fit(train, target_col=tgt_c,
                                       time_col=time_c, id_col=id_c)
                        preds = forecaster.predict(
                            horizon=p["val_weeks"], id_col=id_c, time_col=time_c
                        )

                        actual_vals = val[tgt_c].to_numpy().astype(float)
                        pred_vals   = preds["forecast"].to_numpy().astype(float)
                        min_len     = min(len(actual_vals), len(pred_vals))
                        actual_vals = actual_vals[:min_len]
                        pred_vals   = pred_vals[:min_len]

                        denom     = np.sum(np.abs(actual_vals))
                        wmape     = float(np.sum(np.abs(actual_vals - pred_vals)) / denom) if denom else None
                        norm_bias = float(np.sum(actual_vals - pred_vals) / denom) if denom else None
                        mae       = float(np.mean(np.abs(actual_vals - pred_vals)))

                        series_id_val = pdf[id_c].iloc[0] if id_c in pdf.columns else "unknown"
                        rows.append({
                            id_c:              series_id_val,
                            "model":           model_name,
                            "fold":            fold,
                            "wmape":           wmape,
                            "normalized_bias": norm_bias,
                            "mae":             mae,
                        })
                    except Exception as exc:
                        series_id_val = pdf[id_c].iloc[0] if id_c in pdf.columns else "?"
                        logger.warning(
                            "Backtest failed — series=%s model=%s fold=%d: %s",
                            series_id_val, model_name, fold, exc,
                        )

            return (
                pd.DataFrame(rows)
                if rows
                else pd.DataFrame(
                    columns=[id_c, "model", "fold", "wmape", "normalized_bias", "mae"]
                )
            )

        actuals_sdf = repartition_by_series(actuals_sdf, id_col, num_partitions)
        results_sdf = actuals_sdf.groupby(id_col).applyInPandas(
            _backtest_partition, schema=output_schema
        )
        logger.info(
            "SparkForecastPipeline.run_backtest launched (models=%s, folds=%d)",
            model_names, self.config.backtest.n_folds,
        )
        return results_sdf

    # ── champion selection ────────────────────────────────────────────────────

    def select_champion(self, backtest_results_sdf, primary_metric: str = "wmape"):
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
        Spark DataFrame: [model, mean_wmape, mean_normalized_bias, mean_mae,
                          n_evals, rank].
        """
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        metric_col = f"mean_{primary_metric}"

        leaderboard = (
            backtest_results_sdf
            .groupby("model")
            .agg(
                F.mean("wmape").alias("mean_wmape"),
                F.mean("normalized_bias").alias("mean_normalized_bias"),
                F.mean("mae").alias("mean_mae"),
                F.count("*").alias("n_evals"),
            )
            .withColumn(
                "rank",
                F.rank().over(Window.orderBy(F.col(metric_col).asc_nulls_last())),
            )
            .orderBy("rank")
        )
        return leaderboard
