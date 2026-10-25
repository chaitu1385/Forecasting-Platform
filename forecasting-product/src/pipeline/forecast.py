"""
Forecast pipeline — production inference workflow.

Steps:
  1. Load config.
  2. Build series (with transitions).
  3. Load champion model(s) from backtest results.
  4. Fit champion on all available data.
  5. Forecast the full horizon (39 weeks).
  6. Emit leaf-level forecasts (bottom-up coherent by construction).
  7. Validate before writing frozen forecasts to the output store.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

import polars as pl

from ..config.schema import PlatformConfig
from ..forecasting.base import BaseForecaster
from ..forecasting.registry import registry
from ..observability.context import PipelineContext
from ..observability.metrics import MetricsEmitter
from ..series.builder import SeriesBuilder
from .validation_step import run_post_validation

logger = logging.getLogger(__name__)


class ForecastPipeline:
    """
    End-to-end production forecast pipeline.

    Usage
    -----
    >>> config = load_config("configs/platform_config.yaml")
    >>> pipeline = ForecastPipeline(config)
    >>> forecast_df = pipeline.run(actuals_df, champion_model="lgbm_direct")
    """

    def __init__(self, config: PlatformConfig, context: Optional[PipelineContext] = None):
        self.config = config
        self.context = context or PipelineContext(lob=config.lob)
        self._emitter = MetricsEmitter(
            backend=config.observability.metrics_backend,
            context=self.context,
            prefix=config.observability.metrics_prefix,
        )
        self._series_builder = SeriesBuilder(config)
        self._conformal_residuals: Optional[pl.DataFrame] = None
        self._last_forecast_path: Optional[str] = None
        self._last_validation_result: Optional[dict] = None

    def set_conformal_residuals(self, residuals: pl.DataFrame) -> None:
        """Set conformal residuals from backtest for interval correction."""
        self._conformal_residuals = residuals

    def run(
        self,
        actuals: pl.DataFrame,
        champion_model: Union[str, BaseForecaster, pl.DataFrame] = "naive_seasonal",
        product_master: Optional[pl.DataFrame] = None,
        mapping_table: Optional[pl.DataFrame] = None,
        forecast_origin: Optional[date] = None,
        overrides: Optional[pl.DataFrame] = None,
        external_features: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        Generate production forecasts.

        Parameters
        ----------
        actuals:
            Historical data.
        champion_model:
            Either the registered model name (str), a pre-built
            ``BaseForecaster`` instance, or a multi-horizon champion
            DataFrame (from ``ChampionSelector.select_by_horizon()``).
        product_master:
            Product metadata for transitions.
        mapping_table:
            SKU mappings for transitions.
        forecast_origin:
            Forecast start date.  Defaults to the last date in actuals.
        overrides:
            Planner transition overrides.

        Returns
        -------
        Forecast DataFrame: [series_id, week, forecast] plus optional
        ``forecast_p{q}`` quantile columns if ``config.forecast.quantiles``
        is configured.
        """
        self._last_forecast_path = None
        self._last_validation_result = None
        fc = self.config.forecast
        horizon = fc.horizon_weeks
        run_id = self.context.run_id

        reconcilable = self.config.get_reconcilable_hierarchies()
        if reconcilable and self.config.reconciliation.method != "bottom_up":
            raise ValueError(
                "ForecastPipeline produces one leaf-level forecast and therefore "
                "supports bottom_up reconciliation only. Use the hierarchy API or "
                "a hierarchical forecaster when independent aggregate forecasts "
                f"must be reconciled with method={self.config.reconciliation.method!r}."
            )

        # Step 1: Build series
        logger.info("[%s] Building model-ready series...", run_id)
        with self._emitter.timer("series_build"):
            series = self._series_builder.build(
                actuals=actuals,
                external_features=external_features,
                product_master=product_master,
                mapping_table=mapping_table,
                forecast_origin=forecast_origin,
                overrides=overrides,
            )

        # Step 1b: Log data quality warnings (if report enabled)
        qr = self._series_builder._last_quality_report
        if qr is not None:
            for w in qr.warnings:
                logger.warning("Data quality: %s", w)

        # Multi-horizon stitching: champion_model is a DataFrame
        if isinstance(champion_model, pl.DataFrame):
            return self._run_multi_horizon(
                series, champion_model, forecast_origin
            )

        # Step 2: Resolve forecaster (name → registry lookup, or use directly)
        if isinstance(champion_model, str):
            forecaster: BaseForecaster = registry.build(
                champion_model, frequency=fc.frequency
            )
        else:
            forecaster = champion_model
        base_model_name = forecaster.name
        if fc.constraints.enabled:
            from ..forecasting.constrained import ConstrainedDemandEstimator

            forecaster = ConstrainedDemandEstimator(forecaster, fc.constraints)
        logger.info("Champion model: %s", forecaster.name)

        # Step 3: Fit on all data
        logger.info("[%s] Fitting on %d rows...", run_id, len(series))
        with self._emitter.timer("model_fit"):
            forecaster.fit(
                series,
                target_col=fc.target_column,
                time_col=fc.time_column,
                id_col=fc.series_id_column,
            )

        # Step 3b: Set future features for ML models (if external regressors configured)
        feature_target = getattr(forecaster, "base", forecaster)
        if external_features is not None and hasattr(feature_target, "set_future_features"):
            er_config = fc.external_regressors
            if er_config.enabled and er_config.future_features_path:
                from ..data.regressors import load_external_features
                future_feats = load_external_features(er_config.future_features_path)
                feature_target.set_future_features(
                    future_feats,
                    id_col=fc.series_id_column,
                    time_col=fc.time_column,
                )

        # Step 4: Point forecast
        logger.info("[%s] Forecasting %d periods...", run_id, horizon)
        with self._emitter.timer("model_predict"):
            forecast = forecaster.predict(
                horizon=horizon,
                id_col=fc.series_id_column,
                time_col=fc.time_column,
            )
        # Tag forecast rows with the model name
        if "model" not in forecast.columns:
            forecast = forecast.with_columns(pl.lit(base_model_name).alias("model"))
        self._emitter.gauge("forecast_rows", float(len(forecast)))

        # Step 4b: Quantile forecasts (if configured)
        if fc.quantiles:
            logger.info("Computing quantile forecasts %s...", fc.quantiles)
            qdf = forecaster.predict_quantiles(
                horizon=horizon,
                quantiles=fc.quantiles,
                id_col=fc.series_id_column,
                time_col=fc.time_column,
            )
            # Join quantile columns onto the point forecast frame
            forecast = forecast.join(
                qdf, on=[fc.series_id_column, fc.time_column], how="left"
            )

        # Step 4c: Conformal prediction correction (if calibration configured)
        cal = fc.calibration
        if (fc.quantiles and cal.enabled and cal.conformal_correction
                and self._conformal_residuals is not None):
            from ..evaluation.calibration import apply_conformal_correction
            forecast = apply_conformal_correction(
                forecast,
                self._conformal_residuals,
                fc.quantiles,
                cal.coverage_targets,
                fc.series_id_column,
                model_id=base_model_name if isinstance(champion_model, str) else None,
            )
            logger.info("Applied conformal prediction correction to intervals")

        # Step 5: Validate and write to output
        forecast = self._write_forecast(forecast, forecast_origin)

        # Step 6: Write provenance manifest
        self._write_manifest(
            actuals=actuals,
            champion_model_id=(
                base_model_name
            ),
            forecast=forecast,
        )

        return forecast

    def _run_multi_horizon(
        self,
        series: pl.DataFrame,
        champion_table: pl.DataFrame,
        forecast_origin: Optional[date],
    ) -> pl.DataFrame:
        """
        Generate a stitched forecast from multiple champion models.

        Each model is fit on the full training data, predicts the full
        horizon, and its predictions are sliced to the steps in its
        assigned bucket.  The slices are concatenated into the final
        forecast.
        """
        fc = self.config.forecast
        horizon = fc.horizon_weeks
        id_col = fc.series_id_column
        time_col = fc.time_column

        logger.info(
            "Multi-horizon forecast: %d bucket(s)", len(champion_table)
        )

        forecast_parts = []
        for row in champion_table.iter_rows(named=True):
            model_name = row["model_id"]
            start_step = row["start_step"]
            end_step = row["end_step"]
            bucket_name = row["horizon_bucket"]

            logger.info(
                "  Bucket %s (steps %d-%d): model %s",
                bucket_name, start_step, end_step, model_name,
            )

            forecaster = registry.build(model_name, frequency=fc.frequency)
            forecaster.fit(
                series,
                target_col=fc.target_column,
                time_col=time_col,
                id_col=id_col,
            )
            preds = forecaster.predict(
                horizon=horizon, id_col=id_col, time_col=time_col
            )

            # Add forecast_step: rank by time within each series
            preds = preds.with_columns(
                pl.col(time_col)
                .rank("ordinal")
                .over(id_col)
                .cast(pl.Int32)
                .alias("forecast_step")
            )

            # Slice to this bucket's step range
            bucket_preds = preds.filter(
                pl.col("forecast_step").is_between(start_step, end_step)
            )
            forecast_parts.append(bucket_preds)

        if not forecast_parts:
            return pl.DataFrame()

        forecast = pl.concat(forecast_parts).sort([id_col, time_col])

        # Drop the forecast_step helper column
        if "forecast_step" in forecast.columns:
            forecast = forecast.drop("forecast_step")

        forecast = self._write_forecast(forecast, forecast_origin)

        # Write provenance manifest for multi-horizon run
        self._write_manifest(
            actuals=series,
            champion_model_id="multi_horizon",
            forecast=forecast,
        )

        return forecast

    def _write_manifest(
        self,
        actuals: pl.DataFrame,
        champion_model_id: str,
        forecast: pl.DataFrame,
        backtest_wmape: Optional[float] = None,
    ) -> None:
        """Write a provenance manifest alongside the last forecast file."""
        if self._last_forecast_path is None:
            return
        try:
            from .manifest import build_manifest, write_manifest
            manifest = build_manifest(
                run_id=uuid4().hex[:16],
                config=self.config,
                actuals=actuals,
                series_builder=self._series_builder,
                champion_model_id=champion_model_id,
                forecast=forecast,
                forecast_file=Path(self._last_forecast_path).name,
                backtest_wmape=backtest_wmape,
                post_validation_result=self._last_validation_result,
            )
            write_manifest(manifest, self._last_forecast_path)
        except Exception:
            logger.warning("Failed to write pipeline manifest", exc_info=True)

    def _write_forecast(
        self, forecast: pl.DataFrame, forecast_origin: Optional[date]
    ) -> pl.DataFrame:
        """Validate every forecast path before any publication side effects."""
        if self.config.post_validation.enabled:
            fc = self.config.forecast
            with self._emitter.timer("post_validation"):
                self._last_validation_result = run_post_validation(
                    results_df=forecast,
                    config=self.config.post_validation,
                    lob=self.config.lob,
                    forecast_df=forecast,
                    id_col=fc.series_id_column,
                    time_col=fc.time_column,
                )

        output_path = Path(self.config.output.forecast_path) / self.config.lob
        output_path.mkdir(parents=True, exist_ok=True)

        origin_str = (
            forecast_origin.isoformat()
            if forecast_origin
            else date.today().isoformat()
        )
        filename = f"forecast_{self.config.lob}_{origin_str}.parquet"
        filepath = output_path / filename

        forecast.write_parquet(str(filepath))
        logger.info("Forecast written to %s (%d rows)", filepath, len(forecast))
        self._last_forecast_path = str(filepath)

        return forecast
