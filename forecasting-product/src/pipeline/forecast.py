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
from threading import Lock
from typing import Optional, Union

import polars as pl

from ..config.schema import PlatformConfig
from ..evaluation.calibration import apply_conformal_correction, validate_quantile_forecasts
from ..forecasting.base import BaseForecaster
from ..forecasting.registry import registry
from ..observability.context import PipelineContext
from ..observability.metrics import MetricsEmitter
from ..series.builder import SeriesBuilder
from .manifest import PipelineManifest, build_manifest, model_metadata, runtime_metadata
from .run_store import RunStore, resolve_run_root, utc_timestamp
from .validation_step import ValidationError, run_post_validation

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

    def __init__(
        self, config: PlatformConfig, context: Optional[PipelineContext] = None,
        source_kind: str = "programmatic",
    ):
        self.config = config
        self._parent_context = context
        if source_kind not in {"programmatic", "api", "cli", "spark-local"}:
            raise ValueError("Unknown pipeline source kind")
        self._source_kind = source_kind
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
        self.last_run: Optional[PipelineManifest] = None
        self._run_lock = Lock()

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
        """Generate and atomically publish one independent forecast attempt.

        A supplied constructor context is a parent correlation, not an artifact
        identity. Concurrent reuse of one mutable pipeline instance is rejected.
        """
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("This pipeline instance already has a running attempt")
        try:
            self.context = PipelineContext(
                lob=self.config.lob,
                parent_run_id=self._parent_context.run_id if self._parent_context else None,
            )
            self._emitter.set_context(self.context)
            self._series_builder = SeriesBuilder(self.config)
            self._last_forecast_path = None
            self._last_validation_result = None
            self.last_run = None
            return self._run(
                actuals.clone(),
                champion_model.clone() if isinstance(champion_model, pl.DataFrame) else champion_model,
                product_master.clone() if product_master is not None else None,
                mapping_table.clone() if mapping_table is not None else None,
                forecast_origin,
                overrides.clone() if overrides is not None else None,
                external_features.clone() if external_features is not None else None,
            )
        except Exception as exc:
            logger.error("Forecast attempt failed: run_id=%s error_type=%s", self.context.run_id, type(exc).__name__)
            raise
        finally:
            self._run_lock.release()

    def _run(
        self,
        actuals: pl.DataFrame,
        champion_model: Union[str, BaseForecaster, pl.DataFrame],
        product_master: Optional[pl.DataFrame],
        mapping_table: Optional[pl.DataFrame],
        forecast_origin: Optional[date],
        overrides: Optional[pl.DataFrame],
        external_features: Optional[pl.DataFrame],
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
        optional_inputs = {
            kind: frame for kind, frame in {
                "product_master": product_master, "mapping_table": mapping_table,
                "overrides": overrides, "external_features": external_features,
            }.items() if frame is not None
        }
        if isinstance(champion_model, pl.DataFrame):
            return self._run_multi_horizon(
                actuals, series, champion_model, forecast_origin, optional_inputs,
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
                optional_inputs["future_features"] = future_feats
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
            try:
                validate_quantile_forecasts(forecast, fc.quantiles)

                # Step 4c: Conformal prediction correction (if configured)
                cal = fc.calibration
                if (cal.enabled and cal.conformal_correction
                        and self._conformal_residuals is not None):
                    forecast = apply_conformal_correction(
                        forecast,
                        self._conformal_residuals,
                        fc.quantiles,
                        cal.coverage_targets,
                        fc.series_id_column,
                        model_id=base_model_name,
                    )
                    logger.info("Applied conformal prediction correction to intervals")
                    optional_inputs["conformal_residuals"] = self._conformal_residuals
            except ValueError as exc:
                validation = self.config.post_validation
                if validation.enabled and validation.halt_on_blocker:
                    raise ValidationError(
                        f"[{self.config.lob}] Quantile BLOCKER: {exc}"
                    ) from exc
                raise

        self._publish(
            actuals, series, forecast, base_model_name, forecast_origin,
            [model_metadata(forecaster, fc.frequency, horizon)], optional_inputs,
        )

        return forecast

    def _run_multi_horizon(
        self,
        actuals: pl.DataFrame,
        series: pl.DataFrame,
        champion_table: pl.DataFrame,
        forecast_origin: Optional[date],
        optional_inputs: dict,
    ) -> pl.DataFrame:
        """
        Generate a stitched forecast from multiple champion models.

        Each model is fit on the full training data, predicts the full
        horizon, and its predictions are sliced to the steps in its
        assigned bucket.  The slices are concatenated into the final
        forecast.
        """
        from ..forecasting.constrained import (
            apply_point_constraints,
            prepare_constraint_capacity_map,
        )

        fc = self.config.forecast
        horizon = fc.horizon_weeks
        id_col = fc.series_id_column
        time_col = fc.time_column

        capacity_map = None
        if fc.constraints.enabled:
            if fc.quantiles:
                if fc.constraints.aggregate_max is not None:
                    raise ValueError("unsupported_aggregate_quantiles: stitched forecasts do not support aggregate quantile budgets")
                raise ValueError("unsupported_stitched_quantiles: constrained horizon stitching currently produces point forecasts only")
            prepared_capacities = prepare_constraint_capacity_map(series, fc.constraints, id_col)
            if fc.constraints.capacity_column is not None:
                capacity_map = prepared_capacities

        logger.info(
            "Multi-horizon forecast: %d bucket(s)", len(champion_table)
        )

        forecast_parts = []
        models = []
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
            if "model" not in preds.columns:
                preds = preds.with_columns(pl.lit(model_name).alias("model"))
            models.append(model_metadata(forecaster, fc.frequency, horizon))

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
            raise ValueError("Multi-horizon forecast produced no forecast parts")

        forecast = pl.concat(forecast_parts).sort([id_col, time_col])

        # Drop the forecast_step helper column
        if "forecast_step" in forecast.columns:
            forecast = forecast.drop("forecast_step")

        if fc.constraints.enabled:
            forecast = apply_point_constraints(
                forecast, fc.constraints, id_col, time_col, "forecast", capacity_map,
            )

        optional_inputs["champion_table"] = champion_table
        self._publish(actuals, series, forecast, "multi_horizon", forecast_origin, models, optional_inputs)

        return forecast

    def _publish(
        self,
        actuals: pl.DataFrame,
        prepared: pl.DataFrame,
        forecast: pl.DataFrame,
        champion_model_id: str,
        forecast_origin: Optional[date],
        models: list,
        optional_inputs: dict,
    ) -> None:
        """Validate, then make the complete run visible through its one marker."""
        if forecast.is_empty():
            raise ValueError("Forecast attempt produced no rows")
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

        actuals = self._series_builder.normalized_input_snapshot(actuals)
        manifest = build_manifest(
            run_id=self.context.run_id, config=self.config, actuals=actuals,
            series_builder=self._series_builder, champion_model_id=champion_model_id,
            forecast=forecast, forecast_file="forecast.parquet",
            post_validation_result=self._last_validation_result,
        )
        manifest.run_type = "forecast"
        manifest.started_at = utc_timestamp(self.context.started_at)
        manifest.timestamp = manifest.started_at
        origin = forecast_origin or prepared[self.config.forecast.time_column].max()
        manifest.forecast_origin = origin.isoformat() if origin is not None else None
        manifest.models = models
        manifest.execution = {
            "run_type": "forecast", "model_id": champion_model_id,
            "frequency": self.config.forecast.frequency,
            "horizon_periods": self.config.forecast.horizon_periods, "models": models,
        }
        manifest.source = {**runtime_metadata(), "entrypoint": self._source_kind}
        manifest.reproducibility = {
            "raw_snapshot": "normalized input DataFrame including configured series ID, not original upload bytes",
            "prepared_snapshot": "SeriesBuilder output; models may apply additional preparation",
            "optional_inputs_snapshotted": sorted(optional_inputs),
            "fitted_model_persisted": False,
            "limits": ["No guarantee of deterministic retraining or frozen external provider/model weights",
                       "Credential settings and server locations are deliberately omitted"],
        }
        manifest.total_seconds = self.context.elapsed_seconds
        store = RunStore(resolve_run_root(self.config, "forecast"))
        committed = store.publish(
            manifest, self.config,
            {"input": actuals, "prepared": prepared, "forecast": forecast, **optional_inputs},
            self._last_validation_result or {"skipped": True},
        )
        self.last_run = committed
        self._last_forecast_path = str(store.directory(self.config.lob, committed.run_id) / "forecast.parquet")
        logger.info("Forecast committed: run_id=%s rows=%d", committed.run_id, len(forecast))
