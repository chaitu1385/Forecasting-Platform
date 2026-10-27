"""
Backtest pipeline — full end-to-end model evaluation workflow.

Steps:
  1. Load config.
  2. Build series (with transitions).
  3. Run backtesting across all configured models.
  4. Select champion model(s).
  5. Write results to metric store.
"""

import logging
from datetime import date
from threading import Lock
from typing import Optional

import polars as pl

from ..backtesting.champion import ChampionSelector
from ..backtesting.engine import BacktestEngine
from ..config.schema import PlatformConfig
from ..forecasting.ensemble import WeightedEnsembleForecaster
from ..forecasting.registry import registry
from ..metrics.store import MetricStore
from ..observability.context import PipelineContext
from ..observability.metrics import MetricsEmitter
from ..series.builder import SeriesBuilder
from .manifest import PipelineManifest, build_manifest, model_metadata, runtime_metadata
from .run_store import RunStore, resolve_run_root, utc_timestamp
from .validation_step import run_post_validation

logger = logging.getLogger(__name__)


class BacktestPipeline:
    """
    End-to-end backtest pipeline.

    Usage
    -----
    >>> from src.config import load_config
    >>> config = load_config("configs/platform_config.yaml")
    >>> pipeline = BacktestPipeline(config)
    >>> results = pipeline.run(actuals_df)
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
        self._metric_store = MetricStore(config.output.metrics_path, str(resolve_run_root(config, "backtest")))
        self._backtest_engine = BacktestEngine(config, self._metric_store)
        self._champion_selector = ChampionSelector(config.backtest)
        self.last_run: Optional[PipelineManifest] = None
        self._run_lock = Lock()

    def run(
        self,
        actuals: pl.DataFrame,
        product_master: Optional[pl.DataFrame] = None,
        mapping_table: Optional[pl.DataFrame] = None,
        forecast_origin: Optional[date] = None,
        overrides: Optional[pl.DataFrame] = None,
        external_features: Optional[pl.DataFrame] = None,
    ) -> dict:
        """Evaluate, validate and atomically commit one independent attempt."""
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("This pipeline instance already has a running attempt")
        try:
            self.context = PipelineContext(
                lob=self.config.lob,
                parent_run_id=self._parent_context.run_id if self._parent_context else None,
            )
            self._emitter.set_context(self.context)
            self.last_run = None
            self._series_builder = SeriesBuilder(self.config)
            self._metric_store = MetricStore(
                self.config.output.metrics_path, str(resolve_run_root(self.config, "backtest")),
            )
            self._backtest_engine = BacktestEngine(self.config, self._metric_store)
            self._champion_selector = ChampionSelector(self.config.backtest)
            return self._run(
                actuals.clone(),
                product_master.clone() if product_master is not None else None,
                mapping_table.clone() if mapping_table is not None else None,
                forecast_origin,
                overrides.clone() if overrides is not None else None,
                external_features.clone() if external_features is not None else None,
            )
        except Exception as exc:
            logger.error("Backtest attempt failed: run_id=%s error_type=%s", self.context.run_id, type(exc).__name__)
            raise
        finally:
            self._run_lock.release()

    def _run(
        self,
        actuals: pl.DataFrame,
        product_master: Optional[pl.DataFrame],
        mapping_table: Optional[pl.DataFrame],
        forecast_origin: Optional[date],
        overrides: Optional[pl.DataFrame],
        external_features: Optional[pl.DataFrame],
    ) -> dict:
        """
        Run the full backtest pipeline.

        Returns
        -------
        Dict with keys:
          - "backtest_results": full per-model-fold metric DataFrame
          - "champions": champion model table
          - "leaderboard": aggregated model ranking
        """
        fc = self.config.forecast

        run_id = self.context.run_id

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
        self._emitter.gauge("series_count", float(series[fc.series_id_column].n_unique()))
        logger.info("[%s] Series built: %d rows, %d series",
                     run_id, len(series),
                     series[fc.series_id_column].n_unique())

        # Step 2: Instantiate forecasters from config (frequency-aware)
        freq_kwargs = {"frequency": fc.frequency}
        forecasters = registry.build_from_config(
            fc.forecasters,
            params={name: freq_kwargs for name in fc.forecasters},
        )
        logger.info("Forecasters: %s", [f.name for f in forecasters])

        # Intermittent demand forecasters (optional sparse routing)
        sparse_forecasters = None
        if fc.intermittent_forecasters and fc.sparse_detection:
            sparse_forecasters = registry.build_from_config(
                fc.intermittent_forecasters,
                params={name: freq_kwargs for name in fc.intermittent_forecasters},
            )
            logger.info(
                "Sparse forecasters: %s", [f.name for f in sparse_forecasters]
            )

        # Step 3: Run backtesting
        logger.info("[%s] Running backtesting (%d folds)...", run_id, self.config.backtest.n_folds)
        with self._emitter.timer("backtest_run"):
            results = self._backtest_engine.run(
                series, forecasters, sparse_forecasters=sparse_forecasters,
                run_id=run_id, persist=False,
            )
        self._emitter.counter("backtest_folds_completed", self.config.backtest.n_folds)

        # Surface any model failures
        failures = self._backtest_engine.get_failure_summary()
        if not failures.is_empty():
            logger.warning(
                "Model failures during backtest:\n%s", failures
            )

        if results.is_empty():
            raise ValueError("Backtesting produced no results; no run was committed")

        # Step 4: Select champion / build ensemble
        strategy = self.config.backtest.selection_strategy
        ensemble: Optional[WeightedEnsembleForecaster] = None

        if strategy == "weighted_ensemble":
            logger.info("Building weighted ensemble from backtest results...")
            weights = self._champion_selector.compute_ensemble_weights(results)
            ensemble = WeightedEnsembleForecaster(
                forecasters=forecasters,
                weights=weights,
            )
            # Create a synthetic champion row describing the ensemble
            champions = pl.DataFrame({
                "group_key": [self.config.lob],
                "model_id": ["weighted_ensemble"],
                self.config.backtest.primary_metric: [0.0],
                self.config.backtest.secondary_metric: [0.0],
            })
            logger.info("Ensemble: %s", ensemble)
        elif self.config.backtest.horizon_buckets:
            logger.info("Selecting champion model(s) per horizon bucket...")
            champions = self._champion_selector.select_by_horizon(
                results, self.config.backtest.horizon_buckets
            )
        else:
            logger.info("Selecting champion model(s)...")
            champions = self._champion_selector.select(results)

        # Step 5: Build leaderboard
        leaderboard = MetricStore.leaderboard_from_records(
            results,
            primary_metric=self.config.backtest.primary_metric,
            secondary_metric=self.config.backtest.secondary_metric,
        )

        # Step 5b: Calibration report (if quantiles + calibration enabled)
        calibration_report = None
        conformal_residuals = None
        if fc.quantiles and fc.calibration.enabled and not results.is_empty():
            from ..evaluation.calibration import (
                compute_calibration_report,
                compute_conformal_residuals,
            )
            calibration_report = compute_calibration_report(
                results, fc.quantiles, fc.calibration.coverage_targets,
            )
            for model_id, coverages in calibration_report.model_reports.items():
                for cov in coverages:
                    logger.info(
                        "Calibration [%s] %s%%: nominal=%.0f%%, empirical=%.1f%%, "
                        "miscalibration=%+.1f%%, sharpness=%.1f",
                        model_id, cov.label, cov.nominal * 100,
                        cov.empirical * 100, cov.miscalibration * 100,
                        cov.sharpness,
                    )
            if fc.calibration.conformal_correction:
                conformal_residuals = compute_conformal_residuals(
                    results, fc.quantiles, fc.calibration.coverage_targets,
                    id_col=fc.series_id_column,
                )

        # Step 6: Post-pipeline validation (4-layer + confidence grade)
        validation_result = None
        if self.config.post_validation.enabled and not results.is_empty():
            with self._emitter.timer("post_validation"):
                validation_result = run_post_validation(
                    results_df=results,
                    config=self.config.post_validation,
                    lob=self.config.lob,
                )
            self._emitter.gauge("validation_score", float(validation_result.get("score", -1)))

        model_ids = champions["model_id"].unique().to_list() if "model_id" in champions.columns else []
        selected_model = model_ids[0] if len(model_ids) == 1 else "multi_model"
        actuals = self._series_builder.normalized_input_snapshot(actuals)
        manifest = build_manifest(
            run_id=run_id, config=self.config, actuals=actuals,
            series_builder=self._series_builder, champion_model_id=selected_model,
            forecast=results, forecast_file="",
            post_validation_result=validation_result,
        )
        manifest.run_type = "backtest"
        manifest.started_at = utc_timestamp(self.context.started_at)
        manifest.timestamp = manifest.started_at
        manifest.models = [
            model_metadata(model, fc.frequency, self.config.backtest.val_periods)
            for model in [*forecasters, *(sparse_forecasters or [])]
        ]
        manifest.execution = {
            "run_type": "backtest", "frequency": fc.frequency,
            "horizon_periods": fc.horizon_periods, "validation_periods": self.config.backtest.val_periods,
            "models": manifest.models,
        }
        manifest.source = {**runtime_metadata(), "entrypoint": self._source_kind}
        optional_inputs = {
            kind: frame for kind, frame in {
                "product_master": product_master, "mapping_table": mapping_table,
                "overrides": overrides, "external_features": external_features,
                "conformal_residuals": conformal_residuals,
            }.items() if frame is not None
        }
        manifest.reproducibility = {
            "raw_snapshot": "normalized input DataFrame including configured series ID, not original upload bytes",
            "prepared_snapshot": "SeriesBuilder output; each model may apply additional preparation",
            "optional_inputs_snapshotted": sorted(optional_inputs),
            "fitted_model_persisted": False,
            "model_fold_failures": [
                {key: row[key] for key in ("model_id", "fold", "error_type")}
                for row in failures.to_dicts()
            ],
            "limits": ["No guarantee of deterministic retraining or frozen external provider/model weights",
                       "Credential settings and server locations are deliberately omitted"],
        }
        manifest.total_seconds = self.context.elapsed_seconds
        champion_snapshot = champions if champions.columns else pl.DataFrame(schema={"model_id": pl.Utf8})
        self.last_run = RunStore(resolve_run_root(self.config, "backtest")).publish(
            manifest, self.config,
            {"input": actuals, "prepared": series, "metrics": results, "champions": champion_snapshot,
             "leaderboard": leaderboard, **optional_inputs},
            validation_result or {"skipped": True},
        )

        logger.info("[%s] Backtest pipeline complete.", run_id)
        return {
            "run_id": run_id,
            "manifest": self.last_run,
            "backtest_results": results,
            "champions": champions,
            "leaderboard": leaderboard,
            "ensemble": ensemble,           # None unless selection_strategy="weighted_ensemble"
            "failures": failures,
            "calibration_report": calibration_report,
            "conformal_residuals": conformal_residuals,
            "data_quality_report": self._series_builder._last_quality_report,
            "validation": validation_result,
        }
