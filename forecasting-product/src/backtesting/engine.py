"""
Backtesting engine — runs all forecasters across all CV folds.

For each (model, fold) combination:
  1. Fit on training data.
  2. Predict over the validation horizon.
  3. Compute metrics (WMAPE, Normalized Bias, etc.) per series.
  4. Write results to the metric store.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from multiprocessing import cpu_count, get_context
from typing import List, Optional

import polars as pl

from ..config.schema import PlatformConfig, get_frequency_profile
from ..forecasting.base import BaseForecaster
from ..metrics.definitions import compute_all_metrics
from ..metrics.store import METRIC_SCHEMA, MetricStore
from ..observability.context import new_run_id
from .cross_validator import WalkForwardCV

logger = logging.getLogger(__name__)


def _run_model_in_process(
    train_bytes: bytes,
    val_bytes: bytes,
    forecaster: BaseForecaster,
    fold_index: int,
    run_id: str,
    run_date_iso: str,
    lob: str,
    target_col: str,
    time_col: str,
    id_col: str,
    metrics_list: List[str],
    val_weeks: int,
    frequency: str,
    quantiles: List[float],
    calibration_enabled: bool,
) -> bytes:
    """
    Fit and evaluate a single model on a single fold.

    Runs in a subprocess via ProcessPoolExecutor.  Data is transferred
    as Polars IPC bytes for zero-copy deserialization.
    """
    import io

    import polars as pl

    from ..config.schema import get_frequency_profile
    from ..metrics.definitions import compute_all_metrics

    train = pl.read_ipc(io.BytesIO(train_bytes))
    val = pl.read_ipc(io.BytesIO(val_bytes))
    run_date = date.fromisoformat(run_date_iso)

    model_name = forecaster.name
    forecaster.fit(train, target_col=target_col, time_col=time_col, id_col=id_col)

    predictions = forecaster.predict(horizon=val_weeks, id_col=id_col, time_col=time_col)

    # Quantile predictions
    if quantiles and calibration_enabled:
        qdf = forecaster.predict_quantiles(
            horizon=val_weeks, quantiles=quantiles, id_col=id_col, time_col=time_col,
        )
        q_cols = [c for c in qdf.columns if c.startswith("forecast_p")]
        predictions = predictions.join(
            qdf.select([id_col, time_col] + q_cols), on=[id_col, time_col], how="left",
        )

    merged = val.join(predictions, on=[id_col, time_col], how="inner")
    if merged.is_empty():
        empty = pl.DataFrame()
        buf = io.BytesIO()
        empty.write_ipc(buf)
        return buf.getvalue()

    val_start = val[time_col].min()
    results = []
    for sid in merged[id_col].unique().to_list():
        s = merged.filter(pl.col(id_col) == sid)
        actual = s[target_col]
        forecast = s["forecast"]

        insample = train.filter(pl.col(id_col) == sid)[target_col]
        context = {"insample": insample}
        metrics = compute_all_metrics(actual, forecast, metric_names=metrics_list, context=context)

        q_cols = [c for c in s.columns if c.startswith("forecast_p")]
        for row in s.iter_rows(named=True):
            step_days = get_frequency_profile(frequency)["timedelta_kwargs"]
            period_days = sum(v * (7 if k == "weeks" else 1) for k, v in step_days.items())
            forecast_step = (row[time_col] - val_start).days // period_days + 1

            record = {
                "run_id": run_id, "run_type": "backtest", "run_date": run_date,
                "lob": lob, "model_id": model_name, "fold": fold_index,
                "grain_level": "series", "series_id": sid, "channel": "",
                "target_week": row[time_col], "forecast_step": forecast_step,
                "actual": float(row[target_col]), "forecast": float(row["forecast"]),
            }
            record.update(metrics)
            for qc in q_cols:
                if row[qc] is not None:
                    record[qc] = float(row[qc])
            results.append(record)

    result_df = pl.DataFrame(results)
    buf = io.BytesIO()
    result_df.write_ipc(buf)
    return buf.getvalue()


class BacktestEngine:
    """
    Run walk-forward backtesting for multiple models.

    Produces a detailed metric table and writes to the metric store.
    Model failures are tracked and returned alongside results.
    """

    def __init__(
        self,
        config: PlatformConfig,
        metric_store: Optional[MetricStore] = None,
    ):
        self.config = config
        self.bt_config = config.backtest
        self.metric_store = metric_store or MetricStore(
            config.output.metrics_path
        )
        self._cv = WalkForwardCV(
            n_folds=self.bt_config.n_folds,
            val_weeks=self.bt_config.val_weeks,
            gap_weeks=self.bt_config.gap_weeks,
            frequency=config.forecast.frequency,
        )
        self._failures: List[dict] = []
        self.run_id: Optional[str] = None

    def run(
        self,
        series: pl.DataFrame,
        forecasters: List[BaseForecaster],
        sparse_forecasters: Optional[List[BaseForecaster]] = None,
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
        id_col: Optional[str] = None,
        run_id: Optional[str] = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """
        Run full backtesting.

        Parameters
        ----------
        series:
            Model-ready panel data (from SeriesBuilder).
        forecasters:
            List of forecaster instances to evaluate on dense (regular) series.
        sparse_forecasters:
            Optional list of forecasters for intermittent/sparse series.
            When provided, series are split by the SparseDetector and each
            partition is evaluated with its dedicated model list.

        Returns
        -------
        DataFrame of per-(model, fold, series, week) metrics.
        """
        fc = self.config.forecast
        self.run_id = run_id or new_run_id()
        self._failures = []
        self._neural_notes = []
        target_col = target_col or fc.target_column
        time_col = time_col or fc.time_column
        id_col = id_col or fc.series_id_column

        # Sparse routing: split series into dense and sparse partitions
        if sparse_forecasters:
            from ..series.sparse_detector import SparseDetector
            detector = SparseDetector(
                adi_threshold=fc.sparse_adi_threshold,
                cv2_threshold=fc.sparse_cv2_threshold,
            )
            dense_series, sparse_series = detector.split(
                series, target_col=target_col, id_col=id_col
            )
            n_sparse = sparse_series[id_col].n_unique() if not sparse_series.is_empty() else 0
            n_dense = dense_series[id_col].n_unique() if not dense_series.is_empty() else 0
            logger.info(
                "Sparse routing: %d dense series → %s, %d sparse series → %s",
                n_dense, [f.name for f in forecasters],
                n_sparse, [f.name for f in sparse_forecasters],
            )
        else:
            dense_series = series
            sparse_series = pl.DataFrame()

        splits = self._cv.split_data(series, time_col)
        if not splits:
            logger.warning("No valid CV folds. Check data date range.")
            return pl.DataFrame()

        # Warn early if ML models are unlikely to have enough training data
        # for their lag features (they'll fail silently otherwise).
        self._warn_insufficient_data(splits, forecasters, time_col)

        run_id = self.run_id
        run_date = date.today()
        all_results: List[pl.DataFrame] = []
        self._failures = []

        for fold, _train_full, _val_full in splits:
            logger.info(
                "Fold %d: train %s→%s, val %s→%s (%d/%d train/val rows)",
                fold.fold_index,
                fold.train_start, fold.train_end,
                fold.val_start, fold.val_end,
                len(_train_full), len(_val_full),
            )

            # Build per-partition train/val splits
            partitions: List[tuple] = []
            if not dense_series.is_empty():
                dense_ids = dense_series[id_col].unique().to_list()
                train_d = _train_full.filter(pl.col(id_col).is_in(dense_ids))
                val_d = _val_full.filter(pl.col(id_col).is_in(dense_ids))
                partitions.append((train_d, val_d, forecasters))
            if sparse_forecasters and not sparse_series.is_empty():
                sparse_ids = sparse_series[id_col].unique().to_list()
                train_s = _train_full.filter(pl.col(id_col).is_in(sparse_ids))
                val_s = _val_full.filter(pl.col(id_col).is_in(sparse_ids))
                partitions.append((train_s, val_s, sparse_forecasters))
            if not partitions:
                partitions = [(_train_full, _val_full, forecasters)]

            n_workers = getattr(self.config, "parallelism", None)
            n_workers = n_workers.n_workers if n_workers else 1
            use_parallel = n_workers != 1 and sum(
                len(m) for _, _, m in partitions
            ) > 1

            if use_parallel:
                self._run_fold_parallel(
                    partitions, fold, run_id, run_date,
                    target_col, time_col, id_col,
                    all_results, n_workers,
                )
            else:
                for train, val, models in partitions:
                    if train.is_empty() or val.is_empty():
                        continue
                    for forecaster in models:
                        logger.info(
                            "  Model: %s (fold %d)", forecaster.name, fold.fold_index
                        )
                        try:
                            fold_result = self._run_one(
                                forecaster=forecaster,
                                train=train,
                                val=val,
                                fold_index=fold.fold_index,
                                run_id=run_id,
                                run_date=run_date,
                                target_col=target_col,
                                time_col=time_col,
                                id_col=id_col,
                            )
                            all_results.append(fold_result)
                        except Exception as e:
                            logger.error(
                                "  Model %s fold %d failed: %s",
                                forecaster.name, fold.fold_index, e,
                            )
                            self._failures.append({
                                "model_id": forecaster.name,
                                "fold": fold.fold_index,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                            })

        if self._failures:
            logger.warning(
                "Backtest had %d model-fold failure(s): %s",
                len(self._failures),
                ", ".join(
                    f"{f['model_id']}(fold {f['fold']}): {f['error_type']}"
                    for f in self._failures
                ),
            )

        # Log neural model training notes (CPU-vs-GPU advisory)
        self._neural_notes = []
        for forecaster in forecasters:
            if hasattr(forecaster, "training_notes"):
                notes = forecaster.training_notes()
                self._neural_notes.append(notes)
                if not notes.get("is_production_quality", True):
                    logger.warning(
                        "NEURAL MODEL ADVISORY: %s", notes["recommendation"]
                    )

        if not all_results:
            return pl.DataFrame()

        all_results = [self._normalize_result_schema(r) for r in all_results]
        combined = pl.concat(all_results, how="diagonal")

        # Write to metric store
        if persist:
            self.metric_store.write(
                combined,
                run_type="backtest",
                lob=self.config.lob,
            )
        logger.info(
            "Backtest complete: run_id=%s rows=%d persisted=%s",
            run_id, len(combined), persist,
        )

        return combined

    @property
    def failures(self) -> List[dict]:
        """Return list of model-fold failures from the most recent run."""
        return list(self._failures)

    @property
    def neural_training_notes(self) -> List[dict]:
        """Return neural model training advisories from the most recent run.

        Each entry contains current settings, whether the configuration is
        production-quality, and recommended GPU settings.  Useful for
        backtest reports and model cards.
        """
        return list(getattr(self, "_neural_notes", []))

    def get_failure_summary(self) -> pl.DataFrame:
        """Return failures as a DataFrame for reporting."""
        if not self._failures:
            return pl.DataFrame(schema={
                "model_id": pl.Utf8,
                "fold": pl.Int32,
                "error_type": pl.Utf8,
                "error_message": pl.Utf8,
            })
        return pl.DataFrame(self._failures)

    def _warn_insufficient_data(
        self,
        splits: list,
        forecasters: List[BaseForecaster],
        time_col: str,
    ) -> None:
        """Emit a warning when ML models' max lag exceeds the shortest training fold."""
        from ..forecasting.ml import _DirectMLBase

        min_train_periods = min(len(train) for _fold, train, _val in splits)
        for forecaster in forecasters:
            if not isinstance(forecaster, _DirectMLBase):
                continue
            max_lag = max(forecaster.lags) if forecaster.lags else 0
            if max_lag > 0 and min_train_periods <= max_lag:
                logger.warning(
                    "ML model '%s' has max lag %d but the shortest training "
                    "fold has only %d periods. The model will likely fail or "
                    "produce empty results. Increase data history or reduce "
                    "n_folds/val_periods.",
                    forecaster.name, max_lag, min_train_periods,
                )

    def _run_fold_parallel(
        self,
        partitions,
        fold,
        run_id: str,
        run_date: date,
        target_col: str,
        time_col: str,
        id_col: str,
        all_results: list,
        n_workers: int,
    ) -> None:
        """Run all models within a fold in parallel using ProcessPoolExecutor."""
        import io

        fc = self.config.forecast
        quantiles = fc.quantiles if hasattr(fc, "quantiles") else []
        calibration_enabled = (
            quantiles and hasattr(fc, "calibration") and fc.calibration.enabled
        )
        task_count = sum(
            len(models) for train, val, models in partitions
            if not train.is_empty() and not val.is_empty()
        )
        if not task_count:
            return
        max_workers = min(task_count, cpu_count() if n_workers == -1 else max(1, n_workers))

        futures = {}
        # Fork inherits Polars/native-library locks from the already active parent.
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as executor:
            for train, val, models in partitions:
                if train.is_empty() or val.is_empty():
                    continue
                train_buf = io.BytesIO()
                train.write_ipc(train_buf)
                train_bytes = train_buf.getvalue()
                val_buf = io.BytesIO()
                val.write_ipc(val_buf)
                val_bytes = val_buf.getvalue()

                for forecaster in models:
                    logger.info(
                        "  Model: %s (fold %d) [parallel]",
                        forecaster.name, fold.fold_index,
                    )
                    fut = executor.submit(
                        _run_model_in_process,
                        train_bytes=train_bytes,
                        val_bytes=val_bytes,
                        forecaster=forecaster,
                        fold_index=fold.fold_index,
                        run_id=run_id,
                        run_date_iso=run_date.isoformat(),
                        lob=self.config.lob,
                        target_col=target_col,
                        time_col=time_col,
                        id_col=id_col,
                        metrics_list=self.config.metrics,
                        val_weeks=self.bt_config.val_weeks,
                        frequency=fc.frequency,
                        quantiles=quantiles or [],
                        calibration_enabled=bool(calibration_enabled),
                    )
                    futures[fut] = forecaster.name

            for fut in as_completed(futures):
                model_name = futures[fut]
                try:
                    result_bytes = fut.result()
                    result_df = pl.read_ipc(io.BytesIO(result_bytes))
                    if not result_df.is_empty():
                        all_results.append(result_df)
                except Exception as e:
                    logger.error(
                        "  Model %s fold %d failed: %s",
                        model_name, fold.fold_index, e,
                    )
                    self._failures.append({
                        "model_id": model_name,
                        "fold": fold.fold_index,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    })

    @staticmethod
    def _normalize_result_schema(df: pl.DataFrame) -> pl.DataFrame:
        """Ensure result DataFrame conforms to METRIC_SCHEMA before concat."""
        if df.is_empty():
            return df
        for col, dtype in METRIC_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df

    def _run_one(
        self,
        forecaster: BaseForecaster,
        train: pl.DataFrame,
        val: pl.DataFrame,
        fold_index: int,
        run_id: str,
        run_date: date,
        target_col: str,
        time_col: str,
        id_col: str,
    ) -> pl.DataFrame:
        """Run one (model, fold) combination."""
        # Fit
        forecaster.fit(train, target_col=target_col, time_col=time_col, id_col=id_col)

        # Predict
        horizon = self.bt_config.val_weeks
        predictions = forecaster.predict(horizon, id_col=id_col, time_col=time_col)

        # Quantile predictions (if calibration enabled)
        quantiles = self.config.forecast.quantiles
        if quantiles and self.config.forecast.calibration.enabled:
            qdf = forecaster.predict_quantiles(
                horizon=horizon, quantiles=quantiles,
                id_col=id_col, time_col=time_col,
            )
            # Drop point forecast column from quantile df to avoid join conflict
            q_cols = [c for c in qdf.columns if c.startswith("forecast_p")]
            predictions = predictions.join(
                qdf.select([id_col, time_col] + q_cols),
                on=[id_col, time_col],
                how="left",
            )

        # Join predictions with actuals
        merged = val.join(
            predictions,
            on=[id_col, time_col],
            how="inner",
        )

        if merged.is_empty():
            return pl.DataFrame()

        # Compute per-series metrics
        val_start = val[time_col].min()
        results = []
        for sid in merged[id_col].unique().to_list():
            s = merged.filter(pl.col(id_col) == sid)
            actual = s[target_col]
            forecast = s["forecast"]

            # Build context for context-dependent metrics (e.g. MASE)
            insample = train.filter(pl.col(id_col) == sid)[target_col]
            context = {"insample": insample}

            metrics = compute_all_metrics(
                actual, forecast,
                metric_names=self.config.metrics,
                context=context,
            )

            # Detect quantile columns present in merged data
            q_cols = [c for c in s.columns if c.startswith("forecast_p")]

            for row in s.iter_rows(named=True):
                # forecast_step: 1-indexed week offset from validation start
                step_days = get_frequency_profile(self.config.forecast.frequency)["timedelta_kwargs"]
                period_days = sum(
                    v * (7 if k == "weeks" else 1) for k, v in step_days.items()
                )
                forecast_step = (row[time_col] - val_start).days // period_days + 1

                record = {
                    "run_id": run_id,
                    "run_type": "backtest",
                    "run_date": run_date,
                    "lob": self.config.lob,
                    "model_id": forecaster.name,
                    "fold": fold_index,
                    "grain_level": "series",
                    "series_id": sid,
                    "channel": "",
                    "target_week": row[time_col],
                    "forecast_step": forecast_step,
                    "actual": float(row[target_col]),
                    "forecast": float(row["forecast"]),
                }
                record.update(metrics)
                for qc in q_cols:
                    if row[qc] is not None:
                        record[qc] = float(row[qc])
                results.append(record)

        return pl.DataFrame(results)
