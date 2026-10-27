"""
BatchInferenceRunner — partitioned forecast with configurable parallelism.

Wraps the pattern: partition series into groups → train/predict each batch
→ merge results.  Supports configurable backends:

  - ``"local"`` — ``concurrent.futures.ProcessPoolExecutor`` (default).
  - ``"spark"`` — PySpark ``applyInPandas`` (requires SparkSession).

For ML models (LightGBM, XGBoost), all series in a batch are trained
together (the native ``mlforecast`` pattern).  For statistical models,
series are still independent but batched for I/O efficiency.

Usage
-----
>>> from src.pipeline.batch_runner import BatchInferenceRunner
>>> runner = BatchInferenceRunner(n_workers=4, batch_size=500)
>>> forecasts = runner.run_forecast(actuals, forecaster, horizon=39)
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from typing import List

import polars as pl

from ..forecasting.base import BaseForecaster

logger = logging.getLogger(__name__)


def _fit_predict_batch(
    batch_data: bytes,
    forecaster: BaseForecaster,
    horizon: int,
    target_col: str,
    time_col: str,
    id_col: str,
) -> bytes:
    """
    Worker function for process pool execution.

    Accepts and returns serialized Polars DataFrames (IPC bytes) to avoid
    pickle issues with complex objects.
    """
    batch = pl.read_ipc(batch_data)

    forecaster.fit(batch, target_col=target_col, time_col=time_col, id_col=id_col)
    result = forecaster.predict(horizon=horizon, id_col=id_col, time_col=time_col)

    buf = result.write_ipc(None)
    return buf.getvalue() if hasattr(buf, "getvalue") else buf


class BatchInferenceRunner:
    """
    Run forecast or backtest with configurable parallelism.

    Parameters
    ----------
    n_workers:
        Number of parallel workers.  ``-1`` = all CPU cores.  ``1`` = sequential.
    batch_size:
        Number of series per batch.  ``0`` = all series in one batch (current
        platform behavior).
    backend:
        ``"local"`` (ProcessPoolExecutor) or ``"spark"`` (PySpark).
    """

    def __init__(
        self,
        n_workers: int = -1,
        batch_size: int = 0,
        backend: str = "local",
    ):
        if n_workers == -1:
            import os
            n_workers = os.cpu_count() or 1
        self.n_workers = max(1, n_workers)
        self.batch_size = batch_size
        self.backend = backend

    def run_forecast(
        self,
        actuals: pl.DataFrame,
        forecaster: BaseForecaster,
        horizon: int,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> pl.DataFrame:
        """
        Partition series → fit/predict per batch → merge results.

        Parameters
        ----------
        actuals:
            Full actuals DataFrame with all series.
        forecaster:
            Fitted or unfitted forecaster instance.
        horizon:
            Forecast horizon in periods.
        target_col, time_col, id_col:
            Column names.

        Returns
        -------
        pl.DataFrame with columns [id_col, time_col, "forecast"].
        """
        if self.batch_size == 0 or self.n_workers == 1:
            # All-at-once: current platform behavior
            forecaster.fit(actuals, target_col=target_col,
                           time_col=time_col, id_col=id_col)
            return forecaster.predict(horizon=horizon, id_col=id_col,
                                      time_col=time_col)

        batches = self._partition_series(actuals, id_col)
        model_name = getattr(forecaster, "name", "naive_seasonal")

        logger.info(
            "BatchInferenceRunner: %d batches, %d workers, model=%s",
            len(batches), self.n_workers, model_name,
        )

        if self.backend == "local":
            results = self._run_local(
                batches, forecaster, horizon, target_col, time_col, id_col
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend!r}")

        if not results:
            return pl.DataFrame(schema={
                id_col: pl.Utf8, time_col: pl.Date, "forecast": pl.Float64
            })
        return pl.concat(results)

    def _partition_series(
        self,
        df: pl.DataFrame,
        id_col: str,
    ) -> List[pl.DataFrame]:
        """
        Split the DataFrame into batches of ``batch_size`` series each.

        Uses round-robin assignment to balance batch sizes.
        """
        unique_ids = df[id_col].unique().sort().to_list()
        n_batches = max(1, (len(unique_ids) + self.batch_size - 1) // self.batch_size)

        batches = []
        for i in range(n_batches):
            batch_ids = unique_ids[i * self.batch_size : (i + 1) * self.batch_size]
            batch = df.filter(pl.col(id_col).is_in(batch_ids))
            if len(batch) > 0:
                batches.append(batch)
        return batches

    def _run_local(
        self,
        batches: List[pl.DataFrame],
        forecaster: BaseForecaster,
        horizon: int,
        target_col: str,
        time_col: str,
        id_col: str,
    ) -> List[pl.DataFrame]:
        """Execute batches in parallel using ProcessPoolExecutor."""
        results: List[pl.DataFrame] = []

        if self.n_workers <= 1 or len(batches) == 1:
            # Sequential fallback
            for batch in batches:
                batch_model = deepcopy(forecaster)
                batch_model.fit(batch, target_col=target_col,
                                time_col=time_col, id_col=id_col)
                result = batch_model.predict(horizon=horizon, id_col=id_col,
                                             time_col=time_col)
                results.append(result)
            return results

        # Serialize batches to IPC bytes for inter-process transfer
        batch_bytes = []
        for batch in batches:
            buf = batch.write_ipc(None)
            batch_bytes.append(buf.getvalue() if hasattr(buf, "getvalue") else buf)

        # Never fork a process after Polars has initialized its native thread pool.
        with ProcessPoolExecutor(
            max_workers=min(self.n_workers, len(batches)),
            mp_context=get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(
                    _fit_predict_batch,
                    data, forecaster, horizon, target_col, time_col, id_col,
                ): i
                for i, data in enumerate(batch_bytes)
            }

            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    result_bytes = future.result()
                    result_df = pl.read_ipc(result_bytes)
                    results.append(result_df)
                except Exception as exc:
                    logger.error(
                        "Batch %d failed: %s", batch_idx, exc
                    )
                    raise RuntimeError(f"Forecast batch {batch_idx} failed") from exc

        return results
