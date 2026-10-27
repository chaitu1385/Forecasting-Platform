"""Isolated real process-pool probe; emits parent/worker stacks on a stall."""

import faulthandler
import functools
import json
import logging
import multiprocessing
import os
import signal
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl


def worker_diagnostics() -> None:
    if os.name == "nt":
        faulthandler.dump_traceback_later(30, repeat=True)
    else:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    print(
        f"worker pid={os.getpid()} start_method={multiprocessing.get_start_method()}",
        flush=True,
    )


def main() -> None:
    import src.backtesting.engine as engine_module
    import src.pipeline.batch_runner as batch_module

    logging.basicConfig(level=logging.INFO)
    # Do not start a watchdog thread before fork: the probe must not introduce
    # another inherited native lock. The supervising test signals owned PIDs.
    if os.name == "nt":
        faulthandler.dump_traceback_later(30, repeat=True)
    else:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    for module in (engine_module, batch_module):
        module.ProcessPoolExecutor = functools.partial(
            module.ProcessPoolExecutor, initializer=worker_diagnostics,
        )

    scenario, output_dir = sys.argv[1:]
    output = Path(output_dir)
    data = pl.DataFrame({
        "series_id": [f"S{s}" for s in range(2) for _ in range(80)],
        "week": [date(2022, 1, 2) + timedelta(weeks=w) for _ in range(2) for w in range(80)],
        "quantity": [float(30 + s * 10 + w % 13) for s in range(2) for w in range(80)],
    })
    # Initialize the native pool and file reader before creating any workers.
    history = output / "history.parquet"
    data.write_parquet(history)
    data = pl.read_parquet(history)
    assert data.group_by("series_id").agg(pl.col("quantity").sum()).height == 2
    print(f"parent pid={os.getpid()} scenario={scenario} native_threads={pl.thread_pool_size()}", flush=True)

    if scenario == "backtest":
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app(
            data_dir=str(output / "data"),
            metrics_dir=str(output / "metrics"),
            auth_enabled=False,
        )
        config = (
            "forecast:\n"
            "  forecasters: [naive_seasonal, auto_ets]\n"
            "  quantiles: []\n"
            "backtest:\n"
            "  n_folds: 1\n"
            "  val_weeks: 4\n"
            "parallelism:\n"
            "  n_workers: 2\n"
        )
        with TestClient(app) as client:
            response = client.post(
                "/pipeline/backtest?lob=process_probe",
                files={
                    "file": ("actuals.csv", data.write_csv().encode(), "text/csv"),
                    "config_file": ("config.yaml", config.encode(), "application/x-yaml"),
                },
            )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] == "completed", result
        assert {row["model_id"] for row in result["leaderboard"]} == {
            "naive_seasonal", "auto_ets",
        }, result
    elif scenario == "batch":
        from polars.testing import assert_frame_equal

        from src.forecasting.naive import SeasonalNaiveForecaster
        from src.pipeline.batch_runner import BatchInferenceRunner

        expected = BatchInferenceRunner(n_workers=1).run_forecast(
            data, SeasonalNaiveForecaster(), horizon=4,
        )
        result = BatchInferenceRunner(n_workers=2, batch_size=1).run_forecast(
            data, SeasonalNaiveForecaster(), horizon=4,
        )
        assert_frame_equal(
            result.sort(["series_id", "week"]),
            expected.sort(["series_id", "week"]),
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps({"scenario": scenario, "status": "passed"}), flush=True)


if __name__ == "__main__":
    main()
