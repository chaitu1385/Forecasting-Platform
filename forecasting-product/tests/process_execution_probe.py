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
from pickle import PicklingError

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
    step = timedelta(weeks=1) if scenario == "backtest" else timedelta(days=1)
    data = pl.DataFrame({
        "series_id": [f"S{s}" for s in range(2) for _ in range(80)],
        "week": [date(2022, 1, 2) + step * w for _ in range(2) for w in range(80)],
        "quantity": [float(30 + s * 10 + w % 13) for s in range(2) for w in range(80)],
    })
    # Initialize the native pool and file reader before creating any workers.
    history = output / "history.parquet"
    data.write_parquet(history)
    data = pl.read_parquet(history)
    assert data.group_by("series_id").agg(pl.col("quantity").sum()).height == 2
    print(f"parent pid={os.getpid()} scenario={scenario} native_threads={pl.thread_pool_size()}", flush=True)
    start_method = multiprocessing.get_start_method()

    if scenario == "backtest":
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app(
            data_dir=str(output / "data"),
            metrics_dir=str(output / "metrics"),
            auth_enabled=False,
            dev_mode=True,
            jwt_secret="",
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
    elif scenario == "backtest-payload":
        from dataclasses import replace

        from polars.testing import assert_frame_equal

        from src.backtesting.engine import BacktestEngine
        from src.config.schema import BacktestConfig, ForecastConfig, OutputConfig, ParallelismConfig, PlatformConfig
        from src.forecasting.naive import SeasonalNaiveForecaster
        from src.forecasting.statistical import AutoETSForecaster

        config = PlatformConfig(
            forecast=ForecastConfig(frequency="D", quantiles=[]),
            backtest=BacktestConfig(n_folds=1, val_weeks=4),
            parallelism=ParallelismConfig(n_workers=1),
            output=OutputConfig(metrics_path=str(output / "sequential-metrics")),
        )
        models = [
            SeasonalNaiveForecaster(season_length=3, frequency="D"),
            AutoETSForecaster(season_length=7, frequency="D"),
        ]
        sequential = BacktestEngine(config)
        expected = sequential.run(data, models)
        parallel = BacktestEngine(replace(
            config,
            parallelism=ParallelismConfig(n_workers=2),
            output=OutputConfig(metrics_path=str(output / "parallel-metrics")),
        ))
        result = parallel.run(data, models)
        assert sequential.failures == parallel.failures == []
        assert result.height == expected.height == 16
        keys = ["model_id", "series_id", "target_week"]
        assert_frame_equal(
            result.drop("run_id").sort(keys),
            expected.drop("run_id").sort(keys),
        )
    elif scenario == "batch":
        from polars.testing import assert_frame_equal

        from src.forecasting.naive import SeasonalNaiveForecaster
        from src.pipeline.batch_runner import BatchInferenceRunner

        expected = BatchInferenceRunner(n_workers=1).run_forecast(
            data, SeasonalNaiveForecaster(season_length=3, frequency="D"), horizon=4,
        )
        result = BatchInferenceRunner(n_workers=2, batch_size=1).run_forecast(
            data, SeasonalNaiveForecaster(season_length=3, frequency="D"), horizon=4,
        )
        assert_frame_equal(
            result.sort(["series_id", "week"]),
            expected.sort(["series_id", "week"]),
        )
    elif scenario == "fitted-quantiles":
        from polars.testing import assert_frame_equal

        from src.backtesting.engine import BacktestEngine
        from src.config.schema import BacktestConfig, CalibrationConfig, ForecastConfig, OutputConfig, ParallelismConfig, PlatformConfig
        from src.forecasting.ml import LGBMDirectForecaster
        from src.forecasting.naive import SeasonalNaiveForecaster

        params = {
            "frequency": "D", "lags": [1, 2], "num_threads": 1,
            "lgbm_params": {"n_estimators": 12, "num_leaves": 7, "min_child_samples": 3, "random_state": 17},
        }
        quantiles = [0.1, 0.5, 0.9]
        primed = LGBMDirectForecaster(**params)
        primed.fit(data)
        future = primed.predict_quantiles(4, quantiles)
        assert set(primed._quantile_mlfs) == {0.1, 0.9}
        assert future["week"].min() > data["week"].max()
        original_cache = dict(primed._quantile_mlfs)

        config = PlatformConfig(
            forecast=ForecastConfig(
                frequency="D", quantiles=quantiles,
                calibration=CalibrationConfig(enabled=True),
            ),
            backtest=BacktestConfig(n_folds=2, val_weeks=4),
            parallelism=ParallelismConfig(n_workers=2),
            output=OutputConfig(metrics_path=str(output / "quantile-metrics")),
        )
        engine = BacktestEngine(config)
        results = engine.run(data, [primed, SeasonalNaiveForecaster(frequency="D")])
        assert engine.failures == []
        observed = results.filter(pl.col("model_id") == "lgbm_direct")
        assert observed.height == 16
        q_columns = ["forecast_p10", "forecast_p50", "forecast_p90"]
        for column in q_columns:
            assert observed[column].null_count() == 0, observed

        keys = ["series_id", "week"]
        for fold, train, _val in engine._cv.split_data(data):
            fresh = LGBMDirectForecaster(**params)
            fresh.fit(train)
            expected = fresh.predict(4).join(fresh.predict_quantiles(4, quantiles), on=keys)
            assert set(fresh._quantile_mlfs) == {0.1, 0.9}
            actual = observed.filter(pl.col("fold") == fold.fold_index).select(
                "series_id", pl.col("target_week").alias("week"), "forecast", *q_columns,
            )
            assert_frame_equal(actual.sort(keys), expected.sort(keys))
        assert all(primed._quantile_mlfs[q] is original_cache[q] for q in original_cache)
    elif scenario == "serialization-failure":
        from src.backtesting.engine import BacktestEngine
        from src.config.schema import BacktestConfig, ForecastConfig, OutputConfig, ParallelismConfig, PlatformConfig
        from src.forecasting.naive import SeasonalNaiveForecaster
        from src.forecasting.statistical import AutoETSForecaster
        from src.pipeline.batch_runner import BatchInferenceRunner

        model = SeasonalNaiveForecaster(frequency="D")
        model.unpicklable = lambda: None
        try:
            BatchInferenceRunner(n_workers=2, batch_size=1).run_forecast(data, model, horizon=4)
        except RuntimeError as error:
            assert "Forecast batch" in str(error)
            assert isinstance(error.__cause__, (AttributeError, PicklingError))
            assert "<lambda>" in str(error.__cause__)
        else:
            raise AssertionError("Serialization failure was hidden as a successful partial forecast")

        config = PlatformConfig(
            forecast=ForecastConfig(frequency="D", quantiles=[]),
            backtest=BacktestConfig(n_folds=1, val_weeks=4),
            parallelism=ParallelismConfig(n_workers=2),
            output=OutputConfig(metrics_path=str(output / "failed-model-metrics")),
        )
        engine = BacktestEngine(config)
        result = engine.run(data, [model, AutoETSForecaster(frequency="D")])
        assert result.height == 8
        assert result["model_id"].unique().to_list() == ["auto_ets"]
        assert len(engine.failures) == 1
        assert engine.failures[0]["model_id"] == "naive_seasonal"
        assert engine.failures[0]["error_type"] in {"AttributeError", "PicklingError"}
        assert "<lambda>" in engine.failures[0]["error_message"]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    assert multiprocessing.get_start_method() == start_method
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps({"scenario": scenario, "status": "passed"}), flush=True)


if __name__ == "__main__":
    main()
