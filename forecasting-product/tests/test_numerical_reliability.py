"""Numerical oracles independent of fitted library predictions or golden runs."""

from datetime import date, timedelta

import polars as pl
import pytest

from src.backtesting.champion import ChampionSelector
from src.backtesting.cross_validator import WalkForwardCV
from src.backtesting.engine import BacktestEngine
from src.config.schema import BacktestConfig, HorizonBucket, PlatformConfig
from src.forecasting.feature_manager import MLForecastFeatureManager
from src.forecasting.naive import SeasonalNaiveForecaster
from src.forecasting.registry import registry
from src.metrics.store import MetricStore
from src.pipeline.backtest import BacktestPipeline


def _panel(dates, values=None):
    return pl.DataFrame({
        "series_id": ["s"] * len(dates),
        "week": dates,
        "quantity": values if values is not None else [10.0] * len(dates),
    })


def _months(n=24):
    return [date(2022 + i // 12, i % 12 + 1, 1) for i in range(n)]


def _config(tmp_path, *, frequency="D", gap=0):
    config = PlatformConfig()
    config.forecast.frequency = frequency
    config.forecast.forecasters = ["naive_seasonal"]
    config.backtest.n_folds = 2
    config.backtest.val_weeks = 2
    config.backtest.gap_weeks = gap
    config.output.metrics_path = str(tmp_path / "metrics")
    config.data_quality.min_series_length_weeks = 1
    return config


def _rows(predictions, actuals=(100.0, 100.0), series_ids=("s", "s")):
    rows = []
    for model, values in predictions.items():
        fold_wmape = sum(abs(a - f) for a, f in zip(actuals, values)) / sum(actuals)
        for i, (actual, forecast, sid) in enumerate(zip(actuals, values, series_ids)):
            # Deliberately repeated whole-fold scores: horizon selection must
            # use the raw observations, never this cached summary.
            rows.append({
                "run_id": "oracle",
                "lob": "retail",
                "model_id": model,
                "fold": 0,
                "series_id": sid,
                "target_week": date(2024, 1, 1) + timedelta(days=i),
                "forecast_step": i + 1,
                "actual": actual,
                "forecast": float(forecast),
                "wmape": fold_wmape if len(set(series_ids)) == 1 else (
                    abs(actual - forecast) / actual if actual else None
                ),
                "normalized_bias": (forecast - actual) / actual if actual else None,
            })
    return pl.DataFrame(rows)


def test_real_month_starts_are_not_thirty_day_steps():
    model = SeasonalNaiveForecaster(frequency="M")
    model.fit(_panel(_months()))
    assert model.predict(3)["week"].to_list() == [
        date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1),
    ]


@pytest.mark.parametrize("model_name", ["naive_seasonal", "croston", "croston_sba", "tsb"])
@pytest.mark.parametrize(
    "frequency,history,expected",
    [
        ("D", [date(2024, 1, i) for i in range(1, 11)],
         [date(2024, 1, 11), date(2024, 1, 12)]),
        ("W", [date(2024, 1, 2), date(2024, 1, 9)],
         [date(2024, 1, 16), date(2024, 1, 23)]),
        ("M", [date(2023, 12, 31), date(2024, 1, 31)],
         [date(2024, 2, 29), date(2024, 3, 31)]),
        ("Q", [date(2023, 9, 30), date(2023, 12, 31)],
         [date(2024, 3, 31), date(2024, 6, 30)]),
    ],
)
def test_registry_frequency_reaches_points_and_intervals(model_name, frequency, history, expected):
    model = registry.build(model_name, frequency=frequency)
    model.fit(_panel(history))
    assert model.predict(2).sort("week")["week"].to_list() == expected
    assert model.predict_quantiles(2, [0.1, 0.5, 0.9]).sort("week")["week"].to_list() == expected
    assert model.get_params()["frequency"] == frequency


def test_monthly_cv_has_three_actual_periods_and_a_two_period_gap():
    splits = WalkForwardCV(n_folds=2, val_weeks=3, gap_weeks=2, frequency="M").split_data(
        _panel(_months())
    )
    assert len(splits) == 2
    assert [fold.train_end for fold, _, _ in splits] == [date(2023, 4, 1), date(2023, 7, 1)]
    assert [val["week"].to_list() for _, _, val in splits] == [
        [date(2023, 7, 1), date(2023, 8, 1), date(2023, 9, 1)],
        [date(2023, 10, 1), date(2023, 11, 1), date(2023, 12, 1)],
    ]


def test_configured_monthly_frequency_rejects_thirty_day_pseudomonths():
    dates = [date(2022, 1, 1) + timedelta(days=30 * i) for i in range(24)]
    with pytest.raises(ValueError, match="calendar|frequency|regular|anchor"):
        WalkForwardCV(n_folds=2, val_weeks=3, frequency="M").split(_panel(dates))


def test_feature_future_dates_follow_each_series_origin_and_frequency():
    manager = MLForecastFeatureManager(frequency="M")
    history = _panel([date(2024, 1, 31), date(2024, 2, 29)]).with_columns(pl.lit(2.0).alias("promo"))
    other = _panel([date(2023, 12, 31), date(2024, 1, 31)]).with_columns(
        pl.lit("other").alias("series_id"), pl.lit(3.0).alias("promo")
    )
    manager.prepare_fit(pl.concat([history, other]), "series_id", "week", "quantity")
    future = pl.from_pandas(manager.prepare_predict(2)).with_columns(pl.col("ds").cast(pl.Date))
    assert future.filter(pl.col("unique_id") == "s").sort("ds")["ds"].to_list() == [
        date(2024, 3, 31), date(2024, 4, 30),
    ]
    assert future.filter(pl.col("unique_id") == "other").sort("ds")["ds"].to_list() == [
        date(2024, 2, 29), date(2024, 3, 31),
    ]


def test_gap_is_forecast_before_scoring_and_steps_are_leads_from_training(tmp_path):
    config = _config(tmp_path, gap=2)
    engine = BacktestEngine(config)
    result = engine.run(
        _panel([date(2024, 1, i) for i in range(1, 17)]),
        [SeasonalNaiveForecaster(season_length=1, frequency="D")],
    )
    assert engine.failures == []
    assert result.height == 4
    assert result["forecast_step"].to_list() == [3, 4, 3, 4]
    assert result["wmape"].to_list() == [0.0] * 4


class _BrokenPrediction(SeasonalNaiveForecaster):
    name = "broken"

    def __init__(self, defect):
        super().__init__(season_length=1, frequency="D")
        self.defect = defect

    def predict(self, horizon, id_col="series_id", time_col="week"):
        predictions = super().predict(horizon, id_col=id_col, time_col=time_col)
        if self.defect == "wrong_dates":
            return predictions.with_columns(pl.col(time_col) + pl.duration(days=1))
        if self.defect == "missing":
            return predictions.head(1)
        if self.defect == "duplicate":
            return pl.concat([predictions, predictions.head(1)])
        if self.defect == "nonfinite":
            return predictions.with_columns(pl.lit(float("nan")).alias("forecast"))
        if self.defect == "late_failure" and self._fitted_data.height > 12:
            raise RuntimeError("deliberate later-fold failure")
        return predictions


@pytest.mark.parametrize("defect", ["wrong_dates", "missing", "duplicate", "nonfinite", "late_failure"])
def test_invalid_or_partial_model_cannot_win_on_surviving_rows(tmp_path, defect):
    engine = BacktestEngine(_config(tmp_path))
    results = engine.run(
        _panel([date(2024, 1, i) for i in range(1, 17)]),
        [_BrokenPrediction(defect), SeasonalNaiveForecaster(frequency="D")],
    )
    assert engine.failures, "A missing/invalid prediction must be reported, not silently joined away"
    assert results["model_id"].unique().to_list() == ["naive_seasonal"]


def test_horizon_winners_are_recomputed_from_their_actual_errors():
    records = _rows({"short": [100.0, 150.0], "long": [130.0, 100.0]})
    selected = ChampionSelector(BacktestConfig()).select_by_horizon(
        records, [HorizonBucket("short", 1, 1), HorizonBucket("long", 2, 2)]
    )
    assert {row["horizon_bucket"]: row["model_id"] for row in selected.iter_rows(named=True)} == {
        "short": "short", "long": "long",
    }
    assert selected["wmape"].to_list() == [0.0, 0.0]


def test_volume_and_explicit_macro_wmape_have_different_truthful_winners():
    records = _rows({"A": [900.0, 0.0], "B": [800.0, 1.0]}, (1000.0, 1.0), ("large", "small"))
    winner = ChampionSelector(BacktestConfig()).select(records)
    assert winner["model_id"].item() == "A"
    assert winner["wmape"].item() == pytest.approx(101.0 / 1001.0)
    macro = ChampionSelector(BacktestConfig(primary_metric="macro_wmape")).select(records)
    assert macro["model_id"].item() == "B"
    assert macro["macro_wmape"].item() == pytest.approx(0.1)


def test_bias_as_primary_metric_is_ranked_by_distance_from_zero():
    records = _rows({"under": [50.0, 50.0], "over": [110.0, 110.0]})
    winner = ChampionSelector(
        BacktestConfig(primary_metric="normalized_bias", secondary_metric="mae")
    ).select(records)
    assert winner["model_id"].item() == "over"
    assert winner["normalized_bias"].item() == pytest.approx(0.1)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nonfinite", "actual_disagreement"])
def test_selection_rejects_incomparable_populations(defect):
    records = _rows({"A": [90.0, 90.0], "B": [80.0, 80.0]})
    if defect == "missing":
        records = records.head(3)
    elif defect == "duplicate":
        records = pl.concat([records, records.head(1)])
    elif defect == "nonfinite":
        records = records.with_columns(
            pl.when(pl.col("model_id") == "B").then(float("inf")).otherwise(pl.col("forecast")).alias("forecast")
        )
    else:
        records = records.with_columns(
            pl.when(pl.col("model_id") == "B").then(200.0).otherwise(pl.col("actual")).alias("actual")
        )
    with pytest.raises(ValueError, match="(?i)coverage|duplicate|finite|actual|population"):
        ChampionSelector(BacktestConfig()).select(records)


def test_metric_store_uses_same_volume_math_as_selector(tmp_path):
    records = _rows({"A": [900.0, 0.0], "B": [800.0, 1.0]}, (1000.0, 1.0), ("large", "small"))
    store = MetricStore(str(tmp_path))
    store.write(records.with_columns(pl.lit("backtest").alias("run_type")), "backtest", "retail")
    board = store.leaderboard(lob="retail")
    assert board["model_id"].to_list() == ["A", "B"]
    assert board["wmape"].to_list() == pytest.approx([101 / 1001, 200 / 1001])


@pytest.mark.parametrize("option", ["cleansing", "structural_breaks", "screen"])
def test_backtest_rejects_non_fold_local_learned_preprocessing_before_build(tmp_path, monkeypatch, option):
    config = _config(tmp_path)
    if option == "screen":
        config.forecast.external_regressors.enabled = True
        config.forecast.external_regressors.screen.enabled = True
    else:
        getattr(config.data_quality, option).enabled = True
    pipeline = BacktestPipeline(config)
    monkeypatch.setattr(
        pipeline._series_builder, "build",
        lambda **kwargs: pytest.fail("Unsafe whole-history preprocessing was reached"),
    )
    with pytest.raises(ValueError, match="fold.local"):
        pipeline.run(_panel([date(2024, 1, i) for i in range(1, 17)]))


@pytest.mark.parametrize("model_name", ["naive_seasonal", "auto_arima", "auto_ets", "auto_theta", "mstl"])
def test_explicit_season_length_is_not_replaced_by_a_frequency_default(model_name):
    model = registry.build(model_name, frequency="D", season_length=52)
    assert model.season_length == 52
    assert model.frequency == "D"


@pytest.mark.parametrize("name", ["nbeats", "nhits", "tft"])
def test_neural_registry_preserves_frequency_without_loading_a_backend(name):
    model = registry.build(name, frequency="M")
    assert model.frequency == "M"
    assert model.get_params()["frequency"] == "M"


@pytest.mark.parametrize("model_name", ["auto_ets", "lgbm_direct"])
@pytest.mark.parametrize(
    "frequency,dates,expected",
    [
        ("D", [date(2024, 1, i) for i in range(1, 21)],
         [date(2024, 1, 21), date(2024, 1, 22)]),
        ("W", [date(2023, 1, 3) + timedelta(weeks=i) for i in range(20)],
         [date(2023, 5, 23), date(2023, 5, 30)]),
        ("M", [date(2023, i, 1) for i in range(1, 13)],
         [date(2024, 1, 1), date(2024, 2, 1)]),
        ("M", [date(2023, 5, 31), date(2023, 6, 30), date(2023, 7, 31), date(2023, 8, 31),
               date(2023, 9, 30), date(2023, 10, 31), date(2023, 11, 30), date(2023, 12, 31),
               date(2024, 1, 31), date(2024, 2, 29)],
         [date(2024, 3, 31), date(2024, 4, 30)]),
        ("Q", [date(2022, 3, 31), date(2022, 6, 30), date(2022, 9, 30), date(2022, 12, 31),
               date(2023, 3, 31), date(2023, 6, 30), date(2023, 9, 30), date(2023, 12, 31)],
         [date(2024, 3, 31), date(2024, 6, 30)]),
    ],
)
def test_real_nixtla_models_use_the_same_calendar(model_name, frequency, dates, expected):
    kwargs = {"season_length": 1} if model_name == "auto_ets" else {
        "lags": [1], "lag_transforms": {1: []},
        "lgbm_params": {"n_estimators": 3, "min_child_samples": 1, "num_leaves": 2},
    }
    model = registry.build(model_name, frequency=frequency, **kwargs)
    history = _panel(dates, [float(i + 10) for i in range(len(dates))])
    if model_name == "lgbm_direct":
        history = history.with_columns(pl.lit(2.0).alias("planned_promo"))
    model.fit(history)
    assert model.predict(2).sort("week")["week"].to_list() == expected


def test_volume_metric_truth_includes_zero_demand_false_positive_errors():
    from src.metrics.aggregation import aggregate_metrics
    records = _rows({"A": [110.0, 180.0, 5.0]}, (100.0, 200.0, 0.0), ("s", "s", "zero"))
    board = aggregate_metrics(records, ["wmape", "normalized_bias", "mae", "mape", "rmse"])
    row = board.row(0, named=True)
    assert row["wmape"] == pytest.approx(35 / 300)
    assert row["normalized_bias"] == pytest.approx(-5 / 300)
    assert row["mae"] == pytest.approx(35 / 3)
    assert row["mape"] == pytest.approx(0.1)
    assert row["rmse"] == pytest.approx((525 / 3) ** 0.5)


@pytest.mark.parametrize("metric", ["wmape", "normalized_bias", "mae", "mape", "rmse"])
def test_aggregation_matches_metric_definitions(metric):
    from src.metrics.aggregation import aggregate_metrics
    from src.metrics.definitions import METRIC_REGISTRY
    records = _rows({"A": [130.0, 90.0]})
    expected = METRIC_REGISTRY[metric](records["actual"], records["forecast"])
    assert aggregate_metrics(records, [metric])[metric].item() == pytest.approx(expected)


def test_undefined_primary_and_mixed_runs_are_explicit_errors():
    records = _rows({"A": [90.0, 90.0], "B": [100.0, 100.0]})
    zero = records.with_columns(pl.lit(0.0).alias("actual"))
    with pytest.raises(ValueError, match="undefined"):
        ChampionSelector(BacktestConfig()).select(zero)
    mixed = pl.concat([records, records.with_columns(pl.lit("other-run").alias("run_id"))])
    with pytest.raises(ValueError, match="run_id"):
        ChampionSelector(BacktestConfig()).select(mixed)


def test_insufficient_cv_history_is_not_a_smaller_successful_backtest():
    with pytest.raises(ValueError, match="Insufficient data"):
        WalkForwardCV(n_folds=3, val_weeks=3, frequency="D").split(
            _panel([date(2024, 1, i) for i in range(1, 8)])
        )


def _spark_params(config):
    from src.spark.pipeline import _extract_forecast_params
    params = _extract_forecast_params(config, "croston", 2)
    params.update({
        "model_names": ["croston"], "metrics": config.metrics,
        "run_id": "spark-oracle", "run_date": "2024-01-01", "lob": config.lob,
    })
    return params


def test_pure_spark_worker_preserves_frequency_gap_keys_and_bias_sign(tmp_path):
    from src.metrics.aggregation import aggregate_metrics
    from src.spark.pipeline import backtest_partition, forecast_partition
    config = _config(tmp_path, gap=1)
    params = _spark_params(config)
    history = _panel(
        [date(2024, 1, i) for i in range(1, 17)],
        [20.0] * 12 + [10.0] * 4,
    )
    forecast = pl.from_pandas(forecast_partition(history.to_pandas(), params))
    assert forecast["week"].cast(pl.Date).to_list() == [date(2024, 1, 17), date(2024, 1, 18)]
    remote = pl.from_pandas(backtest_partition(history.to_pandas(), params)).with_columns(
        pl.col("target_week").cast(pl.Date), pl.col("run_date").cast(pl.Date),
    )
    engine = BacktestEngine(config)
    local = engine.run(history, [registry.build("croston", frequency="D")])
    columns = ["fold", "series_id", "target_week", "forecast_step", "actual", "forecast"]
    assert remote.select(columns).to_dicts() == local.select(columns).to_dicts()
    remote_board = aggregate_metrics(remote, ["wmape", "normalized_bias"])
    local_board = aggregate_metrics(local, ["wmape", "normalized_bias"])
    assert remote_board["normalized_bias"].item() > 0
    assert remote_board.select("wmape", "normalized_bias").to_dicts() == local_board.select("wmape", "normalized_bias").to_dicts()


def test_pure_spark_worker_does_not_hide_missing_predictions(tmp_path, monkeypatch):
    from src.spark.pipeline import backtest_partition
    config = _config(tmp_path)
    monkeypatch.setattr(registry, "build", lambda *args, **kwargs: _BrokenPrediction("missing"))
    with pytest.raises(ValueError, match="coverage"):
        backtest_partition(
            _panel([date(2024, 1, i) for i in range(1, 17)]).to_pandas(), _spark_params(config),
        )


def test_shared_run_identity_and_deferred_publication_survive_scoring(tmp_path, monkeypatch):
    engine = BacktestEngine(_config(tmp_path))
    monkeypatch.setattr(
        engine.metric_store, "write",
        lambda *args, **kwargs: pytest.fail("persist=False wrote metrics before run publication"),
    )
    result = engine.run(
        _panel([date(2024, 1, i) for i in range(1, 17)]),
        [SeasonalNaiveForecaster(frequency="D")],
        run_id="caller-owned-run", persist=False,
    )
    assert result["run_id"].unique().to_list() == ["caller-owned-run"]
    assert result.height == 4
    assert engine.failures == []
