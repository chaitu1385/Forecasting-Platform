"""Regression coverage for validation before forecast publication."""

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from conftest import make_actuals
from polars.testing import assert_frame_equal
from src.config.schema import (
    BacktestConfig,
    DataQualityConfig,
    ForecastConfig,
    OutputConfig,
    ParallelismConfig,
    PlatformConfig,
    PostValidationConfig,
)
from src.forecasting.naive import SeasonalNaiveForecaster
from src.metrics.aggregation import aggregate_metrics
from src.pipeline.backtest import BacktestPipeline
from src.pipeline.forecast import ForecastPipeline
from src.pipeline.run_store import MANIFEST_NAME, RunStore, resolve_run_root
from src.pipeline.validation_step import ValidationError, run_post_validation
from src.validation.business_rules import validate_segment_coverage
from src.validation.logical_validator import validate_group_balance
from src.validation.structural_validator import validate_primary_key

pytestmark = pytest.mark.unit

ORIGIN = date(2024, 3, 4)


@pytest.fixture(params=["standard", "multi_horizon"])
def forecast_mode(request):
    return request.param


def _config(
    tmp_path: Path,
    horizon: int = 4,
    id_col: str = "series_id",
    time_col: str = "week",
) -> PlatformConfig:
    return PlatformConfig(
        lob="validation",
        forecast=ForecastConfig(
            horizon_weeks=horizon,
            forecasters=["naive_seasonal"],
            series_id_column=id_col,
            time_column=time_col,
        ),
        data_quality=DataQualityConfig(fill_gaps=False, min_series_length_weeks=1),
        output=OutputConfig(
            forecast_path=str(tmp_path / "forecasts"),
            metrics_path=str(tmp_path / "metrics"),
        ),
        parallelism=ParallelismConfig(n_workers=1),
        post_validation=PostValidationConfig(halt_on_blocker=True),
    )


def _actuals(config: PlatformConfig, value: float = 100.0) -> pl.DataFrame:
    return (
        make_actuals(n_series=2, n_weeks=8)
        .with_columns(pl.lit(value).alias("quantity"))
        .rename({
            "series_id": config.forecast.series_id_column,
            "week": config.forecast.time_column,
        })
    )


def _champion(mode: str, horizon: int = 4) -> str | pl.DataFrame:
    if mode == "standard":
        return "naive_seasonal"
    split = max(1, horizon // 2)
    rows = [{
        "model_id": "naive_seasonal",
        "horizon_bucket": "short",
        "start_step": 1,
        "end_step": split,
    }]
    if split < horizon:
        rows.append({
            "model_id": "naive_seasonal",
            "horizon_bucket": "long",
            "start_step": split + 1,
            "end_step": horizon,
        })
    return pl.DataFrame(rows)


def _business_only(config: PlatformConfig) -> None:
    config.post_validation.structural_checks = False
    config.post_validation.logical_checks = False
    config.post_validation.simpsons_paradox_checks = False


def _published_files(config: PlatformConfig) -> dict[Path, bytes]:
    root = Path(config.output.forecast_path)
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_failed_attempt(pipeline: ForecastPipeline) -> None:
    assert pipeline._last_validation_result is None
    assert pipeline._last_forecast_path is None
    assert pipeline.last_run is None


def _assert_single_published_run(config: PlatformConfig) -> dict:
    store = RunStore(resolve_run_root(config, "forecast"))
    manifests = store.list(lob=config.lob, run_type="forecast")
    assert len(manifests) == 1
    manifest = manifests[0]
    directory = store.directory(config.lob, manifest.run_id)
    expected = {item["file"] for item in manifest.artifacts.values()}
    expected.update({MANIFEST_NAME, ".manifest.pending.json", ".attempt.json"})
    assert {file.name for file in directory.iterdir()} == expected
    assert {"input", "prepared", "forecast", "config", "validation"}.issubset(manifest.artifacts)
    return json.loads((directory / MANIFEST_NAME).read_text())


@pytest.mark.parametrize(
    ("id_col", "time_col"),
    [("series_id", "week"), ("series_id", "date"), ("sku", "period_end")],
)
def test_multi_period_forecast_validates_before_publication(
    tmp_path, forecast_mode, id_col, time_col
):
    config = _config(tmp_path, id_col=id_col, time_col=time_col)
    pipeline = ForecastPipeline(config)

    forecast = pipeline.run(
        _actuals(config), champion_model=_champion(forecast_mode), forecast_origin=ORIGIN
    )

    assert forecast.height == 8
    assert forecast.select([id_col, time_col]).unique().height == 8
    result = pipeline._last_validation_result
    assert result is not None
    assert result["confidence"]["caps_applied"].get("blocker_cap") is not True
    key_result = result["layers"]["structural"]["details"]["primary_key"]
    assert key_result["ok"] is True
    assert key_result["duplicate_count"] == 0
    assert_frame_equal(pl.read_parquet(pipeline._last_forecast_path), forecast)
    manifest = _assert_single_published_run(config)
    assert manifest["run_id"] == pipeline.context.run_id
    assert manifest["post_validation_grade"] == result["grade"]
    assert manifest["post_validation_score"] == result["score"]


@pytest.mark.parametrize("horizon", [1, 4])
def test_real_blocker_prevents_forecast_and_manifest(tmp_path, forecast_mode, horizon):
    config = _config(tmp_path, horizon=horizon)
    _business_only(config)
    actuals = _actuals(config, value=-10.0)
    with pytest.raises(ValidationError, match="BLOCKER"):
        run_post_validation(
            actuals.rename({"quantity": "forecast"}), config.post_validation, config.lob
        )
    pipeline = ForecastPipeline(config)

    with pytest.raises(ValidationError, match="BLOCKER"):
        pipeline.run(
            actuals,
            champion_model=_champion(forecast_mode, horizon),
            forecast_origin=ORIGIN,
        )

    assert not Path(config.output.forecast_path).exists()
    _assert_failed_attempt(pipeline)


@pytest.mark.parametrize("halt_on_blocker", [False, True])
@pytest.mark.parametrize(
    ("layer", "check"),
    [
        ("structural_checks", "structural_validator.validate_primary_key"),
        ("logical_checks", "logical_validator.validate_no_future_dates"),
        ("business_rules_checks", "business_rules.validate_no_negative"),
        ("simpsons_paradox_checks", "simpsons_paradox.check_simpsons_paradox"),
    ],
)
def test_unexpected_checker_error_prevents_publication(
    tmp_path, monkeypatch, forecast_mode, halt_on_blocker, layer, check
):
    config = _config(tmp_path)
    config.post_validation = PostValidationConfig(
        halt_on_blocker=halt_on_blocker,
        structural_checks=False,
        logical_checks=False,
        business_rules_checks=False,
        simpsons_paradox_checks=False,
    )
    setattr(config.post_validation, layer, True)
    failure = TypeError("unexpected validator failure")

    def fail_check(*args, **kwargs):
        raise failure

    # Keep the real validation orchestrator, scoring, and publication path.
    monkeypatch.setattr(f"src.validation.{check}", fail_check)
    pipeline = ForecastPipeline(config)

    with pytest.raises(TypeError, match="unexpected validator failure") as caught:
        pipeline.run(
            _actuals(config), champion_model=_champion(forecast_mode), forecast_origin=ORIGIN
        )

    assert caught.value is failure
    assert not Path(config.output.forecast_path).exists()
    _assert_failed_attempt(pipeline)


@pytest.mark.parametrize("same_origin", [False, True])
@pytest.mark.parametrize("failure_kind", ["blocker", "exception"])
def test_failed_repeat_preserves_published_files(
    tmp_path, monkeypatch, forecast_mode, same_origin, failure_kind
):
    config = _config(tmp_path)
    _business_only(config)
    pipeline = ForecastPipeline(config)
    champion = _champion(forecast_mode)
    pipeline.run(_actuals(config), champion_model=champion, forecast_origin=ORIGIN)
    before = _published_files(config)
    _assert_single_published_run(config)
    assert pipeline._last_validation_result is not None

    if failure_kind == "exception":
        def fail_check(*args, **kwargs):
            raise TypeError("unexpected validator failure")

        monkeypatch.setattr("src.validation.business_rules.validate_no_negative", fail_check)
        error = TypeError
    else:
        error = ValidationError

    with pytest.raises(error):
        pipeline.run(
            _actuals(config, value=-10.0),
            champion_model=champion,
            forecast_origin=ORIGIN if same_origin else ORIGIN + timedelta(weeks=1),
        )

    assert _published_files(config) == before
    _assert_failed_attempt(pipeline)


def test_disabled_validation_preserves_opt_out(tmp_path, monkeypatch, forecast_mode):
    config = _config(tmp_path)
    config.post_validation.enabled = False

    def fail_check(*args, **kwargs):
        pytest.fail("Disabled post-validation must not run any checks")

    monkeypatch.setattr("src.pipeline.validation_step.run_structural_checks", fail_check)
    pipeline = ForecastPipeline(config)
    forecast = pipeline.run(
        _actuals(config, value=-10.0),
        champion_model=_champion(forecast_mode),
        forecast_origin=ORIGIN,
    )

    assert forecast["forecast"].to_list() == [-10.0] * 8
    _assert_single_published_run(config)
    assert pipeline._last_validation_result is None


def test_nonblocking_grade_warns_and_publishes(tmp_path, forecast_mode, caplog):
    config = _config(tmp_path)
    _business_only(config)
    config.post_validation.halt_on_blocker = False
    config.post_validation.min_grade = "A"
    pipeline = ForecastPipeline(config)

    with caplog.at_level(logging.WARNING, logger="src.pipeline.validation_step"):
        forecast = pipeline.run(
            _actuals(config, value=-10.0),
            champion_model=_champion(forecast_mode),
            forecast_origin=ORIGIN,
        )

    assert forecast.height == 8
    _assert_single_published_run(config)
    result = pipeline._last_validation_result
    assert result is not None
    assert result["confidence"]["caps_applied"]["blocker_cap"] is True
    assert "below minimum A" in caplog.text


def test_warning_does_not_halt_publication(tmp_path, forecast_mode, caplog):
    config = _config(tmp_path)
    config.post_validation.min_grade = "A"
    config.post_validation.business_rules_checks = False
    config.post_validation.simpsons_paradox_checks = False
    actuals = _actuals(config)
    offset = date.today() + timedelta(weeks=1) - actuals["week"].min()
    actuals = actuals.with_columns(
        pl.Series("week", [day + offset for day in actuals["week"]])
    )
    pipeline = ForecastPipeline(config)

    with caplog.at_level(logging.WARNING, logger="src.pipeline.validation_step"):
        pipeline.run(actuals, champion_model=_champion(forecast_mode), forecast_origin=ORIGIN)

    result = pipeline._last_validation_result
    future_dates = result["layers"]["logical"]["details"]["no_future_dates"]
    assert future_dates["severity"] == "WARNING"
    assert result["confidence"]["caps_applied"].get("blocker_cap") is not True
    assert "below minimum A" in caplog.text
    _assert_single_published_run(config)


def test_quantile_output_is_validated_before_publication(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.forecast.quantiles = [0.1, 0.9]

    def missing_quantiles(self, horizon, quantiles, id_col, time_col):
        return self.predict(horizon, id_col, time_col).select([id_col, time_col]).with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias(f"forecast_p{int(round(q * 100))}")
                for q in quantiles
            ]
        )

    monkeypatch.setattr(SeasonalNaiveForecaster, "predict_quantiles", missing_quantiles)
    pipeline = ForecastPipeline(config)

    with pytest.raises(ValidationError, match="BLOCKER"):
        pipeline.run(_actuals(config), forecast_origin=ORIGIN)

    assert not Path(config.output.forecast_path).exists()
    _assert_failed_attempt(pipeline)


def test_valid_quantiles_are_published(tmp_path):
    config = _config(tmp_path)
    config.forecast.quantiles = [0.1, 0.9]
    pipeline = ForecastPipeline(config)

    forecast = pipeline.run(make_actuals(n_series=2, n_weeks=104), forecast_origin=ORIGIN)

    assert {"forecast_p10", "forecast_p90"}.issubset(forecast.columns)
    assert forecast.height == 8
    assert pipeline._last_validation_result["layers"]["structural"]["ok"] is True
    assert_frame_equal(pl.read_parquet(pipeline._last_forecast_path), forecast)
    _assert_single_published_run(config)


def test_constraints_run_before_validation(tmp_path):
    config = _config(tmp_path)
    config.forecast.constraints.enabled = True
    _business_only(config)
    pipeline = ForecastPipeline(config)

    forecast = pipeline.run(_actuals(config, value=-10.0), forecast_origin=ORIGIN)

    assert forecast["forecast"].to_list() == [0.0] * 8
    assert pipeline._last_validation_result["layers"]["business"]["details"]["no_negative"]["ok"]
    _assert_single_published_run(config)


def test_reused_pipeline_clears_result_when_validation_disabled(tmp_path, forecast_mode):
    config = _config(tmp_path)
    _business_only(config)
    pipeline = ForecastPipeline(config)
    pipeline.run(_actuals(config), forecast_origin=ORIGIN)
    assert pipeline._last_validation_result is not None

    config.post_validation.enabled = False
    pipeline.run(
        _actuals(config), champion_model=_champion(forecast_mode), forecast_origin=ORIGIN
    )

    assert pipeline._last_validation_result is None
    store = RunStore(resolve_run_root(config, "forecast"))
    assert len(store.list(lob=config.lob)) == 2
    directory = store.directory(config.lob, pipeline.context.run_id)
    manifest = json.loads((directory / MANIFEST_NAME).read_text())
    assert manifest["post_validation_grade"] is None
    assert manifest["post_validation_score"] is None


def test_reused_pipeline_clears_state_before_model_failure(tmp_path):
    config = _config(tmp_path)
    _business_only(config)
    pipeline = ForecastPipeline(config)
    pipeline.run(_actuals(config), forecast_origin=ORIGIN)
    before = _published_files(config)
    assert pipeline._last_validation_result is not None

    with pytest.raises(KeyError, match="not registered"):
        pipeline.run(_actuals(config), champion_model="missing", forecast_origin=ORIGIN)

    assert _published_files(config) == before
    _assert_failed_attempt(pipeline)


@pytest.mark.parametrize(
    ("id_col", "time_col"), [("series_id", "week"), ("sku", "period_end")]
)
def test_multi_horizon_duplicate_periods_are_blocked(tmp_path, id_col, time_col):
    config = _config(tmp_path, id_col=id_col, time_col=time_col)
    champions = _champion("multi_horizon").with_columns(
        pl.when(pl.col("horizon_bucket") == "long")
        .then(2)
        .otherwise(pl.col("start_step"))
        .alias("start_step")
    )
    pipeline = ForecastPipeline(config)

    with pytest.raises(ValidationError, match="BLOCKER"):
        pipeline.run(_actuals(config), champion_model=champions, forecast_origin=ORIGIN)

    assert not Path(config.output.forecast_path).exists()
    _assert_failed_attempt(pipeline)


@pytest.mark.parametrize("time_col", ["target_week", "week", "date"])
def test_standalone_validation_detects_forecast_period(time_col):
    forecast = make_actuals(n_series=2, n_weeks=4).rename(
        {"week": time_col, "quantity": "forecast"}
    )
    result = run_post_validation(
        forecast, PostValidationConfig(halt_on_blocker=True), lob="validation"
    )
    assert result["layers"]["structural"]["details"]["primary_key"]["ok"] is True


def test_backtest_keeps_canonical_keys_for_custom_input_columns(tmp_path):
    config = _config(tmp_path, id_col="sku", time_col="period_end")
    config.backtest = BacktestConfig(n_folds=2, val_weeks=2)
    result = BacktestPipeline(config).run(_actuals(config))

    assert {"series_id", "target_week", "model_id", "fold"}.issubset(
        result["backtest_results"].columns
    )
    validation = result["validation"]
    assert validation is not None
    assert validation["layers"]["structural"]["details"]["primary_key"]["ok"] is True
    assert validation["confidence"]["caps_applied"].get("blocker_cap") is not True


def _backtest_metric_records() -> pl.DataFrame:
    return pl.DataFrame({
        "run_id": ["a" * 32, "a" * 32],
        "lob": ["validation", "validation"],
        "series_id": ["A", "B"],
        "target_week": [ORIGIN, ORIGIN],
        "forecast_step": [1, 1],
        "model_id": ["naive_seasonal", "naive_seasonal"],
        "fold": [0, 0],
        "actual": [100.0, 80.0],
        "forecast": [95.0, 76.0],
        "wmape": [0.05, 0.05],
        "mape": [None, None],
        "mase": [None, None],
        "mase_scale": [None, None],
    })


def _metric_validation_config() -> PostValidationConfig:
    return PostValidationConfig(
        halt_on_blocker=True,
        logical_checks=False,
        business_rules_checks=False,
        simpsons_paradox_checks=False,
    )


@pytest.mark.parametrize("bad_column", [
    "run_id", "lob", "series_id", "target_week", "forecast_step", "model_id", "fold",
    "actual", "forecast", "wmape",
    "unexpected",
])
def test_backtest_completeness_still_blocks_required_or_unknown_nulls(bad_column):
    records = _backtest_metric_records()
    before = records.clone()
    config = _metric_validation_config()
    result = run_post_validation(
        records, config, "validation", requested_metrics=["wmape"],
    )
    completeness = result["layers"]["structural"]["details"]["completeness"]
    assert completeness["ok"] is True
    assert "mape" not in {column["name"] for column in completeness["column_stats"]}
    assert_frame_equal(records, before)

    corrupted = records.with_columns(
        pl.lit(None, dtype=records.schema.get(bad_column, pl.Float64)).alias(bad_column)
    )
    with pytest.raises(ValidationError, match="BLOCKER"):
        run_post_validation(
            corrupted, config, "validation", requested_metrics=["wmape"],
        )


@pytest.mark.parametrize("requested_metrics", [None, ["wmape", "mape"]])
def test_selected_or_unspecified_metric_nulls_are_not_exempt(requested_metrics):
    records = _backtest_metric_records()
    with pytest.raises(ValidationError, match="BLOCKER"):
        run_post_validation(
            records, _metric_validation_config(), "validation",
            requested_metrics=requested_metrics,
        )


def test_partially_populated_unrequested_score_remains_checked():
    records = _backtest_metric_records().with_columns(pl.Series("mape", [0.05, None]))
    with pytest.raises(ValidationError, match="BLOCKER"):
        run_post_validation(
            records, _metric_validation_config(), "validation",
            requested_metrics=["wmape"],
        )


def test_requested_zero_actual_ratios_keep_undefined_rows_and_population_scores():
    records = _backtest_metric_records().with_columns(
        pl.Series("actual", [0.0, 10.0]),
        pl.Series("forecast", [2.0, 8.0]),
        pl.Series("wmape", [None, 0.2]),
        pl.Series("normalized_bias", [None, -0.2]),
        pl.Series("mape", [None, 0.2]),
        pl.Series("mae", [2.0, 2.0]),
        pl.Series("rmse", [2.0, 2.0]),
    )
    before = records.clone()
    result = run_post_validation(
        records, _metric_validation_config(), "validation",
        requested_metrics=["wmape", "normalized_bias", "mape", "mae", "rmse", "mase"],
    )
    metric_checks = result["layers"]["structural"]["details"]["metric_values"]
    assert metric_checks["ok"] is True
    for metric in ("wmape", "normalized_bias", "mape"):
        assert metric_checks["metrics"][metric] == {
            "defined_rows": 1, "undefined_rows": 1, "ok": True,
        }
    assert metric_checks["metrics"]["mase"] == {
        "defined_rows": 0, "undefined_rows": 2, "ok": True,
    }
    assert metric_checks["nullable_mase_scale_rows"] == 2
    assert_frame_equal(records, before)
    population = aggregate_metrics(records, ["wmape", "normalized_bias"])
    assert population["wmape"].item() == pytest.approx(0.4)
    assert population["normalized_bias"].item() == 0


@pytest.mark.parametrize(("column", "value"), [
    ("actual", float("nan")), ("forecast", float("inf")),
    ("wmape", float("nan")), ("wmape", float("inf")), ("wmape", -0.05),
    ("wmape", None), ("mase_scale", 0.0), ("mase_scale", -1.0),
    ("mase_scale", float("nan")), ("mase_scale", float("inf")),
])
def test_backtest_metric_context_rejects_corrupt_values(column, value):
    records = _backtest_metric_records()
    values = records[column].to_list()
    values[0] = value
    records = records.with_columns(pl.Series(column, values, dtype=pl.Float64))
    with pytest.raises(ValidationError, match="BLOCKER"):
        run_post_validation(
            records, _metric_validation_config(), "validation", requested_metrics=["wmape"],
        )


def test_backtest_metric_context_rejects_fabricated_definedness():
    records = _backtest_metric_records().with_columns(
        pl.Series("actual", [0.0, 80.0]), pl.Series("wmape", [0.0, 0.05]),
    )
    with pytest.raises(ValidationError, match="BLOCKER"):
        run_post_validation(
            records, _metric_validation_config(), "validation", requested_metrics=["wmape"],
        )


@pytest.mark.parametrize("missing", [
    "run_id", "lob", "series_id", "target_week", "forecast_step", "model_id", "fold",
    "actual", "forecast", "wmape",
])
def test_backtest_metric_context_rejects_missing_required_columns(missing):
    with pytest.raises(ValidationError, match="BLOCKER"):
        run_post_validation(
            _backtest_metric_records().drop(missing), _metric_validation_config(), "validation",
            requested_metrics=["wmape"],
        )


def test_requested_mase_requires_recorded_scale_column():
    with pytest.raises(ValidationError, match="BLOCKER"):
        run_post_validation(
            _backtest_metric_records().drop("mase_scale"), _metric_validation_config(), "validation",
            requested_metrics=["wmape", "mase"],
        )


def test_backtest_metric_context_rejects_all_missing_keys():
    records = _backtest_metric_records().select("actual", "forecast", "wmape")
    with pytest.raises(ValidationError, match="BLOCKER"):
        run_post_validation(
            records, _metric_validation_config(), "validation", requested_metrics=["wmape"],
        )


@pytest.mark.parametrize("column", ["actual", "forecast", "wmape", "mase_scale"])
def test_backtest_metric_context_rejects_nonnumeric_values(column):
    records = _backtest_metric_records().with_columns(pl.lit("invalid").alias(column))
    with pytest.raises(ValidationError, match="BLOCKER"):
        run_post_validation(
            records, _metric_validation_config(), "validation", requested_metrics=["wmape"],
        )


@pytest.mark.parametrize("invalid", ["empty", "duplicate"])
def test_backtest_metric_context_rejects_invalid_population(invalid):
    records = _backtest_metric_records()
    records = records.head(0) if invalid == "empty" else pl.concat([records, records.head(1)])
    with pytest.raises(ValidationError, match="BLOCKER"):
        run_post_validation(
            records, _metric_validation_config(), "validation", requested_metrics=["wmape"],
        )


@pytest.mark.parametrize("key_columns", [["series_id"], ["series_id", "week"]])
def test_primary_key_duplicate_sample_includes_null_rows(key_columns):
    frame = pl.DataFrame({
        "series_id": [None, None, "A"],
        "week": [ORIGIN] * 3,
    })
    result = validate_primary_key(frame, key_columns)
    assert result["duplicate_count"] == 1
    assert result["null_count"] == 2
    assert result["duplicate_sample"] == [
        {column: frame[column][0] for column in key_columns}
    ]


def test_group_balance_counts_null_group_rows():
    frame = pl.DataFrame({"group": [None, None, "A", "A", "A"]})
    result = validate_group_balance(frame, "group", min_count=2)
    assert result["ok"] is True
    assert result["group_count"] == 2
    assert result["min_count"] == 2
    assert result["max_count"] == 3
    assert result["undersize_groups"] == []


def test_segment_coverage_counts_null_group_rows():
    frame = pl.DataFrame({"group": [None, None, "A", "A", "A"]})
    result = validate_segment_coverage(
        frame, "group", expected_segments=["A"], min_rows_per_segment=2
    )
    assert result["ok"] is True
    assert result["missing_segments"] == []
    assert result["undersized_segments"] == []
