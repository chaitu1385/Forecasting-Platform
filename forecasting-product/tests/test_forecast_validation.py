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
from src.pipeline.backtest import BacktestPipeline
from src.pipeline.forecast import ForecastPipeline
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
    files = _published_files(config)
    assert len(files) == 2
    manifest = json.loads(next(data for path, data in files.items() if path.suffix == ".json"))
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
    assert len(before) == 2
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
    assert len(_published_files(config)) == 2
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
    assert len(_published_files(config)) == 2
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
    assert len(_published_files(config)) == 2


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
    assert len(_published_files(config)) == 2


def test_constraints_run_before_validation(tmp_path):
    config = _config(tmp_path)
    config.forecast.constraints.enabled = True
    _business_only(config)
    pipeline = ForecastPipeline(config)

    forecast = pipeline.run(_actuals(config, value=-10.0), forecast_origin=ORIGIN)

    assert forecast["forecast"].to_list() == [0.0] * 8
    assert pipeline._last_validation_result["layers"]["business"]["details"]["no_negative"]["ok"]
    assert len(_published_files(config)) == 2


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
    files = _published_files(config)
    manifest = json.loads(next(data for path, data in files.items() if path.suffix == ".json"))
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
    # Unrequested metrics have null placeholders in the canonical metric schema.
    config.post_validation.halt_on_blocker = False
    result = BacktestPipeline(config).run(_actuals(config))

    assert {"series_id", "target_week", "model_id", "fold"}.issubset(
        result["backtest_results"].columns
    )
    validation = result["validation"]
    assert validation is not None
    assert validation["layers"]["structural"]["details"]["primary_key"]["ok"] is True


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
