"""Causal interval regressions: preserve point forecasts, order labeled quantiles."""

import hashlib
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from scripts.prepare_m5_weekly import prepare_weekly
from src.config.loader import load_config
from src.config.schema import CalibrationConfig, ForecastConfig, PlatformConfig
from src.forecasting.naive import SeasonalNaiveForecaster
from src.pipeline.forecast import ForecastPipeline

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = [0.1, 0.5, 0.9]


def _series(values):
    return pl.DataFrame({
        "series_id": ["s"] * len(values),
        "week": [date(2024, 1, 1) + timedelta(days=index) for index in range(len(values))],
        "quantity": [None if value is None else float(value) for value in values],
    })


def _assert_ordered_finite(frame, quantiles=QUANTILES):
    columns = [f"forecast_p{int(round(level * 100))}" for level in sorted(quantiles)]
    for column in columns:
        assert frame[column].null_count() == 0
        assert frame[column].is_finite().all()
    for lower, upper in zip(columns, columns[1:]):
        assert (frame[lower] <= frame[upper]).all(), frame.select(columns).to_dicts()


@pytest.mark.parametrize(
    "values,point",
    [([1, 2, 6, 7], [6.0, 7.0]), ([8, 9, 3, 4], [3.0, 4.0])],
)
def test_one_sided_seasonal_residuals_do_not_make_a_labeled_median_cross(values, point):
    model = SeasonalNaiveForecaster(season_length=2, frequency="D")
    model.fit(_series(values))
    before = model.predict(2).sort("week")
    quantiles = model.predict_quantiles(2, QUANTILES).sort("week")
    assert before["forecast"].to_list() == point
    _assert_ordered_finite(quantiles)
    assert quantiles["forecast_p50"].to_list() == point
    assert model.predict(2).sort("week").to_dicts() == before.to_dicts()


@pytest.mark.parametrize("season_length", [1, 2])
def test_naive_quantiles_after_refit_match_a_fresh_model(season_length):
    model = SeasonalNaiveForecaster(season_length=season_length, frequency="D")
    model.fit(_series([1, 2, 6, 7]))
    model.predict_quantiles(4, QUANTILES)
    history = _series([8, 9, 3, 4])
    model.fit(history)
    fresh = SeasonalNaiveForecaster(season_length=season_length, frequency="D")
    fresh.fit(history)
    observed = model.predict_quantiles(4, QUANTILES).sort("week")
    assert observed.to_dicts() == fresh.predict_quantiles(4, QUANTILES).sort("week").to_dicts()
    _assert_ordered_finite(observed)
    assert observed["forecast_p50"].to_list() == model.predict(4).sort("week")["forecast"].to_list()


@pytest.mark.parametrize(
    "season_length,values,points,lower,upper",
    [
        (1, [10, 12, 16, 24, 40], [40, 40], [36.6, 36.6], [47.6, 47.6]),
        (1, [100, 98, 94, 86, 70], [70, 70], [62.4, 62.4], [73.4, 73.4]),
        (2, [10, 100, 12, 105, 16, 112, 24, 123, 40, 140],
         [40, 140], [36.6, 136.6], [47.6, 146.2]),
        (2, [100, 200, 98, 195, 94, 188, 86, 177, 70, 160],
         [70, 160], [62.4, 153.8], [73.4, 163.4]),
    ],
)
def test_residual_centering_preserves_points_and_empirical_interval_width(
    season_length, values, points, lower, upper,
):
    # Even-position residuals are ±[2, 4, 8, 16]; odd ones are ±[5, 7, 11, 17].
    # Linear empirical P10/P90 minus their own median gives these exact offsets.
    model = SeasonalNaiveForecaster(season_length=season_length, frequency="D")
    model.fit(_series(values))
    levels = [0.9, 0.25, 0.5, 0.1, 0.75]
    observed = model.predict_quantiles(2, levels).sort("week")
    assert model.predict(2).sort("week")["forecast"].to_list() == points
    assert observed["forecast_p50"].to_list() == points
    assert observed["forecast_p10"].to_list() == pytest.approx(lower)
    assert observed["forecast_p90"].to_list() == pytest.approx(upper)
    widths = [11.0, 11.0 if season_length == 1 else 9.6]
    assert (observed["forecast_p90"] - observed["forecast_p10"]).to_list() == pytest.approx(widths)
    _assert_ordered_finite(observed, levels)


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), -float("inf")])
def test_nonfinite_refit_fails_without_retaining_previous_predictions(bad_value):
    model = SeasonalNaiveForecaster(season_length=1, frequency="D")
    model.fit(_series([10, 12, 16, 24, 40]))
    with pytest.raises(ValueError, match="finite numeric"):
        model.fit(_series([10, 12, bad_value, 24, 40]))
    assert model.predict(2).is_empty()
    model.fit(_series([100, 98, 94, 86, 70]))
    _assert_ordered_finite(model.predict_quantiles(2, QUANTILES))


def _pipeline(tmp_path, *, quantiles=QUANTILES, calibrated=False):
    config = PlatformConfig(
        lob="interval-regression",
        forecast=ForecastConfig(
            frequency="D", horizon_weeks=2, quantiles=quantiles,
            calibration=CalibrationConfig(enabled=calibrated),
        ),
    )
    config.data_quality.min_series_length_weeks = 0
    config.data_quality.fill_gaps = False
    config.post_validation.enabled = False
    config.output.forecast_path = str(tmp_path / "forecasts")
    config.output.metrics_path = str(tmp_path / "metrics")
    return ForecastPipeline(config)


def test_upper_tail_only_quantiles_need_not_contain_the_raw_point(tmp_path):
    pipeline = _pipeline(tmp_path, quantiles=[0.9, 0.75], calibrated=True)
    pipeline.set_conformal_residuals(pl.DataFrame({"residual_80": [2.0]}))
    result = pipeline.run(
        _series([10, 12, 16, 24, 40]),
        champion_model=SeasonalNaiveForecaster(season_length=1, frequency="D"),
    )
    _assert_ordered_finite(result, [0.9, 0.75])
    assert result["forecast"].to_list() == [40.0, 40.0]
    assert result["forecast_p75"].to_list() == [44.0, 44.0]
    assert (result["forecast"] < result["forecast_p75"]).all()


@pytest.mark.parametrize("prebuilt", [False, True])
def test_pipeline_calibration_uses_the_same_models_residuals_on_reuse(tmp_path, prebuilt):
    pipeline = _pipeline(tmp_path, calibrated=True)
    pipeline.set_conformal_residuals(pl.DataFrame({
        "model_id": ["naive_seasonal", "unrelated_model"],
        "residual_80": [2.0, 100.0],
    }))
    champion = (
        SeasonalNaiveForecaster(season_length=2, frequency="D")
        if prebuilt else "naive_seasonal"
    )
    for history in ([1, 2, 6, 7], [8, 9, 3, 4]):
        result = pipeline.run(_series(history), champion_model=champion).sort("week")
        fresh = SeasonalNaiveForecaster(season_length=2 if prebuilt else 7, frequency="D")
        fresh.fit(_series(history))
        assert result["forecast"].to_list() == fresh.predict(2).sort("week")["forecast"].to_list()
        assert result["forecast_p50"].to_list() == result["forecast"].to_list()
        assert (result["forecast"] - result["forecast_p10"]).to_list() == [2.0, 2.0]
        assert (result["forecast_p90"] - result["forecast"]).to_list() == [2.0, 2.0]
        _assert_ordered_finite(result)


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("fault", ["crossed", "null", "infinite", "missing"])
def test_pipeline_rejects_malformed_quantiles_before_publication(tmp_path, fault, calibrated):
    class MalformedNaive(SeasonalNaiveForecaster):
        def predict_quantiles(self, *args, **kwargs):
            quantiles = super().predict_quantiles(*args, **kwargs)
            if fault == "missing":
                return quantiles.drop("forecast_p50")
            value = (
                pl.col("forecast_p90") + 1 if fault == "crossed"
                else pl.lit(None if fault == "null" else float("inf"))
            )
            return quantiles.with_columns(value.alias("forecast_p50"))

    pipeline = _pipeline(tmp_path, calibrated=calibrated)
    pipeline.set_conformal_residuals(pl.DataFrame({
        "model_id": ["naive_seasonal"], "residual_80": [20.0],
    }))
    with pytest.raises(ValueError, match="[Qq]uantile"):
        pipeline.run(
            _series([1, 2, 6, 7]),
            champion_model=MalformedNaive(season_length=2, frequency="D"),
        )
    assert pipeline._last_forecast_path is None
    assert not (tmp_path / "forecasts").exists()


def test_pipeline_rejects_crossing_calibration_without_replacing_prior_forecast(tmp_path):
    pipeline = _pipeline(tmp_path, calibrated=True)
    model = SeasonalNaiveForecaster(season_length=1, frequency="D")
    history = _series([10, 12, 16, 24, 40])
    pipeline.set_conformal_residuals(pl.DataFrame({"residual_80": [1.0]}))
    pipeline.run(history, champion_model=model)
    published_path = Path(pipeline._last_forecast_path)
    published_bytes = published_path.read_bytes()

    pipeline.set_conformal_residuals(pl.DataFrame({"residual_80": [-4.0]}))
    with pytest.raises(ValueError, match="ordered quantiles"):
        pipeline.run(history, champion_model=model)
    assert pipeline._last_forecast_path is None
    assert published_path.read_bytes() == published_bytes


@pytest.mark.parametrize("calibrated", [False, True])
def test_real_weekly_demo_naive_forecast_has_ordered_finite_quantiles(tmp_path, calibrated):
    """Same preparation/config/120-row inference path, without full daily M5 fitting."""
    csv = tmp_path / "m5_weekly.csv"
    weekly = prepare_weekly(
        ROOT / "tests" / "integration" / "fixtures" / "m5_daily_sample.csv",
        csv, n_series=10,
    )
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == (
        "d14250e3a55d3d00cea3418d2792579b30ecbd0608e4e6950204ca68b5aa60e4"
    )
    config_path = ROOT / "configs" / "m5_weekly_config.yaml"
    assert hashlib.sha256(config_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == (
        "fa6b002e04dc5684f64da5f25cd3a50d413918db03dc5f5587820899f2506b52"
    )
    config = load_config(config_path)
    assert not config.forecast.calibration.enabled
    config.output.forecast_path = str(tmp_path / "forecasts")
    config.output.metrics_path = str(tmp_path / "metrics")
    config.forecast.calibration.enabled = calibrated
    pipeline = ForecastPipeline(config)
    if calibrated:
        pipeline.set_conformal_residuals(pl.DataFrame({
            "model_id": ["naive_seasonal"], "residual_80": [2.0],
        }))
    forecast = pipeline.run(weekly, champion_model="naive_seasonal")
    assert forecast.height == 120
    assert forecast["series_id"].n_unique() == 10
    _assert_ordered_finite(forecast)
    expected_points = []
    for series in weekly.partition_by("series_id"):
        series = series.sort("date")
        for lead in range(12):
            expected_points.append({
                "series_id": series["series_id"][0],
                "date": series["date"][-1] + timedelta(weeks=lead + 1),
                "forecast": series["quantity"][-52 + lead],
            })
    assert forecast.select("series_id", "date", "forecast").sort(["series_id", "date"]).to_dicts() == (
        pl.DataFrame(expected_points).sort(["series_id", "date"]).to_dicts()
    )
    assert (forecast["forecast_p50"] == forecast["forecast"]).all()
    saved = pl.read_parquet(pipeline._last_forecast_path)
    _assert_ordered_finite(saved)
    assert saved.to_dicts() == forecast.to_dicts()
    assert forecast.filter(
        (pl.col("series_id") == "CA_1_FOODS_1_136")
        & (pl.col("date") == date(2016, 4, 25))
    )["forecast"].item() == 6.0
    assert forecast.filter(
        (pl.col("series_id") == "CA_1_FOODS_1_136")
        & (pl.col("date") == date(2016, 5, 2))
    )["forecast"].item() == 3.0
