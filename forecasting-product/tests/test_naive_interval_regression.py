"""Causal interval regressions: preserve point forecasts, order labeled quantiles."""

import hashlib
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from scripts.prepare_m5_weekly import prepare_weekly
from src.config.loader import load_config
from src.forecasting.naive import SeasonalNaiveForecaster
from src.pipeline.forecast import ForecastPipeline


ROOT = Path(__file__).resolve().parents[1]
QUANTILES = [0.1, 0.5, 0.9]


def _series(values):
    return pl.DataFrame({
        "series_id": ["s"] * len(values),
        "week": [date(2024, 1, 1) + timedelta(days=index) for index in range(len(values))],
        "quantity": [float(value) for value in values],
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
    assert model.predict(2).sort("week").to_dicts() == before.to_dicts()


def test_naive_quantiles_after_refit_match_a_fresh_model():
    model = SeasonalNaiveForecaster(season_length=2, frequency="D")
    model.fit(_series([1, 2, 6, 7]))
    model.predict_quantiles(4, QUANTILES)
    history = _series([8, 9, 3, 4])
    model.fit(history)
    fresh = SeasonalNaiveForecaster(season_length=2, frequency="D")
    fresh.fit(history)
    observed = model.predict_quantiles(4, QUANTILES).sort("week")
    assert observed.to_dicts() == fresh.predict_quantiles(4, QUANTILES).sort("week").to_dicts()
    _assert_ordered_finite(observed)


def test_real_weekly_demo_naive_forecast_has_ordered_finite_quantiles(tmp_path):
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
    forecast = ForecastPipeline(config).run(weekly, champion_model="naive_seasonal")
    assert forecast.height == 120
    assert forecast["series_id"].n_unique() == 10
    _assert_ordered_finite(forecast)
    assert forecast.filter(
        (pl.col("series_id") == "CA_1_FOODS_1_136")
        & (pl.col("date") == date(2016, 4, 25))
    )["forecast"].item() == 6.0
    assert forecast.filter(
        (pl.col("series_id") == "CA_1_FOODS_1_136")
        & (pl.col("date") == date(2016, 5, 2))
    )["forecast"].item() == 3.0
