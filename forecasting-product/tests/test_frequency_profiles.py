"""
Tests for multi-frequency support infrastructure.

Validates that FREQUENCY_PROFILES, helper functions, and frequency-aware
config properties work correctly for all supported frequencies (D/W/M/Q).
"""

import unittest
from datetime import date, timedelta

import polars as pl

from src.config.schema import (
    FREQUENCY_PROFILES,
    BacktestConfig,
    DataQualityConfig,
    ForecastConfig,
    freq_timedelta,
    get_frequency_profile,
)
import pytest

pytestmark = pytest.mark.unit


class TestFrequencyProfiles(unittest.TestCase):
    """Tests for the FREQUENCY_PROFILES dict."""

    def test_all_four_frequencies_present(self):
        self.assertEqual(sorted(FREQUENCY_PROFILES.keys()), ["D", "M", "Q", "W"])

    def test_each_profile_has_required_keys(self):
        required = {
            "season_length", "secondary_season", "default_lags",
            "min_series_length", "min_ml_series_length", "default_val_periods",
            "default_horizon", "statsforecast_freq", "timedelta_kwargs",
        }
        for freq, profile in FREQUENCY_PROFILES.items():
            with self.subTest(freq=freq):
                self.assertTrue(
                    required.issubset(profile.keys()),
                    f"Missing keys for {freq}: {required - profile.keys()}"
                )

    def test_default_lags_are_sorted(self):
        for freq, profile in FREQUENCY_PROFILES.items():
            with self.subTest(freq=freq):
                lags = profile["default_lags"]
                self.assertEqual(lags, sorted(lags), f"Lags not sorted for {freq}")


@pytest.mark.parametrize("freq,expected", [("D", 7), ("W", 52), ("M", 12), ("Q", 4)])
def test_season_lengths(freq, expected):
    assert FREQUENCY_PROFILES[freq]["season_length"] == expected


@pytest.mark.parametrize("freq,expected", [("D", "D"), ("W", "W"), ("M", "MS"), ("Q", "QS")])
def test_statsforecast_freq_strings(freq, expected):
    assert FREQUENCY_PROFILES[freq]["statsforecast_freq"] == expected


class TestGetFrequencyProfile(unittest.TestCase):
    """Tests for the get_frequency_profile() helper."""

    def test_valid_frequencies(self):
        for freq in ("D", "W", "M", "Q"):
            profile = get_frequency_profile(freq)
            self.assertIsInstance(profile, dict)
            self.assertIn("season_length", profile)

    def test_invalid_frequency_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_frequency_profile("Y")
        self.assertIn("Unsupported frequency", str(ctx.exception))
        self.assertIn("'Y'", str(ctx.exception))


@pytest.mark.parametrize("freq,periods,expected", [
    ("D", 1, timedelta(days=1)),
    ("D", 7, timedelta(days=7)),
    ("W", 1, timedelta(weeks=1)),
    ("W", 4, timedelta(weeks=4)),
])
def test_freq_timedelta(freq, periods, expected):
    assert freq_timedelta(freq, periods) == expected


@pytest.mark.parametrize("freq", ["M", "Q"])
def test_calendar_frequencies_are_not_fixed_durations(freq):
    with pytest.raises(ValueError, match="calendar"):
        freq_timedelta(freq)
    assert FREQUENCY_PROFILES[freq]["timedelta_kwargs"] is None


class TestFreqTimedeltaArithmetic(unittest.TestCase):
    """Tests for freq_timedelta() date arithmetic."""

    def test_date_arithmetic(self):
        """freq_timedelta should work with date addition."""
        d = date(2024, 1, 1)
        self.assertEqual(d + freq_timedelta("W", 1), date(2024, 1, 8))
        self.assertEqual(d + freq_timedelta("D", 1), date(2024, 1, 2))


@pytest.mark.parametrize("freq,expected_season,expected_sf_freq", [
    ("W", 52, "W"),
    ("D", 7, "D"),
    ("M", 12, "MS"),
    ("Q", 4, "QS"),
])
def test_frequency_config(freq, expected_season, expected_sf_freq):
    fc = ForecastConfig(frequency=freq)
    assert fc.season_length == expected_season
    assert fc.statsforecast_freq == expected_sf_freq


class TestForecastConfigProperties(unittest.TestCase):
    """Tests for frequency-derived ForecastConfig properties."""

    def test_weekly_defaults(self):
        fc = ForecastConfig()
        self.assertEqual(fc.frequency, "W")
        self.assertEqual(fc.horizon_periods, 39)

    def test_daily_lags(self):
        fc = ForecastConfig(frequency="D")
        self.assertEqual(fc.default_lags, [1, 2, 3, 7, 14, 21, 28, 56, 91, 182, 364])

    def test_monthly_lags(self):
        fc = ForecastConfig(frequency="M")
        self.assertEqual(fc.default_lags, [1, 2, 3, 6, 12])


class TestBacktestConfigAliases(unittest.TestCase):
    """Tests for BacktestConfig period aliases."""

    def test_val_periods_alias(self):
        bt = BacktestConfig(val_weeks=8)
        self.assertEqual(bt.val_periods, 8)

    def test_gap_periods_alias(self):
        bt = BacktestConfig(gap_weeks=2)
        self.assertEqual(bt.gap_periods, 2)


class TestDataQualityConfigAlias(unittest.TestCase):
    """Tests for DataQualityConfig min_series_length alias."""

    def test_min_series_length_alias(self):
        dq = DataQualityConfig(min_series_length_weeks=24)
        self.assertEqual(dq.min_series_length, 24)


class TestNaiveForecasterFrequency(unittest.TestCase):
    """Tests that SeasonalNaiveForecaster respects frequency parameter."""

    def test_monthly_season_length(self):
        from src.forecasting.naive import SeasonalNaiveForecaster
        f = SeasonalNaiveForecaster(frequency="M")
        self.assertEqual(f.season_length, 12)
        self.assertEqual(f.frequency, "M")

    def test_weekly_default_unchanged(self):
        from src.forecasting.naive import SeasonalNaiveForecaster
        f = SeasonalNaiveForecaster()
        self.assertEqual(f.season_length, 52)
        self.assertEqual(f.frequency, "W")

    def test_explicit_season_length_preserved(self):
        from src.forecasting.naive import SeasonalNaiveForecaster
        f = SeasonalNaiveForecaster(season_length=26, frequency="M")
        self.assertEqual(f.season_length, 26)

    def test_monthly_predict_dates(self):
        """Real month starts must remain month starts across leap February."""
        from src.forecasting.naive import SeasonalNaiveForecaster
        f = SeasonalNaiveForecaster(frequency="M")
        df = pl.DataFrame({
            "series_id": ["A"] * 24,
            "week": [date(2022 + i // 12, i % 12 + 1, 1) for i in range(24)],
            "quantity": [float(i % 12 + 1) for i in range(24)],
        })
        f.fit(df)
        preds = f.predict(3)
        dates = preds["week"].sort().to_list()
        self.assertEqual(dates, [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)])


class TestRegistryFrequencyFiltering(unittest.TestCase):
    """Frequency is part of every registered forecasting model's contract."""

    def test_build_with_frequency(self):
        from src.forecasting.registry import registry
        f = registry.build("naive_seasonal", frequency="M")
        self.assertEqual(f.frequency, "M")

    def test_build_preserves_intermittent_frequency(self):
        from src.forecasting.registry import registry
        f = registry.build("croston", frequency="M")
        self.assertEqual(f.name, "croston")
        self.assertEqual(f.frequency, "M")


if __name__ == "__main__":
    unittest.main()
