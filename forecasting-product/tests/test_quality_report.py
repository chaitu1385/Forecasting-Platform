"""Tests for the pre-training data quality report."""

from datetime import date, timedelta

import polars as pl
import pytest
from src.config.schema import (
    DataQualityConfig,
    DataQualityReportConfig,
    ForecastConfig,
    PlatformConfig,
)
from src.data.quality_report import DataQualityAnalyzer, DataQualityReport
from src.pipeline.backtest import BacktestPipeline
from src.pipeline.forecast import ForecastPipeline

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _make_config(
    min_weeks: int = 52,
    report_enabled: bool = True,
    sparse_classification: bool = True,
    include_series_detail: bool = True,
) -> PlatformConfig:
    return PlatformConfig(
        forecast=ForecastConfig(
            target_column="quantity",
            time_column="week",
            series_id_column="series_id",
        ),
        data_quality=DataQualityConfig(
            min_series_length_weeks=min_weeks,
            report=DataQualityReportConfig(
                enabled=report_enabled,
                include_series_detail=include_series_detail,
                sparse_classification=sparse_classification,
            ),
        ),
    )


def _make_complete_series(
    n_series: int = 3, n_weeks: int = 104, base_demand: float = 100.0
) -> pl.DataFrame:
    """No gaps, no zeros, steady demand."""
    import random
    rng = random.Random(42)
    rows = []
    start = date(2022, 1, 3)
    for s in range(n_series):
        for w in range(n_weeks):
            rows.append({
                "series_id": f"S{s:03d}",
                "week": start + timedelta(weeks=w),
                "quantity": base_demand + rng.uniform(-5, 5),
            })
    return pl.DataFrame(rows)


def _make_gappy_series(
    n_series: int = 3, n_weeks: int = 104, gap_pct: float = 0.2
) -> pl.DataFrame:
    """Series with some weeks randomly missing."""
    import random
    rng = random.Random(99)
    rows = []
    start = date(2022, 1, 3)
    for s in range(n_series):
        for w in range(n_weeks):
            if rng.random() < gap_pct:
                continue  # skip this week
            rows.append({
                "series_id": f"S{s:03d}",
                "week": start + timedelta(weeks=w),
                "quantity": 50.0 + rng.uniform(-3, 3),
            })
    return pl.DataFrame(rows)


def _make_sparse_series(n_series: int = 3, n_weeks: int = 104) -> pl.DataFrame:
    """Intermittent demand — many zeros with sporadic sales."""
    import random
    rng = random.Random(7)
    rows = []
    start = date(2022, 1, 3)
    for s in range(n_series):
        for w in range(n_weeks):
            # ~80% zeros
            val = 0.0 if rng.random() < 0.80 else rng.uniform(10, 200)
            rows.append({
                "series_id": f"S{s:03d}",
                "week": start + timedelta(weeks=w),
                "quantity": val,
            })
    return pl.DataFrame(rows)


def _make_zero_series(n_weeks: int = 104) -> pl.DataFrame:
    """One series that is all zeros."""
    start = date(2022, 1, 3)
    return pl.DataFrame({
        "series_id": ["ZERO"] * n_weeks,
        "week": [start + timedelta(weeks=w) for w in range(n_weeks)],
        "quantity": [0.0] * n_weeks,
    })


@pytest.mark.parametrize(("frequency", "labels"), [
    ("D", ["2024-02-27", "2024-02-28", "2024-02-29", "2024-03-01"]),
    ("W", ["2024-02-26", "2024-03-04", "2024-03-11", "2024-03-18"]),
    ("M", ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
    ("M", ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"]),
    ("Q", ["2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01"]),
    ("Q", ["2023-10-31", "2024-01-31", "2024-04-30", "2024-07-31"]),
])
def test_completeness_uses_configured_calendar_and_unique_cells(frequency, labels):
    dates = [date.fromisoformat(label) for label in labels]
    panel = pl.DataFrame({
        "series_id": ["A"] * 4 + ["B"] * 4,
        "week": dates + dates,
        "quantity": [10.0] * 8,
    })
    config = _make_config(min_weeks=1)
    config.forecast.frequency = frequency
    analyzer = DataQualityAnalyzer(config)
    gapped = panel.filter(~((pl.col("series_id") == "B") & (pl.col("week") == dates[1])))
    for frame, missing in ((panel, 0), (gapped, 1), (pl.concat([gapped, panel.head(1)]), 1)):
        before = frame.clone()
        report = analyzer.analyze(frame, "week", "quantity", "series_id")
        assert report.total_rows == frame.height
        assert report.total_weeks == 4
        assert report.missing_week_pct == pytest.approx(missing / 8 * 100)
        assert report.series_with_gaps == missing
        assert report.per_series is not None
        rows = {row["series_id"]: row for row in report.per_series.to_dicts()}
        assert rows["A"]["n_weeks"] == 4
        assert rows["A"]["missing_pct"] == 0
        assert rows["B"]["n_weeks"] == 4 - missing
        assert rows["B"]["missing_pct"] == pytest.approx(missing / 4 * 100)
        assert frame.equals(before)


def _daily_panel() -> pl.DataFrame:
    return pl.DataFrame([
        {
            "series_id": f"DAILY_{'A' if series == 0 else 'B'}",
            "date": date(2025, 1, 1) + timedelta(days=index),
            "quantity": 0.0 if index in (160, 190) else float(
                100 + 2 * index + 5 * (index % 4) if series == 0
                else 40 + index + 2 * (index % 4)
            ),
        }
        for series in range(2) for index in range(208)
    ])


def test_daily_complete_grid_and_one_missing_cell_have_exact_gap_rates():
    config = _make_config(min_weeks=1)
    config.forecast.frequency = "D"
    panel = _daily_panel()
    analyzer = DataQualityAnalyzer(config)
    complete: DataQualityReport = analyzer.analyze(panel, "date", "quantity", "series_id")
    assert complete.total_weeks == 208
    assert complete.total_rows == 416
    assert complete.missing_week_pct == 0
    assert complete.series_with_gaps == 0
    assert complete.per_series is not None
    assert complete.per_series.schema["n_weeks"] == pl.Int64
    assert complete.zero_inflation_rate == pytest.approx(4 / 416 * 100)
    assert not any("gap rate" in warning for warning in complete.warnings)
    gapped = analyzer.analyze(panel.slice(1), "date", "quantity", "series_id")
    assert gapped.total_weeks == 208
    assert gapped.missing_week_pct == pytest.approx(1 / 416 * 100)
    assert gapped.series_with_gaps == 1
    assert gapped.per_series is not None
    detail = gapped.per_series.filter(pl.col("series_id") == "DAILY_A").to_dicts()[0]
    assert detail["missing_pct"] == pytest.approx(1 / 208 * 100)


def test_quality_warning_units_are_periods():
    config = _make_config(min_weeks=3)
    config.forecast.frequency = "D"
    frame = pl.DataFrame({
        "series_id": ["A"] * 4 + ["B"],
        "date": [date(2024, 1, day) for day in (1, 2, 3, 4, 1)],
        "quantity": [10.0] * 5,
    })
    report = DataQualityAnalyzer(config).analyze(frame, "date", "quantity", "series_id")
    assert report.missing_week_pct == 37.5
    assert report.short_series_count == 1
    assert any("period cells" in warning for warning in report.warnings)
    assert any("3 periods" in warning for warning in report.warnings)
    assert not any("weeks" in warning for warning in report.warnings)


def test_off_grid_quality_input_is_not_reported_as_healthy():
    config = _make_config(min_weeks=1)
    frame = pl.DataFrame({
        "series_id": ["A"] * 3,
        "week": [date(2024, 1, day) for day in (1, 2, 8)],
        "quantity": [10.0] * 3,
    })
    with pytest.raises(ValueError, match="calendar"):
        DataQualityAnalyzer(config).analyze(frame, "week", "quantity", "series_id")


def test_daily_pipeline_report_does_not_change_forecasts_or_backtest_values(tmp_path, caplog):
    config = _make_config(min_weeks=1)
    config.forecast.frequency = "D"
    config.forecast.time_column = "date"
    config.forecast.horizon_weeks = 51
    config.backtest.n_folds = 2
    config.backtest.val_weeks = 51
    config.backtest.champion_granularity = "series"
    config.parallelism.n_workers = 1
    config.output.runs_path = str(tmp_path / "runs")
    config.output.forecast_path = str(tmp_path / "forecasts")
    config.output.metrics_path = str(tmp_path / "metrics")
    panel = _daily_panel()
    pipeline = ForecastPipeline(config)
    forecast = pipeline.run(panel)
    report = pipeline._series_builder._last_quality_report
    assert report is not None
    assert report.total_weeks == 208
    assert report.missing_week_pct == 0
    assert report.series_with_gaps == 0
    assert forecast.height == 102
    assert forecast["forecast"].sum() == 38946
    assert forecast["date"].min() == date(2025, 7, 28)
    assert forecast["date"].max() == date(2025, 9, 16)
    backtest = BacktestPipeline(config).run(panel)
    assert backtest["data_quality_report"].missing_week_pct == 0
    records = backtest["backtest_results"]
    assert records.height == 204
    assert (records["forecast"] - records["actual"]).abs().sum() == 10024
    assert records["actual"].abs().sum() == 61910
    assert not any("High gap rate" in record.message for record in caplog.records)
    config.data_quality.report.enabled = False
    assert ForecastPipeline(config).run(panel).equals(forecast)
    control = BacktestPipeline(config).run(panel)["backtest_results"]
    columns = ["series_id", "target_week", "model_id", "fold", "forecast_step", "actual", "forecast", "wmape", "normalized_bias", "mase_scale"]
    assert control.select(columns).equals(records.select(columns))


# --------------------------------------------------------------------------- #
#  TestBasicStats
# --------------------------------------------------------------------------- #

class TestBasicStats:
    def test_row_and_series_counts(self):
        df = _make_complete_series(n_series=3, n_weeks=104)
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert report.total_rows == 3 * 104
        assert report.total_series == 3

    def test_date_range(self):
        df = _make_complete_series(n_series=1, n_weeks=52)
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert report.date_range_start == date(2022, 1, 3)
        assert report.date_range_end == date(2022, 1, 3) + timedelta(weeks=51)
        assert report.total_weeks == 52

    def test_empty_dataframe(self):
        df = pl.DataFrame({
            "series_id": pl.Series([], dtype=pl.Utf8),
            "week": pl.Series([], dtype=pl.Date),
            "quantity": pl.Series([], dtype=pl.Float64),
        })
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert report.total_rows == 0
        assert "Empty dataset" in report.warnings

    def test_single_series(self):
        df = _make_complete_series(n_series=1, n_weeks=104)
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert report.total_series == 1
        assert report.total_rows == 104


# --------------------------------------------------------------------------- #
#  TestCompleteness
# --------------------------------------------------------------------------- #

class TestCompleteness:
    def test_no_gaps_zero_missing_pct(self):
        df = _make_complete_series(n_series=2, n_weeks=52)
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert report.missing_week_pct == pytest.approx(0.0, abs=0.1)
        assert report.series_with_gaps == 0

    def test_gaps_detected(self):
        df = _make_gappy_series(n_series=3, n_weeks=104, gap_pct=0.2)
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert report.missing_week_pct > 0
        assert report.series_with_gaps > 0

    def test_short_series_counted(self):
        # 2 full series + 1 short series (only 10 weeks)
        full = _make_complete_series(n_series=2, n_weeks=104)
        start = date(2022, 1, 3)
        short = pl.DataFrame({
            "series_id": ["SHORT"] * 10,
            "week": [start + timedelta(weeks=w) for w in range(10)],
            "quantity": [50.0] * 10,
        })
        df = pl.concat([full, short])
        analyzer = DataQualityAnalyzer(_make_config(min_weeks=52))
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert report.short_series_count == 1

    def test_zero_series_counted(self):
        normal = _make_complete_series(n_series=2, n_weeks=104)
        zeros = _make_zero_series(n_weeks=104)
        df = pl.concat([normal, zeros])
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert report.zero_series_count == 1


# --------------------------------------------------------------------------- #
#  TestValueDistribution
# --------------------------------------------------------------------------- #

class TestValueDistribution:
    def test_zero_inflation_rate(self):
        # Mix: 50% zeros
        start = date(2022, 1, 3)
        rows = []
        for w in range(100):
            rows.append({
                "series_id": "A",
                "week": start + timedelta(weeks=w),
                "quantity": 0.0 if w < 50 else 100.0,
            })
        df = pl.DataFrame(rows)
        analyzer = DataQualityAnalyzer(_make_config(min_weeks=10))
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert report.zero_inflation_rate == pytest.approx(50.0, abs=0.1)

    def test_outlier_count_from_cleansing(self):
        """When a CleansingReport is provided, pull outlier counts from it."""
        from dataclasses import dataclass

        @dataclass
        class FakeCleansing:
            total_outliers: int = 5
            outlier_pct: float = 2.5

        df = _make_complete_series(n_series=1, n_weeks=52)
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(
            df, "week", "quantity", "series_id",
            cleansing_report=FakeCleansing(),
        )
        assert report.outlier_count == 5
        assert report.outlier_pct == pytest.approx(2.5)

    def test_outlier_count_standalone(self):
        """When no cleansing report, run IQR detection."""
        # Inject extreme outliers
        df = _make_complete_series(n_series=1, n_weeks=52, base_demand=100.0)
        # Add a few extreme values
        extreme = pl.DataFrame({
            "series_id": ["S000"] * 3,
            "week": [date(2024, 1, 1) + timedelta(weeks=w) for w in range(3)],
            "quantity": [10000.0, -5000.0, 9000.0],
        })
        df = pl.concat([df, extreme])
        analyzer = DataQualityAnalyzer(_make_config(min_weeks=10))
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert report.outlier_count >= 3


# --------------------------------------------------------------------------- #
#  TestDemandClassification
# --------------------------------------------------------------------------- #

class TestDemandClassification:
    def test_smooth_series_classified(self):
        df = _make_complete_series(n_series=2, n_weeks=104, base_demand=100.0)
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert "smooth" in report.demand_classes
        assert report.demand_classes["smooth"] == 2

    def test_sparse_series_classified(self):
        df = _make_sparse_series(n_series=3, n_weeks=104)
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        # Sparse series should be classified as intermittent or lumpy
        sparse_count = report.demand_classes.get("intermittent", 0) + \
                       report.demand_classes.get("lumpy", 0)
        assert sparse_count > 0


# --------------------------------------------------------------------------- #
#  TestWarnings
# --------------------------------------------------------------------------- #

class TestWarnings:
    def test_high_gap_warning(self):
        df = _make_gappy_series(n_series=3, n_weeks=104, gap_pct=0.4)
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert any("gap rate" in w.lower() for w in report.warnings)

    def test_high_zero_inflation_warning(self):
        df = _make_sparse_series(n_series=3, n_weeks=104)
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert any("zero-inflation" in w.lower() for w in report.warnings)

    def test_no_warnings_clean_data(self):
        df = _make_complete_series(n_series=3, n_weeks=104, base_demand=100.0)
        analyzer = DataQualityAnalyzer(_make_config())
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert len(report.warnings) == 0


# --------------------------------------------------------------------------- #
#  TestPerSeriesDetail
# --------------------------------------------------------------------------- #

class TestPerSeriesDetail:
    def test_per_series_dataframe_present(self):
        df = _make_complete_series(n_series=3, n_weeks=104)
        analyzer = DataQualityAnalyzer(_make_config(include_series_detail=True))
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert isinstance(report.per_series, pl.DataFrame)
        assert report.per_series.height == 3
        assert "n_weeks" in report.per_series.columns
        assert "mean" in report.per_series.columns
        assert "cv" in report.per_series.columns

    def test_per_series_disabled(self):
        df = _make_complete_series(n_series=2, n_weeks=52)
        analyzer = DataQualityAnalyzer(_make_config(include_series_detail=False))
        report = analyzer.analyze(df, "week", "quantity", "series_id")
        assert report.per_series is None


# --------------------------------------------------------------------------- #
#  TestSeriesBuilderIntegration
# --------------------------------------------------------------------------- #

class TestSeriesBuilderIntegration:
    def test_report_stored_on_builder(self):
        from src.series.builder import SeriesBuilder

        config = _make_config(min_weeks=10)
        builder = SeriesBuilder(config)
        df = _make_complete_series(n_series=2, n_weeks=52)
        builder.build(actuals=df)
        from src.data.quality_report import DataQualityReport
        assert isinstance(builder._last_quality_report, DataQualityReport)
        assert builder._last_quality_report.total_series == 2
