"""Fast exact reference checks; never runs a smaller substitute M5 benchmark."""

import csv
import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from tests.integration.m5_reference import (
    candidate_provenance,
    canonical_text_sha256,
    daily_naive_reference,
    exact_totals,
    read_daily_panel,
    score_complete_predictions,
)

ROOT = Path(__file__).resolve().parent / "integration"
REFERENCE = ROOT / "baselines" / "m5_daily_reference.json"
FIXTURE = ROOT / "fixtures" / "m5_daily_sample.csv"
CONFIG = ROOT / "fixtures" / "m5_daily_config.yaml"


def _toy_points():
    panel = {"s": [(date(2024, 1, i), Decimal(i)) for i in range(1, 7)]}
    return daily_naive_reference(panel, n_folds=2, val_periods=2, season_length=2)


def _toy_records():
    return [{
        "model_id": "naive_seasonal", "run_id": "toy",
        "fold": point.fold, "series_id": point.series_id,
        "target_week": point.target_date, "forecast_step": point.forecast_step,
        "actual": float(point.actual), "forecast": float(point.naive_forecast),
    } for point in _toy_points()]


def test_reference_algorithm_has_a_hand_computed_oracle():
    points = _toy_points()
    assert [point.naive_forecast for point in points] == [1, 2, 3, 4]
    assert [point.actual for point in points] == [3, 4, 5, 6]
    assert exact_totals(points) == {
        "observations": 4, "absolute_error": "8", "absolute_actual": "18", "signed_error": "-8",
    }
    scores = score_complete_predictions(_toy_records(), points, {"naive_seasonal"})
    assert scores["naive_seasonal"] == pytest.approx({
        "wmape": 4 / 9, "normalized_bias": -4 / 9, "mae": 2,
    })


def test_checked_in_m5_reference_is_non_skippable_and_exact():
    snapshot = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert canonical_text_sha256(FIXTURE) == snapshot["fixture_sha256"]
    assert canonical_text_sha256(CONFIG) == snapshot["config_sha256"]
    panel = read_daily_panel(FIXTURE)
    assert len(panel) == snapshot["series_count"] == 105
    assert sum(map(len, panel.values())) == snapshot["input_rows"] == 76755
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["forecast"]["frequency"] == "D"
    assert config["forecast"]["forecasters"] == snapshot["models"]
    assert config["backtest"]["n_folds"] == snapshot["n_folds"] == 2
    assert config["backtest"]["val_weeks"] == snapshot["val_periods"] == 28
    assert config["backtest"]["gap_weeks"] == snapshot["gap_periods"] == 0
    points = daily_naive_reference(panel)
    assert exact_totals(points) == snapshot["totals"]
    for expected in snapshot["folds"]:
        fold = [point for point in points if point.fold == expected["fold"]]
        observed = {
            "fold": expected["fold"],
            "first_target": min(point.target_date for point in fold).isoformat(),
            "last_target": max(point.target_date for point in fold).isoformat(),
            **exact_totals(fold),
        }
        assert observed == expected


def test_observed_runtime_metadata_is_not_automatically_an_approved_baseline(monkeypatch):
    for name in ("M5_PROFILE_LOCK", "M5_PROFILE_INSTALL_REPORT", "M5_PROFILE_QUALIFICATION"):
        monkeypatch.delenv(name, raising=False)
    provenance = candidate_provenance(FIXTURE, CONFIG)
    assert provenance["approved_baseline"] is False
    assert provenance["profile_lock_sha256"] is None
    assert provenance["packages"]["polars"]
    assert provenance["packages"]["numpy"]
    assert provenance["fixture_utf8_lf_sha256"] == canonical_text_sha256(FIXTURE)
    assert provenance["config_utf8_lf_sha256"] == canonical_text_sha256(CONFIG)
    json.dumps(provenance, allow_nan=False)


def test_full_fixture_naive_component_matches_exact_reference(tmp_path):
    """Validate this deterministic component; the full M5 fixture still fits all three models."""
    import polars as pl
    from src.backtesting.engine import BacktestEngine
    from src.config.schema import BacktestConfig, ForecastConfig, OutputConfig, PlatformConfig
    from src.forecasting.naive import SeasonalNaiveForecaster
    from src.metrics.aggregation import aggregate_metrics

    config = PlatformConfig(
        lob="m5-reference",
        forecast=ForecastConfig(frequency="D", time_column="date"),
        backtest=BacktestConfig(n_folds=2, val_weeks=28),
        output=OutputConfig(metrics_path=str(tmp_path / "metrics")),
    )
    observations = pl.read_csv(FIXTURE, try_parse_dates=True)
    records = BacktestEngine(config).run(
        observations, [SeasonalNaiveForecaster(frequency="D")],
        run_id="m5-reference", persist=False,
    )
    reference = daily_naive_reference(read_daily_panel(FIXTURE))
    expected = score_complete_predictions(records.iter_rows(named=True), reference, {"naive_seasonal"})
    assert records.height == 5880
    assert records["series_id"].n_unique() == 105
    board = aggregate_metrics(records, ["wmape", "normalized_bias"])
    assert expected["naive_seasonal"]["wmape"] == pytest.approx(7921 / 7945, rel=1e-14, abs=1e-14)
    assert expected["naive_seasonal"]["normalized_bias"] == pytest.approx(479 / 7945, rel=1e-14, abs=1e-14)
    assert board["wmape"].item() == pytest.approx(7921 / 7945, rel=1e-14, abs=1e-14)
    assert board["normalized_bias"].item() == pytest.approx(479 / 7945, rel=1e-14, abs=1e-14)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong_date", "wrong_lead", "changed_actual", "wrong_naive", "nonfinite", "mixed_run"])
def test_reference_never_hides_bad_observations(defect):
    rows = _toy_records()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(dict(rows[0]))
    elif defect == "wrong_date":
        rows[0]["target_week"] += timedelta(days=1)
    elif defect == "wrong_lead":
        rows[0]["forecast_step"] += 1
    elif defect == "changed_actual":
        rows[0]["actual"] += 1
    elif defect == "wrong_naive":
        rows[0]["forecast"] += 1
    elif defect == "nonfinite":
        rows[0]["forecast"] = float("nan")
    else:
        rows[0]["run_id"] = "other"
    with pytest.raises(ValueError):
        score_complete_predictions(rows, _toy_points(), {"naive_seasonal"})


@pytest.mark.parametrize("defect", ["missing", "duplicate"])
def test_reference_input_grid_cannot_silently_change(tmp_path, defect):
    path = tmp_path / "panel.csv"
    dates = [date(2024, 1, i) for i in range(1, 7)]
    if defect == "missing":
        dates.pop(2)
    else:
        dates.append(dates[2])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["series_id", "date", "quantity"])
        writer.writerows(("s", label.isoformat(), 1) for label in dates)
    with pytest.raises(ValueError, match="duplicate or missing"):
        read_daily_panel(path)
