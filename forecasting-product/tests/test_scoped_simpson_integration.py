"""Comparable-cohort Simpson diagnostics preserve undefined metric evidence."""

from datetime import date, timedelta

import polars as pl
import pytest
from src.config.schema import PostValidationConfig
from src.pipeline.validation_step import ValidationError, _run_simpsons_check, run_post_validation
from src.validation.confidence_scoring import score_confidence


def _records(actuals, predictions, scope="lob"):
    rows = []
    for model, series_predictions in predictions.items():
        for sid, values in actuals.items():
            if sid not in series_predictions:
                continue
            for step, (actual, forecast) in enumerate(zip(values, series_predictions[sid]), start=1):
                rows.append({
                    "run_id": "simpson-probe", "lob": "retail", "run_type": "backtest",
                    "model_id": model, "series_id": sid, "fold": 0, "grain_level": "series",
                    "channel": "", "target_week": date(2024, 1, 1) + timedelta(days=step),
                    "forecast_step": step, "actual": float(actual), "forecast": float(forecast),
                    "wmape": abs(forecast - actual) / abs(actual) if actual else None,
                    "competition_scope": scope,
                })
    return pl.DataFrame(rows, schema_overrides={"wmape": pl.Float64})


def _config():
    return PostValidationConfig(
        enabled=True, structural_checks=False, logical_checks=False,
        business_rules_checks=False, simpsons_paradox_checks=True,
        simpsons_segment_columns=["series_id"], halt_on_blocker=True,
    )


def test_disjoint_single_candidate_competitions_are_explicitly_not_applicable():
    records = _records(
        {"dense": [10, 10], "sparse": [0, 0]},
        {"naive_seasonal": {"dense": [11, 11]}, "croston": {"sparse": [3, 3]}},
        scope="series_id",
    )
    before = records.clone()
    result = _run_simpsons_check(records, _config())
    assert result["evaluated"] is False
    assert result["status"] == "not_applicable" and result["severity"] == "WARNING"
    assert result["comparisons_run"] == 0
    assert {item["group_key"] for item in result["unavailable"]} == {"dense", "sparse"}
    assert all(item["reason"] == "single_candidate" for item in result["unavailable"])
    factor = score_confidence(paradox=result)["factors"]["paradox"]
    assert factor["score"] == 50 and factor["unavailable"] is True
    assert factor["evaluated"] is False and factor["issues"]
    assert records.equals(before)


def test_all_zero_complete_comparison_arms_remain_undefined_not_pass():
    records = _records(
        {"a": [0, 0], "b": [0, 0]},
        {"model_a": {"a": [1, 1], "b": [2, 2]}, "model_b": {"a": [3, 3], "b": [4, 4]}},
    )
    result = _run_simpsons_check(records, _config())
    assert result["status"] == "not_evaluable" and result["evaluated"] is False
    assert result["severity"] == "WARNING" and result["any_paradox"] is False
    assert result["comparisons_run"] == 0
    assert result["unavailable"][0]["reason"] == "undefined_comparison_arm"
    assert all(arm["total_volume"] == 0 and arm["wmape"] is None for arm in result["unavailable"][0]["arms"])
    assert score_confidence(paradox=result)["factors"]["paradox"]["score"] == 50


def test_mixed_zero_positive_arms_use_pooled_raw_errors_before_diagnosis():
    records = _records(
        {"large": [0, 100], "small": [0, 10]},
        {"model_a": {"large": [10, 110], "small": [1, 11]},
         "model_b": {"large": [20, 120], "small": [2, 12]}},
    )
    assert records["wmape"].null_count() == 4
    result = _run_simpsons_check(records, _config())
    assert result["evaluated"] is True and result["status"] == "evaluated"
    assert result["severity"] == "PASS" and result["any_paradox"] is False
    comparison = result["comparisons"][0]["result"]
    assert comparison["aggregate_a"] == pytest.approx(0.2)
    assert comparison["aggregate_b"] == pytest.approx(0.4)
    assert score_confidence(paradox=result)["factors"]["paradox"]["score"] == 100


def test_genuine_comparable_paradox_preserves_blocking_policy():
    records = _records(
        {"large": [1000], "small_a": [10], "small_b": [10]},
        {"model_a": {"large": [1900], "small_a": [11], "small_b": [11]},
         "model_b": {"large": [1100], "small_a": [12], "small_b": [12]}},
    )
    result = _run_simpsons_check(records, _config())
    assert result["evaluated"] is True and result["any_paradox"] is True
    assert result["severity"] == "BLOCKER"
    with pytest.raises(ValidationError, match="BLOCKER"):
        run_post_validation(records, _config(), "retail")


@pytest.mark.parametrize("damage", ["missing", "null", "duplicate"])
def test_malformed_eligible_comparisons_remain_actionable_failures(damage):
    records = _records(
        {"a": [100, 100], "b": [100, 100]},
        {"model_a": {"a": [110, 110], "b": [110, 110]},
         "model_b": {"a": [120, 120], "b": [120, 120]}},
    )
    if damage == "missing":
        records = records.head(records.height - 1)
    elif damage == "null":
        records = records.with_columns(
            pl.when(pl.col("model_id") == "model_b").then(None).otherwise(pl.col("forecast")).alias("forecast")
        )
    else:
        records = pl.concat([records, records.head(1)])
    with pytest.raises(ValueError, match="coverage|Duplicate|finite|null|Null"):
        _run_simpsons_check(records, _config())


def test_unexpected_comparison_checker_errors_are_not_caught(monkeypatch):
    from src.pipeline import validation_step

    records = _records(
        {"a": [100, 100], "b": [100, 100]},
        {"model_a": {"a": [110, 110], "b": [110, 110]},
         "model_b": {"a": [120, 120], "b": [120, 120]}},
    )

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic checker defect")

    monkeypatch.setattr(validation_step, "check_simpsons_multi_segment", broken)
    with pytest.raises(RuntimeError, match="synthetic checker defect"):
        run_post_validation(records, _config(), "retail")


def test_unexpected_eligible_checker_error_still_prevents_run_commit(tmp_path, monkeypatch):
    from src.pipeline import validation_step
    from src.pipeline.backtest import BacktestPipeline
    from src.pipeline.run_store import RunStore

    from tests.test_run_scoring_integration import _actuals, _models
    from tests.test_run_scoring_integration import _config as run_config

    config = run_config(tmp_path)
    config.post_validation.enabled = True
    _models(monkeypatch)

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic eligible checker defect")

    monkeypatch.setattr(validation_step, "check_simpsons_multi_segment", broken)
    pipeline = BacktestPipeline(config)
    with pytest.raises(RuntimeError, match="synthetic eligible checker defect"):
        pipeline.run(_actuals())
    assert pipeline.last_run is None
    assert RunStore(config.output.runs_path).list() == []
