"""Authenticated, source-bound contracts for moving-average decomposition."""

import hashlib
import io
import math
import secrets
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import jwt
import polars as pl
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.auth.token import TOKEN_AUDIENCE, TOKEN_ISSUER, create_token
from src.config.schema import PlatformConfig
from src.pipeline.manifest import PipelineManifest
from src.pipeline.run_store import RunStore, new_run_id

HISTORY_VALUES = [97, 110, 123, 127, 140, 153, 157, 170, 183, 187, 200, 213]
FORECAST_VALUES = [209, 217, 230]
ROUTES = ("decomposition", "shap")


def _history() -> pl.DataFrame:
    return pl.DataFrame({
        "series_id": ["hand-example"] * 12,
        "week": [date(2026, 1, 5) + timedelta(weeks=i) for i in range(12)],
        "quantity": [float(value) for value in HISTORY_VALUES],
    })


def _forecast() -> pl.DataFrame:
    return pl.DataFrame({
        "series_id": ["hand-example"] * 3,
        "week": [date(2026, 1, 5) + timedelta(weeks=i) for i in range(12, 15)],
        "forecast": [float(value) for value in FORECAST_VALUES],
        "model": ["recorded-seasonal"] * 3,
    })


@pytest.fixture
def api(tmp_path):
    key = secrets.token_urlsafe(48)
    roots = {
        "data_dir": str(tmp_path / "data"),
        "metrics_dir": str(tmp_path / "metrics"),
        "runs_dir": str(tmp_path / "runs"),
    }
    headers = {
        "Authorization": f"Bearer {create_token('synthetic-viewer', '', 'viewer', key, expiry_hours=1)}",
    }
    app = create_app(**roots, auth_enabled=True, dev_mode=False, jwt_secret=key)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, headers, RunStore(roots["runs_dir"]), roots, key


def _publish(
    store, *, history=None, forecast=None, raw=None, model="recorded-seasonal",
    started="2026-05-01T00:00:00.000000+00:00", config=None,
):
    history = _history() if history is None else history
    forecast = _forecast() if forecast is None else forecast
    raw = history.with_columns(pl.col("quantity") + 1000) if raw is None else raw
    config = PlatformConfig(lob="retail") if config is None else config
    config.forecast.horizon_weeks = 3
    config.post_validation.enabled = False
    manifest = PipelineManifest(
        run_id=new_run_id(), lob="retail", run_type="forecast",
        started_at=started, timestamp=started, champion_model_id=model,
        models=[{"name": model, "parameters": {"season_length": 3}}],
        input_row_count=raw.height, input_series_count=1,
        forecast_row_count=forecast.height, forecast_horizon=3,
    )
    store.publish(
        manifest, config, {"input": raw, "prepared": history, "forecast": forecast},
        {"skipped": True},
    )
    return manifest.run_id


def _post(api, route="decomposition", **params):
    client, headers, *_ = api
    return client.post(
        f"/metrics/retail/{route}", headers=headers,
        params={"season_length": 3, **params},
    )


def _hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(root).rglob("*") if path.is_file()
    }


def test_canonical_reports_actual_method_and_recorded_model(api):
    run_id = _publish(api[2])
    response = _post(api, run_id=run_id, model_name="invented-fitted-model")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["method"] == "moving_average_decomposition"
    assert body["model_specific"] is False
    assert body["forecast_model"] == "recorded-seasonal"
    assert body["run_id"] == run_id
    assert body["artifact_source"] == "committed"


def test_alias_does_not_invent_requested_model_provenance(api):
    run_id = _publish(api[2])
    response = _post(api, "shap", run_id=run_id, model_name="invented-fitted-model")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "recorded-seasonal"
    assert body["method"] == "moving_average_decomposition"
    assert body["model_specific"] is False


def test_alias_uses_prepared_history_not_raw_input(api):
    run_id = _publish(api[2])
    before = _hashes(api[2].root)
    response = _post(api, "shap", run_id=run_id)
    assert response.status_code == 200, response.text
    body = response.json()
    historical = [row for row in body["decomposition_preview"] if not row["is_forecast"]]
    assert [row["value"] for row in historical] == HISTORY_VALUES
    assert body["history_source"] == "committed"
    assert body["history_stage"] == "prepared"
    assert body["history_run_id"] == run_id
    assert _hashes(api[2].root) == before


def test_original_alias_positive_control_reaches_real_decomposition(api):
    run_id = _publish(api[2], raw=_history())
    response = _post(api, "shap", run_id=run_id)
    assert response.status_code == 200, response.text
    rows = response.json()["decomposition_preview"]
    future = [row for row in rows if row["is_forecast"]]
    assert [row["trend"] for row in future] == pytest.approx([205, 215, 225])
    assert [row["seasonal"] for row in future] == pytest.approx([-3, 0, 3])
    assert [row["residual"] for row in future] == pytest.approx([7, 2, 2])


@pytest.mark.parametrize("route", ROUTES)
def test_hand_calculated_components_counts_and_nullable_edges(api, route):
    run_id = _publish(api[2])
    body = _post(api, route, run_id=run_id).json()
    assert body["method"] == "moving_average_decomposition"
    assert body["forecast_model_source"] == "manifest"
    assert body["season_length"] == body["trend_window"] == 3
    assert body["statistics_population"] == "history_rows_for_forecast_series"
    assert body["seasonal_basis"] == "observation_position"
    assert body["std_ddof"] == 1
    assert (body["history_rows"], body["forecast_rows"], body["checked_rows"]) == (12, 3, 15)
    assert body["preview_rows"] == 15 and not body["preview_truncated"]
    components = {component["component"]: component for component in body["components"]}
    for name, magnitude, count, std in (
        ("trend", 155, 10, math.sqrt(8250 / 9)),
        ("seasonal", 2, 12, math.sqrt(72 / 11)),
        ("residual", 0, 10, 0),
    ):
        assert components[name]["mean_abs_value"] == pytest.approx(magnitude)
        assert components[name]["defined_count"] == count
        assert components[name]["undefined_count"] == 12 - count
        assert components[name]["std"] == pytest.approx(std)
    history = [row for row in body["decomposition_preview"] if not row["is_forecast"]]
    assert [row["trend"] for row in history] == [None, *range(110, 201, 10), None]
    assert [row["seasonal"] for row in history] == [-3, 0, 3] * 4
    assert [row["residual"] for row in history] == [None, *([0] * 10), None]


def test_canonical_alias_parity_and_legacy_field_names(api):
    run_id = _publish(api[2])
    canonical = _post(api, run_id=run_id).json()
    alias = _post(api, "shap", run_id=run_id, model_name="not-recorded", top_k=1).json()
    assert {key: alias[key] for key in canonical} == canonical
    assert alias["model"] == canonical["forecast_model"]
    assert alias["feature_importance"] == [
        {"feature": "trend", **{key: value for key, value in canonical["components"][0].items() if key != "component"}},
    ]
    assert "model" not in canonical and "feature_importance" not in canonical


def test_openapi_canonical_and_alias_have_typed_contracts(api):
    spec = api[0].get("/openapi.json").json()
    canonical = spec["paths"]["/metrics/{lob}/decomposition"]["post"]
    alias = spec["paths"]["/metrics/{lob}/shap"]["post"]
    assert not canonical.get("deprecated", False) and alias["deprecated"]
    assert canonical["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("DecompositionSummaryResponse")
    assert alias["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("DecompositionCompatibilityResponse")
    assert "model_specific" in spec["components"]["schemas"]["DecompositionSummaryResponse"]["properties"]


@pytest.mark.parametrize("route", ROUTES)
def test_exact_old_run_never_uses_new_forecast_history_or_model(api, route):
    first = _publish(api[2], model="old-recorded")
    first_result = _post(api, route, run_id=first).json()
    second = _publish(
        api[2], model="new-recorded", started="2026-06-01T00:00:00.000000+00:00",
        history=_history().with_columns(pl.col("quantity") + 100),
        forecast=_forecast().with_columns(pl.col("forecast") + 100),
    )
    before = _hashes(api[2].root)
    assert _post(api, route, run_id=first).json() == first_result
    latest = _post(api, route).json()
    assert latest["run_id"] == latest["history_run_id"] == second
    assert latest["forecast_model"] == "new-recorded"
    assert latest["components"][0]["mean_abs_value"] == pytest.approx(255)
    assert _hashes(api[2].root) == before


def _publish_backtest(store):
    manifest = PipelineManifest(
        run_id=new_run_id(), run_type="backtest", lob="retail",
        started_at="2026-07-01T00:00:00.000000+00:00",
        models=[{"name": "backtest-only", "parameters": {}}],
        input_row_count=12, forecast_row_count=1, forecast_horizon=3,
    )
    config = PlatformConfig(lob="retail")
    config.forecast.horizon_weeks = 3
    config.post_validation.enabled = False
    store.publish(manifest, config, {
        "input": _history(), "prepared": _history(),
        "metrics": pl.DataFrame({"run_id": [manifest.run_id], "run_type": ["backtest"], "lob": ["retail"]}),
        "champions": pl.DataFrame({"model": ["backtest-only"]}),
        "leaderboard": pl.DataFrame({"model": ["backtest-only"]}),
    }, {"skipped": True})
    return manifest.run_id


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("selector", ["missing", "backtest", "invalid", "cross_lob"])
def test_explicit_bad_run_never_falls_back_to_available_forecast(api, route, selector):
    existing = _publish(api[2])
    requested = _publish_backtest(api[2]) if selector == "backtest" else (
        "../invalid" if selector == "invalid" else existing if selector == "cross_lob" else new_run_id()
    )
    before = _hashes(api[2].root)
    lob = "other" if selector == "cross_lob" else "retail"
    response = api[0].post(
        f"/metrics/{lob}/{route}", headers=api[1], params={"run_id": requested, "season_length": 3},
    )
    assert response.status_code == (400 if selector == "invalid" else 404)
    assert _hashes(api[2].root) == before


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("member", ["forecast.parquet", "prepared.parquet", "config.json", "manifest.json"])
def test_corrupt_exact_run_is_integrity_error_not_empty_or_latest(api, route, member):
    old = _publish(api[2])
    _publish(api[2], started="2026-06-01T00:00:00.000000+00:00")
    damaged = api[2].directory("retail", old) / member
    damaged.write_bytes(b"synthetic-corruption")
    before = _hashes(api[2].root)
    response = _post(api, route, run_id=old)
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "run_integrity_error"
    assert _hashes(api[2].root) == before


def _upload(api, frame, route="decomposition", *, fmt="parquet", **params):
    if fmt == "csv":
        content = frame.write_csv().encode()
    else:
        buffer = io.BytesIO()
        frame.write_parquet(buffer)
        content = buffer.getvalue()
    return api[0].post(
        f"/metrics/retail/{route}", headers=api[1], params={"season_length": 3, **params},
        files={"file": (f"synthetic.{fmt}", content)},
    )


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_uploaded_history_is_explicit_override_not_committed_snapshot(api, route, fmt):
    run_id = _publish(api[2])
    before = _hashes(api[2].root)
    response = _upload(api, _history().with_columns(pl.col("quantity") + 50), route, fmt=fmt, run_id=run_id)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_id"] == run_id and body["artifact_source"] == "committed"
    assert body["history_source"] == body["history_stage"] == "uploaded"
    assert body["history_run_id"] is None
    assert body["components"][0]["mean_abs_value"] == pytest.approx(205)
    assert _hashes(api[2].root) == before


@pytest.mark.parametrize("route", ROUTES)
def test_recorded_custom_columns_and_separate_uploaded_columns(api, route):
    config = PlatformConfig(lob="retail")
    config.forecast.series_id_column = "item"
    config.forecast.time_column = "period"
    config.forecast.target_column = "units"
    history = _history().rename({"series_id": "item", "week": "period", "quantity": "units"})
    forecast = _forecast().rename({"series_id": "item", "week": "period"})
    run_id = _publish(api[2], history=history, raw=history, forecast=forecast, config=config)
    expected = _post(api, route, run_id=run_id)
    assert expected.status_code == 200, expected.text
    uploaded = _upload(
        api, _history(), route, fmt="csv", run_id=run_id,
        id_col="series_id", time_col="week", target_col="quantity",
    )
    assert uploaded.status_code == 200, uploaded.text
    assert uploaded.json()["decomposition_preview"] == expected.json()["decomposition_preview"]


def _legacy(api, *, model=True):
    data = Path(api[3]["data_dir"])
    forecasts = data / "forecasts" / "retail"
    history = data / "history" / "retail"
    forecasts.mkdir(parents=True)
    history.mkdir(parents=True)
    frame = _forecast() if model else _forecast().drop("model")
    frame.write_parquet(forecasts / "forecast_retail_2026-04-01.parquet")
    _history().write_parquet(history / "history.parquet")


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("model", [True, False])
def test_legacy_identity_is_unverified_and_model_is_recorded_or_unknown(api, route, model):
    _legacy(api, model=model)
    response = _post(api, route, model_name="never-a-recorded-model")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["artifact_source"] == body["history_source"] == "legacy"
    assert body["history_stage"] == "unknown"
    assert body["run_id"] is body["history_run_id"] is None
    assert body["forecast_model"] == ("recorded-seasonal" if model else None)
    assert body["forecast_model_source"] == ("forecast_frame" if model else "unknown")


def test_explicit_legacy_does_not_switch_to_a_new_committed_run(api):
    _legacy(api)
    run_id = _publish(api[2], model="committed-model")
    assert _post(api, legacy=True).json()["artifact_source"] == "legacy"
    assert _post(api, legacy=True, run_id=run_id).status_code == 400


@pytest.mark.parametrize("models,expected", [
    (["frame-model"] * 3, "frame-model"), (["a", "a", "b"], None),
    (["a", None, "a"], None), ([""] * 3, None),
])
def test_model_fallback_never_guesses_from_current_config_or_partial_frame(api, models, expected):
    run_id = _publish(api[2], model="", forecast=_forecast().with_columns(pl.Series("model", models)))
    response = _post(api, run_id=run_id, model_name="caller-claim")
    assert response.status_code == 200, response.text
    assert response.json()["forecast_model"] == expected


def _long_panels():
    history = pl.DataFrame({
        "series_id": ["hand-example"] * 64,
        "week": [date(2024, 1, 1) + timedelta(weeks=i) for i in range(64)],
        "quantity": [100.0 + i for i in range(64)],
    })
    forecast = pl.DataFrame({
        "series_id": ["hand-example"] * 64,
        "week": [date(2024, 1, 1) + timedelta(weeks=i) for i in range(64, 128)],
        "forecast": [164.0 + i for i in range(64)],
    })
    return history, forecast


@pytest.mark.parametrize("route", ROUTES)
def test_full_frame_counts_and_statistics_do_not_use_preview(api, route):
    history, forecast = _long_panels()
    run_id = _publish(api[2], history=history, forecast=forecast)
    body = _post(api, route, run_id=run_id).json()
    assert (body["checked_rows"], body["preview_rows"]) == (128, 50)
    assert body["preview_truncated"]
    assert body["components"][0]["defined_count"] == 62
    assert body["components"][0]["mean_abs_value"] == pytest.approx(131.5)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("panel", ["history", "forecast"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None])
def test_nonfinite_or_null_input_after_preview_boundary_is_rejected(api, route, panel, value):
    history, forecast = _long_panels()
    column = "quantity" if panel == "history" else "forecast"
    frame = history if panel == "history" else forecast
    values = frame[column].to_list()
    values[60] = value
    frame = frame.with_columns(pl.Series(column, values))
    run_id = _publish(
        api[2], history=frame if panel == "history" else history,
        forecast=frame if panel == "forecast" else forecast,
    )
    before = _hashes(api[2].root)
    response = _post(api, route, run_id=run_id)
    assert response.status_code == 400, response.text
    assert _hashes(api[2].root) == before


@pytest.mark.parametrize("route", ROUTES)
def test_missing_forecast_series_history_cannot_be_silently_dropped(api, route):
    extra = _forecast().with_columns(pl.lit("missing-history").alias("series_id"))
    run_id = _publish(api[2], forecast=pl.concat([_forecast(), extra]))
    response = _post(api, route, run_id=run_id)
    assert response.status_code == 400
    assert "Every forecast series" in response.json()["detail"]


def test_extra_history_is_explicitly_excluded_from_statistics(api):
    history = pl.concat([_history(), _history().with_columns(
        pl.lit("history-only").alias("series_id"), pl.lit(10000.0).alias("quantity"),
    )])
    run_id = _publish(api[2], history=history)
    body = _post(api, run_id=run_id).json()
    assert body["excluded_history_rows"] == body["history_rows"] == 12
    assert body["series_count"] == 1
    assert body["components"][0]["mean_abs_value"] == pytest.approx(155)


@pytest.mark.parametrize("panel", ["history", "forecast"])
def test_duplicate_keys_are_rejected(api, panel):
    history, forecast = _history(), _forecast()
    if panel == "history":
        history = pl.concat([history, history.head(1)])
    else:
        forecast = pl.concat([forecast, forecast.head(1)])
    run_id = _publish(api[2], history=history, forecast=forecast)
    assert _post(api, run_id=run_id).status_code == 400


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4])
def test_empty_or_short_history_never_becomes_zero_components(api, length):
    run_id = _publish(api[2], history=_history().head(length))
    response = _post(api, run_id=run_id)
    assert response.status_code == 400


def test_minimum_supported_history_has_all_seasonal_positions(api):
    run_id = _publish(api[2], history=_history().head(5))
    response = _post(api, run_id=run_id)
    assert response.status_code == 200, response.text
    assert response.json()["components"][0]["defined_count"] == 3


@pytest.mark.parametrize("season,length,window", [(1, 2, 1), (2, 4, 3), (4, 8, 5), (52, 104, 53)])
def test_effective_odd_and_even_windows_with_sufficient_history(api, season, length, window):
    history = pl.DataFrame({
        "series_id": ["hand-example"] * length,
        "week": [date(2020, 1, 1) + timedelta(weeks=i) for i in range(length)],
        "quantity": [20.0] * length,
    })
    run_id = _publish(api[2], history=history)
    response = _post(api, run_id=run_id, season_length=season)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["trend_window"] == window
    assert body["components"][0]["defined_count"] == length - 2 * (season // 2)
    assert body["components"][0]["mean_abs_value"] == pytest.approx(20)


def _window_history():
    return pl.DataFrame({
        "series_id": ["hand-example"] * 104,
        "week": [date(2020, 1, 1) + timedelta(weeks=i) for i in range(104)],
        "quantity": [20.0] * 104,
    })


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("season", [6, 7, 12, 21])
def test_known_unsupported_window_is_actionable_400_before_domain(api, monkeypatch, route, season):
    from src.analytics.explainer import ForecastExplainer

    run_id = _publish(api[2], history=_window_history())
    before = _hashes(api[2].root)
    response = _post(api, route, run_id=run_id, season_length=season)
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["code"] == "unsupported_decomposition_window"
    assert "last-three" in response.json()["detail"]["message"]
    blocked = Mock(side_effect=AssertionError("Unsupported window reached domain math"))
    monkeypatch.setattr(ForecastExplainer, "decompose", blocked)
    assert _post(api, route, run_id=run_id, season_length=season).status_code == 400
    blocked.assert_not_called()
    assert _hashes(api[2].root) == before


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("season", [5, 22, 52])
def test_supported_window_boundaries_keep_real_math_and_stored_bytes(api, route, season):
    run_id = _publish(api[2], history=_window_history())
    before = _hashes(api[2].root)
    response = _post(api, route, run_id=run_id, season_length=season)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["components"][0]["mean_abs_value"] == pytest.approx(20)
    assert body["components"][0]["defined_count"] == 104 - 2 * (season // 2)
    assert body["checked_rows"] == 107
    assert _hashes(api[2].root) == before


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("season,recent_window,status", [(12, 6, 200), (22, 16, 400)])
def test_admission_is_derived_from_actual_recent_window_not_season_whitelist(
    api, monkeypatch, route, season, recent_window, status,
):
    from src.analytics.explainer import ForecastExplainer

    real_init = ForecastExplainer.__init__

    def configured(self, season_length=52, trend_window=12):
        real_init(self, season_length=season_length, trend_window=recent_window)

    monkeypatch.setattr(ForecastExplainer, "__init__", configured)
    run_id = _publish(api[2], history=_window_history())
    response = _post(api, route, run_id=run_id, season_length=season)
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["components"][0]["mean_abs_value"] == pytest.approx(20)
    else:
        assert response.json()["detail"]["code"] == "unsupported_decomposition_window"


def test_empty_forecast_and_overlapping_time_ranges_are_rejected(api):
    empty = _publish(api[2], forecast=_forecast().head(0))
    assert _post(api, run_id=empty).status_code == 400
    overlap = _publish(api[2], forecast=_forecast().with_columns(
        pl.Series("week", _history()["week"].tail(3)),
    ))
    assert _post(api, run_id=overlap).status_code == 400


@pytest.mark.parametrize("parameter,value,status", [
    ("season_length", 0, 400), ("season_length", -2, 400),
    ("season_length", 1000000, 400), ("season_length", 10**18, 400),
    ("season_length", "bad", 422),
    ("season_length", "1.5", 422), ("top_k", 0, 400),
    ("top_k", -1, 400), ("top_k", "bad", 422),
])
def test_invalid_typed_or_business_parameters_are_not_accepted(api, parameter, value, status):
    run_id = _publish(api[2])
    response = _post(api, run_id=run_id, **{parameter: value})
    assert response.status_code == status


@pytest.mark.parametrize("value", [None, "", " ", 1.5, True, 2**53])
def test_unsupported_or_lossy_identifiers_are_rejected(api, value):
    history = _history().with_columns(pl.lit(value).alias("series_id"))
    forecast = _forecast().with_columns(pl.lit(value).alias("series_id"))
    run_id = _publish(api[2], history=history, forecast=forecast)
    assert _post(api, run_id=run_id).status_code == 400


@pytest.mark.parametrize("value", ["000123", 0, 2**53 - 1])
def test_exact_string_and_safe_integer_identifiers_are_preserved(api, value):
    history = _history().with_columns(pl.lit(value).alias("series_id"))
    forecast = _forecast().with_columns(pl.lit(value).alias("series_id"))
    run_id = _publish(api[2], history=history, forecast=forecast)
    response = _post(api, run_id=run_id)
    assert response.status_code == 200, response.text
    assert {row["series_id"] for row in response.json()["decomposition_preview"]} == {value}
    if isinstance(value, str):
        uploaded = _upload(api, history, fmt="csv", run_id=run_id)
        assert uploaded.status_code == 200
        assert uploaded.json()["decomposition_preview"] == response.json()["decomposition_preview"]


def _datetime_panels(unit="us", zone=None, offset=0):
    def convert(frame):
        return frame.with_columns(
            (pl.col("week").cast(pl.Datetime(unit)).cast(pl.Int64) + offset).cast(pl.Datetime(unit, zone)),
        )
    return convert(_history()), convert(_forecast())


@pytest.mark.parametrize("unit,zone,offset", [
    ("ms", None, 1), ("us", None, 123456), ("ns", None, 123456000),
    ("us", "UTC", 123456), ("us", "America/New_York", 123456),
])
def test_supported_time_precision_preserves_every_instant(api, unit, zone, offset):
    history, forecast = _datetime_panels(unit, zone, offset)
    run_id = _publish(api[2], history=history, forecast=forecast)
    response = _post(api, run_id=run_id)
    assert response.status_code == 200, response.text
    observed = [datetime.fromisoformat(row["week"]) for row in response.json()["decomposition_preview"]]
    assert observed == [*history["week"].to_list(), *forecast["week"].to_list()]


@pytest.mark.parametrize("panel", ["history", "forecast"])
def test_submicrosecond_parquet_keys_are_rejected_before_domain_call(api, monkeypatch, panel):
    history, forecast = _datetime_panels("ns")
    frame = history if panel == "history" else forecast
    frame = frame.with_columns((pl.col("week").cast(pl.Int64) + 1).cast(pl.Datetime("ns")))
    run_id = _publish(
        api[2], history=frame if panel == "history" else history,
        forecast=frame if panel == "forecast" else forecast,
    )
    domain = Mock(side_effect=AssertionError("Lossy input reached domain math"))
    monkeypatch.setattr("src.analytics.explainer.ForecastExplainer.decompose", domain)
    response = _post(api, run_id=run_id)
    assert response.status_code == 400
    domain.assert_not_called()


@pytest.mark.parametrize("fraction,status", [("123456789", 400), ("000000001", 400), ("123456000", 200), ("123456", 200)])
def test_csv_fractional_precision_is_checked_before_inference(api, fraction, status):
    history, forecast = _datetime_panels("us", offset=123456)
    run_id = _publish(api[2], history=history, forecast=forecast)
    raw = history.with_columns((pl.col("week").dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{fraction}").alias("week"))
    response = _upload(api, raw, fmt="csv", run_id=run_id)
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["decomposition_preview"][0]["week"].endswith(".123456")


@pytest.mark.parametrize("bad_time", [None, "not-a-date", "2026-02-30", 123])
def test_invalid_temporal_keys_are_client_errors(api, bad_time):
    run_id = _publish(api[2])
    history = _history().with_columns(pl.lit(bad_time).alias("week"))
    assert _upload(api, history, run_id=run_id).status_code == 400


def test_mixed_date_datetime_and_timezone_awareness_are_not_coerced(api):
    history, forecast = _datetime_panels()
    run_id = _publish(api[2], history=_history(), forecast=forecast)
    assert _post(api, run_id=run_id).status_code == 400
    aware = history.with_columns(pl.col("week").dt.replace_time_zone("UTC"))
    run_id = _publish(api[2], history=aware, forecast=forecast)
    assert _post(api, run_id=run_id).status_code == 400


@pytest.mark.parametrize("defect", ["drop", "duplicate", "nan", "unexpected_null", "wrong_value", "wrong_component"])
def test_full_result_is_checked_before_preview_or_statistics(api, monkeypatch, caplog, defect):
    from src.analytics.explainer import ForecastExplainer

    history, forecast = _long_panels()
    run_id = _publish(api[2], history=history, forecast=forecast)
    real = ForecastExplainer.decompose

    def damaged(self, *args, **kwargs):
        result = real(self, *args, **kwargs)
        if defect == "drop":
            return result.head(-1)
        if defect == "duplicate":
            return pl.concat([result.head(-1), result.head(1)])
        column = "value" if defect == "wrong_value" else "trend"
        values = result[column].to_list()
        values[-1] = {
            "nan": float("nan"), "unexpected_null": None, "wrong_value": 1.0, "wrong_component": 1.0,
        }[defect]
        return result.with_columns(pl.Series(column, values))

    monkeypatch.setattr(ForecastExplainer, "decompose", damaged)
    before = _hashes(api[2].root)
    response = _post(api, run_id=run_id)
    assert response.status_code == 500, response.text
    assert "Forecast decomposition failed" in caplog.text
    assert "components" not in response.json()
    assert _hashes(api[2].root) == before


@pytest.mark.parametrize("exception", [RuntimeError, ValueError, OSError])
def test_unexpected_computation_errors_are_logged_and_bounded(api, monkeypatch, caplog, exception):
    run_id = _publish(api[2])
    monkeypatch.setattr(
        "src.analytics.explainer.ForecastExplainer.decompose",
        Mock(side_effect=exception("private-path-or-value")),
    )
    response = _post(api, run_id=run_id)
    assert response.status_code == 500
    assert "Forecast decomposition failed" in caplog.text
    assert "private-path-or-value" not in response.text


def test_nonfinite_computation_or_statistics_never_become_zero_or_null_success(api):
    run_id = _publish(api[2], history=_history().with_columns(pl.col("quantity") * 1e199))
    assert _post(api, run_id=run_id).status_code == 500


@pytest.mark.parametrize("route", ROUTES)
def test_no_fitted_model_shap_or_provider_path_is_called(api, monkeypatch, route):
    from src.forecasting.registry import ForecasterRegistry

    run_id = _publish(api[2])
    blocked = Mock(side_effect=AssertionError("Model or provider path must not execute"))
    monkeypatch.setattr("src.analytics.explainer.ForecastExplainer.explain_ml", blocked)
    monkeypatch.setattr(ForecasterRegistry, "build", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    response = _post(api, route, run_id=run_id)
    assert response.status_code == 200, response.text
    blocked.assert_not_called()


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("identity,status", [("anonymous", 401), ("expired", 401), ("inactive", 403), ("viewer", 200)])
def test_real_authentication_and_view_metrics_permission(api, route, identity, status):
    run_id = _publish(api[2])
    now = int(time.time())
    token = jwt.encode({
        "user_id": "synthetic-viewer", "email": "", "role": "viewer",
        "is_active": identity != "inactive", "iat": now - 7200 if identity == "expired" else now,
        "exp": now - 3600 if identity == "expired" else now + 3600,
        "iss": TOKEN_ISSUER, "aud": TOKEN_AUDIENCE,
    }, api[4], algorithm="HS256")
    headers = {} if identity == "anonymous" else {"Authorization": f"Bearer {token}"}
    response = api[0].post(
        f"/metrics/retail/{route}", headers=headers, params={"run_id": run_id, "season_length": 3},
    )
    assert response.status_code == status
    assert token not in response.text and api[4] not in response.text


@pytest.mark.parametrize("lob", ["..bad", "CON", "retail.", "bad%5Cpath"])
def test_path_containment_is_retained(api, lob):
    response = api[0].post(f"/metrics/{lob}/decomposition", headers=api[1])
    assert response.status_code == 400


def test_upload_limit_is_still_enforced_and_parse_errors_are_bounded(api, monkeypatch):
    from src.api.deps import validate_upload_size

    run_id = _publish(api[2])

    async def limited(file):
        return await validate_upload_size(file, max_bytes=8)

    monkeypatch.setattr("src.api.routers.analytics.validate_upload_size", limited)
    response = _upload(api, _history(), run_id=run_id)
    assert response.status_code == 413
    response = api[0].post(
        "/metrics/retail/decomposition", headers=api[1], params={"run_id": run_id, "season_length": 3},
        files={"file": ("bad.parquet", b"broken")},
    )
    assert response.status_code == 400
    assert "broken" not in response.text
