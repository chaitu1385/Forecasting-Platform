"""Real ASGI regressions for planner intent and fail-closed config uploads."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.config.schema import PlatformConfig
from src.forecasting.intermittent import CrostonForecaster
from src.pipeline.run_store import RunStore

pytestmark = pytest.mark.integration


def _config() -> dict:
    return {
        "forecast": {
            "frequency": "W",
            "horizon_periods": 5,
            "forecasters": ["naive_seasonal", "croston"],
        },
        "backtest": {"n_folds": 1, "val_periods": 4, "gap_periods": 0},
        "data_quality": {"fill_gaps": False, "min_series_length_periods": 1},
        "parallelism": {"n_workers": 1},
        "post_validation": {"enabled": False},
    }


def _actuals(n_weeks: int = 60) -> pl.DataFrame:
    return pl.DataFrame({
        "series_id": ["sku"] * n_weeks,
        "week": [date(2024, 1, 1) + timedelta(weeks=i) for i in range(n_weeks)],
        "quantity": [float(10 + i * 3) for i in range(n_weeks)],
    })


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Even an accidental fallback to relative paths must stay inside the fixture.
    monkeypatch.chdir(tmp_path)
    app = create_app(
        data_dir=str(tmp_path / "data"),
        metrics_dir=str(tmp_path / "data" / "metrics"),
        audit_log_path=str(tmp_path / "audit"),
        auth_enabled=False,
        dev_mode=True,
        jwt_secret="",
    )
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _run(client, endpoint="forecast", *, config=None, raw_config=None, params=None, actuals=None):
    df = _actuals() if actuals is None else actuals
    files = {"file": ("actuals.csv", df.write_csv().encode(), "text/csv")}
    if config is not None:
        raw_config = yaml.safe_dump(config).encode()
    if raw_config is not None:
        files["config_file"] = ("plan.yaml", raw_config, "text/yaml")
    return client.post(
        f"/pipeline/{endpoint}",
        params={"lob": "retail", **(params or {})},
        files=files,
    )


def _snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_backtest_winner_and_five_period_plan_are_preserved(client):
    config = _config()
    backtest = _run(client, "backtest", config=config)
    assert backtest.status_code == 200, backtest.text
    winner = backtest.json()["champion_model"]
    assert winner == "croston"

    # The UI sends this selected winner and the same YAML, not a naive fallback.
    forecast = _run(client, config=config, params={"model_id": winner})
    assert forecast.status_code == 200, forecast.text
    body = forecast.json()
    assert body["model_id"] == winner
    assert body["horizon"] == body["forecast_rows"] == 5
    assert {row["model"] for row in body["forecast_preview"]} == {"croston"}
    model = CrostonForecaster()
    model.fit(_actuals())
    expected = model.predict(5)
    assert [row["forecast"] for row in body["forecast_preview"]] == expected["forecast"].to_list()
    execution = body["execution"]
    assert execution["model_id"] == winner
    assert execution["model_source"] == "request"
    assert execution["horizon_periods"] == 5
    assert execution["horizon_source"] == "config"
    assert execution["config_source"] == "upload"
    assert execution["config"]["frequency"] == "W"
    assert execution["config"]["fill_gaps"] is False
    assert execution["config"]["min_series_length_periods"] == 1
    assert backtest.json()["execution"]["horizon_periods"] == 5
    assert backtest.json()["execution"]["model_source"] == "backtest"
    assert backtest.json()["execution"]["config"]["backtest_val_periods"] == 4

    store = RunStore(client.app.state.runs_dir)
    stored = store.list("retail", "forecast")
    assert len(stored) == 1
    assert store.frame("retail", body["run_id"], "forecast").height == 5
    manifest = store.get("retail", body["run_id"], "forecast")
    assert manifest.run_id == stored[0].run_id
    assert body["run_id"] != backtest.json()["run_id"]
    assert manifest.champion_model_id == winner
    assert manifest.forecast_horizon == 5


@pytest.mark.parametrize("metrics", [
    ["wmape", "normalized_bias"],
    ["wmape", "normalized_bias", "mae", "rmse"],
])
def test_strict_backtest_preserves_unrequested_metric_placeholders(client, metrics):
    config = _config()
    config["forecast"]["forecasters"] = ["naive_seasonal"]
    config["backtest"] = {"n_folds": 2, "val_periods": 8, "gap_periods": 0}
    config["metrics"] = metrics
    config["post_validation"] = {"enabled": True, "halt_on_blocker": True}

    response = _run(client, "backtest", config=config, actuals=_actuals(104))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["champion_model"] == "naive_seasonal"
    assert body["execution"]["config"]["post_validation_enabled"] is True
    assert body["execution"]["config"]["halt_on_blocker"] is True
    store = RunStore(client.app.state.runs_dir)
    records = store.frame("retail", body["run_id"], "metrics")
    assert records.height == 16
    assert records["mape"].null_count() == records.height
    assert records["mase"].null_count() == records.height
    for metric in metrics:
        assert records[metric].is_finite().all()


@pytest.mark.parametrize("undefined", ["zero_actuals", "supplementary_mase"])
def test_strict_backtest_preserves_legitimately_undefined_scores(client, undefined):
    config = _config()
    config["forecast"]["forecasters"] = ["naive_seasonal"]
    config["backtest"] = {"n_folds": 2, "val_periods": 8, "gap_periods": 0}
    config["metrics"] = ["wmape", "normalized_bias", "mae", "rmse", "mase"]
    config["post_validation"] = {"enabled": True, "halt_on_blocker": True}
    actuals = _actuals(104)
    if undefined == "zero_actuals":
        actuals = actuals.with_row_index().with_columns(
            pl.when(pl.col("index") % 4 == 0).then(0.0).otherwise(pl.col("quantity")).alias("quantity"),
        ).drop("index")
    else:
        actuals = actuals.with_columns(pl.lit(100.0).alias("quantity"))

    response = _run(client, "backtest", config=config, actuals=actuals)

    assert response.status_code == 200, response.text
    body = response.json()
    store = RunStore(client.app.state.runs_dir)
    records = store.frame("retail", body["run_id"], "metrics")
    assert records.height == 16
    if undefined == "zero_actuals":
        assert records["wmape"].null_count() == records["normalized_bias"].null_count() == 4
        volume = records["actual"].abs().sum()
        absolute_error = (records["forecast"] - records["actual"]).abs().sum()
        assert body["best_wmape"] == pytest.approx(absolute_error / volume)
    else:
        assert records["mase_scale"].null_count() == records["mase"].null_count() == 16
        assert body["execution"]["config"]["primary_metric"] == "wmape"
        assert body["best_wmape"] == 0
        config["backtest"]["primary_metric"] = "mase"
        rejected = _run(client, "backtest", config=config, actuals=actuals)
        assert rejected.status_code == 500
        assert rejected.json()["detail"]["outcome"] == "not_committed"
        assert [run.run_id for run in store.list("retail", "backtest")] == [body["run_id"]]


@pytest.mark.parametrize("horizon_key", ["horizon_periods", "horizon_weeks"])
def test_omitted_query_horizon_uses_config_alias(client, horizon_key):
    config = _config()
    config["forecast"] = {"forecasters": ["croston"], horizon_key: 5}
    response = _run(client, config=config)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["horizon"] == body["forecast_rows"] == 5
    assert body["model_id"] == "croston"
    assert body["execution"]["model_source"] == "config"


def test_explicit_request_overrides_valid_config(client):
    config = _config()
    config["forecast"]["forecasters"] = ["croston"]
    response = _run(client, config=config, params={"model_id": "naive_seasonal", "horizon": 3})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_id"] == "naive_seasonal"
    assert body["horizon"] == body["forecast_rows"] == 3
    assert body["execution"]["model_source"] == "request"
    assert body["execution"]["horizon_source"] == "request"
    assert body["execution"]["horizon_periods"] == 3
    assert {row["model"] for row in body["forecast_preview"]} == {"naive_seasonal"}


@pytest.mark.parametrize("raw_config", [None, b"{}"])
def test_no_config_and_empty_mapping_use_truthful_platform_defaults(client, raw_config):
    response = _run(client, raw_config=raw_config)
    assert response.status_code == 200, response.text
    body = response.json()
    defaults = PlatformConfig()
    assert body["model_id"] == defaults.forecast.forecasters[0]
    assert body["horizon"] == body["forecast_rows"] == defaults.forecast.horizon_periods
    assert body["execution"]["config_source"] == ("default" if raw_config is None else "upload")
    assert body["execution"]["config"]["post_validation_enabled"] is True
    assert body.get("champion_model") is None


def test_backtest_without_config_remains_supported(client):
    response = _run(client, "backtest")
    assert response.status_code == 200, response.text
    assert response.json()["champion_model"] == "naive_seasonal"
    assert response.json()["execution"]["config_source"] == "default"


@pytest.mark.parametrize("metric", ["wmape", "normalized_bias", "mape", "mae", "rmse", "mase"])
def test_registered_selection_metrics_remain_supported(client, metric):
    config = _config()
    secondary = "mae" if metric == "normalized_bias" else "normalized_bias"
    config["backtest"].update({"primary_metric": metric, "secondary_metric": secondary})
    config["metrics"] = [metric, secondary]
    response = _run(client, "backtest", config=config)
    assert response.status_code == 200, response.text
    assert response.json()["execution"]["config"]["primary_metric"] == metric
    assert response.json()["execution"]["config"]["secondary_metric"] == secondary


def test_legacy_best_metric_strategy_reports_effective_champion_strategy(client):
    config = _config()
    config["backtest"]["selection_strategy"] = "best_metric"
    response = _run(client, "backtest", config=config)
    assert response.status_code == 200, response.text
    assert response.json()["champion_model"] == "croston"
    assert response.json()["execution"]["config"]["selection_strategy"] == "champion"


def test_ensemble_backtest_does_not_advertise_a_standalone_winner(client):
    config = _config()
    config["backtest"]["selection_strategy"] = "weighted_ensemble"
    response = _run(client, "backtest", config=config)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["execution"]["model_id"] == "weighted_ensemble"
    assert body["execution"]["config"]["selection_strategy"] == "weighted_ensemble"
    assert body.get("champion_model") is None
    assert {row["model_id"] for row in body["leaderboard"]} == {"naive_seasonal", "croston"}


def test_configured_constraints_remain_supported_for_a_standalone_model(client):
    config = _config()
    config["forecast"]["constraints"] = {"enabled": True, "max_capacity": 95.0}
    response = _run(client, config=config, params={"model_id": "croston"})
    assert response.status_code == 200, response.text
    assert response.json()["execution"]["config"]["constraints_enabled"] is True
    assert response.json()["forecast_rows"] == 5
    assert {row["forecast"] for row in response.json()["forecast_preview"]} == {95.0}


def test_period_alias_precedence_and_configured_columns(client):
    config = _config()
    config["forecast"].update({
        "horizon_weeks": 9,
        "series_id_column": "item",
        "time_column": "period",
        "target_column": "sales",
    })
    config["backtest"].update({"val_weeks": 7, "gap_weeks": 2})
    df = _actuals().rename({"series_id": "item", "week": "period", "quantity": "sales"})
    backtest = _run(client, "backtest", config=config, actuals=df)
    assert backtest.status_code == 200, backtest.text
    assert backtest.json()["champion_model"] == "croston"
    response = _run(client, config=config, actuals=df, params={"model_id": "croston"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["forecast_rows"] == 5
    assert body["series_count"] == 1
    assert "item" in body["forecast_preview"][0]
    assert "period" in body["forecast_preview"][0]
    assert body["execution"]["config"]["series_id_column"] == "item"
    assert body["execution"]["config"]["time_column"] == "period"
    assert body["execution"]["config"]["target_column"] == "sales"
    assert backtest.json()["execution"]["config"]["backtest_val_periods"] == 4
    assert backtest.json()["execution"]["config"]["backtest_gap_periods"] == 0


@pytest.mark.parametrize("frequency", ["D", "W", "M", "Q"])
def test_configured_frequency_and_horizon_reach_forecast(client, frequency):
    config = _config()
    config["forecast"].update({"frequency": frequency, "forecasters": ["naive_seasonal"]})
    if frequency in ("M", "Q"):
        months = [i * (3 if frequency == "Q" else 1) for i in range(65)]
        dates = [date(2024 + month // 12, month % 12 + 1, 1) for month in months]
    else:
        dates = [date(2024, 1, 1) + timedelta(days=i * (7 if frequency == "W" else 1)) for i in range(65)]
    df = _actuals().with_columns(pl.Series("week", dates[:60]))
    response = _run(client, config=config, actuals=df)
    assert response.status_code == 200, response.text
    assert response.json()["forecast_rows"] == 5
    assert response.json()["execution"]["config"]["frequency"] == frequency
    assert sorted(row["week"] for row in response.json()["forecast_preview"]) == [
        value.isoformat() for value in dates[60:]
    ]


def test_explicit_macro_wmape_is_a_valid_configured_selection_criterion(client):
    config = _config()
    config["backtest"]["primary_metric"] = "macro_wmape"
    config["metrics"] = ["macro_wmape", "normalized_bias"]
    response = _run(client, "backtest", config=config)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["champion_model"] == "croston"
    assert all(row["macro_wmape"] is not None for row in result["leaderboard"])
    assert all(row["scoring_policy"] == "raw-observations-v1" for row in result["leaderboard"])


@pytest.mark.parametrize("frequency,expected", [("D", 70.0), ("W", 0.0), (None, 0.0)])
def test_cleansing_audit_uses_explicit_frequency_and_preserves_legacy_default(client, frequency, expected):
    source = pl.DataFrame({
        "series_id": ["sku"] * 9,
        "week": [date(2024, 1, index) for index in range(1, 10)],
        "quantity": [70.0] + [10.0] * 6 + [0.0, 10.0],
    })
    params = {"outlier_action": "flag_only", "stockout_detection": True, "min_zero_run": 1}
    if frequency is not None:
        params["frequency"] = frequency
    response = client.post(
        "/series/cleansing-audit", params=params,
        files={"file": ("actuals.csv", source.write_csv().encode(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    day_eight = next(row for row in result["cleansed_preview"] if row["week"] == "2024-01-08")
    assert day_eight["quantity"] == expected
    assert result["frequency"] == (frequency or "W")


def test_cleansing_audit_rejects_an_unsupported_frequency(client):
    response = client.post("/series/cleansing-audit", params={"frequency": "UNKNOWN"})
    assert response.status_code == 422, response.text
    assert any("frequency" in error["loc"] for error in response.json()["detail"])


INVALID_CONFIGS = [
    pytest.param(b"forecast: [", id="syntax"),
    pytest.param(b"", id="empty-upload"),
    pytest.param(b"null", id="null-document"),
    pytest.param(b"[]", id="list-document"),
    pytest.param(b"42", id="scalar-document"),
    pytest.param(b"forecast: null", id="null-section"),
    pytest.param(b"forecast: 3", id="scalar-section"),
    pytest.param(b"forecast:\n  horizon_periods: '5'", id="string-horizon"),
    pytest.param(b"forecast:\n  horizon_periods: true", id="boolean-horizon"),
    pytest.param(b"forecast:\n  horizon_periods: 2.5", id="fractional-horizon"),
    pytest.param(b"forecast:\n  horizon_periods: 0", id="zero-horizon"),
    pytest.param(b"forecast:\n  frequency: X", id="unknown-frequency"),
    pytest.param(b"forecast:\n  forecasters: croston", id="scalar-models"),
    pytest.param(b"forecast:\n  forecasters: []", id="empty-models"),
    pytest.param(b"forecast:\n  forecasters: [not_a_model]", id="unknown-model"),
    pytest.param(b"forecast:\n  forecasters: [weighted_ensemble]", id="composite-candidate"),
    pytest.param(b"forecast:\n  intermittent_forecasters: [not_a_model]", id="unknown-sparse-model"),
    pytest.param(b"forecast:\n  quantiles: [1.5]", id="invalid-quantile"),
    pytest.param(b"forecast:\n  quantiles: [.nan]", id="nonfinite-quantile"),
    pytest.param(b"forecast:\n  time_column: ''", id="empty-column"),
    pytest.param(b"data_quality:\n  fill_gaps: 'false'", id="string-boolean"),
    pytest.param(b"backtest:\n  n_folds: 0", id="zero-folds"),
    pytest.param(b"backtest:\n  n_folds: true", id="boolean-folds"),
    pytest.param(b"backtest:\n  gap_periods: -1", id="negative-gap"),
    pytest.param(b"backtest:\n  selection_strategy: champon", id="unknown-strategy"),
    pytest.param(b"backtest:\n  champion_granularity: typo", id="unknown-granularity"),
    pytest.param(b"backtest:\n  primary_metric: unknown", id="unknown-primary"),
    pytest.param(b"backtest:\n  secondary_metric: unknown", id="unknown-secondary"),
    pytest.param(b"metrics: [unknown]", id="unknown-metric"),
    pytest.param(b"backtest:\n  primary_metric: mae", id="uncomputed-selection-metric"),
    pytest.param(b"backtest:\n  primary_metric: normalized_bias", id="duplicate-selection-metric"),
    pytest.param(b"backtest:\n  horizon_buckets: [{name: short, start_step: one, end_step: 4}]", id="nested-type"),
    pytest.param(b"post_validation:\n  enabled: 'false'", id="validation-type"),
    pytest.param(b"post_validation:\n  min_grade: Z", id="validation-grade"),
    pytest.param(b"output: [bad]", id="output-type"),
    pytest.param(b"\xff\xfe", id="encoding"),
]


@pytest.mark.parametrize("endpoint", ["backtest", "forecast"])
@pytest.mark.parametrize("raw_config", INVALID_CONFIGS)
def test_invalid_uploaded_config_is_4xx_without_any_publication(client, tmp_path, endpoint, raw_config):
    before = _snapshot(tmp_path)
    response = _run(client, endpoint, raw_config=raw_config)
    assert response.status_code == 400, response.text
    assert "config" in response.json()["detail"].lower()
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("endpoint", ["backtest", "forecast"])
def test_invalid_config_cannot_overwrite_existing_results(client, tmp_path, endpoint):
    seeded = _run(client, endpoint, config=_config(), params={"model_id": "croston", "horizon": 5})
    assert seeded.status_code == 200, seeded.text
    before = _snapshot(tmp_path)
    assert before
    response = _run(client, endpoint, raw_config=b"forecast:\n  frequency: INVALID")
    assert response.status_code == 400
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("params", [
    {"horizon": 0},
    {"horizon": -5},
    {"horizon": "abc"},
    {"horizon": "2.5"},
    {"model_id": "not_a_model"},
    {"model_id": ""},
    {"model_id": " "},
    pytest.param({"model_id": "weighted_ensemble"}, id="composite-ensemble"),
    pytest.param({"model_id": "constrained_demand"}, id="composite-wrapper"),
])
def test_invalid_explicit_intent_fails_before_write(client, tmp_path, params):
    before = _snapshot(tmp_path)
    response = _run(client, config=_config(), params=params)
    assert response.status_code in (400, 422), response.text
    assert _snapshot(tmp_path) == before


def test_override_does_not_hide_invalid_uploaded_config(client, tmp_path):
    response = _run(
        client,
        raw_config=b"forecast:\n  horizon_periods: 0",
        params={"model_id": "croston", "horizon": 5},
    )
    assert response.status_code == 400, response.text
    assert not list((tmp_path / "data").rglob("*"))


def test_invalid_config_preserves_existing_forecast_artifacts(client, tmp_path):
    successful = _run(client, config=_config())
    assert successful.status_code == 200, successful.text
    before = _snapshot(tmp_path)
    entries = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert any(name.endswith(".parquet") for name in before)
    response = _run(
        client,
        raw_config=b"forecast:\n  horizon_periods: 0",
        params={"model_id": "croston", "horizon": 5},
    )
    assert response.status_code == 400, response.text
    assert _snapshot(tmp_path) == before
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == entries


def test_execution_settings_are_allowlisted_not_raw_config(client, tmp_path):
    config = _config()
    config.update({
        "description": "PRIVATE_DESCRIPTION",
        "unrecognized_metadata": {"credential": "PRIVATE_CREDENTIAL"},
        "output": {"forecast_path": str(tmp_path / "private-output")},
        "ai": {"api_key_env_var": "PRIVATE_KEY_NAME"},
        "observability": {"alerts": {"webhook_url": "https://example.invalid/PRIVATE_WEBHOOK"}},
    })
    response = _run(client, config=config)
    assert response.status_code == 200, response.text
    execution = response.json()["execution"]
    assert execution["config"]["forecasters"] == ["naive_seasonal", "croston"]
    assert "PRIVATE_" not in response.text
    assert str(tmp_path) not in response.text
    assert "output" not in execution["config"]


def test_execution_contract_is_in_openapi(client):
    schema = client.get("/openapi.json").json()
    response = schema["paths"]["/pipeline/forecast"]["post"]["responses"]["200"]
    assert "$ref" in response["content"]["application/json"]["schema"]
    contract = schema["components"]["schemas"]["PipelineRunResponse"]
    assert "execution" in contract["required"]
