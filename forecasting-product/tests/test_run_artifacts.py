"""Real pipeline/API contracts for immutable run identity and restart retrieval."""

import json
import secrets
import shutil
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.auth.token import create_token
from src.config.schema import PlatformConfig
from src.pipeline.backtest import BacktestPipeline
from src.pipeline.forecast import ForecastPipeline
from src.pipeline.run_store import MANIFEST_NAME, RunStore, resolve_run_root


def _actuals(value=10.0):
    return pl.DataFrame({
        "series_id": ["synthetic"] * 64,
        "week": [date(2024, 1, 1) + timedelta(weeks=index) for index in range(64)],
        "quantity": [value] * 64,
    })


def _config():
    return {
        "lob": "retail",
        "forecast": {"forecasters": ["naive_seasonal"], "horizon_periods": 2},
        "backtest": {"n_folds": 1, "val_periods": 2},
        "parallelism": {"n_workers": 1},
        "data_quality": {"fill_gaps": False, "min_series_length_periods": 1},
        "post_validation": {"enabled": False},
    }


@pytest.fixture
def api(tmp_path):
    key = secrets.token_urlsafe(48)
    roots = {"data_dir": str(tmp_path / "data"), "metrics_dir": str(tmp_path / "legacy-metrics"),
             "runs_dir": str(tmp_path / "authority")}
    credentials = {
        role: {"Authorization": f"Bearer {create_token('synthetic-' + role, '', role, key, expiry_hours=1)}"}
        for role in ("admin", "viewer")
    }
    app = create_app(**roots, auth_enabled=True, dev_mode=False, jwt_secret=key)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, credentials, roots, key


def _run(client, headers, value=10.0, operation="forecast", config=None):
    return client.post(
        f"/pipeline/{operation}", params={"lob": "retail"}, headers=headers,
        files={
            "file": ("synthetic.csv", _actuals(value).write_csv().encode(), "text/csv"),
            "config_file": ("plan.yaml", yaml.safe_dump(config or _config()).encode(), "text/yaml"),
        },
    )


def test_same_origin_runs_preserve_values_and_exact_identity_after_fresh_app(api):
    client, headers, roots, key = api
    first = _run(client, headers["admin"], 10.0)
    second = _run(client, headers["admin"], 20.0)
    assert first.status_code == second.status_code == 200
    first_id, second_id = first.json()["run_id"], second.json()["run_id"]
    assert len(first_id) == len(second_id) == 32 and first_id != second_id
    before = {}
    for run_id, value in ((first_id, 10.0), (second_id, 20.0)):
        response = client.get("/forecast/retail", params={"run_id": run_id}, headers=headers["viewer"])
        assert response.status_code == 200
        assert response.json()["run_id"] == run_id
        assert {point["forecast"] for point in response.json()["points"]} == {value}
        metadata = client.get(f"/pipeline/runs/retail/{run_id}", headers=headers["viewer"])
        assert metadata.status_code == 200
        saved = metadata.json()
        assert saved["input_data_hash"] == response.json()["input_data_hash"]
        assert saved["config_hash"] == response.json()["config_hash"]
        configuration = client.get(f"/pipeline/runs/retail/{run_id}/config", headers=headers["viewer"])
        assert configuration.json()["config_hash"] == saved["config_hash"]
        history = client.get("/series/retail/synthetic/history", params={"run_id": run_id}, headers=headers["viewer"])
        assert history.status_code == 200
        assert history.json()["run_id"] == run_id
        assert {point["value"] for point in history.json()["points"]} == {value}
        before[run_id] = (saved, response.json(), history.json(), configuration.json())
    listing = client.get("/pipeline/runs?lob=retail&run_type=forecast", headers=headers["viewer"])
    assert [row["run_id"] for row in listing.json()["runs"]] == [second_id, first_id]
    assert client.get("/forecast/retail", headers=headers["viewer"]).json()["run_id"] == second_id
    with TestClient(create_app(**roots, auth_enabled=True, dev_mode=False, jwt_secret=key)) as restarted:
        for run_id, expected in before.items():
            observed = (
                restarted.get(f"/pipeline/runs/retail/{run_id}", headers=headers["viewer"]).json(),
                restarted.get("/forecast/retail", params={"run_id": run_id}, headers=headers["viewer"]).json(),
                restarted.get("/series/retail/synthetic/history", params={"run_id": run_id}, headers=headers["viewer"]).json(),
                restarted.get(f"/pipeline/runs/retail/{run_id}/config", headers=headers["viewer"]).json(),
            )
            assert observed == expected


def test_backtest_metrics_context_and_manifest_use_one_identity(api):
    client, headers, roots, _ = api
    response = _run(client, headers["admin"], operation="backtest")
    assert response.status_code == 200, response.text
    run_id = response.json()["run_id"]
    store = RunStore(roots["runs_dir"])
    manifest = store.get("retail", run_id, "backtest")
    assert store.frame("retail", run_id, "metrics")["run_id"].unique().to_list() == [run_id]
    board = client.get("/metrics/leaderboard/retail", params={"run_id": run_id}, headers=headers["viewer"])
    assert board.status_code == 200 and board.json()["run_id"] == manifest.run_id
    wrong_type = client.get("/forecast/retail", params={"run_id": run_id}, headers=headers["viewer"])
    assert wrong_type.status_code == 404


@pytest.mark.parametrize("operation", ["forecast", "backtest"])
def test_pipeline_instance_reuse_has_fresh_attempts_in_its_custom_root(tmp_path, operation):
    config = PlatformConfig(lob="retail")
    config.forecast.horizon_weeks = 2
    config.backtest.n_folds = 1
    config.backtest.val_weeks = 2
    config.parallelism.n_workers = 1
    config.data_quality.fill_gaps = False
    config.data_quality.min_series_length_weeks = 1
    config.post_validation.enabled = False
    config.output.forecast_path = str(tmp_path / "forecast-device")
    config.output.metrics_path = str(tmp_path / "metric-device")
    pipeline = ForecastPipeline(config) if operation == "forecast" else BacktestPipeline(config)
    pipeline.run(_actuals(10.0))
    first = pipeline.context.run_id
    pipeline.run(_actuals(20.0))
    second = pipeline.context.run_id
    assert first != second and pipeline.last_run.run_id == second
    store = RunStore(resolve_run_root(config, operation))
    assert {item.run_id for item in store.list()} == {first, second}
    assert store.frame("retail", first, "input")["quantity"].unique().to_list() == [10.0]
    assert store.frame("retail", second, "input")["quantity"].unique().to_list() == [20.0]
    assert store.root.is_relative_to(tmp_path)
    if operation == "backtest":
        assert pipeline._backtest_engine.run_id == second


def test_precommit_api_failure_never_claims_completed_and_keeps_old_run(api, monkeypatch):
    from src.pipeline import run_store

    client, headers, roots, _ = api
    first = _run(client, headers["admin"])
    run_id = first.json()["run_id"]
    store = RunStore(roots["runs_dir"])
    before = store.frame("retail", run_id, "forecast").to_dicts()
    original = run_store._write_new

    def reject_config(path, writer):
        if path.name == "config.json":
            raise OSError("synthetic snapshot failure")
        return original(path, writer)

    monkeypatch.setattr(run_store, "_write_new", reject_config)
    failed = _run(client, headers["admin"], 20.0)
    assert failed.status_code == 500
    detail = failed.json()["detail"]
    assert detail["outcome"] == "not_committed" and detail["run_id"] != run_id
    assert client.get(detail["links"]["metadata"], headers=headers["viewer"]).status_code == 404
    assert [item.run_id for item in store.list()] == [run_id]
    assert store.frame("retail", run_id, "forecast").to_dicts() == before


def test_postcommit_api_failure_explicitly_returns_committed_identity(api, monkeypatch):
    from src.pipeline import run_store

    client, headers, roots, _ = api
    original = run_store.os.link

    def lose_receipt(source, destination):
        original(source, destination)
        if destination.name == MANIFEST_NAME:
            raise OSError("synthetic response/confirmation loss")

    monkeypatch.setattr(run_store.os, "link", lose_receipt)
    response = _run(client, headers["admin"])
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["outcome"] == "committed"
    assert client.get(detail["links"]["metadata"], headers=headers["viewer"]).status_code == 200
    assert [item.run_id for item in RunStore(roots["runs_dir"]).list()] == [detail["run_id"]]


def test_new_retrieval_endpoints_require_auth_and_reject_wrong_namespace(api):
    client, headers, _, _ = api
    run_id = _run(client, headers["admin"]).json()["run_id"]
    for suffix in ("", "/config", "/input"):
        url = f"/pipeline/runs/retail/{run_id}{suffix}"
        assert client.get(url).status_code == 401
        assert client.get(url, headers={"Authorization": "Bearer invalid"}).status_code == 401
        assert client.get(url, headers=headers["viewer"]).status_code == 200
    assert client.get("/pipeline/runs").status_code == 401
    assert client.get(f"/pipeline/runs/another_lob/{run_id}", headers=headers["viewer"]).status_code == 404
    assert _run(client, headers["viewer"]).status_code == 403
    for identifier in ("../outside", "A" * 32, "not-a-run", "a" * 32 + "\\..\\outside"):
        result = client.get("/forecast/retail", params={"run_id": identifier}, headers=headers["viewer"])
        assert result.status_code == 400


def test_safe_config_and_metadata_do_not_save_or_return_credentials_or_paths(api):
    client, headers, roots, key = api
    config = _config()
    private = "PRIVATE_DO_NOT_PERSIST"
    config.update({
        "description": private,
        "ai": {"api_key_env_var": private},
        "observability": {"alerts": {"webhook_url": f"https://user:{private}@example.invalid/hook?token={private}"}},
        "output": {"forecast_path": private, "metrics_path": private, "runs_path": private},
    })
    response = _run(client, headers["admin"], config=config)
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    configuration = client.get(f"/pipeline/runs/retail/{run_id}/config", headers=headers["viewer"])
    metadata = client.get(f"/pipeline/runs/retail/{run_id}", headers=headers["viewer"])
    text = response.text + configuration.text + metadata.text
    assert private not in text and key not in text and roots["runs_dir"] not in text
    directory = RunStore(roots["runs_dir"]).directory("retail", run_id)
    for file in directory.glob("*.json"):
        content = file.read_text()
        assert private not in content and key not in content and roots["runs_dir"] not in content


def test_corrupt_commit_is_diagnosed_instead_of_falling_back(api):
    client, headers, roots, _ = api
    _run(client, headers["admin"], 10.0)
    run_id = _run(client, headers["admin"], 20.0).json()["run_id"]
    directory = RunStore(roots["runs_dir"]).directory("retail", run_id)
    (directory / "forecast.parquet").write_bytes(b"synthetic corruption")
    for url in ("/forecast/retail", "/pipeline/runs?lob=retail", f"/pipeline/runs/retail/{run_id}"):
        result = client.get(url, headers=headers["viewer"])
        assert result.status_code == 500
        assert result.json()["detail"]["code"] == "run_integrity_error"


def test_offline_copy_restore_uses_exact_immutable_identity(api, tmp_path):
    client, headers, roots, _ = api
    run_id = _run(client, headers["admin"]).json()["run_id"]
    original = RunStore(roots["runs_dir"])
    restored_root = tmp_path / "isolated-restore"
    shutil.copytree(roots["runs_dir"], restored_root)
    restored = RunStore(restored_root)
    assert restored.get("retail", run_id) == original.get("retail", run_id)
    assert restored.frame("retail", run_id, "forecast").to_dicts() == original.frame("retail", run_id, "forecast").to_dicts()
    assert restored.json_member("retail", run_id, "config") == original.json_member("retail", run_id, "config")


def test_spark_local_adapter_preserves_forecasts_without_claiming_a_run_bundle(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace

    import pandas as pd

    from src.fabric.deployment import DeploymentOrchestrator
    from src.pipeline import run_store
    from src.pipeline.readers import RunReader

    class LocalFrame:
        def __init__(self, value):
            self.value = value

        def toPandas(self):
            return pd.DataFrame({
                "series_id": ["synthetic"], "week": [date(2024, 1, 8)], "forecast": [self.value],
            })

    config = PlatformConfig(lob="retail")
    config.output.forecast_path = str(tmp_path / "legacy-forecasts")
    deploy = DeploymentOrchestrator(None, config, SimpleNamespace(lob="retail", workspace=""))
    identities = iter(["f" * 32, "0" * 32])
    monkeypatch.setattr(run_store, "new_run_id", lambda: next(identities))
    deploy._write_forecasts(LocalFrame(10.0), "2024-01-01", "1" * 32)
    deploy._write_forecasts(LocalFrame(20.0), "2024-01-01", "2" * 32)
    files = list((tmp_path / "legacy-forecasts" / "retail").glob("forecast_*.parquet"))
    assert len(files) == 2
    assert {pl.read_parquet(file)["forecast"][0] for file in files} == {10.0, 20.0}
    for file in files:
        timestamp = 1_700_000_000_000_000_000 + int(pl.read_parquet(file)["forecast"][0]) * 1_000_000
        os.utime(file, ns=(timestamp, timestamp))
    selected = RunReader(tmp_path / "canonical-runs", tmp_path / "legacy-forecasts").forecast("retail")
    assert selected.source == "legacy" and selected.run_id is None
    assert selected.forecast_origin == "2024-01-01"
    assert selected.frame["forecast"].to_list() == [20.0]


def test_bi_export_retains_both_run_identities_without_long_filenames(tmp_path):
    from src.analytics.bi_export import BIExporter

    frame = pl.DataFrame({"series_id": ["synthetic"], "week": [date(2024, 1, 8)]})
    exporter = BIExporter(str(tmp_path / "exports"))
    paths = [
        exporter.export_forecast_vs_actual(
            frame.with_columns(pl.lit(10.0).alias("forecast")),
            frame.with_columns(pl.lit(12.0).alias("quantity")),
            "retail", run_id="1" * 32, actuals_run_id="2" * 32,
        )
        for _ in range(2)
    ]
    assert paths[0] != paths[1]
    for path in paths:
        saved = pl.read_parquet(path, hive_partitioning=False)
        assert saved["source_run_id"].to_list() == ["1" * 32]
        assert saved["actuals_run_id"].to_list() == ["2" * 32]
        assert saved["forecast"].to_list() == [10.0]
        assert saved["quantity"].to_list() == [12.0]
        assert len(path.name) == 10 + 1 + 32 + len(".parquet")


@pytest.mark.parametrize("time_column", ["week", "date", "ds", "timestamp", "month", "quarter"])
def test_legacy_history_aliases_preserve_actual_http_values(api, time_column):
    client, headers, roots, _ = api
    directory = Path(roots["data_dir"]) / "history" / "retail"
    directory.mkdir(parents=True)
    pl.DataFrame({
        "series_id": ["synthetic", "synthetic"],
        time_column: [date(2024, 4, 1), date(2024, 1, 1)],
        "y": [31.5, 12.0],
    }).write_parquet(directory / "actuals.parquet")
    response = client.get("/series/retail/synthetic/history?legacy=true", headers=headers["viewer"])
    assert response.status_code == 200, response.text
    assert response.json()["run_id"] is None
    assert response.json()["artifact_source"] == "legacy"
    assert response.json()["points"] == [
        {"week": "2024-01-01", "value": 12.0},
        {"week": "2024-04-01", "value": 31.5},
    ]


@pytest.mark.parametrize("time_column", ["week", "date", "ds", "timestamp", "month", "quarter"])
def test_legacy_series_aliases_preserve_classification_inputs(api, time_column):
    client, headers, roots, _ = api
    directory = Path(roots["data_dir"]) / "history" / "retail"
    directory.mkdir(parents=True)
    pl.DataFrame({
        "series_id": ["synthetic"] * 12,
        time_column: [date(2024, 1, 1) + timedelta(days=index) for index in range(12)],
        "y": [float(index + 1) for index in range(12)],
    }).write_parquet(directory / "actuals.parquet")
    response = client.get("/series/retail?legacy=true", headers=headers["viewer"])
    assert response.status_code == 200, response.text
    assert response.json()["series_count"] == 1
    observed = response.json()["series"][0]
    assert observed["series_id"] == "synthetic" and observed["n_observations"] == 12
    assert observed["adi"] == 1.0 and observed["demand_class"] == "smooth"


def test_legacy_history_keeps_original_column_precedence(api):
    client, headers, roots, _ = api
    directory = Path(roots["data_dir"]) / "history" / "retail"
    directory.mkdir(parents=True)
    pl.DataFrame({
        "sku_id": ["synthetic"], "item_id": ["not-selected"],
        "ds": [date(2024, 1, 1)], "timestamp": [date(2025, 1, 1)],
        "period": [date(2026, 1, 1)], "value": [12.5], "sales": [999.0], "y": [-999.0],
    }).write_parquet(directory / "actuals.parquet")
    response = client.get("/series/retail/synthetic/history?legacy=true", headers=headers["viewer"])
    assert response.status_code == 200
    assert response.json()["points"] == [{"week": "2024-01-01", "value": 12.5}]


def _capture_explanation_provider(monkeypatch):
    from src.ai import nl_query

    engine = nl_query.NaturalLanguageQueryEngine.__new__(nl_query.NaturalLanguageQueryEngine)
    engine._client = object()
    captured = {}
    gather = engine._gather_context

    def gather_context(*args):
        context, sources = gather(*args)
        captured["metrics"] = args[3]
        return context, sources

    def provider(system_prompt, user_prompt):
        captured["prompt"] = user_prompt
        return "### ANSWER\nSynthetic response.\n### CONFIDENCE\nlow\n### SUPPORTING_DATA\n{}"

    monkeypatch.setattr(engine, "_gather_context", gather_context)
    monkeypatch.setattr(engine, "_call_claude", provider)
    monkeypatch.setattr(nl_query, "NaturalLanguageQueryEngine", lambda: engine)
    return captured


def test_legacy_ai_never_attaches_unbound_committed_accuracy(api, monkeypatch):
    client, headers, roots, _ = api
    data = Path(roots["data_dir"])
    history = data / "history" / "retail"
    forecast = data / "forecasts" / "retail"
    history.mkdir(parents=True)
    forecast.mkdir(parents=True)
    _actuals(10.0).write_parquet(history / "actuals.parquet")
    _actuals(10.0).rename({"quantity": "forecast"}).write_parquet(
        forecast / "forecast_retail_2024-01-01.parquet"
    )
    unrelated = _run(client, headers["admin"], value=99.0, operation="backtest")
    assert unrelated.status_code == 200
    metrics = RunStore(roots["runs_dir"]).frame("retail", unrelated.json()["run_id"], "metrics")
    assert metrics["actual"].unique().to_list() == [99.0]
    captured = _capture_explanation_provider(monkeypatch)
    response = client.post("/ai/explain", headers=headers["viewer"], json={
        "lob": "retail", "series_id": "synthetic", "question": "Explain this forecast.", "legacy": True,
    })
    assert response.status_code == 200
    assert captured["metrics"] is None
    assert '"metrics"' not in captured["prompt"]
    assert '"lineage"' in captured["prompt"] and '"unknown"' in captured["prompt"]
    assert "metrics" not in response.json()["sources_used"]
    assert {"history", "forecast"}.issubset(response.json()["sources_used"])


def test_committed_ai_keeps_only_matching_run_accuracy(api, monkeypatch):
    client, headers, _, _ = api
    assert _run(client, headers["admin"], value=10.0, operation="backtest").status_code == 200
    selected = _run(client, headers["admin"], value=10.0)
    assert selected.status_code == 200
    assert _run(client, headers["admin"], value=99.0, operation="backtest").status_code == 200
    captured = _capture_explanation_provider(monkeypatch)
    response = client.post("/ai/explain", headers=headers["viewer"], json={
        "lob": "retail", "series_id": "synthetic", "question": "Explain this forecast.",
        "run_id": selected.json()["run_id"],
    })
    assert response.status_code == 200
    assert captured["metrics"]["actual"].unique().to_list() == [10.0]
    assert "metrics" in response.json()["sources_used"]
