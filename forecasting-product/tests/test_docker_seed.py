"""Seeder lifecycle tests; all HTTP and child-process boundaries are local mocks."""

import csv
import importlib.util
import io
import json
import re
import subprocess
import sys
import urllib.error
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def seeder(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts" / "docker_seed.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("API_DATA_DIR", "state")
    monkeypatch.setenv("API_METRICS_DIR", "state/metrics")
    monkeypatch.setenv("API_RUNS_DIR", "state/runs")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("API_DEV_MODE", "true")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    spec = importlib.util.spec_from_file_location("seed_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    demo = tmp_path / "state" / "demo" / "m5_weekly.csv"
    demo.parent.mkdir(parents=True)
    demo.write_text("series_id,date,quantity\nS1,2024-01-01,10\n")
    return module


def _mock_server(seeder, monkeypatch):
    process = MagicMock()
    process.poll.return_value = None
    process.wait.return_value = 0
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(seeder.subprocess, "Popen", popen)
    monkeypatch.setattr(seeder, "wait_for_api", lambda **kwargs: True)
    return process, popen


def test_cold_and_restart_use_identical_absolute_storage(seeder, tmp_path, monkeypatch):
    _, popen = _mock_server(seeder, monkeypatch)
    seed = MagicMock(return_value="naive_seasonal")
    monkeypatch.setattr(seeder, "seed_via_api", seed)
    seeder.main()
    cold_command = popen.call_args.args[0]
    assert cold_command[cold_command.index("--data-dir") + 1] == str(tmp_path / "state")
    assert cold_command[cold_command.index("--metrics-dir") + 1] == str(tmp_path / "state" / "metrics")
    assert cold_command[cold_command.index("--runs-dir") + 1] == str(tmp_path / "state" / "runs")
    assert (tmp_path / "state" / ".seeded").read_text().endswith("champion=naive_seasonal\n")

    seeder.main()
    assert popen.call_args.args[0] == cold_command
    assert popen.call_count == 2
    seed.assert_called_once()


def test_failed_seed_stops_and_reaps_server_without_sentinel(seeder, tmp_path, monkeypatch):
    process, _ = _mock_server(seeder, monkeypatch)
    monkeypatch.setattr(seeder, "seed_via_api", MagicMock(side_effect=RuntimeError("forecast failed")))
    with pytest.raises(RuntimeError, match="forecast failed"):
        seeder.main()
    process.terminate.assert_called_once()
    process.wait.assert_called()
    assert not (tmp_path / "state" / ".seeded").exists()


def test_unhealthy_server_is_stopped_and_reaped(seeder, monkeypatch):
    process, _ = _mock_server(seeder, monkeypatch)
    monkeypatch.setattr(seeder, "wait_for_api", lambda **kwargs: False)
    with pytest.raises((RuntimeError, SystemExit)):
        seeder.main()
    process.terminate.assert_called_once()
    process.wait.assert_called()


def test_server_that_ignores_termination_is_killed_and_reaped(seeder, monkeypatch):
    process, _ = _mock_server(seeder, monkeypatch)
    process.wait.side_effect = [subprocess.TimeoutExpired("serve", 10), 0]
    monkeypatch.setattr(seeder, "seed_via_api", MagicMock(side_effect=RuntimeError("failed")))
    with pytest.raises(RuntimeError, match="failed"):
        seeder.main()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


@pytest.mark.parametrize(
    "backtest,forecast",
    [
        ({"status": "failed", "champion_model": "naive_seasonal"}, {}),
        ({"status": "completed"}, {}),
        ({"status": "completed", "champion_model": "naive_seasonal"}, {"status": "failed"}),
        ({"status": "completed", "champion_model": "naive_seasonal"}, {"status": "completed", "forecast_rows": 0}),
    ],
)
def test_incomplete_pipeline_response_is_not_seed_success(seeder, tmp_path, monkeypatch, backtest, forecast):
    csv = tmp_path / "input.csv"
    config = tmp_path / "config.yaml"
    csv.write_text("x")
    config.write_text("lob: demo")
    responses = [io.BytesIO(json.dumps(value).encode()) for value in (backtest, forecast)]
    monkeypatch.setattr("urllib.request.urlopen", MagicMock(side_effect=responses))
    with pytest.raises(RuntimeError):
        seeder.seed_via_api(csv, config, "demo")


def test_forecast_http_failure_is_not_seed_success(seeder, tmp_path, monkeypatch):
    csv = tmp_path / "input.csv"
    config = tmp_path / "config.yaml"
    csv.write_text("x")
    config.write_text("lob: demo")
    responses = [
        io.BytesIO(b'{"status":"completed","champion_model":"naive_seasonal"}'),
        urllib.error.HTTPError("http://127.0.0.1", 500, "forecast failed", {}, None),
    ]
    monkeypatch.setattr("urllib.request.urlopen", MagicMock(side_effect=responses))
    with pytest.raises(RuntimeError, match="Forecast"):
        seeder.seed_via_api(csv, config, "demo")


def test_seed_requires_both_successful_responses(seeder, tmp_path, monkeypatch):
    csv = tmp_path / "input.csv"
    config = tmp_path / "config.yaml"
    csv.write_text("x")
    config.write_text("lob: demo")
    responses = [
        io.BytesIO(b'{"status":"completed","champion_model":"naive_seasonal"}'),
        io.BytesIO(b'{"status":"completed","forecast_rows":12}'),
    ]
    urlopen = MagicMock(side_effect=responses)
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert seeder.seed_via_api(csv, config, "demo") == "naive_seasonal"
    assert urlopen.call_count == 2


def test_readiness_closes_http_response(seeder, monkeypatch):
    response = MagicMock()
    monkeypatch.setattr("urllib.request.urlopen", MagicMock(return_value=response))
    assert seeder.wait_for_api(timeout=1)
    response.__exit__.assert_called_once()


@pytest.mark.parametrize("fault", [None, "crossed", "missing", "null", "infinite", "point_mismatch"])
def test_docker_smoke_checks_actual_interval_invariants(tmp_path, fault):
    source = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "docker-smoke.sh"
    validation = source.read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    forecast_id, backtest_id = "a" * 32, "b" * 32
    points = [{
        "series_id": f"series-{series}", "week": f"2026-01-{period + 1:02d}",
        "forecast": 10.0, "forecast_p10": 9.0, "forecast_p50": 10.0,
        "forecast_p90": 11.0, "model": "naive_seasonal",
    } for series in range(10) for period in range(12)]
    if fault == "crossed":
        points[0].update(forecast=6.0, forecast_p50=6.0, forecast_p10=11.0, forecast_p90=11.0)
    elif fault == "missing":
        points[0].pop("forecast_p50")
    elif fault == "null":
        points[0]["forecast_p50"] = None
    elif fault == "infinite":
        points[0]["forecast_p90"] = float("inf")
    elif fault == "point_mismatch":
        points[0]["forecast_p50"] = 10.5
    documents = {
        "health": {"status": "ok", "checks": {"storage": True}},
        "forecast": {"series_count": 10, "points": points, "artifact_source": "committed", "run_id": forecast_id},
        "leaderboard": {"entries": [{"model_id": "naive_seasonal"}], "artifact_source": "committed", "run_id": backtest_id},
        "runs": {"runs": [
            {"run_type": "forecast", "run_id": forecast_id},
            {"run_type": "backtest", "run_id": backtest_id},
        ]},
    }
    for name, document in documents.items():
        (tmp_path / f"cold-{name}.json").write_text(json.dumps(document))
    result = subprocess.run(
        [sys.executable, "-", str(tmp_path), "cold"], input=validation,
        text=True, capture_output=True, timeout=30,
    )
    if fault is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0, "Invalid intervals must fail the actual Docker smoke assertion"


@pytest.mark.parametrize("fault", [None, "other_model", "point_shift", "wrong_run", "duplicate_key", "wrong_horizon"])
def test_explicit_naive_smoke_requires_its_model_points_identity_and_config(tmp_path, fault):
    source = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "docker-smoke.sh"
    validation = source.read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    new_id, old_id, backtest_id = "c" * 32, "a" * 32, "b" * 32
    start = date(2024, 1, 1)
    points = []
    with (tmp_path / "explicit-naive-input.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["series_id", "date", "quantity"])
        for series in range(10):
            writer.writerows(
                (f"series-{series}", (start + timedelta(weeks=period)).isoformat(), series * 100 + period)
                for period in range(64)
            )
            for period in range(12):
                value = float(series * 100 + 12 + period)
                points.append({
                    "series_id": f"series-{series}", "week": (start + timedelta(weeks=64 + period)).isoformat(),
                    "forecast": value, "forecast_p10": value - 1, "forecast_p50": value,
                    "forecast_p90": value + 1, "model": "naive_seasonal",
                })
    if fault == "other_model":
        for point in points:
            point["model"] = "auto_ets"
    elif fault == "point_shift":
        for column in ("forecast", "forecast_p10", "forecast_p50", "forecast_p90"):
            points[0][column] += 1
    elif fault == "duplicate_key":
        points[0] = dict(points[1])
    documents = {
        "health": {"status": "ok", "checks": {"storage": True}},
        "forecast": {
            "series_count": 10, "points": points, "artifact_source": "committed",
            "run_id": old_id if fault == "wrong_run" else new_id, "config_hash": "d" * 64,
        },
        "leaderboard": {"entries": [{"model_id": "auto_ets"}], "artifact_source": "committed", "run_id": backtest_id},
        "runs": {"runs": [
            {"run_type": "forecast", "run_id": new_id},
            {"run_type": "forecast", "run_id": old_id},
            {"run_type": "backtest", "run_id": backtest_id},
        ]},
        "config": {
            "run_id": new_id, "config_hash": "d" * 64,
            "configuration": {"settings": {"forecast": {
                "frequency": "W", "horizon_weeks": 11 if fault == "wrong_horizon" else 12,
            }}},
        },
    }
    for name, document in documents.items():
        (tmp_path / f"naive-{name}.json").write_text(json.dumps(document))
    result = subprocess.run(
        [sys.executable, "-", str(tmp_path), "naive", new_id], input=validation,
        text=True, capture_output=True, timeout=30,
    )
    if fault is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0, "The explicit flow cannot pass on a different model/run or changed points"


@pytest.mark.parametrize("fault", [None, "old_artifact_changed", "no_new_bundle"])
def test_explicit_naive_smoke_requires_a_new_bundle_and_immutable_prior_artifacts(tmp_path, fault):
    source = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "docker-smoke.sh"
    programs = re.findall(r"<<'PY'\n(.*?)\nPY", source.read_text(), re.DOTALL)
    validation, = [program for program in programs if "prior.keys() < after.keys()" in program]
    prior = {"runs/old/forecast.parquet": "a" * 64, "runs/old/manifest.json": "b" * 64}
    after = {**prior, "runs/new/forecast.parquet": "c" * 64, "runs/new/manifest.json": "d" * 64}
    if fault == "old_artifact_changed":
        after["runs/old/forecast.parquet"] = "e" * 64
    elif fault == "no_new_bundle":
        after = dict(prior)
    (tmp_path / "restart-artifacts.json").write_text(json.dumps(prior))
    (tmp_path / "naive-artifacts.json").write_text(json.dumps(after))
    result = subprocess.run(
        [sys.executable, "-", str(tmp_path)], input=validation,
        text=True, capture_output=True, timeout=30,
    )
    if fault is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0, "Publishing Naive cannot replace a previous artifact or reuse its bundle"
