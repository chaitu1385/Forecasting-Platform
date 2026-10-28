"""Shipped CLI and custom/cross-device roots use one explicitly chosen authority."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.loader import parse_config_yaml
from src.pipeline.backtest import BacktestPipeline
from src.pipeline.run_store import RunStore
from test_run_artifacts import _actuals, _config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("operation", ["forecast", "backtest"])
@pytest.mark.parametrize("explicit", [False, True])
def test_shipped_cli_preserves_custom_roots_and_exact_run_identity(tmp_path, operation, explicit):
    settings = _config()
    settings["forecast"]["forecasters"] = ["croston"]
    settings["output"] = {
        "forecast_path": str(tmp_path / "custom-forecast"),
        "metrics_path": str(tmp_path / "custom-metrics"),
    }
    config = tmp_path / "config.yaml"
    data = tmp_path / "input.csv"
    config.write_text(yaml.safe_dump(settings))
    _actuals(10.0).write_csv(data)
    root = tmp_path / "shared-authority" if explicit else Path(
        settings["output"]["forecast_path" if operation == "forecast" else "metrics_path"]
    ) / "runs"
    command = [
        sys.executable, str(ROOT / "scripts" / f"run_{operation}.py"),
        "--config", str(config), "--data", str(data),
    ]
    if explicit:
        command.extend(["--runs-dir", str(root)])
    environment = {
        **os.environ, "POLARS_MAX_THREADS": "2", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
    }
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    store = RunStore(root)
    manifests = store.list("retail", operation)
    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.run_id in result.stdout
    assert manifest.source["entrypoint"] == "cli"
    assert manifest.champion_model_id == "croston"
    assert not (tmp_path / "data" / "runs").exists()
    inspect = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "inspect_runs.py"), "--runs-dir", str(root),
         "--lob", "retail", "--run-id", manifest.run_id],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert inspect.returncode == 0, inspect.stderr
    assert manifest.run_id in inspect.stdout


def test_explicit_authority_does_not_depend_on_legacy_root_filesystems(tmp_path, record_property):
    candidates = []
    if os.name == "posix":
        candidates.append(Path("/dev/shm"))
    if os.environ.get("RUNNER_TEMP"):
        candidates.append(Path(os.environ["RUNNER_TEMP"]))
    other_parent = next((
        path for path in candidates
        if path.is_dir() and path.stat().st_dev != tmp_path.stat().st_dev
    ), None)
    record_property("cross_device_exercised", other_parent is not None)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        assert other_parent is not None, "Hosted filesystem gate requires a real second temporary filesystem"
    with tempfile.TemporaryDirectory(prefix="forecast-other-device-", dir=other_parent or tmp_path) as directory:
        settings = _config()
        settings["output"] = {
            "forecast_path": str(tmp_path / "unused-forecast-root"),
            "metrics_path": str(Path(directory) / "unused-metric-root"),
            "runs_path": str(tmp_path / "authority"),
        }
        config = parse_config_yaml(yaml.safe_dump(settings))
        assert config.output.runs_path == settings["output"]["runs_path"]
        pipeline = BacktestPipeline(config)
        result = pipeline.run(_actuals())
        manifest = RunStore(config.output.runs_path).get("retail", result["run_id"], "backtest")
        assert manifest.run_id == pipeline.context.run_id
        assert not Path(config.output.forecast_path).exists()
        assert not Path(config.output.metrics_path).exists()
