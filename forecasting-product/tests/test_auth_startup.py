"""Authentication checks against the shipped launcher, not a mocked app factory."""

import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.auth.token import create_token


ROOT = Path(__file__).resolve().parents[1]


def _request(base, path, *, body=None, token=None, extra_headers=None):
    headers = dict(extra_headers or {})
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    content = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        content = json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=content, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _stop(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=True, capture_output=True,
        )
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


@contextmanager
def _process(tmp_path, settings, *, extra_args=(), entrypoint="serve.py"):
    data = tmp_path / "data"
    metrics = data / "metrics"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {
        **os.environ,
        "POLARS_MAX_THREADS": "2",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "PYTHONUTF8": "1",
        "API_RATE_LIMIT": "1000",
        "API_HOST": "127.0.0.1",
        "API_PORT": str(port),
        "API_DATA_DIR": str(data),
        "API_METRICS_DIR": str(metrics),
    }
    for key in ("AUTH_ENABLED", "JWT_SECRET_KEY", "API_DEV_MODE"):
        env.pop(key, None)
    env.update(settings)
    base = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        command = [sys.executable, str(ROOT / "scripts" / entrypoint)]
        if entrypoint == "serve.py":
            command.extend([
                "--host", "127.0.0.1", "--port", str(port),
                "--data-dir", str(data), "--metrics-dir", str(metrics),
                *extra_args,
            ])
        process = subprocess.Popen(
            command,
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        try:
            yield process, base, data, log_path
        finally:
            _stop(process)
            log.flush()
            secret = settings.get("JWT_SECRET_KEY")
            if secret and secret in log_path.read_text(encoding="utf-8"):
                pytest.fail("Signing configuration was exposed in server logs")


@contextmanager
def _server(tmp_path, settings, *, extra_args=(), entrypoint="serve.py"):
    with _process(tmp_path, settings, extra_args=extra_args, entrypoint=entrypoint) as (process, base, data, _log):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail("Shipped API launcher exited before readiness")
            try:
                status, _ = _request(base, "/health")
                if status == 200:
                    break
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.1)
        else:
            pytest.fail("Shipped API launcher did not become ready within 45 seconds")
        yield base, data


def test_serve_enforces_documented_auth_environment(tmp_path):
    settings = {
        "AUTH_ENABLED": "true",
        "JWT_SECRET_KEY": secrets.token_urlsafe(48),
        "API_DEV_MODE": "0",
    }
    with _server(tmp_path, settings) as (base, data):
        status, _ = _request(
            base, "/overrides",
            body={"old_sku": "SYNTHETIC_OLD", "new_sku": "SYNTHETIC_NEW", "proportion": 0.5},
        )
        assert status == 401
        assert not list(data.glob("overrides*"))


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"AUTH_ENABLED": "false"},
        {"AUTH_ENABLED": "false", "API_DEV_MODE": "false"},
        {"AUTH_ENABLED": "true", "API_DEV_MODE": "true"},
        {"AUTH_ENABLED": "invalid"},
        {"API_DEV_MODE": "not-a-boolean"},
        {"JWT_SECRET_KEY": "weak"},
    ],
)
def test_invalid_production_configuration_exits_without_opening_api(tmp_path, settings):
    with _process(tmp_path, settings) as (process, _base, data, _log):
        try:
            result = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail("Invalid authentication configuration did not stop the launcher")
        assert result != 0
        assert not data.exists()


@pytest.mark.parametrize("extra_args", [(), ("--workers", "2"), ("--reload",)])
def test_default_production_and_worker_launches_use_verified_identity(tmp_path, extra_args):
    key = secrets.token_urlsafe(48)
    with _server(tmp_path, {"JWT_SECRET_KEY": key}, extra_args=extra_args) as (base, _data):
        assert _request(base, "/health")[1]["status"] == "ok"
        assert _request(base, "/auth/status") == (200, {"mode": "production", "auth_required": True})
        assert _request(base, "/auth/me")[0] == 401
        admin = create_token("synthetic-admin", "", "admin", key, expiry_hours=1)
        viewer = create_token("synthetic-viewer", "", "viewer", key, expiry_hours=1)
        for token, role in ((admin, "admin"), (viewer, "viewer")):
            status, identity = _request(base, "/auth/me", token=token)
            assert status == 200 and identity["role"] == role
        assert _request(base, "/overrides", token=viewer, body={
            "old_sku": "SYNTHETIC_OLD", "new_sku": "SYNTHETIC_NEW", "proportion": 0.5,
        })[0] == 403
        assert _request(base, "/overrides", token=admin, body={
            "old_sku": "SYNTHETIC_OLD", "new_sku": "SYNTHETIC_NEW", "proportion": 0.5,
        })[0] == 200


def test_explicit_demo_launcher_supports_only_simulated_roles(tmp_path):
    with _server(tmp_path, {}, extra_args=("--demo",)) as (base, _data):
        assert _request(base, "/auth/status") == (200, {"mode": "demo", "auth_required": False})
        status, identity = _request(base, "/auth/me", extra_headers={
            "X-Demo-User": "demo-viewer", "X-Demo-Role": "viewer",
        })
        assert status == 200 and identity["role"] == "viewer"
        assert _request(base, "/auth/token?username=demo&role=admin", body={})[0] == 404


@pytest.mark.parametrize("settings", [
    {"AUTH_ENABLED": "true"},
    {"API_DEV_MODE": "false"},
    {"AUTH_ENABLED": "invalid"},
    {"AUTH_ENABLED": "true", "API_DEV_MODE": "false", "JWT_SECRET_KEY": "weak"},
])
def test_demo_flag_does_not_ignore_contradictory_environment(tmp_path, settings):
    with _process(tmp_path, settings, extra_args=("--demo",)) as (process, _base, _data, _log):
        assert process.wait(timeout=15) != 0


def test_production_seeder_launches_secure_api_without_demo_data(tmp_path):
    key = secrets.token_urlsafe(48)
    with _server(tmp_path, {"JWT_SECRET_KEY": key}, entrypoint="docker_seed.py") as (base, data):
        assert _request(base, "/auth/status") == (200, {"mode": "production", "auth_required": True})
        assert _request(base, "/auth/me")[0] == 401
        assert not (data / ".seeded").exists()
        assert not (data / "demo").exists()


def test_seeder_rejects_missing_secret_even_with_existing_marker(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    marker = data / ".seeded"
    marker.write_text("existing-marker")
    with _process(tmp_path, {}, entrypoint="docker_seed.py") as (process, _base, _data, _log):
        assert process.wait(timeout=15) != 0
    assert marker.read_text() == "existing-marker"
