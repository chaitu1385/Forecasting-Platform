"""Verify the shipped production container with fresh synthetic credentials/data."""

import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main() -> None:
    image = sys.argv[1]
    artifacts = Path(sys.argv[2] if len(sys.argv) > 2 else "docker-smoke-results")
    artifacts.mkdir(parents=True, exist_ok=True)
    report = artifacts / "production-auth.json"
    if report.exists():
        raise FileExistsError("Refusing to overwrite an existing authentication smoke report")
    name = "fp-auth-" + uuid.uuid4().hex
    volume = name + "-data"
    owned_container = False
    owned_volume = False
    key = secrets.token_urlsafe(48)
    private_values = [key]
    results = {}

    def safe(text: str) -> str:
        if any(value in text for value in private_values):
            raise RuntimeError("Credential material was exposed; refusing to publish diagnostics")
        return text

    def docker(*args, environment=None, input_text=None, check=True, timeout=60):
        response = subprocess.run(
            ["docker", *args], input=input_text, env=environment,
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        if check and response.returncode:
            raise RuntimeError(f"Docker {args[0]} failed: {safe(response.stderr)}")
        return response

    def remove_container(label):
        nonlocal owned_container
        if not owned_container:
            return
        try:
            response = docker("logs", name)
            log = response.stdout + response.stderr
            (artifacts / f"auth-{label}.log").write_text(safe(log), encoding="utf-8")
        finally:
            docker("stop", "--time", "15", name)
            docker("rm", name)
            owned_container = False

    def start(settings, persistent=False):
        nonlocal owned_container
        command = ["run", "-d", "--name", name, "--cpus=2", "--memory=2g", "-p", "127.0.0.1::8000"]
        environment = dict(os.environ)
        environment.pop("JWT_SECRET_KEY", None)
        for setting, value in settings.items():
            if setting == "JWT_SECRET_KEY":
                environment[setting] = value
                command.extend(["--env", setting])
            else:
                command.extend(["--env", f"{setting}={value}"])
        if persistent:
            command.extend(["--mount", f"type=volume,source={volume},target=/app/forecasting-product/data"])
        owned_container = True
        docker(*command, image, environment=environment)

    def request(base, endpoint, token=None, body=None):
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            base + endpoint, headers=headers,
            data=None if body is None else json.dumps(body).encode(),
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def ready():
        port = docker("port", name, "8000/tcp").stdout.strip()
        base = f"http://{port}"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if docker("inspect", "--format", "{{.State.Running}}", name).stdout.strip() != "true":
                raise RuntimeError("Production container exited before readiness")
            try:
                status, health = request(base, "/health")
                if status == 200:
                    assert health["status"] == "ok" and all(health["checks"].values())
                    assert set(health) == {"status", "version", "checks"}
                    return base
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        raise TimeoutError("Production container did not become ready within 90 seconds")

    try:
        if docker("container", "inspect", name, check=False).returncode == 0:
            raise RuntimeError("Refusing to reuse an existing smoke container")
        if docker("volume", "inspect", volume, check=False).returncode == 0:
            raise RuntimeError("Refusing to reuse an existing smoke volume")
        invalid = {
            "missing-secret": {},
            "weak-secret": {"JWT_SECRET_KEY": "weak"},
            "auth-disabled-without-demo": {"AUTH_ENABLED": "false"},
            "malformed-boolean": {"API_DEV_MODE": "false-ish"},
        }
        for label, settings in invalid.items():
            start(settings)
            code = int(docker("wait", name, timeout=30).stdout.strip())
            assert code != 0, "Invalid authentication configuration started a successful container"
            results[label] = {"exit_code": code}
            remove_container(label)

        docker("volume", "create", volume)
        owned_volume = True
        start({"JWT_SECRET_KEY": key}, persistent=True)
        base = ready()
        assert request(base, "/auth/status") == (200, {"mode": "production", "auth_required": True})
        assert request(base, "/auth/me")[0] == 401
        assert request(base, "/auth/me", token="not-a-valid-token")[0] == 401
        assert request(base, "/auth/token?username=untrusted&role=admin", body={})[0] == 404
        docker("exec", "-i", name, "python", "-", input_text="""
from datetime import date
from pathlib import Path
import polars as pl
root = Path('/app/forecasting-product/data')
assert not (root / '.seeded').exists() and not (root / 'demo').exists()
output = root / 'forecasts' / 'auth_smoke'
output.mkdir(parents=True)
pl.DataFrame({'series_id': ['SYNTHETIC'], 'week': [date(2024, 1, 1)],
              'forecast': [10.0], 'model': ['naive_seasonal']}).write_parquet(
                  output / 'forecast_auth_smoke_2024-01-01.parquet')
""")
        with tempfile.TemporaryDirectory(prefix="forecast-auth-private-") as temporary:
            tokens = {}
            for role in ("admin", "viewer"):
                issuance = docker(
                    "exec", name, "python", "forecasting-product/scripts/issue_token.py",
                    "--user-id", f"smoke-{role}", "--role", role, "--expiry-hours", "1",
                    "--output", f"/tmp/smoke-{role}.jwt",
                )
                output = Path(temporary) / f"{role}.jwt"
                docker("cp", f"{name}:/tmp/smoke-{role}.jwt", str(output))
                tokens[role] = output.read_text(encoding="utf-8").strip()
                private_values.append(tokens[role])
                safe(issuance.stdout + issuance.stderr)
            body = {"old_sku": "SYNTHETIC_OLD", "new_sku": "SYNTHETIC_NEW", "proportion": 0.5}
            for role, token in tokens.items():
                status, identity = request(base, "/auth/me", token)
                assert status == 200 and identity["role"] == role and identity["username"] == f"smoke-{role}"
                assert request(base, "/forecast/auth_smoke", token)[0] == 200
            assert request(base, "/overrides", tokens["viewer"], body)[0] == 403
            assert request(base, "/overrides", tokens["admin"], body)[0] == 200
            assert request(base, "/forecast/auth_smoke")[0] == 401
            results["production"] = {
                "mode": "production", "anonymous": 401, "invalid_token": 401,
                "network_issuer": 404, "admin_write": 200, "viewer_read": 200, "viewer_write": 403,
                "demo_seeding": False, "operator_output": "private-file-only",
            }
            remove_container("production")
            start({"JWT_SECRET_KEY": key}, persistent=True)
            base = ready()
            status, forecast = request(base, "/forecast/auth_smoke", tokens["viewer"])
            assert status == 200 and forecast["points"][0]["forecast"] == 10.0
            assert request(base, "/forecast/auth_smoke")[0] == 401
            results["production_restart"] = {"authenticated_read": 200, "anonymous": 401}
        with report.open("x", encoding="utf-8") as output:
            output.write(safe(json.dumps(results, indent=2)) + "\n")
        print("Production container: invalid configurations rejected; JWT roles and authenticated restart verified.")
    finally:
        try:
            remove_container("final")
        finally:
            if owned_volume:
                docker("volume", "rm", volume)


if __name__ == "__main__":
    main()
