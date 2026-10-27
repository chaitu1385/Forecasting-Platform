"""Real JWT/RBAC, explicit demo, startup settings and private operator output."""

import json
import os
import secrets
import stat
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import jwt
import polars as pl
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.auth.issuer import write_private_token
from src.auth.models import Permission, Role
from src.auth.rbac import get_current_user
from src.auth.settings import AuthSettings
from src.auth.token import TOKEN_AUDIENCE, TOKEN_ISSUER, create_token, decode_token


@pytest.fixture
def signing_key():
    return secrets.token_urlsafe(48)


@pytest.fixture
def production(tmp_path, signing_key):
    data = tmp_path / "data"
    app = create_app(
        data_dir=str(data), jwt_secret=signing_key,
        auth_enabled=True, dev_mode=False,
    )
    forecast_dir = data / "forecasts" / "auth_test"
    forecast_dir.mkdir(parents=True)
    pl.DataFrame({
        "series_id": ["SYNTHETIC"],
        "week": [date(2024, 1, 1)],
        "forecast": [10.0],
        "model": ["naive_seasonal"],
    }).write_parquet(forecast_dir / "forecast_auth_test_2024-01-01.parquet")
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, data


def _headers(token):
    return {"Authorization": f"Bearer {token}"}


def _raw_token(signing_key, **changes):
    now = int(time.time())
    payload = {
        "user_id": "synthetic-user", "email": "", "role": "viewer",
        "is_active": True, "iat": now, "exp": now + 3600,
        "iss": TOKEN_ISSUER, "aud": TOKEN_AUDIENCE,
    }
    payload.update(changes)
    return jwt.encode(payload, signing_key, algorithm="HS256")


def _assert_not_exposed(values, text):
    if any(value and value in text for value in values):
        pytest.fail("Authentication material was exposed")


@pytest.mark.parametrize("setting", ["AUTH_ENABLED", "API_DEV_MODE"])
@pytest.mark.parametrize("value", ["", "yes", "no", "enabled", "2", "none"])
def test_environment_booleans_are_strict(setting, value, signing_key):
    environment = {"AUTH_ENABLED": "true", "API_DEV_MODE": "false", "JWT_SECRET_KEY": signing_key}
    environment[setting] = value
    with pytest.raises(ValueError, match=setting):
        AuthSettings.from_env(environment=environment)


@pytest.mark.parametrize("enabled,disabled", [("true", "false"), ("1", "0"), ("TRUE", "FALSE")])
def test_supported_boolean_spellings(enabled, disabled, signing_key):
    settings = AuthSettings.from_env(environment={
        "AUTH_ENABLED": enabled, "API_DEV_MODE": disabled, "JWT_SECRET_KEY": signing_key,
    })
    assert settings.auth_enabled and not settings.dev_mode
    _assert_not_exposed([signing_key], repr(settings))


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"AUTH_ENABLED": "false"},
        {"AUTH_ENABLED": "false", "API_DEV_MODE": "0"},
        {"AUTH_ENABLED": "true", "API_DEV_MODE": "1"},
        {"API_DEV_MODE": "1"},
    ],
)
def test_missing_or_conflicting_settings_do_not_enable_demo(environment):
    with pytest.raises(ValueError):
        AuthSettings.from_env(environment=environment)


@pytest.mark.parametrize(
    "key",
    ["", "short", " " * 64, "x" * 64, "change-me-to-a-real-secret-before-production", "a b" * 32],
)
def test_weak_signing_configuration_is_rejected(key):
    with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
        AuthSettings.from_env(environment={"JWT_SECRET_KEY": key})


def test_explicit_demo_has_no_signing_key(signing_key):
    settings = AuthSettings.from_env(environment={"AUTH_ENABLED": "false", "API_DEV_MODE": "1"})
    assert settings.mode == "demo" and settings.jwt_secret == ""
    with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
        AuthSettings.from_env(environment={
            "AUTH_ENABLED": "false", "API_DEV_MODE": "true", "JWT_SECRET_KEY": signing_key,
        })


def test_programmatic_false_is_not_an_implicit_demo(tmp_path):
    with pytest.raises(ValueError, match="explicit demo"):
        create_app(data_dir=str(tmp_path / "uncreated"), auth_enabled=False, dev_mode=False, jwt_secret="")
    assert not (tmp_path / "uncreated").exists()


def test_factory_reads_the_same_documented_environment(tmp_path, signing_key, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_DEV_MODE", "0")
    monkeypatch.setenv("JWT_SECRET_KEY", signing_key)
    with TestClient(create_app(data_dir=str(tmp_path))) as client:
        assert client.get("/auth/status").json() == {"mode": "production", "auth_required": True}
        assert client.get("/auth/me").status_code == 401


def test_unconfigured_dependencies_never_infer_anonymous_admin():
    app = FastAPI()
    app.state.auth_enabled = False

    @app.get("/protected")
    async def protected(user=Depends(get_current_user)):
        return {"role": user.role}

    with TestClient(app) as client:
        assert client.get("/protected").status_code == 503


@pytest.mark.parametrize("path", ["/auth/me", "/forecast/auth_test", "/audit"])
def test_anonymous_production_requests_are_rejected(production, path):
    client, _ = production
    response = client.get(path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_public_mode_health_and_production_issuer_boundary(production, signing_key):
    client, _ = production
    assert client.get("/health").status_code == 200
    response = client.get("/auth/status")
    assert response.json() == {"mode": "production", "auth_required": True}
    _assert_not_exposed([signing_key], response.text + client.get("/health").text)
    assert client.post("/auth/token?username=untrusted&role=admin").status_code == 404


@pytest.mark.parametrize("role", list(Role))
def test_real_signed_role_matrix(production, signing_key, role):
    client, data = production
    token = create_token("synthetic-user", "", role.value, signing_key, expiry_hours=1)
    headers = _headers(token)
    identity = client.get("/auth/me", headers=headers)
    assert identity.status_code == 200
    assert identity.json()["role"] == role.value
    assert identity.json()["username"] == "synthetic-user"
    assert identity.json()["expires_at"] > time.time()
    _assert_not_exposed([token, signing_key], identity.text)
    assert client.get("/forecast/auth_test", headers=headers).status_code == 200
    response = client.post(
        "/overrides", headers=headers,
        json={"old_sku": "SYNTHETIC_OLD", "new_sku": "SYNTHETIC_NEW", "proportion": 0.5},
    )
    expected = 200 if role in {Role.ADMIN, Role.DATA_SCIENTIST, Role.PLANNER} else 403
    assert response.status_code == expected
    if expected == 403:
        assert not list(data.glob("overrides*"))


@pytest.mark.parametrize(
    "changes",
    [
        {"exp": 1}, {"role": "unknown"}, {"role": None}, {"user_id": ""},
        {"user_id": None}, {"is_active": "false"}, {"is_active": 1},
        {"exp": "9999999999"}, {"iat": []}, {"exp": True},
        {"aud": "another-service"}, {"iss": "another-issuer"},
        {"exp": int(time.time()) + 48 * 3600},
    ],
)
def test_invalid_signed_claims_are_not_defaulted_to_a_role(production, signing_key, changes):
    client, _ = production
    token = _raw_token(signing_key, **changes)
    assert client.get("/auth/me", headers=_headers(token)).status_code == 401


def test_missing_expiry_is_rejected(production, signing_key):
    client, _ = production
    payload = jwt.decode(_raw_token(signing_key), signing_key, algorithms=["HS256"], audience=TOKEN_AUDIENCE)
    del payload["exp"]
    token = jwt.encode(payload, signing_key, algorithm="HS256")
    assert client.get("/auth/me", headers=_headers(token)).status_code == 401


def test_expired_tampered_wrong_key_and_wrong_algorithm_tokens(production, signing_key, caplog):
    client, _ = production
    now = int(time.time())
    expired = _raw_token(signing_key, iat=now - 100, exp=now - 1)
    original = _raw_token(signing_key)
    parts = original.split(".")
    parts[2] = ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
    tampered = ".".join(parts)
    wrong_key = _raw_token(secrets.token_urlsafe(48))
    payload = jwt.decode(original, signing_key, algorithms=["HS256"], audience=TOKEN_AUDIENCE)
    wrong_algorithm = jwt.encode(payload, signing_key, algorithm="HS512")
    unsigned = jwt.encode(payload, "", algorithm="none")
    for token in (expired, tampered, wrong_key, wrong_algorithm, unsigned, "malformed"):
        response = client.get("/auth/me", headers=_headers(token))
        assert response.status_code == 401
        _assert_not_exposed([token, signing_key], response.text + caplog.text)


def test_inactive_identity_is_denied(production, signing_key):
    client, _ = production
    token = _raw_token(signing_key, is_active=False)
    assert client.get("/auth/me", headers=_headers(token)).status_code == 403


def test_demo_headers_never_override_production_identity(production, signing_key):
    client, _ = production
    demo = {"X-Demo-User": "pretend-admin", "X-Demo-Role": "admin"}
    assert client.get("/auth/me", headers=demo).status_code == 401
    token = _raw_token(signing_key, permissions=["manage_users"])
    identity = client.get("/auth/me", headers={**demo, **_headers(token)})
    assert identity.status_code == 200
    assert identity.json()["role"] == "viewer"
    assert Permission.MANAGE_USERS.value not in identity.json()["permissions"]


def test_explicit_demo_roles_are_verified_as_simulations(tmp_path):
    app = create_app(data_dir=str(tmp_path), auth_enabled=False, dev_mode=True, jwt_secret="")
    with TestClient(app) as client:
        assert client.get("/auth/status").json() == {"mode": "demo", "auth_required": False}
        identity = client.get("/auth/me", headers={"X-Demo-User": "demo-viewer", "X-Demo-Role": "viewer"})
        assert identity.json()["role"] == "viewer" and identity.json()["expires_at"] is None
        assert client.get("/auth/me").json()["role"] == "admin"
        assert client.get("/auth/me", headers=_headers("do-not-send-credentials-to-demo")).status_code == 401
        assert client.post("/auth/token?username=untrusted&role=admin").status_code == 404


def _assert_private(path):
    if os.name == "posix":
        info = path.stat()
        assert stat.S_IMODE(info.st_mode) == 0o600
        assert info.st_uid == os.geteuid()
        return
    environment = {**os.environ, "CHECK_PRIVATE_FILE": str(path)}
    script = r"""
$file = Get-Acl -LiteralPath $env:CHECK_PRIVATE_FILE
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$rules = @($file.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier]))
if (-not $file.AreAccessRulesProtected -or $rules.Count -ne 1 -or
    $rules[0].IdentityReference.Value -ne $sid.Value -or $rules[0].IsInherited -or
    $rules[0].AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow) { exit 1 }
"""
    powershell = Path(os.environ["SystemRoot"]) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    result = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", script],
        env=environment, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, "Output does not have a protected current-user-only Windows ACL"


def test_private_output_is_new_owner_only_and_cannot_overwrite(tmp_path):
    output = tmp_path / "credential.jwt"
    write_private_token(output, "synthetic-not-an-actual-credential")
    _assert_private(output)
    with pytest.raises(FileExistsError):
        write_private_token(output, "replacement")
    assert output.read_text().strip() == "synthetic-not-an-actual-credential"


def test_insecure_file_permissions_fail_before_credential_write(tmp_path, monkeypatch):
    from src.auth import issuer

    output = tmp_path / "rejected.jwt"
    if os.name == "nt":
        monkeypatch.setattr(
            issuer, "_WINDOWS_WRITE",
            issuer._WINDOWS_WRITE.replace(
                "$stream = [System.IO.FileStream]::new(",
                "$security.AddAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new("
                "[System.Security.Principal.SecurityIdentifier]::new('S-1-1-0'),"
                "[System.Security.AccessControl.FileSystemRights]::Read,"
                "[System.Security.AccessControl.AccessControlType]::Allow))\n"
                "    $stream = [System.IO.FileStream]::new(",
            ),
        )
    else:
        original_open = os.open
        original_chmod = os.fchmod

        def insecure_open(path, flags, mode):
            descriptor = original_open(path, flags, 0o600)
            original_chmod(descriptor, 0o666)
            return descriptor

        monkeypatch.setattr(issuer.os, "open", insecure_open)
        monkeypatch.setattr(issuer.os, "fchmod", lambda descriptor, mode: None)
    with pytest.raises(RuntimeError, match="token output"):
        write_private_token(output, "synthetic-not-an-actual-credential")
    assert not output.exists() or output.stat().st_size == 0


def test_operator_cli_never_prints_token_or_key(tmp_path, signing_key):
    output = tmp_path / "operator.jwt"
    script = Path(__file__).resolve().parents[1] / "scripts" / "issue_token.py"
    environment = {
        **os.environ, "AUTH_ENABLED": "true", "API_DEV_MODE": "false", "JWT_SECRET_KEY": signing_key,
    }
    command = [
        sys.executable, str(script), "--user-id", "synthetic-operator",
        "--role", "viewer", "--output", str(output),
    ]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, "Operator output failed"
    token = output.read_text().strip()
    _assert_not_exposed([token, signing_key], result.stdout + result.stderr)
    payload = decode_token(token, signing_key)
    assert payload["user_id"] == "synthetic-operator" and payload["role"] == "viewer"
    assert payload["exp"] - payload["iat"] == 3600
    _assert_private(output)
    rejected = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0
    _assert_not_exposed([token, signing_key], rejected.stdout + rejected.stderr)
