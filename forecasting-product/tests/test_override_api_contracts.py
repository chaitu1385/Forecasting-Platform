"""Authenticated stable-ID override mutations on real DuckDB, Parquet and legacy CSV."""

import hashlib
import json
import logging
import secrets
import time
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.auth.token import TOKEN_AUDIENCE, TOKEN_ISSUER, create_token
from src.overrides import store as storage
from src.pipeline.run_store import RunStore
from test_override_store import _open_store
from test_run_store import _publish


class _PrivateKey(str):
    def __repr__(self):
        return "<private synthetic signing key>"


@pytest.fixture(params=["duckdb", "parquet", "csv"])
def backend(request):
    return request.param


@pytest.fixture
def production(tmp_path, backend, monkeypatch):
    key = _PrivateKey(secrets.token_urlsafe(48))
    data = tmp_path / "data"
    roots = {
        "data_dir": str(data), "metrics_dir": str(data / "metrics"),
        "runs_dir": str(data / "runs"), "audit_log_path": str(data / "audit"),
    }
    # Select a real backend, not a replacement persistence or authentication implementation.
    monkeypatch.setattr(
        storage, "get_override_store",
        lambda path: _open_store(backend, Path(f"{path}.{backend}")),
    )
    app = create_app(**roots, jwt_secret=key, auth_enabled=True, dev_mode=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, data, key, roots
        client.headers.clear()


def _headers(key, role="admin"):
    token = create_token(f"override-{role}", "", role, key, expiry_hours=1)
    return {"Authorization": "Bearer " + token}


def _create(client, key):
    response = client.post("/overrides", headers=_headers(key), json={
        "old_sku": "synthetic-old", "new_sku": "synthetic-new",
        "proportion": 0.75, "notes": "before", "effective_date": "2026-01-02",
        "scenario": "custom", "ramp_shape": "sigmoid",
    })
    assert response.status_code == 200
    return response.json()["override_id"]


def _rows(client, key):
    response = client.get("/overrides", headers=_headers(key))
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == len(payload["overrides"])
    return {row["override_id"]: row for row in payload["overrides"]}


def test_valid_id_update_preserves_identity_and_survives_fresh_app(production, record_property):
    client, _, key, roots = production
    override_id = _create(client, key)
    other_id = _create(client, key)
    before = _rows(client, key)
    assert len(before) == 2 and before[override_id]["proportion"] == 0.75
    response = client.put(
        f"/overrides/{override_id}", headers=_headers(key),
        json={"proportion": 0.25, "notes": "after"},
    )
    record_property("original_defect", "Valid existing-ID update returned 404 at 72b7f1f")
    assert response.status_code == 200
    assert response.json() == {"override_id": override_id, "status": "updated"}
    expected = {**before[override_id], "proportion": 0.25, "notes": "after"}
    assert _rows(client, key) == {override_id: expected, other_id: before[other_id]}
    with TestClient(create_app(
        **roots, jwt_secret=key, auth_enabled=True, dev_mode=False,
    )) as restarted:
        assert _rows(restarted, key) == {override_id: expected, other_id: before[other_id]}


@pytest.mark.parametrize("changes", [
    {"proportion": 0}, {"notes": None}, {"effective_date": None},
    {"notes": "", "scenario": "", "ramp_shape": ""},
    {"effective_date": "2028-02-29"},
])
def test_patch_only_changes_supplied_fields(production, changes):
    client, _, key, _ = production
    override_id = _create(client, key)
    before = _rows(client, key)[override_id]
    response = client.put(f"/overrides/{override_id}", headers=_headers(key), json=changes)
    assert response.status_code == 200
    assert _rows(client, key) == {override_id: {**before, **changes}}


@pytest.mark.parametrize("changes,status", [
    ({}, 400), ({"effective_date": ""}, 400), ({"effective_date": "2026-02-30"}, 400),
    ({"override_id": "replacement"}, 422), ({"old_sku": "replacement"}, 422),
    ({"new_sku": "replacement"}, 422), ({"created_by": "replacement"}, 422),
    ({"created_at": "2026-01-01"}, 422), ({"status": "approved"}, 422),
    ({"approved_by": "replacement"}, 422), ({"approved_at": None}, 422),
    ({"extra": "unknown"}, 422), ({"proportion": None}, 422), ({"proportion": True}, 422),
    ({"proportion": "0.25"}, 422), ({"proportion": -0.25}, 422), ({"proportion": 1.25}, 422),
    ({"scenario": None}, 422), ({"ramp_shape": None}, 422), ({"notes": 7}, 422),
    ({"effective_date": 20260101}, 422),
])
def test_invalid_patch_is_not_a_success_or_mutation(production, changes, status):
    client, _, key, _ = production
    override_id = _create(client, key)
    before = _rows(client, key)
    response = client.put(f"/overrides/{override_id}", headers=_headers(key), json=changes)
    assert response.status_code == status
    assert _rows(client, key) == before


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_wire_numbers_are_rejected_without_server_failure(production, value):
    client, _, key, _ = production
    override_id = _create(client, key)
    before = _rows(client, key)
    response = client.put(
        f"/overrides/{override_id}", headers={**_headers(key), "Content-Type": "application/json"},
        content='{"proportion": ' + value + "}",
    )
    assert response.status_code == 422
    assert _rows(client, key) == before
    created = client.post(
        "/overrides", headers={**_headers(key), "Content-Type": "application/json"},
        content='{"old_sku":"old","new_sku":"new","proportion":' + value + "}",
    )
    assert created.status_code == 422
    assert _rows(client, key) == before


@pytest.mark.parametrize("populated", [False, True])
def test_missing_update_delete_are_404_without_creating_a_store(production, populated):
    client, data, key, _ = production
    if populated:
        _create(client, key)
    before = _rows(client, key)
    assert client.put(
        "/overrides/OVR-MISSING", headers=_headers(key), json={"notes": "missing"},
    ).status_code == 404
    assert client.delete("/overrides/OVR-MISSING", headers=_headers(key)).status_code == 404
    assert _rows(client, key) == before
    if not populated:
        assert not any((data / f"overrides.{suffix}").exists() for suffix in ("duckdb", "parquet", "csv"))


@pytest.mark.parametrize("role,status", [
    ("anonymous", 401), ("expired", 401), ("viewer", 403), ("manager", 403),
    ("planner", 200), ("data_scientist", 200), ("admin", 200),
])
def test_update_enforces_real_signed_permission_matrix(production, role, status):
    client, _, key, _ = production
    override_id = _create(client, key)
    before = _rows(client, key)
    if role == "anonymous":
        headers = {}
    elif role == "expired":
        now = int(time.time())
        expired = jwt.encode({
            "user_id": "expired-override-user", "email": "", "role": "admin", "is_active": True,
            "iat": now - 120, "exp": now - 60, "iss": TOKEN_ISSUER, "aud": TOKEN_AUDIENCE,
        }, key, algorithm="HS256")
        headers = {"Authorization": "Bearer " + expired}
    else:
        headers = _headers(key, role)
    response = client.put(
        f"/overrides/{override_id}", headers=headers, json={"proportion": 0.25},
    )
    assert response.status_code == status
    expected = {override_id: {**before[override_id], "proportion": 0.25}} if status == 200 else before
    assert _rows(client, key) == expected


@pytest.mark.parametrize("role,create_status,delete_status", [
    ("viewer", 403, 403), ("planner", 200, 403), ("data_scientist", 200, 200), ("admin", 200, 200),
])
def test_create_delete_permissions_remain_distinct(production, role, create_status, delete_status):
    client, _, key, _ = production
    override_id = _create(client, key)
    created = client.post("/overrides", headers=_headers(key, role), json={
        "old_sku": "other-old", "new_sku": "other-new", "proportion": 0.5,
    })
    assert created.status_code == create_status
    before = _rows(client, key)
    deleted = client.delete(f"/overrides/{override_id}", headers=_headers(key, role))
    assert deleted.status_code == delete_status
    expected = {key: row for key, row in before.items() if key != override_id} if delete_status == 200 else before
    assert _rows(client, key) == expected


@pytest.mark.parametrize("backend", ["parquet", "csv"])
@pytest.mark.parametrize("operation", ["post", "put", "delete"])
def test_file_writer_contention_is_explicit_409_without_automatic_retry(production, backend, operation):
    client, data, key, _ = production
    override_id = _create(client, key)
    before = _rows(client, key)
    with storage._override_file_lock(data / f"overrides.{backend}"):
        if operation == "post":
            response = client.post("/overrides", headers=_headers(key), json={
                "old_sku": "other-old", "new_sku": "other-new", "proportion": 0.5,
            })
        elif operation == "put":
            response = client.put(f"/overrides/{override_id}", headers=_headers(key), json={"notes": "after"})
        else:
            response = client.delete(f"/overrides/{override_id}", headers=_headers(key))
        assert response.status_code == 409
    assert _rows(client, key) == before


@pytest.mark.parametrize("backend", ["parquet", "csv"])
def test_real_replace_failure_is_logged_500_and_preserves_the_old_file(production, backend, monkeypatch, caplog):
    client, data, key, _ = production
    override_id = _create(client, key)
    path = data / f"overrides.{backend}"
    before = path.read_bytes()

    def fail(*args):
        raise OSError("synthetic-private-storage-path")

    with monkeypatch.context() as fault, caplog.at_level(logging.ERROR):
        fault.setattr(storage.os, "replace", fail)
        response = client.put(
            f"/overrides/{override_id}", headers=_headers(key), json={"notes": "after"},
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to update overrides."}
    assert path.read_bytes() == before
    assert any(record.exc_info and record.name.endswith("routers.overrides") for record in caplog.records)
    assert _rows(client, key)[override_id]["notes"] == "before"


@pytest.mark.parametrize("operation", ["get", "post", "put", "delete"])
def test_store_open_failure_is_logged_and_never_exposes_storage_details(production, operation, monkeypatch, caplog):
    client, _, key, _ = production

    def fail(path):
        raise OSError("synthetic-private-storage-path")

    monkeypatch.setattr(storage, "get_override_store", fail)
    with caplog.at_level(logging.ERROR):
        if operation == "post":
            response = client.post("/overrides", headers=_headers(key), json={
                "old_sku": "old", "new_sku": "new", "proportion": 0.5,
            })
        elif operation == "put":
            response = client.put("/overrides/OVR-MISSING", headers=_headers(key), json={"notes": "after"})
        else:
            response = getattr(client, operation)(
                "/overrides" if operation == "get" else "/overrides/OVR-MISSING", headers=_headers(key),
            )
    assert response.status_code == 500
    assert "synthetic-private-storage-path" not in response.text
    assert any(record.exc_info and record.name.endswith("routers.overrides") for record in caplog.records)


def test_override_crud_does_not_rewrite_immutable_forecast_artifacts(production, record_property):
    client, data, key, _ = production
    runs = RunStore(data / "runs")
    manifest = _publish(runs, 17.0)
    directory = runs.directory("retail", manifest.run_id)

    def hashes():
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()}

    before = hashes()
    override_id = _create(client, key)
    assert client.put(
        f"/overrides/{override_id}", headers=_headers(key), json={"proportion": 0, "notes": None},
    ).status_code == 200
    assert client.delete(f"/overrides/{override_id}", headers=_headers(key)).status_code == 200
    assert hashes() == before
    assert runs.get("retail", manifest.run_id) == manifest
    assert runs.frame("retail", manifest.run_id, "forecast")["forecast"].to_list() == [17.0]
    record_property("unchanged_run_artifact_sha256", json.dumps(before, sort_keys=True))
