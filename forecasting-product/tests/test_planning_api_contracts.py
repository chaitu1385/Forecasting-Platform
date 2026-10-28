"""F8 hierarchy acceptance: real authenticated uploads, domain math and full-frame checks."""

import hashlib
import io
import secrets
from datetime import date, timedelta

import jwt
import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.deps import validate_upload_size
from src.auth.token import create_token
from src.hierarchy.reconciler import Reconciler


@pytest.fixture
def client(tmp_path):
    app = create_app(
        data_dir=str(tmp_path / "data"),
        metrics_dir=str(tmp_path / "metrics"),
        runs_dir=str(tmp_path / "runs"),
        audit_log_path=str(tmp_path / "audit"),
        jwt_secret=secrets.token_urlsafe(48),
        auth_enabled=True,
        dev_mode=False,
    )
    with TestClient(app, raise_server_exceptions=False) as connection:
        connection.headers["Authorization"] = f"Bearer {_token(connection, 'admin')}"
        assert connection.get("/auth/me").json()["role"] == "admin"
        yield connection
        connection.headers.clear()


def _token(client, role, expired=False):
    key = client.app.state.jwt_secret
    token = create_token("hierarchy-synthetic", "", role, key, expiry_hours=1)
    if expired:
        claims = jwt.decode(token, options={"verify_signature": False})
        claims["iat"] -= 7200
        claims["exp"] -= 7200
        token = jwt.encode(claims, key, algorithm="HS256")
    return token


def _leaves(periods=2):
    return pl.DataFrame([
        {
            "department": "food",
            "item": item,
            "week": date(2026, 1, 5) + timedelta(weeks=period),
            "forecast": value,
        }
        for period in range(periods)
        for item, value in zip(("a", "b"), (10.0, 20.0) if period == 0 else (4.0, 6.0))
    ])


def _history():
    return pl.DataFrame({
        "department": ["food", "food"], "item": ["a", "b"],
        "week": [date(2025, 12, 1)] * 2, "quantity": [10.0, 30.0],
    })


def _roots():
    return pl.DataFrame({
        "department": ["food"], "week": [date(2026, 1, 5)], "forecast": [120.0],
    })


def _upload(frame, filename="synthetic.csv"):
    if filename.endswith(".parquet"):
        content = io.BytesIO()
        frame.write_parquet(content)
        return filename, content.getvalue(), "application/octet-stream"
    return filename, frame.write_csv().encode(), "text/csv"


def _reconcile(client, frame=None, *, history=None, method="bottom_up", filename="synthetic.csv", **params):
    files = {"file": _upload(_leaves() if frame is None else frame, filename)}
    if history is not None:
        files["actuals_file"] = _upload(history, "history.parquet")
    return client.post(
        "/hierarchy/reconcile",
        params={"levels": "department,item", "id_column": "item", "method": method, **params},
        files=files,
    )


def _values(response, leaf="item", value="forecast", time="week"):
    return {(row[leaf], row[time]): row[value] for row in response.json()["reconciled_preview"]}


@pytest.mark.parametrize("levels,leaf", [("department,item", "item"), ("category,sku", "sku")])
def test_001_bottom_up_exact_keys_values_and_independent_parent_totals(client, levels, leaf):
    frame = _leaves()
    if leaf == "sku":
        frame = frame.rename({"department": "category", "item": "sku"})
    response = _reconcile(client, frame, levels=levels, id_column=leaf)
    assert response.status_code == 200, response.text
    assert _values(response, leaf) == {
        ("a", "2026-01-05"): 10.0, ("b", "2026-01-05"): 20.0,
        ("a", "2026-01-12"): 4.0, ("b", "2026-01-12"): 6.0,
    }
    result = response.json()
    rows = result["reconciled_preview"]
    assert len(rows) == result["input_rows"] == result["output_rows"] == result["rows"] == 4
    assert all(set(row) == {leaf, "week", "forecast"} for row in rows)
    assert sum(row["forecast"] for row in rows if row["week"] == "2026-01-05") == 30.0
    assert sum(row["forecast"] for row in rows if row["week"] == "2026-01-12") == 10.0
    assert result["before_total"] == result["after_total"] == 40.0
    assert result["requested_method"] == result["effective_method"] == result["method"] == "bottom_up"
    assert result["source_level"] == result["output_level"] == leaf
    assert result["coherence"] == {
        "scope": "leaf_derived_parent_totals", "passed": True, "checked_rows": 4,
        "checked_periods": 2, "checked_groups": 2, "value_columns": ["forecast"],
        "max_abs_error": 0.0,
    }


def test_002_top_down_uses_real_earlier_history_and_conserves_root(client):
    response = _reconcile(client, _roots(), history=_history(), method="top_down")
    assert response.status_code == 200, response.text
    assert _values(response) == {("a", "2026-01-05"): 30.0, ("b", "2026-01-05"): 90.0}
    result = response.json()
    assert result["before_total"] == result["after_total"] == 120.0
    assert result["source_level"] == "department" and result["output_level"] == "item"
    assert result["requested_method"] == result["effective_method"] == "top_down"
    assert result["input_rows"] == 1 and result["output_rows"] == 2
    assert result["coherence"]["scope"] == "root_forecast_conservation"
    assert result["coherence"]["checked_groups"] == 1
    assert result["coherence"]["max_abs_error"] == 0.0


def test_002_top_down_multiple_roots_periods_and_requested_quantiles(client):
    history = pl.concat([
        _history(),
        _history().with_columns(
            pl.lit("care").alias("department"), pl.Series("item", ["c", "d"]),
            pl.Series("quantity", [1.0, 3.0]),
        ),
    ]).rename({"quantity": "sales"})
    original = history.clone()
    roots = pl.DataFrame({
        "department": ["food", "care", "food", "care"],
        "week": [date(2026, 1, 5)] * 2 + [date(2026, 1, 12)] * 2,
        "forecast": [120.0, 40.0, 80.0, 80.0],
    }).with_columns(
        (pl.col("forecast") * 0.5).alias("forecast_p10"),
        (pl.col("forecast") * 1.5).alias("forecast_p90"),
        pl.lit(999.0).alias("unused_quantile"),
    )
    response = _reconcile(
        client, roots, history=history, method="top_down", history_value_column="sales",
        value_columns="forecast_p10,forecast,forecast_p90",
    )
    assert response.status_code == 200, response.text
    assert _values(response) == {
        ("a", "2026-01-05"): 30, ("b", "2026-01-05"): 90,
        ("c", "2026-01-05"): 10, ("d", "2026-01-05"): 30,
        ("a", "2026-01-12"): 20, ("b", "2026-01-12"): 60,
        ("c", "2026-01-12"): 20, ("d", "2026-01-12"): 60,
    }
    for row in response.json()["reconciled_preview"]:
        assert set(row) == {"item", "week", "forecast", "forecast_p10", "forecast_p90"}
        assert row["forecast_p10"] == row["forecast"] * 0.5
        assert row["forecast_p90"] == row["forecast"] * 1.5
    assert history.equals(original)
    assert response.json()["coherence"]["checked_groups"] == 4


@pytest.mark.parametrize("method,effective", [("ols", "ols"), ("wls", "wls_structural"), ("mint", "wls_structural")])
def test_003_linear_methods_match_independent_small_matrix_oracle(client, method, effective):
    points = np.array([[10.0, 4.0], [20.0, 6.0], [5.0, 8.0], [15.0, 12.0]])
    summing = np.vstack([
        np.ones(4), [1, 1, 0, 0], [0, 0, 1, 1], np.eye(4),
    ])
    inverse_weights = np.eye(7) if method == "ols" else np.diag(1 / summing.sum(axis=1))
    weighted = summing.T @ inverse_weights
    normal = weighted @ summing
    regularizer = 1e-8 * np.trace(normal) / 4
    expected = np.linalg.solve(normal + regularizer * np.eye(4), weighted @ (summing @ points))
    frame = pl.DataFrame([
        {
            "department": "food", "category": "dry" if index < 2 else "fresh",
            "item": item, "week": date(2026, 1, 5) + timedelta(weeks=period),
            "forecast": points[index, period],
            "forecast_p10": points[index, period] * 0.5,
            "forecast_p90": points[index, period] * 1.5,
        }
        for index, item in enumerate(("a", "b", "c", "d")) for period in range(2)
    ])
    response = _reconcile(
        client, frame, method=method, levels="department,category,item",
        value_columns="forecast,forecast_p10,forecast_p90",
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["requested_method"] == method and result["effective_method"] == effective
    assert result["coherence"]["scope"] == "leaf_derived_parent_totals"
    assert result["coherence"]["checked_groups"] == 6
    assert result["input_rows"] == result["output_rows"] == 8
    assert any("no independent upper" in warning for warning in result["warnings"])
    if method in {"wls", "mint"}:
        assert any("structural" in warning for warning in result["warnings"])
    if method == "mint":
        assert any("not residual-covariance MinT" in warning for warning in result["warnings"])
    for column, scale in (("forecast", 1.0), ("forecast_p10", 0.5), ("forecast_p90", 1.5)):
        actual = _values(response, value=column)
        assert len(actual) == 8
        for index, item in enumerate(("a", "b", "c", "d")):
            for period, week in enumerate(("2026-01-05", "2026-01-12")):
                np.testing.assert_allclose(actual[item, week], expected[index, period] * scale, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "params",
    [
        {"levels": ""}, {"levels": "department,,item"}, {"levels": "department,item,item"},
        {"levels": "department,absent"}, {"value_columns": ""}, {"value_columns": "forecast,forecast"},
        {"value_columns": "forecast,absent"}, {"time_column": "absent"},
        {"time_column": "item"}, {"id_column": "department"}, {"id_column": "absent"},
        {"value_columns": "item"}, {"method": "unrecognized"}, {"method": "middle_out"},
    ],
)
def test_004_invalid_request_contract_is_actionable_400(client, params):
    response = _reconcile(client, **params)
    assert response.status_code == 400, response.text
    code = "unsupported_reconciliation_mode" if "method" in params else "invalid_hierarchy_input"
    assert code in response.json()["detail"]


@pytest.mark.parametrize(
    "change",
    [
        lambda df: df.head(0),
        lambda df: df.drop("forecast"),
        lambda df: df.with_columns(pl.lit(None).alias("item")),
        lambda df: df.with_columns(pl.lit("").alias("department")),
        lambda df: df.with_columns(pl.lit(None).alias("week")),
        lambda df: df.with_columns(pl.lit("not-a-date").alias("week")),
        lambda df: df.with_columns(pl.lit(None).alias("forecast")),
        lambda df: df.with_columns(pl.lit(float("nan")).alias("forecast")),
        lambda df: df.with_columns(pl.lit(float("inf")).alias("forecast")),
        lambda df: df.with_columns(pl.lit("not-a-number").alias("forecast")),
        lambda df: df.with_columns(pl.lit(True).alias("forecast")),
        lambda df: pl.concat([df, df.head(1)]),
        lambda df: df.slice(1),
        lambda df: df.with_columns(pl.col("week").cast(pl.String).alias("department")),
    ],
    ids=["empty", "missing-value", "null-id", "empty-parent", "null-time", "bad-time", "null-value",
         "nan", "infinity", "text", "boolean", "duplicate", "missing-period", "multiple-parents"],
)
def test_004_invalid_panels_never_zero_pad_or_infer_mappings(client, change):
    response = _reconcile(client, change(_leaves()))
    assert response.status_code == 400, response.text
    assert "invalid_hierarchy_input" in response.json()["detail"]


@pytest.mark.parametrize("method", ["ols", "wls", "mint"])
@pytest.mark.parametrize("collision", ["cross-level", "reserved-leaf"])
def test_004_linear_node_identity_ambiguity_is_rejected(client, method, collision):
    frame = _leaves()
    if collision == "cross-level":
        frame = frame.with_columns(pl.lit("a").alias("department"))
    else:
        frame = frame.with_columns(pl.col("item").replace({"a": "node_key"}))
    response = _reconcile(client, frame, method=method)
    assert response.status_code == 400, response.text


@pytest.mark.parametrize("method", ["bottom_up", "top_down", "ols", "wls", "mint"])
def test_004_custom_identity_alias_numeric_keys_and_column_names(client, method):
    history = _history().with_columns(
        pl.Series("item", [1, 2]), pl.Series("code", ["code-a", "code-b"]),
    ).rename({"week": "period", "quantity": "sales"})
    if method == "top_down":
        frame = _roots().rename({"week": "period", "forecast": "amount"})
    else:
        frame = _leaves().with_columns(
            pl.Series("item", [1, 2, 1, 2]), pl.Series("code", ["code-a", "code-b"] * 2),
        ).rename({"week": "period", "forecast": "amount"})
    response = _reconcile(
        client, frame, history=history if method == "top_down" else None, method=method,
        id_column="code", time_column="period", value_columns="amount",
        history_value_column="sales", filename="forecast.parquet",
    )
    assert response.status_code == 200, response.text
    assert {row["item"] for row in response.json()["reconciled_preview"]} == {1, 2}
    assert all(set(row) == {"item", "period", "amount"} for row in response.json()["reconciled_preview"])


@pytest.mark.parametrize(
    "history",
    [
        None,
        _history().head(0),
        _history().with_columns(pl.lit(0.0).alias("quantity")),
        _history().with_columns(pl.Series("quantity", [0.0, 30.0])),
        _history().with_columns(pl.Series("quantity", [-10.0, 30.0])),
        _history().with_columns(pl.lit(float("inf")).alias("quantity")),
        _history().drop("quantity"),
        _history().with_columns(pl.lit(date(2026, 1, 5)).alias("week")),
        _history().with_columns(pl.lit(date(2026, 1, 12)).alias("week")),
        pl.concat([_history(), _history().head(1)]),
        pl.concat([_history(), _history().head(1).with_columns(pl.lit(date(2025, 12, 8)).alias("week"))]),
    ],
    ids=["absent", "empty", "zero-root", "zero-leaf", "negative-leaf", "nonfinite", "missing-target",
         "overlap", "future", "duplicate", "incomplete-history"],
)
def test_002_top_down_rejects_invalid_history_without_equal_shares(client, history):
    response = _reconcile(client, _roots(), history=history, method="top_down")
    assert response.status_code == 400, response.text
    assert "invalid_hierarchy_input" in response.json()["detail"]


def test_002_top_down_rejects_unknown_or_missing_root_forecasts(client):
    response = _reconcile(
        client, _roots().with_columns(pl.lit("unknown").alias("department")),
        history=_history(), method="top_down",
    )
    assert response.status_code == 400
    history = pl.concat([
        _history(),
        _history().with_columns(pl.lit("care").alias("department"), pl.Series("item", ["c", "d"])),
    ])
    assert _reconcile(client, _roots(), history=history, method="top_down").status_code == 400


def test_004_history_is_not_silently_used_by_leaf_only_methods(client):
    response = _reconcile(client, history=_history())
    assert response.status_code == 400
    assert "only for top_down" in response.json()["detail"]


@pytest.mark.parametrize("bad", ["crossed", "nonfinite", "unsupported-name"])
def test_004_requested_quantiles_are_validated_without_sorting(client, bad):
    frame = _leaves().with_columns(
        pl.lit(20.0 if bad == "crossed" else float("nan") if bad == "nonfinite" else 1.0).alias("forecast_p10"),
        pl.lit(10.0).alias("forecast_p90"),
    )
    columns = "forecast,forecast_p10,forecast_p90"
    if bad == "unsupported-name":
        frame = frame.rename({"forecast_p10": "forecast_p010"})
        columns = "forecast,forecast_p010,forecast_p90"
    assert _reconcile(client, frame, value_columns=columns).status_code == 400


@pytest.mark.parametrize("method", ["bottom_up", "top_down", "ols", "wls", "mint"])
def test_004_unrequested_columns_are_not_returned_as_adjusted_values(client, method):
    frame = _roots() if method == "top_down" else _leaves()
    frame = frame.with_columns(
        pl.lit(float("nan")).alias("forecast_p10"), pl.lit(-10.0).alias("forecast_p90"),
        pl.lit("unrequested").alias("notes"),
    )
    response = _reconcile(client, frame, method=method, history=_history() if method == "top_down" else None)
    assert response.status_code == 200, response.text
    assert all(set(row) == {"item", "week", "forecast"} for row in response.json()["reconciled_preview"])


def test_004_validation_and_metadata_cover_more_than_the_preview(client):
    frame = _leaves(101)
    response = _reconcile(client, frame, validate="true")
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["reconciled_preview"]) == 200
    assert result["input_rows"] == result["output_rows"] == result["rows"] == 202
    assert result["coherence"]["checked_rows"] == 202
    assert result["coherence"]["checked_periods"] == result["coherence"]["checked_groups"] == 101
    assert result["before_total"] == result["after_total"] == 1030.0
    assert result["validation"]["skipped"] is False
    assert result["validation"]["layers"]["paradox"] is None


@pytest.mark.parametrize("fault", ["nan", "duplicate", "missing", "unknown", "period", "crossed-quantiles"])
def test_004_invalid_output_beyond_preview_fails_closed(client, monkeypatch, fault):
    original = Reconciler.reconcile

    def corrupted(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        if fault == "missing":
            return output.slice(0, output.height - 1)
        if fault == "duplicate":
            return pl.concat([output.slice(0, output.height - 1), output.head(1)])
        column = "item" if fault == "unknown" else "week" if fault == "period" else (
            "forecast_p10" if fault == "crossed-quantiles" else "forecast"
        )
        values = output[column].to_list()
        values[-1] = "unknown" if fault == "unknown" else date(2035, 1, 1) if fault == "period" else (
            1e6 if fault == "crossed-quantiles" else float("nan")
        )
        return output.with_columns(pl.Series(column, values))

    monkeypatch.setattr(Reconciler, "reconcile", corrupted)
    frame = _leaves(101).with_columns(
        (pl.col("forecast") * 0.5).alias("forecast_p10"),
        (pl.col("forecast") * 1.5).alias("forecast_p90"),
    )
    response = _reconcile(client, frame, value_columns="forecast,forecast_p10,forecast_p90")
    assert response.status_code == 500, response.text
    assert response.json()["detail"].startswith("reconciliation_failed:")


def test_004_top_down_checks_measured_conservation_not_only_keys(client, monkeypatch):
    original = Reconciler.reconcile

    def corrupted(self, *args, **kwargs):
        return original(self, *args, **kwargs).with_columns(pl.col("forecast") + 1)

    monkeypatch.setattr(Reconciler, "reconcile", corrupted)
    response = _reconcile(client, _roots(), history=_history(), method="top_down")
    assert response.status_code == 500


def test_004_optional_checker_failure_is_not_a_success_shaped_skip(client, monkeypatch, caplog):
    from src.pipeline import validation_step

    def failed(*args, **kwargs):
        raise RuntimeError("private synthetic diagnostic")

    monkeypatch.setattr(validation_step, "run_post_validation", failed)
    response = _reconcile(client, validate="true")
    assert response.status_code == 500
    assert "private synthetic diagnostic" not in response.text
    assert "full-frame output validation failed" in caplog.text


@pytest.mark.parametrize("role,status", [("admin", 200), ("data_scientist", 200), ("viewer", 403), ("planner", 403), ("manager", 403)])
def test_022_real_role_permissions_remain_enforced(client, role, status):
    client.headers["Authorization"] = f"Bearer {_token(client, role)}"
    response = _reconcile(client)
    assert response.status_code == status, response.text
    assert client.app.state.jwt_secret not in response.text
    assert client.headers["Authorization"] not in response.text


@pytest.mark.parametrize("credential", ["anonymous", "expired"])
def test_022_anonymous_and_expired_requests_are_denied(client, credential):
    if credential == "anonymous":
        del client.headers["Authorization"]
    else:
        client.headers["Authorization"] = f"Bearer {_token(client, 'admin', expired=True)}"
    response = _reconcile(client)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("filename", ["../outside.csv", "/absolute.csv", "invalid.json"])
def test_022_upload_filenames_cannot_select_server_paths(client, filename):
    response = _reconcile(client, filename=filename)
    assert response.status_code == 400
    assert response.json()["detail"].startswith("invalid_hierarchy_file:")


@pytest.mark.parametrize("part", ["file", "actuals_file"])
def test_022_each_upload_uses_the_existing_size_boundary(client, monkeypatch, part):
    monkeypatch.setattr(validate_upload_size, "__defaults__", (1024,))
    files = {"file": _upload(_roots()), "actuals_file": _upload(_history())}
    files[part] = ("oversized.csv", b"x" * 1025, "text/csv")
    response = client.post(
        "/hierarchy/reconcile", params={"levels": "department,item", "id_column": "item", "method": "top_down"},
        files=files,
    )
    assert response.status_code == 413


def test_022_malformed_upload_and_typed_query_are_not_internal_errors(client):
    response = client.post(
        "/hierarchy/reconcile", params={"levels": "department,item", "id_column": "item"},
        files={"file": ("invalid.parquet", b"not parquet", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert _reconcile(client, validate="not-a-boolean").status_code == 422


def test_022_hierarchy_leaves_committed_run_bytes_history_and_configuration_unchanged(client):
    from test_run_artifacts import _run

    run = _run(client, dict(client.headers))
    assert run.status_code == 200, run.text
    run_id = run.json()["run_id"]
    endpoints = [
        f"/forecast/retail?run_id={run_id}",
        f"/pipeline/runs/retail/{run_id}/config",
        f"/series/retail/synthetic/history?run_id={run_id}",
    ]
    snapshots = [client.get(endpoint).json() for endpoint in endpoints]
    root = client.app.state.runs_dir
    original = {path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    assert original
    assert _reconcile(client).status_code == 200
    assert _reconcile(client, _leaves().slice(1)).status_code == 400
    assert [client.get(endpoint).json() for endpoint in endpoints] == snapshots
    assert {path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()} == original


def test_004_openapi_declares_the_effective_method_and_history_contract(client):
    spec = client.get("/openapi.json").json()
    route = spec["paths"]["/hierarchy/reconcile"]["post"]
    assert route["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/HierarchyReconcileResponse")
    properties = spec["components"]["schemas"]["HierarchyReconcileResponse"]["properties"]
    assert {"method", "requested_method", "effective_method", "source_level", "output_level",
            "input_rows", "output_rows", "coherence", "warnings", "validation"} <= properties.keys()
    assert spec["components"]["schemas"]["HierarchyEffectiveMethod"]["enum"] == ["bottom_up", "top_down", "ols", "wls_structural"]
    assert any(parameter["name"] == "history_value_column" for parameter in route["parameters"])
    body_ref = route["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    assert "actuals_file" in spec["components"]["schemas"][body_ref]["properties"]
