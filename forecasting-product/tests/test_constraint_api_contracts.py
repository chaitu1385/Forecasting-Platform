"""Real authenticated constraint/API contracts and immutable publication controls."""

import io
import json
import math
import secrets
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import jwt
import polars as pl
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.deps import MAX_UPLOAD_BYTES
from src.auth.token import TOKEN_AUDIENCE, TOKEN_ISSUER, create_token
from src.config.schema import ConstraintConfig, PlatformConfig
from src.forecasting.constrained import ConstrainedDemandEstimator, apply_point_constraints
from src.forecasting.naive import SeasonalNaiveForecaster
from src.pipeline.forecast import ForecastPipeline
from src.pipeline.run_store import RunStore


@dataclass(repr=False)
class _PrivateAPI:
    client: TestClient
    root: Path
    signing_key: str = field(repr=False)

    def headers(self, role="data_scientist"):
        if role is None:
            return {}
        if role == "expired":
            now = int(time.time())
            token = jwt.encode({
                "user_id": "constraint-test", "email": "", "role": "data_scientist",
                "is_active": True, "iat": now - 7200, "exp": now - 3600,
                "iss": TOKEN_ISSUER, "aud": TOKEN_AUDIENCE,
            }, self.signing_key, algorithm="HS256")
        else:
            token = create_token("constraint-test", "", role, self.signing_key, expiry_hours=1)
        return {"Authorization": "Bearer " + token}

    def constrain(self, frame, params=None, *, extension="parquet", role="data_scientist"):
        if extension == "parquet":
            content = io.BytesIO()
            frame.write_parquet(content)
            raw = content.getvalue()
        else:
            raw = frame.write_csv().encode()
        response = self.client.post(
            "/forecast/constrain", params=params, headers=self.headers(role),
            files={"file": (f"synthetic.{extension}", raw, "application/octet-stream")},
        )
        if self.signing_key in response.text or str(self.root) in response.text:
            pytest.fail("Constraint response exposed private fixture material")
        return response


@pytest.fixture
def private_api(tmp_path):
    key = secrets.token_urlsafe(48)
    app = create_app(
        data_dir=str(tmp_path / "data"), metrics_dir=str(tmp_path / "metrics"),
        runs_dir=str(tmp_path / "runs"), audit_log_path=str(tmp_path / "audit"),
        jwt_secret=key, auth_enabled=True, dev_mode=False,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield _PrivateAPI(client, tmp_path, key)


def _frame(values=(100.0, 100.0)):
    return pl.DataFrame({
        "series_id": [chr(ord("a") + index) for index in range(len(values))],
        "week": [date(2026, 1, 5)] * len(values),
        "forecast": values,
    })


@pytest.mark.parametrize("proportional,expected", [(True, {"a": 25.0, "b": 25.0}), (False, {"a": 0.0, "b": 50.0})])
@pytest.mark.parametrize("extension", ["csv", "parquet"])
@pytest.mark.parametrize("custom_columns", [False, True])
def test_authenticated_strategy_hand_oracle_api_helper_and_real_wrapper_parity(
    private_api, proportional, expected, extension, custom_columns,
):
    id_col, time_col, value_col = ("sku", "period", "demand") if custom_columns else ("series_id", "week", "forecast")
    actuals = pl.DataFrame({
        id_col: ["a", "a", "b", "b"],
        time_col: [date(2025, 12, 22), date(2025, 12, 29)] * 2,
        "quantity": [100.0] * 4,
    })
    settings = ConstraintConfig(enabled=True, aggregate_max=50, proportional_redistribution=proportional)
    wrapper = ConstrainedDemandEstimator(SeasonalNaiveForecaster(season_length=1), settings)
    wrapper.fit(actuals, id_col=id_col, time_col=time_col)
    frame = wrapper.base.predict(1, id_col=id_col, time_col=time_col).rename({"forecast": value_col})
    frame = frame.reverse().with_columns(pl.Series("_period_total", ["untouched-b", "untouched-a"]))
    response = private_api.constrain(
        frame, {"aggregate_max": 50, "proportional": proportional,
                "id_col": id_col, "time_col": time_col, "value_col": value_col},
        extension=extension,
    )
    assert response.status_code == 200, response.text
    result = response.json()
    output = {row[id_col]: row[value_col] for row in result["constrained_preview"]}
    assert output == expected
    assert result["before_total"] == 200
    assert result["after_total"] == 50
    assert result["allocation_strategy"] == ("proportional" if proportional else "largest_first")
    assert result["constraints_applied"]["proportional"] is proportional
    assert result["checked_rows"] == 2 and result["checked_periods"] == 1
    assert result["max_budget_excess"] == 0
    assert result["constraint_scope"] == "point"
    shared = apply_point_constraints(frame, settings, id_col, time_col, value_col)
    assert result["constrained_preview"] == json.loads(shared.write_json())
    wrapped = wrapper.predict(1, id_col=id_col, time_col=time_col)
    assert dict(zip(wrapped[id_col], wrapped["forecast"])) == expected
    assert [row[id_col] for row in result["constrained_preview"]] == frame[id_col].to_list()


def test_default_strategy_remains_proportional_and_totals_are_not_rounded(private_api):
    response = private_api.constrain(_frame(), {"aggregate_max": 50})
    assert response.status_code == 200
    assert [row["forecast"] for row in response.json()["constrained_preview"]] == [25, 25]
    values = [0.123456789123, 0.234567891234]
    response = private_api.constrain(_frame(values))
    assert response.status_code == 200
    data = response.json()
    assert data["before_total"] == data["after_total"] == math.fsum(values)
    assert data["rows_modified"] == 0
    assert data["allocation_strategy"] == "element_wise"
    assert data["max_budget_excess"] is None


@pytest.mark.parametrize(
    "values,params,expected",
    [
        ([100.0, 2.0], {"min_demand": 10, "aggregate_max": 30}, [20, 10]),
        ([-1.0, 2.0], {"min_demand": 0, "max_capacity": 1}, [0, 1]),
        ([100.0, 100.0], {"aggregate_max": 0, "proportional": False}, [0, 0]),
        ([100.0, 100.0], {"max_capacity": 20, "aggregate_max": 50}, [20, 20]),
    ],
)
def test_feasible_floor_cap_and_budget_results(private_api, values, params, expected):
    response = private_api.constrain(_frame(values), params)
    assert response.status_code == 200
    assert [row["forecast"] for row in response.json()["constrained_preview"]] == expected


@pytest.mark.parametrize("field_name", ["min_demand", "max_capacity", "aggregate_max"])
@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
def test_nonnegative_finite_limit_semantics_are_http400(private_api, field_name, value):
    response = private_api.constrain(_frame(), {field_name: value})
    assert response.status_code == 400
    assert "invalid_constraint_limit" in response.json()["detail"]


@pytest.mark.parametrize(
    "params",
    [{"min_demand": "invalid"}, {"max_capacity": "invalid"}, {"aggregate_max": "invalid"}, {"proportional": "invalid"}],
)
def test_malformed_typed_parameters_keep_fastapi422(private_api, params):
    assert private_api.constrain(_frame(), params).status_code == 422


@pytest.mark.parametrize("params", [{"min_demand": 40, "aggregate_max": 50}, {"min_demand": 10, "max_capacity": 9}])
def test_infeasible_requests_never_claim_applied(private_api, params):
    response = private_api.constrain(_frame(), params)
    assert response.status_code == 400
    assert "infeasible_constraints" in response.json()["detail"]
    assert "constraints_applied" not in response.json()


@pytest.mark.parametrize(
    "frame",
    [
        _frame().head(0), _frame().drop("forecast"), _frame().drop("series_id"), _frame().drop("week"),
        pl.concat([_frame(), _frame().head(1)]),
        _frame().with_columns(pl.Series("series_id", ["a", None])),
        _frame().with_columns(pl.Series("week", [date(2026, 1, 5), None])),
        _frame([1.0, None]), _frame([1.0, float("nan")]), _frame([1.0, float("inf")]),
        _frame(["100", "1"]), _frame([True, False]),
    ],
    ids=["empty", "missing-value", "missing-id", "missing-time", "duplicate", "null-id",
         "null-time", "null-value", "nan-value", "infinite-value", "string-value", "boolean-value"],
)
def test_invalid_semantic_frames_are_http400(private_api, frame):
    response = private_api.constrain(frame)
    assert response.status_code == 400
    assert "invalid_constraint_frame" in response.json()["detail"]


@pytest.mark.parametrize("params", [{"id_col": "week"}, {"time_col": "forecast"}, {"value_col": ""}])
def test_ambiguous_column_configuration_is_http400(private_api, params):
    assert private_api.constrain(_frame(), params).status_code == 400


@pytest.mark.parametrize("proportional", [True, False])
def test_full_population_checked_before_200_row_preview(private_api, proportional):
    frame = pl.DataFrame([
        {"series_id": f"item-{i:03}", "week": date(2026, 1, 5) + timedelta(weeks=period),
         "forecast": 10000.0 if i == 200 else 100.0, "_cap": i}
        for i in range(201) for period in range(2)
    ])
    response = private_api.constrain(frame, {"aggregate_max": 70, "proportional": proportional})
    assert response.status_code == 200
    data = response.json()
    assert len(data["constrained_preview"]) == 200
    assert data["checked_rows"] == 402
    assert data["checked_periods"] == 2
    assert data["before_total"] == 60000
    assert data["after_total"] == pytest.approx(140, abs=1e-9, rel=1e-12)
    assert data["max_budget_excess"] <= 1e-9 + 1e-12 * 70
    assert data["rows_modified"] == 402


def test_invalid_and_infeasible_rows_beyond_preview_are_rejected(private_api):
    frame = pl.DataFrame({
        "series_id": [f"item-{i}" for i in range(201)],
        "week": [date(2026, 1, 5)] * 201,
        "forecast": [100.0] * 200 + [float("nan")],
    })
    assert private_api.constrain(frame, {"aggregate_max": 200}).status_code == 400
    frame = frame.with_columns(pl.lit(100.0).alias("forecast"))
    first_period = frame.head(200).with_columns(pl.lit(date(2025, 12, 29)).alias("week"))
    response = private_api.constrain(pl.concat([first_period, frame]), {"min_demand": 1, "aggregate_max": 200})
    assert response.status_code == 400
    assert "infeasible_constraints" in response.json()["detail"]


def test_exact_nanosecond_keys_survive_parquet_and_json(private_api):
    frame = pl.DataFrame({
        "series_id": ["a", "a"], "week": pl.Series([1, 2], dtype=pl.Datetime("ns")),
        "forecast": [100.0, 100.0], "_q_list": [["first"], ["second"]],
    })
    response = private_api.constrain(frame, {"aggregate_max": 50})
    assert response.status_code == 200
    data = response.json()
    assert data["checked_periods"] == 2
    assert data["after_total"] == 100
    assert data["constrained_preview"] == [
        {"series_id": "a", "week": "1970-01-01 00:00:00.000000001", "forecast": 50, "_q_list": ["first"]},
        {"series_id": "a", "week": "1970-01-01 00:00:00.000000002", "forecast": 50, "_q_list": ["second"]},
    ]


def test_csv_timestamp_inference_cannot_merge_distinct_periods(private_api):
    frame = pl.DataFrame({
        "series_id": ["a", "b"],
        "week": ["2026-01-05T00:00:00.000000001", "2026-01-05T00:00:00.000000002"],
        "forecast": [100.0, 100.0],
    })
    response = private_api.constrain(frame, {"aggregate_max": 50}, extension="csv")
    assert response.status_code == 200
    data = response.json()
    assert data["checked_periods"] == 2
    assert data["after_total"] == 100
    assert [row["week"] for row in data["constrained_preview"]] == frame["week"].to_list()
    assert [row["forecast"] for row in data["constrained_preview"]] == [50, 50]


def test_element_only_quantiles_are_clipped_and_scope_is_truthful(private_api):
    frame = _frame([5.0, 15.0]).with_columns(
        pl.Series("forecast_p80", [10.0, 20.0]), pl.Series("forecast_p90", [30.0, 40.0]),
    )
    response = private_api.constrain(frame, {"max_capacity": 25})
    assert response.status_code == 200
    data = response.json()
    assert data["constraint_scope"] == "point_and_quantiles"
    assert data["allocation_strategy"] == "element_wise"
    assert [row["forecast"] for row in data["constrained_preview"]] == [5, 15]
    assert [row["forecast_p80"] for row in data["constrained_preview"]] == [10, 20]
    assert [row["forecast_p90"] for row in data["constrained_preview"]] == [25, 25]


@pytest.mark.parametrize("proportional", [True, False])
def test_aggregate_quantiles_fail_even_when_under_budget(private_api, proportional):
    frame = _frame().with_columns(pl.lit(120.0).alias("forecast_p90"))
    response = private_api.constrain(frame, {"aggregate_max": 1000, "proportional": proportional})
    assert response.status_code == 400
    assert "unsupported_aggregate_quantiles" in response.json()["detail"]
    assert "constrained_preview" not in response.json()


@pytest.mark.parametrize("values", [[200.0, 200.0], [float("nan"), 10.0], [None, 10.0]])
def test_crossed_or_nonfinite_quantiles_are_not_disguised_by_clipping(private_api, values):
    frame = _frame().with_columns(
        pl.Series("forecast_p10", values), pl.Series("forecast_p90", [100.0, 100.0]),
    )
    assert private_api.constrain(frame, {"min_demand": 10, "max_capacity": 10}).status_code == 400


@pytest.mark.parametrize("role,status", [(None, 401), ("expired", 401), ("viewer", 403), ("data_scientist", 200), ("admin", 200)])
def test_real_run_pipeline_permissions(private_api, role, status):
    response = private_api.constrain(_frame(), {"aggregate_max": 50}, role=role)
    assert response.status_code == status
    assert not (private_api.root / "runs").exists()


def test_malformed_upload_is400_and_the_real_size_limit_is413(private_api):
    response = private_api.client.post(
        "/forecast/constrain", headers=private_api.headers(),
        files={"file": ("synthetic.parquet", b"not a parquet file", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "invalid_constraint_file" in response.json()["detail"]
    response = private_api.client.post(
        "/forecast/constrain", headers=private_api.headers(),
        files={"file": ("oversize.csv", b"x" * (MAX_UPLOAD_BYTES + 1), "text/csv")},
    )
    assert response.status_code == 413


def test_file_path_is_not_a_server_file_input(private_api):
    canary = private_api.root / "outside.csv"
    content = b"series_id,week,forecast\nsecret,2026-01-05,999\n"
    canary.write_bytes(content)
    response = private_api.client.post(
        "/forecast/constrain", headers=private_api.headers(), data={"file": str(canary)},
    )
    assert response.status_code == 422
    response = private_api.client.post(
        "/forecast/constrain", headers=private_api.headers(), params={"aggregate_max": 50},
        files={"file": (str(canary), _frame().write_csv().encode(), "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["before_total"] == 200
    assert canary.read_bytes() == content


def test_unexpected_computation_error_is_logged500_not_success(private_api, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise RuntimeError("private-computation-detail")

    monkeypatch.setattr("src.forecasting.constrained.apply_point_constraints", fail)
    with caplog.at_level("ERROR", logger="src.api.routers.analytics"):
        response = private_api.constrain(_frame())
    assert response.status_code == 500
    assert response.json() == {"detail": "Constraint computation failed"}
    assert "Constraint computation failed" in caplog.text
    assert "private-computation-detail" not in response.text


def _pipeline(root):
    settings = PlatformConfig(lob="constraint_test")
    settings.forecast.horizon_weeks = 1
    settings.data_quality.fill_gaps = False
    settings.data_quality.min_series_length_weeks = 1
    settings.post_validation.enabled = False
    settings.parallelism.n_workers = 1
    settings.output.forecast_path = str(root / "forecasts")
    settings.output.metrics_path = str(root / "metrics")
    settings.output.runs_path = str(root / "runs")
    actuals = pl.DataFrame([
        {"series_id": sid, "week": date(2024, 1, 1) + timedelta(weeks=i), "quantity": 100.0}
        for sid in ("a", "b") for i in range(64)
    ])
    return ForecastPipeline(settings), actuals


def _snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_constraint_success_and_errors_do_not_mutate_committed_history(private_api):
    pipeline, actuals = _pipeline(private_api.root)
    stored_forecast = pipeline.run(actuals)
    run_id = pipeline.last_run.run_id
    store = RunStore(private_api.root / "runs")
    before = _snapshot(store.root)
    for frame, params, status in [
        (stored_forecast, {"aggregate_max": 50, "proportional": False}, 200),
        (stored_forecast, {"min_demand": 40, "aggregate_max": 50}, 400),
        (stored_forecast.with_columns(pl.lit(120.0).alias("forecast_p90")), {"aggregate_max": 50}, 400),
    ]:
        assert private_api.constrain(frame, params).status_code == status
        assert _snapshot(store.root) == before
        assert store.frame("constraint_test", run_id, "forecast").equals(stored_forecast)
    response = private_api.client.get(
        "/forecast/constraint_test", params={"run_id": run_id}, headers=private_api.headers("viewer"),
    )
    assert response.status_code == 200
    assert [row["forecast"] for row in response.json()["points"]] == [100, 100]


@pytest.mark.parametrize("failure", ["infeasible_constraints", "unsupported_aggregate_quantiles"])
def test_pipeline_constraint_failure_precedes_publication_and_preserves_previous_run(tmp_path, failure):
    pipeline, actuals = _pipeline(tmp_path)
    forecast = pipeline.run(actuals)
    old_id = pipeline.last_run.run_id
    store = RunStore(tmp_path / "runs")
    before = _snapshot(store.root)
    settings = pipeline.config.forecast
    settings.constraints = ConstraintConfig(enabled=True, aggregate_max=50)
    if failure == "infeasible_constraints":
        settings.constraints.min_demand = 40
    else:
        settings.quantiles = [0.1, 0.5, 0.9]
    with pytest.raises(ValueError, match=failure):
        pipeline.run(actuals)
    assert pipeline.last_run is None
    assert _snapshot(store.root) == before
    assert store.frame("constraint_test", old_id, "forecast").equals(forecast)


def test_openapi_declares_actual_strategy_scope_and_measurements(private_api):
    schema = private_api.client.get("/openapi.json").json()
    response = schema["paths"]["/forecast/constrain"]["post"]["responses"]["200"]
    assert response["content"]["application/json"]["schema"]["$ref"].endswith("/ConstrainResponse")
    contract = schema["components"]["schemas"]["ConstrainResponse"]
    assert {"allocation_strategy", "checked_rows", "checked_periods", "constraint_scope"} <= set(contract["required"])
    assert contract["properties"]["allocation_strategy"]["enum"] == ["element_wise", "proportional", "largest_first"]
