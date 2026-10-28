"""Stored-run rankings remain evidence while new runs use the reviewed scorer."""

import hashlib
from dataclasses import replace
from datetime import date, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient
from src.analytics.bi_export import BIExporter
from src.analytics.notebook_api import ForecastAnalytics
from src.api.app import create_app
from src.config.schema import PlatformConfig
from src.forecasting.base import BaseForecaster
from src.forecasting.registry import registry
from src.metrics.aggregation import SCORING_POLICY
from src.metrics.store import MetricStore
from src.pipeline.backtest import BacktestPipeline
from src.pipeline.manifest import PipelineManifest, model_metadata
from src.pipeline.run_store import RunPublicationError, RunStore, new_run_id
from src.utils.calendar import PeriodCalendar


class FixedCandidate(BaseForecaster):
    def __init__(self, name: str, frequency: str = "D"):
        self.name = name
        self.frequency = frequency
        self.values = {"large": 900.0, "small": 0.0} if name == "candidate_a" else {"large": 800.0, "small": 1.0}

    def fit(self, df, target_col="quantity", time_col="week", id_col="series_id"):
        self.calendar = PeriodCalendar.from_dates(df[time_col].unique().to_list(), self.frequency)
        self.origins = dict(df.group_by(id_col).agg(pl.col(time_col).max()).iter_rows())

    def predict(self, horizon, id_col="series_id", time_col="week"):
        return pl.DataFrame([
            {id_col: sid, time_col: label, "forecast": self.values[sid], "model": self.name}
            for sid, origin in self.origins.items()
            for label in self.calendar.future_dates(origin, horizon)
        ])

    def get_params(self):
        return {"frequency": self.frequency, "values_by_series": dict(self.values)}


def _actuals():
    return pl.DataFrame([
        {"series_id": sid, "week": date(2024, 1, 1) + timedelta(days=index), "quantity": value}
        for sid, value in (("large", 1000.0), ("small", 1.0))
        for index in range(64)
    ])


def _config(tmp_path):
    config = PlatformConfig(lob="retail")
    config.forecast.frequency = "D"
    config.forecast.horizon_weeks = 1
    config.forecast.forecasters = ["candidate_a", "candidate_b"]
    config.backtest.n_folds = 1
    config.backtest.val_weeks = 1
    config.backtest.champion_granularity = "lob"
    config.parallelism.n_workers = 1
    config.data_quality.min_series_length_weeks = 1
    config.data_quality.fill_gaps = False
    config.post_validation.enabled = False
    config.metrics = ["wmape", "macro_wmape", "normalized_bias"]
    config.output.metrics_path = str(tmp_path / "legacy-metrics")
    config.output.runs_path = str(tmp_path / "runs")
    return config


def _archive(config, recorded_policy=False):
    run_id = new_run_id()
    actuals = _actuals()
    rows = []
    for model, predictions, old_score in (
        ("candidate_a", {"large": 900.0, "small": 0.0}, 0.55),
        ("candidate_b", {"large": 800.0, "small": 1.0}, 0.10),
    ):
        for sid, actual in (("large", 1000.0), ("small", 1.0)):
            rows.append({
                "run_id": run_id, "run_type": "backtest", "lob": "retail",
                "run_date": date(2024, 3, 4), "model_id": model, "fold": 0,
                "grain_level": "series", "channel": "", "series_id": sid,
                "target_week": date(2024, 3, 4), "forecast_step": 1,
                "actual": actual, "forecast": predictions[sid],
                "wmape": old_score, "normalized_bias": -old_score,
            })
    board = pl.DataFrame({
        "model_id": ["candidate_b", "candidate_a"], "rank": [1, 2],
        "wmape": [0.10, 0.55], "normalized_bias": [-0.10, -0.55], "n_series": [2, 2],
    })
    models = [model_metadata(FixedCandidate(name), "D", 1) for name in config.forecast.forecasters]
    execution = {
        "run_type": "backtest", "frequency": "D", "horizon_periods": 1,
        "validation_periods": 1, "models": models,
    }
    if recorded_policy:
        execution["scoring_contract"] = {
            "version": 1, "policy": "archived-macro-v0", "primary_metric": "wmape",
            "secondary_metric": "normalized_bias", "competition_scope": "lob",
        }
    manifest = PipelineManifest(
        run_id=run_id, run_type="backtest", lob="retail",
        timestamp="2024-03-04T00:00:00.000000+00:00", started_at="2024-03-04T00:00:00.000000+00:00",
        input_row_count=actuals.height, input_series_count=2, forecast_row_count=len(rows),
        forecast_horizon=1, champion_model_id="candidate_b", models=models, execution=execution,
    )
    store = RunStore(config.output.runs_path)
    saved = store.publish(
        manifest, config,
        {"input": actuals, "prepared": actuals.clone(), "metrics": pl.DataFrame(rows),
         "leaderboard": board, "champions": board.head(1)},
        {"skipped": True},
    )
    return store, saved, board


def _app(tmp_path, config):
    return create_app(
        data_dir=str(tmp_path / "data"), metrics_dir=config.output.metrics_path,
        runs_dir=config.output.runs_path, auth_enabled=False, dev_mode=True, jwt_secret="",
    )


def _models(monkeypatch):
    monkeypatch.setattr(
        registry, "build_from_config",
        lambda names, params=None: [FixedCandidate(name, (params or {}).get(name, {}).get("frequency", "D")) for name in names],
    )


@pytest.mark.parametrize("recorded_policy", [False, True])
def test_archived_leaderboard_is_not_rescored_across_readers(tmp_path, recorded_policy):
    config = _config(tmp_path)
    authority, archived, expected = _archive(config, recorded_policy)
    directory = authority.directory("retail", archived.run_id)
    before = {file.name: file.read_bytes() for file in directory.iterdir() if file.is_file()}
    store = MetricStore(config.output.metrics_path, runs_path=config.output.runs_path)
    library = store.leaderboard(lob="retail", run_id=archived.run_id)
    notebook = ForecastAnalytics(
        config.output.metrics_path, runs_path=config.output.runs_path,
    ).model_leaderboard(lob="retail", run_id=archived.run_id)
    exported = BIExporter(str(tmp_path / "exports")).export_leaderboard(
        store, "retail", run_id=archived.run_id,
    )
    for frame in (library, notebook):
        assert frame.to_dicts() == expected.to_dicts()
    assert pl.read_parquet(exported).select(expected.columns).to_dicts() == expected.to_dicts()
    with TestClient(_app(tmp_path, config)) as client:
        response = client.get("/metrics/leaderboard/retail", params={"run_id": archived.run_id})
        assert response.status_code == 200
        body = response.json()
        assert [row["model"] for row in body["entries"]] == ["candidate_b", "candidate_a"]
        assert [row["wmape"] for row in body["entries"]] == [0.10, 0.55]
        assert [row["rank"] for row in body["entries"]] == [1, 2]
        assert body["scoring_provenance"]["source"] == "stored"
        assert body["scoring_provenance"]["status"] == ("recorded" if recorded_policy else "unknown")
        assert body["scoring_provenance"]["policy"] == ("archived-macro-v0" if recorded_policy else None)
        assert body["scoring_provenance"]["primary_metric"] == ("wmape" if recorded_policy else None)
        assert body["scoring_provenance"]["competition_scope"] == ("lob" if recorded_policy else None)
        report = client.post("/governance/export/leaderboard", params={"lob": "retail", "run_id": archived.run_id})
        assert report.status_code == 200
        output = tmp_path / "data" / "bi_exports" / report.json()["export_path"]
        assert pl.read_parquet(output).select(expected.columns).to_dicts() == expected.to_dicts()
    assert {file.name: file.read_bytes() for file in directory.iterdir() if file.is_file()} == before


def test_new_run_stores_policy_and_default_reads_new_pooled_ranking(tmp_path, monkeypatch):
    config = _config(tmp_path)
    authority, archived, old_board = _archive(config)
    before = (authority.directory("retail", archived.run_id) / "leaderboard.parquet").read_bytes()
    _models(monkeypatch)
    result = BacktestPipeline(config).run(_actuals())
    saved = authority.get("retail", result["run_id"])
    contract = saved.execution["scoring_contract"]
    assert contract == {
        "version": 1, "policy": SCORING_POLICY, "primary_metric": "wmape",
        "secondary_metric": "normalized_bias", "competition_scope": "lob",
    }
    configuration = authority.json_member("retail", saved.run_id, "config")
    assert configuration["execution"]["scoring_contract"] == contract
    assert saved.config_hash == hashlib.sha256(
        (authority.directory("retail", saved.run_id) / "config.json").read_bytes()
    ).hexdigest()
    board = authority.frame("retail", saved.run_id, "leaderboard")
    assert board["model_id"].to_list() == ["candidate_a", "candidate_b"]
    assert board["wmape"].to_list() == pytest.approx([101 / 1001, 200 / 1001])
    assert board["rank"].to_list() == [1, 2]
    with TestClient(_app(tmp_path, config)) as client:
        latest = client.get("/metrics/leaderboard/retail").json()
        assert latest["run_id"] == saved.run_id
        assert latest["scoring_provenance"]["policy"] == SCORING_POLICY
        assert latest["entries"][0]["model"] == "candidate_a"
        old = client.get("/metrics/leaderboard/retail", params={"run_id": archived.run_id}).json()
        assert [row["wmape"] for row in old["entries"]] == old_board["wmape"].to_list()
        assert old["scoring_provenance"]["status"] == "unknown"
    assert (authority.directory("retail", archived.run_id) / "leaderboard.parquet").read_bytes() == before


def test_new_macro_and_cohort_results_keep_their_actual_criteria_and_scope(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.backtest.primary_metric = "macro_wmape"
    config.backtest.champion_granularity = "series"
    _models(monkeypatch)
    result = BacktestPipeline(config).run(_actuals())
    with TestClient(_app(tmp_path, config)) as client:
        response = client.get("/metrics/leaderboard/retail", params={"run_id": result["run_id"]})
        assert response.status_code == 200
        body = response.json()
        assert body["scoring_provenance"]["primary_metric"] == "macro_wmape"
        assert body["scoring_provenance"]["competition_scope"] == "series_id"
        assert {row["group_key"] for row in body["entries"]} == {"large", "small"}
        winners = {row["group_key"]: row["model"] for row in body["entries"] if row["rank"] == 1}
        assert winners == {"large": "candidate_a", "small": "candidate_b"}
        assert all(row["wmape"] is None and row["scores"]["macro_wmape"] is not None for row in body["entries"])


def test_multiple_archived_runs_require_explicit_scope_without_recomputation(tmp_path):
    config = _config(tmp_path)
    _, archived, expected = _archive(config)
    _archive(config)
    store = MetricStore(config.output.metrics_path, runs_path=config.output.runs_path)
    with pytest.raises(ValueError, match="run_id|latest"):
        store.leaderboard(lob="retail")
    assert store.leaderboard(lob="retail", run_id=archived.run_id).to_dicts() == expected.to_dicts()
    with pytest.raises(ValueError, match="stored|Stored"):
        store.leaderboard(lob="retail", run_id=archived.run_id, primary_metric="macro_wmape")


def test_stored_read_paths_do_not_invoke_the_current_scorer(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _, archived, expected = _archive(config)

    def forbidden(*args, **kwargs):
        pytest.fail("A stored leaderboard was sent to today's scorer")

    monkeypatch.setattr(MetricStore, "leaderboard_from_records", staticmethod(forbidden))
    store = MetricStore(config.output.metrics_path, runs_path=config.output.runs_path)
    assert store.leaderboard(lob="retail", latest=True).to_dicts() == expected.to_dicts()
    with TestClient(_app(tmp_path, config)) as client:
        for params in ({}, {"run_id": archived.run_id}):
            assert client.get("/metrics/leaderboard/retail", params=params).status_code == 200
        assert client.post("/governance/export/leaderboard", params={
            "lob": "retail", "run_id": archived.run_id,
        }).status_code == 200


def test_inconsistent_scoring_configuration_never_commits(tmp_path):
    config = _config(tmp_path)
    authority, archived, _ = _archive(config, recorded_policy=True)
    execution = {**archived.execution, "scoring_contract": {
        **archived.execution["scoring_contract"], "primary_metric": "macro_wmape",
    }}
    candidate = replace(archived, run_id=new_run_id(), execution=execution)
    frames = {name: authority.frame("retail", archived.run_id, name) for name in (
        "input", "prepared", "metrics", "champions", "leaderboard",
    )}
    frames["metrics"] = frames["metrics"].with_columns(pl.lit(candidate.run_id).alias("run_id"))
    with pytest.raises(RunPublicationError) as rejected:
        authority.publish(candidate, config, frames, {"skipped": True})
    assert rejected.value.outcome == "not_committed"
    assert [item.run_id for item in authority.list("retail")] == [archived.run_id]
