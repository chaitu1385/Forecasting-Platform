"""Local publication visibility, failure boundaries and immutable provenance."""

import hashlib
import json
import os
import subprocess
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.config.schema import PlatformConfig
from src.pipeline.manifest import PipelineManifest, canonical_json, configuration_snapshot
from src.pipeline.run_store import (
    MANIFEST_NAME, RunIntegrityError, RunNotFoundError, RunPublicationError,
    RunStore, atomic_frame_append, new_run_id, resolve_run_root,
)


def _bundle(value=10.0, *, run_id=None, started_at="2026-01-01T12:00:00.000000+00:00", lob="retail"):
    config = PlatformConfig(lob=lob)
    config.forecast.horizon_weeks = 1
    config.post_validation.enabled = False
    actuals = pl.DataFrame({
        "series_id": ["synthetic"], "week": [date(2024, 1, 1)], "quantity": [value],
    })
    forecast = pl.DataFrame({
        "series_id": ["synthetic"], "week": [date(2024, 1, 8)],
        "forecast": [value], "model": ["naive_seasonal"],
    })
    manifest = PipelineManifest(
        run_id=run_id or new_run_id(), run_type="forecast", lob=lob,
        started_at=started_at, timestamp=started_at, input_row_count=1,
        input_series_count=1, champion_model_id="naive_seasonal",
        forecast_row_count=1, forecast_horizon=1,
        models=[{"name": "naive_seasonal", "parameters": {"season_length": 52}}],
        execution={"model_id": "naive_seasonal", "horizon_periods": 1, "frequency": "W"},
    )
    return manifest, config, {"input": actuals, "prepared": actuals.clone(), "forecast": forecast}


def _publish(store, value=10.0, **kwargs):
    manifest, config, frames = _bundle(value, **kwargs)
    return store.publish(manifest, config, frames, {"skipped": True})


def _snapshot(directory):
    return {str(file.relative_to(directory)): file.read_bytes() for file in directory.rglob("*") if file.is_file()}


def test_complete_bundle_agrees_with_its_actual_stored_bytes(tmp_path):
    store = RunStore(tmp_path / "runs")
    committed = _publish(store)
    loaded = store.get("retail", committed.run_id, "forecast")
    assert loaded == committed
    assert loaded.schema_version == 1 and loaded.status == "completed"
    assert loaded.input_snapshot_stage == "normalized_raw_dataframe"
    for descriptor in loaded.artifacts.values():
        file = store.directory("retail", loaded.run_id) / descriptor["file"]
        assert descriptor["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    assert loaded.config_hash == hashlib.sha256(canonical_json(store.json_member("retail", loaded.run_id, "config"))).hexdigest()
    assert store.frame("retail", loaded.run_id, "forecast")["forecast"].to_list() == [10.0]
    with pytest.raises(RunNotFoundError):
        store.get("retail", loaded.run_id, "backtest")


def test_case_variant_publication_cannot_poison_successful_runs(tmp_path, record_property):
    store = RunStore(tmp_path / "runs")
    first = _publish(store, 10.0, lob="retail")
    before = _snapshot(store.directory("retail", first.run_id))
    aliases_existing_partition = (store.root / "RETAIL").exists()
    record_property("case_alias_filesystem", aliases_existing_partition)
    rejected = None
    try:
        second = _publish(store, 20.0, lob="RETAIL")
    except RunPublicationError as exc:
        rejected = exc
    assert store.get("retail", first.run_id).lob == "retail"
    if rejected is None:
        assert store.get("RETAIL", second.run_id).lob == "RETAIL"
    published = store.list()
    if aliases_existing_partition:
        assert rejected is not None and rejected.outcome == "not_committed"
        assert [item.run_id for item in published] == [first.run_id]
        assert store.latest("retail", "forecast").run_id == first.run_id
    else:
        assert rejected is None
        assert {item.lob for item in published} == {"retail", "RETAIL"}
        assert store.get("RETAIL", second.run_id).lob == "RETAIL"
        assert store.frame("RETAIL", second.run_id, "forecast")["forecast"].to_list() == [20.0]
    assert store.get("retail", first.run_id).lob == "retail"
    assert _snapshot(store.directory("retail", first.run_id)) == before


def test_real_ensemble_component_parameters_are_recorded_without_mutating_predictions():
    from src.forecasting.ensemble import WeightedEnsembleForecaster
    from src.forecasting.naive import SeasonalNaiveForecaster
    from src.pipeline.manifest import model_metadata

    actuals = pl.DataFrame({
        "series_id": ["synthetic"] * 8,
        "week": [date(2024, 1, index + 1) for index in range(8)],
        "quantity": [float(index + 1) for index in range(8)],
    })
    snapshots = []
    for season in (2, 3):
        model = WeightedEnsembleForecaster(
            [SeasonalNaiveForecaster(season_length=season)], {"naive_seasonal": 1.0},
        )
        model.fit(actuals)
        before = model.predict(2)
        snapshots.append(model_metadata(model, "W", 2))
        assert model.predict(2).equals(before)
    assert snapshots[0] != snapshots[1]
    config = PlatformConfig(lob="retail")
    digests = [
        hashlib.sha256(canonical_json(configuration_snapshot(config, {"models": [snapshot]}))).hexdigest()
        for snapshot in snapshots
    ]
    assert digests[0] != digests[1]
    assert snapshots[0]["components"][0]["parameters"]["season_length"] == 2
    assert snapshots[1]["components"][0]["parameters"]["season_length"] == 3


def test_nested_supported_composites_redact_component_secrets_and_report_omissions(monkeypatch):
    from src.config.schema import ConstraintConfig
    from src.forecasting.constrained import ConstrainedDemandEstimator
    from src.forecasting.ensemble import WeightedEnsembleForecaster
    from src.forecasting.naive import SeasonalNaiveForecaster
    from src.pipeline.manifest import model_metadata

    leaf = SeasonalNaiveForecaster(season_length=2)
    parameters = leaf.get_params()
    secret = "synthetic-component-credential-do-not-persist"
    private = r"C:\private-component\weights"
    leaf._fitted_private_state = "fitted-state-not-a-declared-parameter"
    monkeypatch.setattr(leaf, "get_params", lambda: {
        **parameters, "api_key": secret,
        "options": {"endpoint": "https://operator:password@example.invalid/model", "weights_path": private},
    })
    model = ConstrainedDemandEstimator(
        WeightedEnsembleForecaster([leaf], {"naive_seasonal": 1.0}), ConstraintConfig(),
    )
    snapshot = model_metadata(model, "W", 2)
    encoded = canonical_json(snapshot).decode()
    for forbidden in (secret, private, "operator:password", leaf._fitted_private_state):
        assert forbidden not in encoded
    component = snapshot["components"][0]["components"][0]
    assert component["parameters"]["season_length"] == 2
    assert {"parameters.api_key", "parameters.options.endpoint", "parameters.options.weights_path"}.issubset(
        component["parameters_omitted"]
    )


def test_stacked_constraints_persist_each_nodes_own_parameters_and_config_identity(tmp_path):
    from src.config.schema import ConstraintConfig
    from src.forecasting.constrained import ConstrainedDemandEstimator
    from src.forecasting.naive import SeasonalNaiveForecaster
    from src.pipeline.manifest import model_metadata

    store = RunStore(tmp_path / "runs")
    saved = []
    configurations = []
    predictions = []
    for cap in (10.0, 20.0):
        manifest, config, frames = _bundle(50.0)
        model = ConstrainedDemandEstimator(
            ConstrainedDemandEstimator(
                SeasonalNaiveForecaster(season_length=1),
                ConstraintConfig(enabled=True, max_capacity=30.0),
            ),
            ConstraintConfig(enabled=True, max_capacity=cap),
        )
        model.fit(frames["input"])
        before = model.predict(1)
        manifest.models = [model_metadata(model, "W", 1)]
        manifest.execution["models"] = manifest.models
        assert model.predict(1).equals(before)
        frames["forecast"] = before
        committed = store.publish(manifest, config, frames, {"skipped": True})
        saved.append(store.get("retail", committed.run_id))
        configurations.append(store.json_member("retail", committed.run_id, "config"))
        predictions.append(store.frame("retail", committed.run_id, "forecast")["forecast"].to_list())
    assert predictions == [[10.0], [20.0]]
    assert configurations[0]["settings"] == configurations[1]["settings"]
    assert [item.models[0]["parameters"]["max_capacity"] for item in saved] == [10.0, 20.0]
    assert all(item.models[0]["components"][0]["parameters"]["max_capacity"] == 30.0 for item in saved)
    assert [item["execution"]["models"][0]["parameters"]["max_capacity"] for item in configurations] == [10.0, 20.0]
    assert saved[0].config_hash != saved[1].config_hash
    assert saved[0].models != saved[1].models


def test_composite_metadata_rejects_cycles_before_reading_recursive_parameters():
    from src.config.schema import ConstraintConfig
    from src.forecasting.constrained import ConstrainedDemandEstimator
    from src.forecasting.naive import SeasonalNaiveForecaster
    from src.pipeline.manifest import model_metadata

    model = ConstrainedDemandEstimator(SeasonalNaiveForecaster(), ConstraintConfig())
    model.base = model
    with pytest.raises(ValueError, match="acyclic"):
        model_metadata(model, "W", 2)


def test_corrupt_nested_model_metadata_is_an_explicit_integrity_error(tmp_path):
    store = RunStore(tmp_path / "runs")
    committed = _publish(store)
    marker = store.directory("retail", committed.run_id) / MANIFEST_NAME
    changed = json.loads(marker.read_text())
    changed["models"][0]["components"] = [{"name": "invalid-child", "parameters": []}]
    marker.write_bytes(canonical_json(changed))
    with pytest.raises(RunIntegrityError, match="invalid_model_metadata"):
        store.get("retail", committed.run_id)


@pytest.mark.parametrize("filename", [
    "input.parquet", "prepared.parquet", "forecast.parquet", "config.json",
    "validation.json", ".manifest.pending.json",
])
def test_each_required_write_failure_stays_invisible_and_preserves_old_bytes(tmp_path, monkeypatch, filename):
    from src.pipeline import run_store

    store = RunStore(tmp_path / "runs")
    previous = _publish(store)
    old_directory = store.directory("retail", previous.run_id)
    before = _snapshot(old_directory)
    original = run_store._write_new

    def fail_member(path, writer):
        if path.name == filename:
            raise OSError("synthetic write failure")
        return original(path, writer)

    monkeypatch.setattr(run_store, "_write_new", fail_member)
    attempted, config, frames = _bundle(20.0)
    with pytest.raises(RunPublicationError) as error:
        store.publish(attempted, config, frames, {"skipped": True})
    assert error.value.outcome == "not_committed"
    assert error.value.run_id == attempted.run_id
    assert [run.run_id for run in store.list()] == [previous.run_id]
    assert _snapshot(old_directory) == before
    with pytest.raises(RunNotFoundError):
        store.get("retail", attempted.run_id)
    assert json.loads((store.directory("retail", attempted.run_id) / "failure.json").read_text())["status"] == "not_committed"


def test_manifest_serialization_failure_never_commits(tmp_path, monkeypatch):
    from src.pipeline import run_store

    store = RunStore(tmp_path / "runs")
    original = run_store.canonical_json

    def fail_manifest(value):
        if value.get("status") == "completed":
            raise TypeError("synthetic manifest serialization failure")
        return original(value)

    monkeypatch.setattr(run_store, "canonical_json", fail_manifest)
    with pytest.raises(RunPublicationError) as error:
        _publish(store)
    assert error.value.stage == "manifest"
    assert error.value.outcome == "not_committed"
    assert store.list() == []


def test_failed_commit_link_does_not_publish_or_overwrite(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs")
    previous = _publish(store)
    before = _snapshot(store.directory("retail", previous.run_id))

    def reject_link(*args, **kwargs):
        raise OSError("hard links unavailable in synthetic filesystem")

    monkeypatch.setattr(os, "link", reject_link)
    with pytest.raises(RunPublicationError) as error:
        _publish(store, 20.0)
    assert error.value.stage == "commit" and error.value.outcome == "not_committed"
    assert [run.run_id for run in store.list()] == [previous.run_id]
    assert _snapshot(store.directory("retail", previous.run_id)) == before


@pytest.mark.parametrize("failure", ["link_receipt", "directory_fsync"])
def test_post_commit_failure_reports_committed_and_never_deletes_the_run(tmp_path, monkeypatch, failure):
    from src.pipeline import run_store

    store = RunStore(tmp_path / "runs")
    original_link = os.link
    original_sync = run_store._fsync_directory

    def link_then_fail(source, destination):
        original_link(source, destination)
        raise OSError("synthetic lost commit receipt")

    def fail_confirmation(directory):
        if (directory / MANIFEST_NAME).exists():
            raise OSError("synthetic post-commit directory sync failure")
        return original_sync(directory)

    if failure == "link_receipt":
        monkeypatch.setattr(os, "link", link_then_fail)
    else:
        monkeypatch.setattr(run_store, "_fsync_directory", fail_confirmation)
    attempted, config, frames = _bundle(20.0)
    with pytest.raises(RunPublicationError) as error:
        store.publish(attempted, config, frames, {"skipped": True})
    assert error.value.outcome == "committed"
    assert error.value.run_id == attempted.run_id
    assert [run.run_id for run in store.list()] == [attempted.run_id]
    assert store.frame("retail", attempted.run_id, "forecast")["forecast"].to_list() == [20.0]
    assert not (store.directory("retail", attempted.run_id) / "failure.json").exists()


def test_id_collision_never_modifies_the_existing_run(tmp_path):
    store = RunStore(tmp_path / "runs")
    existing = _publish(store)
    before = _snapshot(store.root)
    with pytest.raises(RunPublicationError) as error:
        _publish(store, 20.0, run_id=existing.run_id)
    assert error.value.outcome == "not_committed"
    assert _snapshot(store.root) == before


def test_older_attempt_finishing_last_cannot_displace_newer_run(tmp_path):
    store = RunStore(tmp_path / "runs")
    newer = _publish(store, 20.0, started_at="2026-01-01T12:00:01.000000+00:00")
    older = _publish(store, 10.0, started_at="2026-01-01T12:00:00.000000+00:00")
    assert store.latest("retail", "forecast").run_id == newer.run_id
    assert store.frame("retail", older.run_id, "forecast")["forecast"].to_list() == [10.0]


def test_unknown_version_or_corrupt_latest_is_not_a_healthy_empty_or_old_result(tmp_path):
    store = RunStore(tmp_path / "runs")
    _publish(store, 10.0, started_at="2026-01-01T12:00:00.000000+00:00")
    current = _publish(store, 20.0, started_at="2026-01-01T12:00:01.000000+00:00")
    file = store.directory("retail", current.run_id) / MANIFEST_NAME
    payload = json.loads(file.read_text())
    payload["schema_version"] = 999
    file.write_text(json.dumps(payload))
    with pytest.raises(RunIntegrityError, match="unsupported_schema_version"):
        store.latest("retail", "forecast")
    file.write_text("{")
    with pytest.raises(RunIntegrityError, match="invalid_manifest_json"):
        store.list()


def test_member_corruption_and_manifest_path_injection_are_explicit_errors(tmp_path):
    store = RunStore(tmp_path / "runs")
    committed = _publish(store)
    directory = store.directory("retail", committed.run_id)
    forecast = directory / "forecast.parquet"
    forecast.write_bytes(b"corrupt")
    with pytest.raises(RunIntegrityError, match="artifact_checksum_mismatch"):
        store.get("retail", committed.run_id)
    marker = directory / MANIFEST_NAME
    payload = json.loads(marker.read_text())
    payload["artifacts"]["forecast"]["file"] = "../outside.parquet"
    marker.write_text(json.dumps(payload))
    with pytest.raises(RunIntegrityError, match="unsafe_artifact_member"):
        store.get("retail", committed.run_id)


@pytest.mark.parametrize("identifier", ["../outside", "a" * 31, "A" * 32, "/absolute", "a" * 32 + "\\file"])
def test_run_selectors_are_not_filesystem_paths(tmp_path, identifier):
    with pytest.raises(ValueError):
        RunStore(tmp_path).get("retail", identifier)


def test_custom_output_root_fallback_is_operation_local_and_explicit_authority_wins(tmp_path):
    config = PlatformConfig()
    config.output.forecast_path = str(tmp_path / "custom-forecast")
    config.output.metrics_path = str(tmp_path / "other-metrics")
    assert resolve_run_root(config, "forecast") == tmp_path / "custom-forecast" / "runs"
    assert resolve_run_root(config, "backtest") == tmp_path / "other-metrics" / "runs"
    config.output.runs_path = str(tmp_path / "explicit")
    assert resolve_run_root(config, "forecast") == resolve_run_root(config, "backtest") == tmp_path / "explicit"


def test_configuration_snapshot_excludes_secrets_and_locations_and_has_stable_identity():
    config = PlatformConfig(lob="retail")
    config.description = "PRIVATE_DESCRIPTION"
    config.ai.api_key_env_var = "PRIVATE_KEY_NAME"
    config.observability.alerts.webhook_url = "https://private:credential@example.invalid/hook?token=PRIVATE_TOKEN"
    config.forecast.external_regressors.future_features_path = "C:\\private\\future.parquet"
    config.transition.override_store_path = "/private/overrides"
    config.output.runs_path = "/private/runs"
    snapshot = canonical_json(configuration_snapshot(config))
    assert b"PRIVATE_" not in snapshot and b"private" not in snapshot
    assert snapshot == canonical_json(configuration_snapshot(config))
    config.forecast.horizon_weeks += 1
    assert snapshot != canonical_json(configuration_snapshot(config))


def test_standalone_metric_appends_are_atomic_and_never_second_named(tmp_path):
    frame = pl.DataFrame({"value": [1.0]})
    first = atomic_frame_append(frame, tmp_path / "metrics")
    before = first.read_bytes()
    second = atomic_frame_append(pl.DataFrame({"value": [2.0]}), tmp_path / "metrics")
    assert first != second and first.read_bytes() == before
    assert pl.read_parquet(first)["value"].to_list() == [1.0]
    assert pl.read_parquet(second)["value"].to_list() == [2.0]
    assert len(list((tmp_path / "metrics").glob("*.parquet"))) == 2


def test_directory_symlink_or_windows_junction_cannot_redirect_run_lookup(tmp_path):
    outside = RunStore(tmp_path / "outside")
    committed = _publish(outside)
    store = RunStore(tmp_path / "authority")
    partition = store.root / "retail"
    partition.mkdir(parents=True)
    link = partition / committed.run_id
    target = outside.directory("retail", committed.run_id)
    if os.name == "nt":
        result = subprocess.run(
            [os.environ["COMSPEC"], "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, "Could not create the owned test-only NTFS junction"
    else:
        link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RunIntegrityError, match="symlink_or_reparse_point"):
        store.get("retail", committed.run_id)
    with pytest.raises(RunIntegrityError, match="symlink_or_reparse_point"):
        store.list()


def test_run_config_disagreement_is_rejected_before_visibility(tmp_path):
    store = RunStore(tmp_path / "runs")
    manifest, config, frames = _bundle()
    manifest.forecast_horizon += 1
    with pytest.raises(RunPublicationError) as error:
        store.publish(manifest, config, frames, {"skipped": True})
    assert error.value.outcome == "not_committed"
    assert store.list() == []


def test_no_hash_fallback_or_secret_model_parameter_serialization(tmp_path):
    from src.pipeline.manifest import model_metadata

    class Model:
        name = "synthetic"
        frequency = "W"

        def get_params(self):
            return {"alpha": 0.1, "api_key": "PRIVATE_CREDENTIAL", "data_path": "/private/model",
                    "endpoint": "https://user:PRIVATE_CREDENTIAL@example.invalid"}

    metadata = model_metadata(Model(), "W", 4)
    encoded = canonical_json(metadata)
    assert b"PRIVATE_CREDENTIAL" not in encoded and b"/private/model" not in encoded
    assert metadata["parameters"]["alpha"] == 0.1
    assert metadata["parameters_omitted"]
    with pytest.raises(TypeError, match="Unsupported metadata"):
        canonical_json({"unserializable": object()})
