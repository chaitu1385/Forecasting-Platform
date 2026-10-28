"""Allowlisted installed-distribution and loaded-runtime provenance."""

import importlib
import importlib.metadata
import json

import polars as pl
import pytest
from src.pipeline.manifest import runtime_metadata


@pytest.mark.parametrize("version", ["3.2.0", "3.4.1"])
def test_cpu_xgboost_is_recorded_under_the_existing_logical_version_key(monkeypatch, version):
    original = importlib.metadata.version

    def installed(name):
        if name == "xgboost":
            raise importlib.metadata.PackageNotFoundError(name)
        if name == "xgboost-cpu":
            return version
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", installed)
    observed = runtime_metadata()
    assert observed["runtime_versions"]["xgboost"] == version
    assert observed["runtime_distributions"]["xgboost-cpu"] == version
    assert "xgboost" not in observed["runtime_distributions"]


@pytest.mark.parametrize("version", ["3.2.0", None])
def test_standard_xgboost_and_absent_optional_backend_remain_explicit(monkeypatch, version):
    original = importlib.metadata.version

    def installed(name):
        if name == "xgboost" and version is not None:
            return version
        if name in {"xgboost", "xgboost-cpu"}:
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", installed)
    observed = runtime_metadata()
    assert observed["runtime_versions"]["xgboost"] == version
    assert observed["runtime_distributions"].get("xgboost") == version
    assert "xgboost-cpu" not in observed["runtime_distributions"]


def test_polars_provenance_identifies_the_actual_loaded_runtime():
    observed = runtime_metadata()
    variant = importlib.import_module("polars._plr").RUNTIME_REPR
    distribution = {
        "rt32": "polars-runtime-32",
        "rt64": "polars-runtime-64",
        "rtcompat": "polars-runtime-compat",
    }[variant]
    assert observed["runtime_versions"]["polars"] == pl.__version__
    assert observed["polars_runtime"] == {
        "variant": variant,
        "distribution": distribution,
        "version": importlib.metadata.version(distribution),
    }


def test_polars_variant_comes_from_loader_not_a_guessed_default(monkeypatch):
    runtime = importlib.import_module("polars._plr")
    original = importlib.metadata.version
    monkeypatch.setattr(runtime, "RUNTIME_REPR", "rtcompat")
    monkeypatch.setattr(
        importlib.metadata, "version",
        lambda name: "2.0.0" if name == "polars-runtime-compat" else original(name),
    )
    observed = runtime_metadata()
    assert observed["polars_runtime"]["distribution"] == "polars-runtime-compat"
    assert observed["runtime_distributions"]["polars-runtime-compat"] == "2.0.0"
    assert "polars-runtime-32" not in observed["runtime_distributions"]


@pytest.mark.parametrize("variant", ["unrecognized-runtime", None])
def test_unknown_polars_runtime_is_not_guessed(monkeypatch, variant):
    monkeypatch.setattr(importlib.import_module("polars._plr"), "RUNTIME_REPR", variant)
    with pytest.raises(ValueError, match="runtime identity"):
        runtime_metadata()


def test_loaded_polars_runtime_requires_distribution_metadata(monkeypatch):
    original = importlib.metadata.version

    def installed(name):
        if name.startswith("polars-runtime-"):
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", installed)
    with pytest.raises(ValueError, match="runtime distribution metadata is missing"):
        runtime_metadata()


def test_ambiguous_xgboost_installations_are_not_silently_mislabeled(monkeypatch):
    original = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version",
        lambda name: "3.4.1" if name in {"xgboost", "xgboost-cpu"} else original(name),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        runtime_metadata()


def test_runtime_provenance_does_not_dump_environment_or_paths(monkeypatch):
    secret = "synthetic-provenance-sentinel-do-not-persist"
    private = r"C:\private-runtime-cache\secret"
    monkeypatch.setenv("JWT_SECRET_KEY", secret)
    monkeypatch.setenv("PROVENANCE_TEST_PRIVATE_CACHE", private)
    encoded = json.dumps(runtime_metadata())
    assert secret not in encoded and private not in encoded
    assert "JWT_SECRET_KEY" not in encoded and "PROVENANCE_TEST_PRIVATE_CACHE" not in encoded
