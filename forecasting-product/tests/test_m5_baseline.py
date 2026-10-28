"""Bounded gate controls and immutable approved-reference bindings; no model fits."""

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path

import pytest

from tests.integration import baseline as gates

REFERENCE = json.loads(
    (Path(__file__).parent / "integration" / "baselines" / "m5_daily_reference.json")
    .read_text(encoding="utf-8")
)

INITIAL_REFERENCES = (
    ("py311", "3.11.18", "9c8258c1079cae76dcca1841b190dc35966574c42ba3e0543a477101cc1d954a"),
    ("py312", "3.12.16", "354db091f450cd94eb36cb1fcc7100f6d1703cbe3f4d620e678847e52306ee15"),
    ("py313", "3.13.18", "44458444276842c2ac99d53793af5695367b7b46a25765535e8743d6064defc0"),
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@pytest.mark.parametrize("profile,python_version,digest", INITIAL_REFERENCES)
def test_initial_linux_reference_preserves_its_reviewed_capture(monkeypatch, profile, python_version, digest):
    """Verify the stored audit binding, not current-machine native qualification."""
    path = gates.BASELINES_DIR / f"m5_daily_baseline_{profile}_linux.json"
    raw = path.read_bytes()
    assert b"\r" not in raw
    document = json.loads(raw)
    candidate = document["provenance"]
    assert document["approved_baseline"] is True and document["seed"] is None
    assert document["approval"] == {
        "reviewed_by": "GitHub Copilot, independent coordinator review",
        "review_reference": "https://github.com/chaitu1385/Forecasting-Platform/pull/124#issuecomment-6455007868",
        "approved_at": "2026-10-27T22:19:15Z",
        "candidate_sha256": digest,
    }
    assert candidate["approved_baseline"] is False
    assert _digest(candidate) == digest
    assert candidate["ci_source_sha"] == "0af8188d49671ee8c0ef843a3213e51efbd33c37"
    assert candidate["ci_run_id"] == "40587318909"
    assert candidate["python"] == python_version
    assert candidate["system"] == "Linux" and candidate["process_machine"] == "x86_64"
    assert candidate["profile_lock_name"] == f"{profile}/dev.txt"
    assert document["per_model_wmape"] == {
        model: metrics["wmape"] for model, metrics in candidate["scalar_reference_scores"].items()
    }
    assert document["champion_wmape"] == document["per_model_wmape"][candidate["champion_model"]]
    assert document["fva_vs_naive"] == document["naive_wmape"] - document["champion_wmape"]
    assert "First empirical reference" in document["notes"]
    monkeypatch.setenv("M5_BASELINE_PATH", str(path))
    loaded = gates.require_baseline("daily", unittest.TestCase(), observed=candidate)
    assert loaded.raw == document
    for other, _, _ in INITIAL_REFERENCES:
        if other == profile:
            continue
        observed = json.loads(
            (gates.BASELINES_DIR / f"m5_daily_baseline_{other}_linux.json").read_text()
        )["provenance"]
        with pytest.raises(AssertionError, match="incompatible"):
            gates.verify_baseline_compatibility(unittest.TestCase(), loaded, observed)


def test_profile_specific_references_do_not_become_an_implicit_generic_reference(monkeypatch):
    monkeypatch.delenv("M5_BASELINE_PATH", raising=False)
    assert gates.baseline_path("daily") == gates.BASELINES_DIR / "m5_daily_baseline.json"
    assert gates.load_baseline("daily") is None


def _candidate():
    scores = {
        model: {"wmape": wmape, "normalized_bias": bias, "mae": wmape * 7945 / 5880}
        for model, wmape, bias in [
            ("naive_seasonal", 7921 / 7945, 479 / 7945),
            ("auto_arima", 0.8, 0.1),
            ("lgbm_direct", 0.75, 0.0),
        ]
    }
    packages = {
        name: None if name in ("polars-runtime-compat", "xgboost") else "1.0"
        for name in gates.PROVENANCE_PACKAGES
    }
    return {
        "approved_baseline": False,
        "fixture_utf8_lf_sha256": REFERENCE["fixture_sha256"],
        "config_utf8_lf_sha256": REFERENCE["config_sha256"],
        "scoring_policy": "raw-observations-v1",
        "models": sorted(scores),
        "input_rows": 76755, "series_count": 105, "folds": 2,
        "validation_periods": 28, "gap_periods": 0, "scored_observations_per_model": 5880,
        "scalar_reference_scores": scores, "champion_model": "lgbm_direct",
        "python": "3.12.0", "system": "Linux", "process_machine": "x86_64",
        "os_release": "synthetic-unit-host", "interpreter_platform": "linux-x86_64",
        "packages": packages,
        "raw_predictions_sha256": "6" * 64, "raw_predictions_rows": 17640,
        "ci_run_id": "123", "ci_source_sha": "a" * 40, "ci_runner_arch": "X64",
        "profile_lock_name": "py312/dev.txt", "profile_lock_sha256": "1" * 64,
        "profile_manifest_sha256": "2" * 64,
        "profile_install_report_sha256": "3" * 64,
        "profile_qualification_sha256": "4" * 64,
        "profile_package_versions": {name: value for name, value in packages.items() if value is not None},
        "profile_wheels": {
            name: {"version": "1.0", "filename": f"{name}-unit.whl", "sha256": "5" * 64}
            for name, value in packages.items() if value is not None
        },
    }


def _document():
    candidate = _candidate()
    return {
        "schema_version": 2, "kind": "m5_empirical_reference",
        "frequency": "daily", "approved_baseline": True,
        "hash_format": REFERENCE["hash_format"],
        "fixture_sha256": REFERENCE["fixture_sha256"],
        "config_sha256": REFERENCE["config_sha256"],
        "per_model_wmape": {
            model: score["wmape"] for model, score in candidate["scalar_reference_scores"].items()
        },
        "champion_model": "lgbm_direct", "champion_wmape": 0.75,
        "naive_wmape": 7921 / 7945, "fva_vs_naive": 7921 / 7945 - 0.75,
        "provenance": candidate,
        "approval": {
            "reviewed_by": "unit-test-reviewer",
            "review_reference": "unit-test-only-not-a-real-approval",
            "approved_at": "2026-01-01T00:00:00+00:00",
            "candidate_sha256": _digest(candidate),
        },
    }


def test_missing_baseline_is_an_explicit_failure_not_a_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "BASELINES_DIR", tmp_path)
    monkeypatch.delenv("M5_BASELINE_PATH", raising=False)
    with pytest.raises(AssertionError, match="approved empirical"):
        try:
            gates.require_baseline("daily", unittest.TestCase())
        except unittest.SkipTest as exc:
            pytest.fail(f"Missing empirical reference was skipped: {exc}")


@pytest.mark.parametrize(
    "defect",
    ["unapproved", "missing_approval", "unbound_approval", "wrong_fixture",
     "wrong_policy", "wrong_coverage", "missing_model", "nonfinite", "wrong_winner",
     "missing_profile", "missing_wheel", "wrong_fva"],
)
def test_malformed_or_unapproved_reference_cannot_be_loaded(defect):
    document = _document()
    if defect == "unapproved":
        document["approved_baseline"] = False
    elif defect == "missing_approval":
        document.pop("approval")
    elif defect == "unbound_approval":
        document["approval"]["candidate_sha256"] = "0" * 64
    elif defect == "wrong_fixture":
        document["fixture_sha256"] = "0" * 64
    elif defect == "wrong_policy":
        document["provenance"]["scoring_policy"] = "mean-of-observation-wmape"
    elif defect == "wrong_coverage":
        document["provenance"]["scored_observations_per_model"] -= 1
    elif defect == "missing_model":
        document["per_model_wmape"].pop("auto_arima")
    elif defect == "nonfinite":
        document["champion_wmape"] = float("nan")
    elif defect == "wrong_winner":
        document["champion_model"] = "auto_arima"
    elif defect == "missing_profile":
        document["provenance"]["profile_lock_sha256"] = None
    elif defect == "missing_wheel":
        document["provenance"]["profile_wheels"].pop("numpy")
    else:
        document["fva_vs_naive"] = 0.01
    with pytest.raises(ValueError):
        gates.Baseline.from_dict(document)


def test_complete_model_regression_and_fva_assertions_keep_original_thresholds():
    reference = gates.Baseline.from_dict(_document())
    case = unittest.TestCase()
    gates.assert_no_regression(case, reference, 0.75, reference.per_model_wmape)
    gates.assert_fva_preserved(case, reference, reference.naive_wmape, 0.75)
    with pytest.raises(AssertionError, match="Champion WMAPE regression"):
        gates.assert_no_regression(case, reference, 0.75 * 1.051, reference.per_model_wmape)
    degraded = dict(reference.per_model_wmape, auto_arima=0.8 * 1.101)
    with pytest.raises(AssertionError, match="auto_arima"):
        gates.assert_no_regression(case, reference, 0.75, degraded)
    with pytest.raises(AssertionError, match="FVA regression"):
        gates.assert_fva_preserved(
            case, reference, reference.naive_wmape,
            reference.naive_wmape - reference.fva_vs_naive * 0.79,
        )
    with pytest.raises(AssertionError, match="non-positive"):
        gates.assert_fva_preserved(case, reference, reference.naive_wmape, reference.naive_wmape)
    assert reference.seed is None


def test_missing_observed_model_is_not_silently_excluded():
    reference = gates.Baseline.from_dict(_document())
    observed = dict(reference.per_model_wmape)
    observed.pop("auto_arima")
    with pytest.raises(AssertionError, match="model"):
        gates.assert_no_regression(unittest.TestCase(), reference, 0.75, observed)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1])
def test_observed_score_must_be_finite_and_nonnegative(value):
    reference = gates.Baseline.from_dict(_document())
    observed = dict(reference.per_model_wmape, auto_arima=value)
    with pytest.raises((AssertionError, ValueError), match="finite|nonnegative"):
        gates.assert_no_regression(unittest.TestCase(), reference, 0.75, observed)


def test_reference_fva_cannot_be_missing_or_nonpositive():
    reference = gates.Baseline.from_dict(_document())
    reference.fva_vs_naive = None
    with pytest.raises((AssertionError, ValueError), match="FVA|fva"):
        gates.assert_fva_preserved(unittest.TestCase(), reference, 1.0, 0.75)


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile_lock_sha256", "9" * 64),
        ("python", "3.11.0"),
        ("system", "Windows"),
        ("packages", {"numpy": "99.0", "xgboost": None, "xgboost-cpu": "1.0"}),
        ("profile_wheels", {}),
        ("scoring_policy", "macro-wmape"),
        ("series_count", 104),
    ],
)
def test_wrong_profile_runtime_policy_or_population_fails_compatibility(field, value):
    reference = gates.Baseline.from_dict(_document())
    observed = _candidate()
    observed[field] = value
    with pytest.raises((AssertionError, ValueError)):
        gates.verify_baseline_compatibility(unittest.TestCase(), reference, observed)


def test_new_source_run_is_allowed_only_with_the_same_qualified_environment():
    reference = gates.Baseline.from_dict(_document())
    observed = _candidate()
    observed["ci_run_id"] = "456"
    observed["ci_source_sha"] = "b" * 40
    observed["profile_install_report_sha256"] = "8" * 64
    gates.verify_baseline_compatibility(unittest.TestCase(), reference, observed)


def test_same_qualified_profile_can_have_a_different_kernel_audit_descriptor():
    document = _document()
    document["provenance"]["os_release"] = "Linux-6.8.0-1030-azure-x86_64-with-glibc2.39"
    document["approval"]["candidate_sha256"] = _digest(document["provenance"])
    reference = gates.Baseline.from_dict(document)
    observed = deepcopy(document["provenance"])
    observed["os_release"] = "Linux-6.8.0-1031-azure-x86_64-with-glibc2.39"
    observed["ci_run_id"] = "456"
    observed["ci_source_sha"] = "b" * 40
    gates.verify_baseline_compatibility(unittest.TestCase(), reference, observed)
    assert reference.raw["provenance"]["os_release"] != observed["os_release"]


@pytest.mark.parametrize(
    "field,value",
    [("system", "Windows"), ("process_machine", "aarch64"), ("interpreter_platform", "win-amd64")],
)
def test_kernel_tolerance_does_not_accept_other_os_architecture_or_interpreter(field, value):
    reference = gates.Baseline.from_dict(_document())
    observed = _candidate()
    observed["os_release"] = "different-kernel-audit-descriptor"
    observed[field] = value
    with pytest.raises((AssertionError, ValueError), match="incompatible|native x64"):
        gates.verify_baseline_compatibility(unittest.TestCase(), reference, observed)


def test_input_hashes_normalize_only_checkout_line_endings(tmp_path):
    reference = gates.Baseline.from_dict(_document())
    fixture, config = tmp_path / "input.csv", tmp_path / "config.yaml"
    fixture.write_bytes(b"a,b\r\n1,2\r\n")
    config.write_bytes(b"frequency: D\r\n")
    reference.fixture_sha256 = hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    reference.config_sha256 = hashlib.sha256(b"frequency: D\n").hexdigest()
    gates.verify_fixture_hash(unittest.TestCase(), reference, fixture, config)
    fixture.write_bytes(b"a,b\n1,3\n")
    with pytest.raises(AssertionError, match="Fixture"):
        gates.verify_fixture_hash(unittest.TestCase(), reference, fixture, config)


def test_capture_draft_cannot_approve_or_invent_a_seed():
    document = gates.baseline_document(_candidate(), notes="unit-only draft")
    assert document["approved_baseline"] is False
    assert document["approval"] is None
    assert document["seed"] is None
    assert document["champion_wmape"] == 0.75
    assert document["per_model_wmape"] == _document()["per_model_wmape"]
    with pytest.raises(ValueError, match="approved"):
        gates.Baseline.from_dict(document)


@pytest.mark.parametrize("method", ["test_baseline_no_regression", "test_baseline_fva_preserved"])
@pytest.mark.parametrize("defect", [None, "missing_champion", "missing_model", "duplicate_model"])
def test_original_empirical_test_methods_keep_complete_model_gates(monkeypatch, method, defect):
    from tests.integration.test_m5_daily_backend import TestBacktestPipeline

    reference = gates.Baseline.from_dict(_document())
    monkeypatch.setattr(gates, "require_baseline", lambda *args, **kwargs: reference)
    case = TestBacktestPipeline(methodName=method)
    case._backtest_provenance = _candidate()
    result = {
        "best_wmape": 0.75, "champion_model": "lgbm_direct",
        "leaderboard": [
            {"model_id": model, "wmape": score}
            for model, score in reference.per_model_wmape.items()
        ],
    }
    if defect == "missing_champion":
        result["best_wmape"] = None
    elif defect == "missing_model":
        result["leaderboard"].pop()
    elif defect == "duplicate_model":
        result["leaderboard"][0] = deepcopy(result["leaderboard"][1])
    case._backtest_result = result
    if defect:
        with pytest.raises(AssertionError):
            getattr(case, method)()
    else:
        getattr(case, method)()


def _case_with_rescored_champion(monkeypatch, method, rescored_wmape, api_wmape):
    """Supply synthetic unit evidence to the real gate methods, without fitting models."""
    from tests.integration.test_m5_daily_backend import TestBacktestPipeline

    reference = gates.Baseline.from_dict(_document())
    observed = _candidate()
    observed["scalar_reference_scores"]["lgbm_direct"]["wmape"] = rescored_wmape
    observed["scalar_reference_scores"]["lgbm_direct"]["mae"] = rescored_wmape * 7945 / 5880
    observed["raw_predictions_sha256"] = "7" * 64
    gates.verify_baseline_compatibility(unittest.TestCase(), reference, observed)
    monkeypatch.setattr(gates, "require_baseline", lambda *args, **kwargs: reference)
    case = TestBacktestPipeline(methodName=method)
    case._backtest_provenance = observed
    case._backtest_result = {
        "best_wmape": api_wmape, "champion_model": "lgbm_direct",
        "leaderboard": [
            {"model_id": model, "wmape": score["wmape"]}
            for model, score in observed["scalar_reference_scores"].items()
        ],
    }
    return case


@pytest.mark.parametrize("method", ["test_baseline_no_regression", "test_baseline_fva_preserved"])
def test_original_gates_reject_stale_finite_api_summary(monkeypatch, method):
    case = _case_with_rescored_champion(monkeypatch, method, 0.80, 0.75)
    with pytest.raises(AssertionError, match="API best_wmape.*independent"):
        getattr(case, method)()


@pytest.mark.parametrize("method", ["test_baseline_no_regression", "test_baseline_fva_preserved"])
@pytest.mark.parametrize("missing", ["absent", "null"])
def test_original_gates_reject_missing_api_summary(monkeypatch, method, missing):
    case = _case_with_rescored_champion(monkeypatch, method, 0.80, None)
    if missing == "absent":
        case._backtest_result.pop("best_wmape")
    with pytest.raises(AssertionError, match="champion WMAPE|best_wmape"):
        getattr(case, method)()


@pytest.mark.parametrize(
    "method,message",
    [("test_baseline_no_regression", "Champion WMAPE regression"),
     ("test_baseline_fva_preserved", "FVA regression")],
)
def test_original_gates_reject_rescored_point_eight_with_consistent_api_summary(monkeypatch, method, message):
    case = _case_with_rescored_champion(monkeypatch, method, 0.80, 0.80)
    with pytest.raises(AssertionError, match=message):
        getattr(case, method)()


@pytest.mark.parametrize(
    "method,message",
    [("test_baseline_no_regression", "Champion WMAPE regression"),
     ("test_baseline_fva_preserved", "FVA regression")],
)
def test_original_gates_use_authoritative_score_even_within_summary_roundoff_policy(monkeypatch, method, message):
    reference = gates.Baseline.from_dict(_document())
    boundary = (
        reference.champion_wmape * (1 + gates.CHAMPION_REGRESSION_TOLERANCE)
        if method == "test_baseline_no_regression"
        else reference.naive_wmape - reference.fva_vs_naive * gates.MIN_FVA_RATIO
    )
    case = _case_with_rescored_champion(
        monkeypatch, method, boundary + 2e-13, boundary - 2e-13,
    )
    with pytest.raises(AssertionError, match=message):
        getattr(case, method)()


def _native_inputs(tmp_path, monkeypatch):
    from tests.integration import m5_reference

    root = tmp_path
    runtime = {
        "python": "3.12.0", "system": "Linux", "process_machine": "x86_64",
        "interpreter_platform": "linux-x86_64",
    }
    versions = {"numpy": "1.0", "xgboost-cpu": "1.0"}
    lock = root / "requirements" / "py312" / "dev.txt"
    lock.parent.mkdir(parents=True)
    digest = "5" * 64
    lock.write_text(
        "".join(f"{name}=={version} \\\n    --hash=sha256:{digest}\n" for name, version in versions.items()),
        encoding="utf-8",
    )
    manifest = root / "requirements" / "manifest.json"
    manifest.write_text(json.dumps({"profiles": {"py312/dev.txt": {
        "python": runtime["python"], "sha256": gates.sha256_file(lock),
    }}}), encoding="utf-8")
    native = root / "native.json"
    native.write_text(json.dumps({
        "profile": "dev", "python": runtime["python"], "os": "linux",
        "host_architecture": "x86_64", "interpreter_platform": "linux-x86_64",
        "packages": versions, "xgboost_build": {"USE_CUDA": False},
        "synthetic_native_outputs": {"lgbm_direct": {}, "xgboost_direct": {}},
    }), encoding="utf-8")
    install = root / "install.json"
    install.write_text(json.dumps({
        "environment": {
            "python_full_version": runtime["python"], "sys_platform": "linux",
            "platform_machine": "x86_64",
        },
        "install": [{
            "metadata": {"name": name, "version": version}, "is_yanked": False,
            "download_info": {
                "url": f"https://unit.invalid/{name}-unit.whl",
                "archive_info": {"hashes": {"sha256": digest}},
            },
        } for name, version in versions.items()],
    }), encoding="utf-8")
    monkeypatch.setenv("M5_PROFILE_LOCK", str(lock))
    monkeypatch.setenv("M5_PROFILE_INSTALL_REPORT", str(install))
    monkeypatch.setenv("M5_PROFILE_QUALIFICATION", str(native))
    monkeypatch.setattr(m5_reference.importlib.metadata, "version", versions.__getitem__)
    return root, runtime, versions, lock, native, install


def test_profile_capture_binds_real_bytes_installed_versions_and_selected_wheels(tmp_path, monkeypatch):
    from tests.integration.m5_reference import _profile_provenance

    root, runtime, versions, lock, native, install = _native_inputs(tmp_path, monkeypatch)
    result = _profile_provenance(root, runtime, versions)
    assert result["profile_lock_name"] == "py312/dev.txt"
    assert result["profile_lock_sha256"] == gates.sha256_file(lock)
    assert result["profile_qualification_sha256"] == gates.sha256_file(native)
    assert result["profile_install_report_sha256"] == gates.sha256_file(install)
    assert result["profile_package_versions"] == versions
    assert set(result["profile_wheels"]) == set(versions)


@pytest.mark.parametrize(
    "defect", ["lock_bytes", "python", "installed_version", "missing_wheel",
               "wheel_hash", "native_os", "install_os", "yanked", "missing_input"],
)
def test_profile_capture_rejects_incompatible_or_incomplete_evidence(tmp_path, monkeypatch, defect):
    from tests.integration import m5_reference

    root, runtime, versions, lock, native, install = _native_inputs(tmp_path, monkeypatch)
    if defect == "lock_bytes":
        lock.write_bytes(lock.read_bytes() + b"\n")
    elif defect == "python":
        runtime["python"] = "3.12.1"
    elif defect == "installed_version":
        monkeypatch.setattr(m5_reference.importlib.metadata, "version", lambda name: "2.0")
    elif defect == "native_os":
        data = json.loads(native.read_text())
        data["os"] = "win32"
        native.write_text(json.dumps(data))
    elif defect == "missing_input":
        monkeypatch.delenv("M5_PROFILE_INSTALL_REPORT")
    else:
        data = json.loads(install.read_text())
        if defect == "missing_wheel":
            data["install"].pop()
        elif defect == "wheel_hash":
            data["install"][0]["download_info"]["archive_info"]["hashes"]["sha256"] = "0" * 64
        elif defect == "install_os":
            data["environment"]["sys_platform"] = "win32"
        else:
            data["install"][0]["is_yanked"] = True
        install.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        m5_reference._profile_provenance(root, runtime, versions)


def test_legacy_capture_command_only_writes_a_new_unapproved_draft(tmp_path, monkeypatch, capsys):
    from scripts.bless_m5_baseline import bless

    candidate, draft = tmp_path / "candidate.json", tmp_path / "review-draft.json"
    candidate.write_text(json.dumps(_candidate()), encoding="utf-8")
    assert bless("daily", "unit only", False, candidate=candidate, output=draft) == 0
    captured = json.loads(draft.read_text())
    assert captured["approved_baseline"] is False
    assert captured["approval"] is None
    assert captured["seed"] is None
    saved = draft.read_bytes()
    with pytest.raises(FileExistsError):
        bless("daily", "", False, candidate=candidate, output=draft)
    assert draft.read_bytes() == saved
    reference_path = tmp_path / "protected-reference.json"
    monkeypatch.setenv("M5_BASELINE_PATH", str(reference_path))
    with pytest.raises(ValueError, match="unapproved"):
        bless("daily", "", False, candidate=candidate, output=reference_path)
    assert not reference_path.exists()
    assert "UNAPPROVED" in capsys.readouterr().out
