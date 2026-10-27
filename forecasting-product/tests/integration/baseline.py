"""Fail-closed empirical M5 gates, separate from the input-only naive oracle.

An approved reference binds independently reviewed native-profile evidence.
Missing references, incomplete populations and incompatible runtimes fail;
capture never approves a reference or invents a deterministic seed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unittest
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .m5_reference import PROVENANCE_PACKAGES, canonical_text_sha256

BASELINES_DIR = Path(__file__).resolve().parent / "baselines"
REFERENCE_PATH = BASELINES_DIR / "m5_daily_reference.json"
SCORING_POLICY = "raw-observations-v1"

# ── Tolerances (can be tightened once baselines are stable) ───────────────────
# Champion WMAPE may degrade by at most this fraction before the gate fires.
CHAMPION_REGRESSION_TOLERANCE = 0.05  # 5 % relative
# Any model's WMAPE may degrade by at most this fraction.
PER_MODEL_REGRESSION_TOLERANCE = 0.10  # 10 % relative
# Minimum fraction of the approved FVA (naive - champion) to preserve.
MIN_FVA_RATIO = 0.80  # accept ≥ 80 % of the blessed FVA improvement
# P90 coverage must stay within this many percentage points of baseline.
CALIBRATION_TOLERANCE_PP = 0.05


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Baseline:
    """Parsed baseline JSON with convenience accessors."""

    frequency: str
    fixture_sha256: str
    config_sha256: str
    seed: int | None
    per_model_wmape: dict[str, float]
    champion_model: str
    champion_wmape: float
    naive_wmape: float | None = None
    fva_vs_naive: float | None = None
    calibration_p50_coverage: float | None = None
    calibration_p90_coverage: float | None = None
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Baseline":
        _validate_document(data)
        return cls(
            frequency=data["frequency"],
            fixture_sha256=data["fixture_sha256"],
            config_sha256=data["config_sha256"],
            seed=data.get("seed"),
            per_model_wmape={k: float(v) for k, v in data["per_model_wmape"].items()},
            champion_model=data["champion_model"],
            champion_wmape=float(data["champion_wmape"]),
            naive_wmape=_opt_float(data.get("naive_wmape")),
            fva_vs_naive=_opt_float(data.get("fva_vs_naive")),
            calibration_p50_coverage=_opt_float(data.get("calibration_p50_coverage")),
            calibration_p90_coverage=_opt_float(data.get("calibration_p90_coverage")),
            notes=data.get("notes", ""),
            raw=data,
        )


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def candidate_sha256(candidate: dict) -> str:
    """Bind review to the entire captured record, not just its rounded scores."""
    return hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, label: str, *, nonnegative: bool = True) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float))
        and math.isfinite(value) and (not nonnegative or value >= 0),
        f"{label} must be finite" + (" and nonnegative" if nonnegative else ""),
    )
    return float(value)


def _sha(value: Any, label: str) -> None:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"Missing or invalid {label} SHA-256")


def _validate_candidate(candidate: dict, *, require_profile: bool) -> None:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    expected = {
        "approved_baseline": False,
        "fixture_utf8_lf_sha256": reference["fixture_sha256"],
        "config_utf8_lf_sha256": reference["config_sha256"],
        "scoring_policy": SCORING_POLICY, "models": sorted(reference["models"]),
        "input_rows": reference["input_rows"], "series_count": reference["series_count"],
        "folds": reference["n_folds"], "validation_periods": reference["val_periods"],
        "gap_periods": reference["gap_periods"],
        "scored_observations_per_model": reference["totals"]["observations"],
    }
    for key, value in expected.items():
        _require(candidate.get(key) == value, f"M5 candidate has wrong or missing {key}")
    scores = candidate.get("scalar_reference_scores", {})
    _require(set(scores) == set(reference["models"]), "M5 candidate must include every model")
    for model, metrics in scores.items():
        for metric in ("wmape", "normalized_bias", "mae"):
            _finite(metrics.get(metric), f"{model} {metric}", nonnegative=metric != "normalized_bias")
    totals = reference["totals"]
    naive_truth = {
        "wmape": float(totals["absolute_error"]) / float(totals["absolute_actual"]),
        "normalized_bias": float(totals["signed_error"]) / float(totals["absolute_actual"]),
        "mae": float(totals["absolute_error"]) / totals["observations"],
    }
    for metric, truth in naive_truth.items():
        _require(math.isclose(scores["naive_seasonal"][metric], truth, rel_tol=1e-12, abs_tol=1e-12),
                 f"M5 naive {metric} differs from the independent input-only oracle")
    winner = min(scores, key=lambda model: (
        scores[model]["wmape"], abs(scores[model]["normalized_bias"]), model,
    ))
    _require(candidate.get("champion_model") == winner, "M5 candidate winner is inconsistent")
    if not require_profile:
        return
    _sha(candidate.get("raw_predictions_sha256"), "raw predictions")
    _require(candidate.get("raw_predictions_rows")
             == len(reference["models"]) * reference["totals"]["observations"],
             "Missing or incomplete raw prediction evidence")
    for key in ("profile_lock_sha256", "profile_manifest_sha256",
                "profile_install_report_sha256", "profile_qualification_sha256"):
        _sha(candidate.get(key), key)
    for key in ("python", "system", "process_machine", "interpreter_platform", "os_release"):
        _require(isinstance(candidate.get(key), str) and bool(candidate[key].strip()),
                 f"Missing native runtime {key}")
    _require(candidate["system"] in ("Linux", "Windows")
             and candidate["process_machine"].lower() in ("amd64", "x86_64")
             and candidate.get("ci_runner_arch") == "X64", "M5 evidence requires a native x64 CI runner")
    _require(str(candidate.get("ci_run_id", "")).isdigit(), "Missing native CI run identity")
    _require(re.fullmatch(r"[0-9a-f]{40}", candidate.get("ci_source_sha") or "") is not None,
             "Missing native CI source commit")
    minor = "".join(candidate["python"].split(".")[:2])
    _require(minor in ("311", "312", "313")
             and candidate.get("profile_lock_name") == f"py{minor}/dev.txt",
             "M5 evidence requires the matching Python dev profile")
    packages = candidate.get("packages", {})
    pinned = candidate.get("profile_package_versions", {})
    wheels = candidate.get("profile_wheels", {})
    for name in PROVENANCE_PACKAGES:
        if name in ("polars-runtime-compat", "xgboost"):
            _require(packages.get(name) is None, f"Unexpected unqualified distribution {name}")
            continue
        version = packages.get(name)
        _require(isinstance(version, str) and bool(version) and pinned.get(name) == version,
                 f"Missing or inconsistent installed profile version for {name}")
        wheel = wheels.get(name, {})
        _require(wheel.get("version") == version
                 and str(wheel.get("filename", "")).endswith(".whl"),
                 f"Missing native wheel evidence for {name}")
        _sha(wheel.get("sha256"), f"{name} wheel")


def baseline_document(candidate: dict, notes: str = "") -> dict:
    """Prepare an UNAPPROVED review draft from an already captured full run."""
    _validate_candidate(candidate, require_profile=False)
    per_model = {model: score["wmape"] for model, score in candidate["scalar_reference_scores"].items()}
    champion = candidate["champion_model"]
    return {
        "schema_version": 2, "kind": "m5_empirical_reference", "frequency": "daily",
        "approved_baseline": False, "approval": None,
        "hash_format": "sha256_utf8_bytes_with_crlf_normalized_to_lf",
        "fixture_sha256": candidate["fixture_utf8_lf_sha256"],
        "config_sha256": candidate["config_utf8_lf_sha256"],
        "seed": None, "notes": notes, "per_model_wmape": per_model,
        "champion_model": champion, "champion_wmape": per_model[champion],
        "naive_wmape": per_model["naive_seasonal"],
        "fva_vs_naive": per_model["naive_seasonal"] - per_model[champion],
        "provenance": candidate,
    }


def _validate_document(data: dict) -> None:
    _require(data.get("schema_version") == 2 and data.get("kind") == "m5_empirical_reference",
             "Expected a versioned M5 empirical reference, not the naive-only oracle")
    _require(data.get("approved_baseline") is True, "M5 empirical reference is not approved")
    _require(data.get("frequency") == "daily", "Only the full daily M5 reference is supported")
    _require(data.get("seed") is None, "No seed override was captured; do not invent one")
    candidate = data.get("provenance", {})
    _validate_candidate(candidate, require_profile=True)
    approval = data.get("approval") or {}
    for key in ("reviewed_by", "review_reference", "approved_at"):
        _require(isinstance(approval.get(key), str) and bool(approval[key].strip()),
                 f"Missing independent approval {key}")
    approved_at = datetime.fromisoformat(approval["approved_at"].replace("Z", "+00:00"))
    _require(approved_at.tzinfo is not None, "Approval timestamp requires a timezone")
    _require(approval.get("candidate_sha256") == candidate_sha256(candidate),
             "Independent approval is not bound to this candidate")
    expected = baseline_document(candidate)
    for key in ("hash_format", "fixture_sha256", "config_sha256", "champion_model"):
        _require(data.get(key) == expected[key], f"Reference {key} differs from captured evidence")
    models = data.get("per_model_wmape", {})
    _require(set(models) == set(expected["per_model_wmape"]), "Reference must retain every model")
    for label, observed, truth in [
        *((model, models[model], value) for model, value in expected["per_model_wmape"].items()),
        *((key, data.get(key), expected[key]) for key in ("champion_wmape", "naive_wmape", "fva_vs_naive")),
    ]:
        _require(math.isclose(_finite(observed, label), truth, rel_tol=1e-12, abs_tol=1e-12),
                 f"Reference {label} differs from captured evidence")
    _require(data["fva_vs_naive"] > 0, "Reference FVA must be positive")


def verify_baseline_compatibility(
    testcase: unittest.TestCase, baseline: Baseline, observed: dict,
) -> None:
    """Compare like-for-like profiles/populations, never different OS/wheel runs."""
    _validate_candidate(observed, require_profile=True)
    expected = baseline.raw["provenance"]
    for key in (
        "fixture_utf8_lf_sha256", "config_utf8_lf_sha256", "scoring_policy", "models",
        "input_rows", "series_count", "folds", "validation_periods", "gap_periods",
        "scored_observations_per_model", "python", "system", "process_machine", "interpreter_platform",
        "os_release", "packages", "profile_lock_name", "profile_lock_sha256",
        "profile_manifest_sha256", "profile_package_versions", "profile_wheels",
    ):
        testcase.assertEqual(observed[key], expected[key], f"M5 baseline incompatible {key}")


# ── File IO + hashing ─────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, streamed to tolerate large fixtures."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def baseline_path(frequency: str) -> Path:
    """Allow an explicit reviewed per-native-profile reference; never guess one."""
    override = os.environ.get("M5_BASELINE_PATH")
    return Path(override) if override else BASELINES_DIR / f"m5_{frequency}_baseline.json"


def load_baseline(frequency: str) -> Baseline | None:
    """Load the blessed baseline for *frequency* or return ``None`` if absent."""
    _require(frequency == "daily", "Only the full daily M5 empirical reference is supported")
    p = baseline_path(frequency)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Baseline.from_dict(data)


def require_baseline(
    frequency: str, testcase: unittest.TestCase, *, observed: dict | None = None,
) -> Baseline:
    """Require an approved, compatible reference; absent evidence is a failure."""
    bl = load_baseline(frequency)
    if bl is None:
        testcase.fail(
            f"No approved empirical M5 baseline at {baseline_path(frequency)}. "
            "Capture a complete native locked-profile run and obtain independent review; "
            "the input-only naive oracle does not replace either empirical gate."
        )
    if observed is None:
        testcase.fail("Current native M5 provenance is required for baseline compatibility")
    verify_baseline_compatibility(testcase, bl, observed)
    return bl


# ── Assertion helpers ─────────────────────────────────────────────────────────

def verify_fixture_hash(
    testcase: unittest.TestCase,
    baseline: Baseline,
    fixture_path: Path,
    config_path: Path,
) -> None:
    """Fail if fixture or config contents have drifted from the blessed hashes.

    Normalize only CRLF/LF, as in the independent fixture reference. Runtime
    provenance also retains raw input hashes; lock/wheel hashes are always raw.
    """
    fixture_hash = canonical_text_sha256(fixture_path)
    config_hash = canonical_text_sha256(config_path)
    testcase.assertEqual(
        fixture_hash,
        baseline.fixture_sha256,
        f"Fixture {fixture_path.name} has changed since baseline was blessed.\n"
        f"  baseline: {baseline.fixture_sha256}\n"
        f"  current:  {fixture_hash}\n"
        "A changed fixture requires a newly reviewed empirical reference.",
    )
    testcase.assertEqual(
        config_hash,
        baseline.config_sha256,
        f"Config {config_path.name} has changed since baseline was blessed.\n"
        f"  baseline: {baseline.config_sha256}\n"
        f"  current:  {config_hash}\n"
        "A changed config requires a newly reviewed empirical reference.",
    )


def assert_no_regression(
    testcase: unittest.TestCase,
    baseline: Baseline,
    observed_champion_wmape: float,
    observed_per_model_wmape: dict[str, float],
) -> None:
    """Layered regression gates against a blessed baseline.

    1. Champion WMAPE must not exceed ``baseline.champion_wmape *
       (1 + CHAMPION_REGRESSION_TOLERANCE)``.
    2. Every blessed model's WMAPE must not exceed its baseline value by more
       than ``PER_MODEL_REGRESSION_TOLERANCE`` (relative). Every model is required.
    """
    _finite(observed_champion_wmape, "Observed champion WMAPE")
    testcase.assertEqual(
        set(observed_per_model_wmape), set(baseline.per_model_wmape),
        "Observed model population differs from the complete empirical reference",
    )
    champion_limit = baseline.champion_wmape * (1 + CHAMPION_REGRESSION_TOLERANCE)
    testcase.assertLessEqual(
        observed_champion_wmape,
        champion_limit,
        f"Champion WMAPE regression: observed={observed_champion_wmape:.4f} "
        f"> baseline×{1 + CHAMPION_REGRESSION_TOLERANCE:.2f}={champion_limit:.4f} "
        f"(baseline={baseline.champion_wmape:.4f}).",
    )

    for model, baseline_wmape in baseline.per_model_wmape.items():
        observed = _finite(observed_per_model_wmape[model], f"Observed {model} WMAPE")
        limit = baseline_wmape * (1 + PER_MODEL_REGRESSION_TOLERANCE)
        testcase.assertLessEqual(
            observed,
            limit,
            f"Model '{model}' WMAPE regression: observed={observed:.4f} "
            f"> baseline×{1 + PER_MODEL_REGRESSION_TOLERANCE:.2f}={limit:.4f} "
            f"(baseline={baseline_wmape:.4f}).",
        )


def assert_fva_preserved(
    testcase: unittest.TestCase,
    baseline: Baseline,
    observed_naive_wmape: float,
    observed_champion_wmape: float,
) -> None:
    """Assert that the champion still beats the naive baseline by a meaningful
    margin (≥ ``MIN_FVA_RATIO`` of the blessed improvement, and strictly > 0).

    This is the *business-value* gate: forecasting must add value over the
    dumbest baseline, not just converge.
    """
    _finite(observed_naive_wmape, "Observed naive WMAPE")
    _finite(observed_champion_wmape, "Observed champion WMAPE")
    reference_fva = _finite(baseline.fva_vs_naive, "Reference FVA")
    _require(reference_fva > 0, "Reference FVA must be positive")
    observed_fva = observed_naive_wmape - observed_champion_wmape
    testcase.assertGreater(
        observed_fva,
        0.0,
        f"Champion ({observed_champion_wmape:.4f}) did not beat naive "
        f"({observed_naive_wmape:.4f}) — forecast value added is non-positive.",
    )
    required = reference_fva * MIN_FVA_RATIO
    testcase.assertGreaterEqual(
        observed_fva,
        required,
        f"FVA regression: observed improvement over naive={observed_fva:.4f} "
        f"< {MIN_FVA_RATIO:.0%} of baseline improvement={baseline.fva_vs_naive:.4f} "
        f"(required ≥ {required:.4f}).",
    )


def assert_calibration_preserved(
    testcase: unittest.TestCase,
    baseline: Baseline,
    observed_p90_coverage: float,
) -> None:
    """Assert that P90 interval coverage is within tolerance of baseline.

    Skipped when the baseline does not record calibration.
    """
    if baseline.calibration_p90_coverage is None:
        testcase.skipTest("Baseline does not record P90 calibration coverage.")
    delta = abs(observed_p90_coverage - baseline.calibration_p90_coverage)
    testcase.assertLessEqual(
        delta,
        CALIBRATION_TOLERANCE_PP,
        f"P90 coverage drift: observed={observed_p90_coverage:.3f}, "
        f"baseline={baseline.calibration_p90_coverage:.3f}, "
        f"delta={delta:.3f} > tol={CALIBRATION_TOLERANCE_PP}.",
    )
