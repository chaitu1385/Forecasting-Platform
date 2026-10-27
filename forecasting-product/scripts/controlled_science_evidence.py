"""Temporary evidence-only comparison; never an install profile or production change."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REFERENCE = ROOT / "scripts" / "recorded_science_tuple.json"


def reference_science(manifest: dict, minor: str) -> dict[str, str]:
    from packaging.markers import Marker

    packages = {**manifest["common_science"], **manifest["interpreters"][minor]["science"]}
    markers = manifest["platform_markers"]
    return {
        name: version for name, version in packages.items()
        if name not in markers or Marker(markers[name]).evaluate()
    }


def fixture() -> list[dict]:
    return [
        {
            "series_id": f"probe-{series}",
            "week": date(2023, 1, 2) + timedelta(weeks=period),
            "quantity": float(100 + series * 20 + period % 4 * 3 + period // 8),
        }
        for series in range(2) for period in range(80)
    ]


def capture(variant: str, output: Path) -> None:
    from scripts.check_dependency_profile import require_native_runtime
    from scripts.lock_dependencies import _read_pins

    require_native_runtime()
    import numpy as np
    import polars as pl
    import xgboost
    from src.forecasting.registry import registry

    manifest = json.loads(REFERENCE.read_text())
    minor = f"py{sys.version_info.major}{sys.version_info.minor}"
    expected = (
        reference_science(manifest, minor)
        if variant == "recorded-master-tuple"
        else {name: pin[0] for name, pin in _read_pins(ROOT / "requirements" / minor / "core.txt").items()}
    )
    for package, version in expected.items():
        if importlib.metadata.version(package) != version:
            raise AssertionError(f"{variant}: wrong installed version for {package}")
    distribution = "xgboost" if variant == "recorded-master-tuple" else "xgboost-cpu"
    if xgboost.__version__ != importlib.metadata.version(distribution):
        raise AssertionError("Loaded XGBoost does not match the recorded distribution")
    build = xgboost.build_info()
    if variant == "supported-cpu-profile" and build["USE_CUDA"]:
        raise AssertionError("The supported production profile must remain CPU-only")
    rows = fixture()
    data = pl.DataFrame(rows)
    outputs = {}
    for name, parameter, device in (
        ("lgbm_direct", "lgbm_params", {"device_type": "cpu"}),
        ("xgboost_direct", "xgb_params", {"device": "cpu"}),
    ):
        model = registry.build(
            name, lags=[1, 2], num_threads=1,
            **{parameter: {"n_estimators": 12, "random_state": 17, **device}},
        )
        model.fit(data)
        point = model.predict(4).sort(["series_id", "week"])
        interval = model.predict_quantiles(4, [0.1, 0.5, 0.9]).sort(["series_id", "week"])
        assert point.height == interval.height == 8
        assert np.isfinite(point["forecast"].to_numpy()).all()
        assert np.isfinite(interval.select("forecast_p10", "forecast_p50", "forecast_p90").to_numpy()).all()
        assert not interval.filter(
            (pl.col("forecast_p10") > pl.col("forecast_p50"))
            | (pl.col("forecast_p50") > pl.col("forecast_p90"))
        ).height
        outputs[name] = {
            "parameters": model.get_params(),
            "points": point.to_dicts(),
            "intervals": interval.to_dicts(),
        }
    report = {
        "variant": variant,
        "python": platform.python_version(),
        "os": platform.platform(),
        "sys_platform": sys.platform,
        "architecture": platform.machine(),
        "interpreter_platform": sysconfig.get_platform(),
        "input_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest(),
        "input_scale": {"rows": 160, "series": 2, "target_min": 100, "target_max": 138},
        "installed_expected_versions": expected,
        "xgboost_distribution": distribution,
        "xgboost_build": {
            key: build[key] for key in ("USE_CUDA", "USE_OPENMP", "GCC_VERSION", "MSVC_VERSION")
            if key in build
        },
        "outputs": outputs,
    }
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")


def differences(left: dict, right: dict) -> dict:
    assert left["input_sha256"] == right["input_sha256"]
    assert left["python"] == right["python"] and left["os"] == right["os"]
    results = {}
    for model in left["outputs"]:
        assert left["outputs"][model]["parameters"] == right["outputs"][model]["parameters"]
        metrics = {}
        for kind in ("points", "intervals"):
            a, b = left["outputs"][model][kind], right["outputs"][model][kind]
            assert len(a) == len(b) == 8
            absolute = relative = 0.0
            for old, new in zip(a, b):
                assert old.keys() == new.keys()
                for key in old:
                    if key in ("forecast", "forecast_p10", "forecast_p50", "forecast_p90"):
                        assert math.isfinite(old[key]) and math.isfinite(new[key])
                        delta = abs(old[key] - new[key])
                        absolute = max(absolute, delta)
                        relative = max(relative, delta / max(abs(old[key]), 1e-12))
                    else:
                        assert old[key] == new[key]
            metrics[kind] = {"max_absolute_difference": absolute, "max_relative_difference": relative}
        results[model] = metrics
    return results


def run(output: Path) -> None:
    from scripts.check_dependency_profile import require_native_runtime
    from scripts.lock_dependencies import (
        _check_resolution_preserved,
        _check_upstream,
        _read_pins,
    )

    require_native_runtime()
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise RuntimeError("This controlled experiment is limited to native Linux/Windows Python 3.11/3.12")
    output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(REFERENCE.read_text())
    minor = f"py{sys.version_info.major}{sys.version_info.minor}"
    current_lock = ROOT / "requirements" / minor / "core.txt"
    source_hashes = {}
    for source in sorted((ROOT / "src").rglob("*.py")):
        relative = source.relative_to(ROOT.parent).as_posix()
        original = subprocess.check_output(
            ["git", "show", f"{reference['baseline_commit']}:{relative}"], cwd=ROOT.parent,
        )
        candidate = subprocess.check_output(
            ["git", "show", f"{reference['supported_commit']}:{relative}"], cwd=ROOT.parent,
        )
        assert original == candidate == source.read_bytes(), relative
        source_hashes[relative] = hashlib.sha256(candidate).hexdigest()
    baseline = {name: pin[0] for name, pin in _read_pins(current_lock).items()}
    for replaced in ("xgboost-cpu", "polars", "polars-runtime-32"):
        baseline.pop(replaced)
    baseline.update(reference_science(reference, minor))
    input_path = output / "baseline-input.in"
    input_path.write_text("\n".join(f"{name}=={version}" for name, version in sorted(baseline.items())) + "\n")
    resolved = output / "baseline-resolved.txt"
    locked = output / "baseline-locked.txt"
    metadata = {
        **reference, "interpreter_profile": minor, "source_hashes": source_hashes,
        "experiment_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip(),
        "active_reference_science": reference_science(reference, minor),
        "native_platform": sys.platform,
        "runner_identity": {
            key: os.environ[key] for key in ("RUNNER_OS", "RUNNER_ARCH", "ImageOS", "ImageVersion")
            if key in os.environ
        },
        "limitation": "Recorded numerical tuple only; current common nonnumerical dependencies, current tooling, common native OS/Python patch. Not a full historical environment replay. Windows uses applicable native artifacts for versions recorded on Linux, not a historically qualified Windows environment.",
    }
    (output / "experiment-identity.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="controlled-science-") as work:
        work = Path(work)
        wheels = work / "baseline-wheels"
        compile_command = [
            sys.executable, "-m", "piptools", "compile", str(input_path),
            "--resolver=backtracking", "--allow-unsafe", "--strip-extras",
            "--no-emit-index-url", "--no-emit-trusted-host", "--no-emit-find-links",
            "--no-annotate", "--quiet",
        ]
        subprocess.run([*compile_command, "--output-file", str(resolved)], check=True)
        if {name: pin[0] for name, pin in _read_pins(resolved).items()} != baseline:
            raise AssertionError("Reference resolution added an unpinned package or changed a recorded/common version")
        release_cache = {}
        _check_upstream(resolved, release_cache)
        subprocess.run([
            sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
            "-r", str(resolved), "--dest", str(wheels), "--quiet",
        ], check=True)
        subprocess.run([
            *compile_command, "--no-index", "--find-links", str(wheels),
            "--constraint", str(resolved), "--generate-hashes", "--output-file", str(locked),
        ], check=True)
        _check_resolution_preserved(resolved, locked)
        _check_upstream(locked, release_cache)
        subprocess.run([
            sys.executable, "-m", "pip_audit", "--disable-pip", "--no-deps",
            "--progress-spinner", "off", "-r", str(locked),
        ], check=True)
        captured = {}
        for variant, requirements in (
            ("recorded-master-tuple", locked), ("supported-cpu-profile", current_lock),
        ):
            environment = work / variant
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            pip = [sys.executable, "-m", "pip", "--python", str(python)]
            install_report = work / f"{variant}-pip-report.json"
            subprocess.run([
                *pip, "install", "--require-hashes", "--only-binary=:all:", "-r", str(requirements),
                "--report", str(install_report), "--quiet",
            ], check=True)
            receipt = json.loads(install_report.read_text(encoding="utf-8"))
            wheel_provenance = {
                "environment": receipt["environment"],
                "packages": [
                    {
                        "name": item["metadata"]["name"],
                        "version": item["metadata"]["version"],
                        "download": item["download_info"],
                        "requires_dist": item["metadata"].get("requires_dist", []),
                    }
                    for item in receipt["install"]
                ],
            }
            (output / f"{variant}-wheel-provenance.json").write_text(json.dumps(wheel_provenance, indent=2) + "\n")
            subprocess.run([*pip, "check"], check=True)
            frozen = subprocess.check_output([*pip, "freeze", "--all"], text=True)
            (output / f"{variant}-installed.txt").write_text(frozen)
            runs = []
            for iteration in (1, 2):
                destination = output / f"{variant}-run{iteration}.json"
                subprocess.run([
                    str(python), str(Path(__file__).resolve()),
                    "--capture", variant, "--output", str(destination),
                ], check=True)
                runs.append(json.loads(destination.read_text()))
            repeatability = differences(*runs)
            assert all(value["max_absolute_difference"] <= 1e-12 for model in repeatability.values() for value in model.values())
            captured[variant] = runs[0]
            (output / f"{variant}-repeatability.json").write_text(json.dumps(repeatability, indent=2) + "\n")
        comparison = differences(captured["recorded-master-tuple"], captured["supported-cpu-profile"])
        report = {
            "scope": metadata["limitation"],
            "original_master": reference["baseline_commit"],
            "supported_candidate": reference["supported_commit"],
            "same_native_os_and_python": True,
            "same_model_source": True,
            "repeatability_within_each_environment_passed": True,
            "comparisons": comparison,
            "requires_numerical_interpretation": any(
                value["max_absolute_difference"] > 1e-12
                for model in comparison.values() for value in model.values()
            ),
            "acceptance": "Workflow success proves protocol/identity/finite ordered outputs only; actual differences require review and are not mathematical correctness.",
        }
        (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", choices=("recorded-master-tuple", "supported-cpu-profile"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.capture:
        capture(args.capture, args.output)
    else:
        run(args.output.resolve())
