"""Library-independent arithmetic and key-coverage oracles for the M5 fixture.

No production scorer, model, date helper, NumPy or DataFrame operation is used.
The exact seasonal-naive reference is not an empirical learned-model baseline.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sysconfig
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import unquote, urlsplit

PROVENANCE_PACKAGES = (
    "numpy", "scipy", "pandas", "polars", "polars-runtime-32", "polars-runtime-compat",
    "pyarrow", "statsforecast", "mlforecast", "hierarchicalforecast", "lightgbm",
    "xgboost", "xgboost-cpu", "scikit-learn", "duckdb", "coreforecast", "utilsforecast",
    "fastapi", "starlette",
)


@dataclass(frozen=True)
class ReferencePoint:
    fold: int
    series_id: str
    target_date: date
    forecast_step: int
    actual: Decimal
    naive_forecast: Decimal

    @property
    def key(self) -> tuple:
        return self.fold, self.series_id, self.target_date, self.forecast_step


def canonical_text_sha256(path: Path) -> str:
    """Hash exact text bytes except Git's platform-dependent CRLF/LF conversion."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_daily_panel(path: Path) -> dict[str, list[tuple[date, Decimal]]]:
    """Read the checked-in daily panel, rejecting missing or duplicate days."""
    panel = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            sid = row["series_id"]
            value = Decimal(row["quantity"])
            if not sid or not value.is_finite():
                raise ValueError("Reference inputs require nonempty IDs and finite actuals")
            panel[sid].append((date.fromisoformat(row["date"]), value))
    if not panel:
        raise ValueError("Reference input is empty")
    common_dates = None
    for sid, observations in panel.items():
        observations.sort()
        dates = [label for label, _ in observations]
        expected = [dates[0] + timedelta(days=index) for index in range(len(dates))]
        if dates != expected:
            raise ValueError(f"Reference series {sid!r} has duplicate or missing daily keys")
        if common_dates is not None and dates != common_dates:
            raise ValueError("Reference series must have the same complete daily population")
        common_dates = dates
    return dict(panel)


def daily_naive_reference(
    panel: Mapping[str, list[tuple[date, Decimal]]],
    *,
    n_folds: int = 2,
    val_periods: int = 28,
    gap_periods: int = 0,
    season_length: int = 7,
) -> list[ReferencePoint]:
    """Repeat the last training seasonal cycle; reserve every requested fold."""
    if min(n_folds, val_periods, season_length) < 1 or gap_periods < 0:
        raise ValueError("Invalid reference fold, season or gap settings")
    points = []
    for fold in range(n_folds):
        for sid in sorted(panel):
            observations = panel[sid]
            val_start = len(observations) - (n_folds - fold) * val_periods
            train_end = val_start - gap_periods
            if train_end < season_length:
                raise ValueError("Insufficient reference history for the complete seasonal cycle")
            for index in range(val_start, val_start + val_periods):
                label, actual = observations[index]
                lead = index - train_end + 1
                forecast = observations[train_end - season_length + (lead - 1) % season_length][1]
                points.append(ReferencePoint(fold, sid, label, lead, actual, forecast))
    return points


def exact_totals(points: Iterable[ReferencePoint]) -> dict:
    """Exact decimal sufficient statistics; ratios need no fitted-library golden."""
    points = list(points)
    return {
        "observations": len(points),
        "absolute_error": str(sum((abs(point.naive_forecast - point.actual) for point in points), Decimal(0))),
        "absolute_actual": str(sum((abs(point.actual) for point in points), Decimal(0))),
        "signed_error": str(sum((point.naive_forecast - point.actual for point in points), Decimal(0))),
    }


def score_complete_predictions(
    rows: Iterable[dict],
    reference: list[ReferencePoint],
    model_names: set[str],
) -> dict[str, dict[str, float]]:
    """Independently verify all model keys/actuals and score raw forecast errors.

    The naive model's prediction is also checked point-for-point against the
    exact reference. Learned predictions are not treated as an approved golden.
    """
    expected = {point.key: point for point in reference}
    if len(expected) != len(reference) or not expected or not model_names:
        raise ValueError("Reference population must be nonempty and unique")
    seen = {model: set() for model in model_names}
    errors = {model: [] for model in model_names}
    run_ids = set()
    for row in rows:
        model = row["model_id"]
        if model not in model_names:
            raise ValueError(f"Unexpected reference model {model!r}")
        key = row["fold"], row["series_id"], row["target_week"], row["forecast_step"]
        if key not in expected or key in seen[model]:
            raise ValueError(f"Incorrect or duplicate {model!r} observation key: {key!r}")
        point = expected[key]
        actual, forecast = float(row["actual"]), float(row["forecast"])
        if not math.isfinite(actual) or not math.isfinite(forecast):
            raise ValueError("Reference scoring requires finite observations")
        if Decimal(str(actual)) != point.actual:
            raise ValueError("Scored actual differs from the original fixture target")
        if model == "naive_seasonal" and Decimal(str(forecast)) != point.naive_forecast:
            raise ValueError(f"Seasonal-naive prediction differs from the exact oracle at {key!r}")
        if not row.get("run_id"):
            raise ValueError("Reference scoring requires a run identity")
        run_ids.add(row["run_id"])
        seen[model].add(key)
        errors[model].append(forecast - actual)
    if len(run_ids) != 1:
        raise ValueError("Reference scoring requires exactly one run")
    volume = float(sum((abs(point.actual) for point in reference), Decimal(0)))
    if volume == 0:
        raise ValueError("Reference WMAPE is undefined for zero-volume actuals")
    scores = {}
    for model in sorted(model_names):
        if seen[model] != expected.keys():
            raise ValueError(f"Incomplete reference coverage for {model!r}")
        absolute_error = math.fsum(abs(error) for error in errors[model])
        scores[model] = {
            "wmape": absolute_error / volume,
            "normalized_bias": math.fsum(errors[model]) / volume,
            "mae": absolute_error / len(reference),
        }
    return scores


def _profile_provenance(root: Path, runtime: dict, versions: dict) -> dict:
    """Bind optional native evidence to real lock bytes and installed wheels.

    These are the existing V lock manifest, pip ``--report`` and native profile
    probe formats. Reading them does not approve their numerical output.
    """
    variables = ("M5_PROFILE_LOCK", "M5_PROFILE_INSTALL_REPORT", "M5_PROFILE_QUALIFICATION")
    supplied = [os.environ.get(name) for name in variables]
    if not any(supplied):
        return {"profile_lock_sha256": None}
    if not all(supplied):
        raise ValueError(f"Native M5 provenance requires all of {', '.join(variables)}")
    lock, install_path, qualification_path = (root / value for value in supplied)
    name = lock.resolve().relative_to((root / "requirements").resolve()).as_posix()
    minor = "".join(runtime["python"].split(".")[:2])
    if name != f"py{minor}/dev.txt":
        raise ValueError("M5 profile must be the current Python minor's dev lock")
    manifest_path = root / "requirements" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["profiles"][name]
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    if entry["sha256"] != lock_sha or entry["python"] != runtime["python"]:
        raise ValueError("M5 profile lock hash or exact Python version differs from its manifest")

    pins, hashes, current = {}, {}, None
    for line in lock.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", line)
        if match:
            current = re.sub(r"[-_.]+", "-", match[1]).lower()
            pins[current] = match[2]
            hashes[current] = set()
        if current:
            hashes[current].update(re.findall(r"--hash=sha256:([0-9a-f]{64})", line))
    if not pins:
        raise ValueError("M5 profile has no exact package pins")
    for package, version in pins.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"M5 locked package is not installed: {package}") from exc
        if installed != version:
            raise ValueError(f"M5 installed {package} differs from its exact lock pin")

    native = json.loads(qualification_path.read_text(encoding="utf-8"))
    expected_os = {"Linux": "linux", "Windows": "win32"}.get(runtime["system"])
    if (
        expected_os is None or native.get("profile") != "dev"
        or native.get("python") != runtime["python"] or native.get("os") != expected_os
        or native.get("host_architecture") != runtime["process_machine"]
        or native.get("interpreter_platform") != runtime["interpreter_platform"]
        or native.get("xgboost_build", {}).get("USE_CUDA") is not False
        or set(native.get("synthetic_native_outputs", {})) != {"lgbm_direct", "xgboost_direct"}
        or not native.get("packages")
        or any(versions.get(package) != value for package, value in native["packages"].items())
    ):
        raise ValueError("Native profile qualification does not describe this runtime")

    install = json.loads(install_path.read_text(encoding="utf-8"))
    environment = install["environment"]
    if (
        environment.get("python_full_version") != runtime["python"]
        or environment.get("sys_platform") != expected_os
        or environment.get("platform_machine") != runtime["process_machine"]
    ):
        raise ValueError("Native pip install report does not describe this runtime")
    installed_wheels = {}
    for item in install["install"]:
        package = re.sub(r"[-_.]+", "-", item["metadata"]["name"]).lower()
        if package not in versions or versions[package] is None:
            continue
        download = item["download_info"]
        digest = download.get("archive_info", {}).get("hashes", {}).get("sha256")
        filename = unquote(urlsplit(download["url"]).path).rsplit("/", 1)[-1]
        if (
            package in installed_wheels or item.get("is_yanked", False)
            or item["metadata"]["version"] != versions[package]
            or digest not in hashes.get(package, set()) or not filename.endswith(".whl")
        ):
            raise ValueError(f"M5 selected wheel is incompatible with its lock: {package}")
        installed_wheels[package] = {
            "version": versions[package], "filename": filename, "sha256": digest,
        }
    if set(installed_wheels) != {name for name, value in versions.items() if value is not None}:
        raise ValueError("M5 install report is missing numerical/runtime wheel evidence")
    return {
        "profile_lock_name": name, "profile_lock_sha256": lock_sha,
        "profile_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "profile_package_versions": pins, "profile_wheels": installed_wheels,
        "profile_install_report_sha256": hashlib.sha256(install_path.read_bytes()).hexdigest(),
        "profile_qualification_sha256": hashlib.sha256(qualification_path.read_bytes()).hexdigest(),
    }


def candidate_provenance(fixture: Path, config: Path) -> dict:
    """Describe an observed run without approving or inferring a locked profile."""
    versions = {}
    for package in PROVENANCE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    runtime = {
        "python": platform.python_version(),
        "system": platform.system(),
        "process_machine": platform.machine(),
        "interpreter_platform": sysconfig.get_platform(),
        "os_release": platform.platform(),
    }
    return {
        "approved_baseline": False,
        **runtime,
        **_profile_provenance(Path(__file__).resolve().parents[2], runtime, versions),
        "packages": versions,
        "fixture_utf8_lf_sha256": canonical_text_sha256(fixture),
        "fixture_raw_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "config_utf8_lf_sha256": canonical_text_sha256(config),
        "config_raw_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "ci_run_id": os.environ.get("GITHUB_RUN_ID"),
        "ci_source_sha": os.environ.get("GITHUB_SHA"),
        "ci_runner_arch": os.environ.get("RUNNER_ARCH"),
    }
