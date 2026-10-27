"""Verify clean native x64 profile restores without provider calls or model downloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("core", "api", "dev", "ai", "fabric", "spark")


def require_native_runtime() -> None:
    """Do not mistake x64 emulation on an ARM host for native qualification."""
    if sys.platform not in ("linux", "win32") or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError(
            "Profile qualification requires native Linux/Windows x64; "
            f"host={platform.machine()}, interpreter={sysconfig.get_platform()}"
        )
    if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
        raise RuntimeError("Only Python 3.11, 3.12 and 3.13 profiles are qualified")


def check_profile(profile: str, output: Path) -> None:
    """Import the real native stack and exercise bounded point/interval models."""
    require_native_runtime()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    sys.path.insert(0, str(ROOT))
    for module in (
        "numpy", "scipy", "pandas", "pyarrow", "polars", "statsforecast",
        "mlforecast", "hierarchicalforecast", "lightgbm", "xgboost", "sklearn",
    ):
        importlib.import_module(module)
    if profile in ("api", "dev"):
        importlib.import_module("src.api.app")
        for module in ("fastapi", "starlette", "uvicorn", "python_multipart", "jwt", "duckdb"):
            importlib.import_module(module)
    if profile in ("api", "dev", "ai"):
        importlib.import_module("anthropic")
    if profile == "fabric":
        importlib.import_module("src.fabric.notebook_adapter")
        importlib.import_module("holidays")
    if profile == "spark":
        importlib.import_module("pyspark")
        importlib.import_module("delta")

    import numpy as np
    import polars as pl
    from src.forecasting.base import BaseForecaster
    from src.forecasting.registry import registry

    data = pl.DataFrame([
        {
            "series_id": f"probe-{series}",
            "week": date(2023, 1, 2) + timedelta(weeks=period),
            "quantity": float(100 + series * 20 + period % 4 * 3 + period // 8),
        }
        for series in range(2) for period in range(80)
    ])
    def collect(model: BaseForecaster) -> dict[str, object]:
        model.fit(data)
        points = model.predict(4).sort(["series_id", "week"])
        intervals = model.predict_quantiles(4, [0.1, 0.5, 0.9]).sort(["series_id", "week"])
        if points.height != 8 or intervals.height != 8:
            raise AssertionError(f"{model.name}: native output shape changed")
        for frame, columns in (
            (points, ["forecast"]),
            (intervals, ["forecast_p10", "forecast_p50", "forecast_p90"]),
        ):
            if not np.isfinite(frame.select(columns).to_numpy()).all():
                raise AssertionError(f"{model.name}: nonfinite native output")
        if intervals.filter(
            (pl.col("forecast_p10") > pl.col("forecast_p50"))
            | (pl.col("forecast_p50") > pl.col("forecast_p90"))
        ).height:
            raise AssertionError(f"{model.name}: crossed prediction intervals")
        return {
            "points": points.to_dicts(), "intervals": intervals.to_dicts(),
            "parameters": model.get_params(),
        }

    results = {}
    for model_name, parameter in (
        ("lgbm_direct", "lgbm_params"),
        ("xgboost_direct", "xgb_params"),
    ):
        model = registry.build(
            model_name, lags=[1, 2], num_threads=1,
            **{parameter: {"n_estimators": 12, "random_state": 17}},
        )
        results[model_name] = collect(model)
    diagnostic = registry.build(
        "xgboost_direct", lags=[1, 2], num_threads=1,
        xgb_params={
            "n_estimators": 12, "random_state": 17,
            "subsample": 1.0, "colsample_bytree": 1.0,
        },
    )
    xgboost = importlib.import_module("xgboost")
    build = xgboost.build_info()
    if build["USE_CUDA"] or xgboost.__version__ != importlib.metadata.version("xgboost-cpu"):
        raise AssertionError("The loaded XGBoost runtime does not match the CPU-only profile")
    packages = (
        "numpy", "scipy", "pandas", "pyarrow", "polars", "polars-runtime-32",
        "statsforecast", "mlforecast", "hierarchicalforecast", "lightgbm",
        "xgboost-cpu", "scikit-learn",
    )
    report = {
        "profile": profile,
        "python": platform.python_version(),
        "os": sys.platform,
        "host_architecture": platform.machine(),
        "interpreter_platform": sysconfig.get_platform(),
        "polars_runtime": importlib.import_module("polars._plr").RUNTIME_REPR,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "input_sha256": hashlib.sha256(
            json.dumps(data.to_dicts(), sort_keys=True, default=str).encode()
        ).hexdigest(),
        "input_scale": {
            "observations": data.height,
            "series": data["series_id"].n_unique(),
            "target_min": data["quantity"].min(),
            "target_max": data["quantity"].max(),
        },
        "xgboost_build": {
            key: build[key]
            for key in ("USE_CUDA", "USE_OPENMP", "GCC_VERSION", "MSVC_VERSION")
            if key in build
        },
        "synthetic_native_outputs": results,
        "diagnostics": {"xgboost_without_sampling": collect(diagnostic)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Native {profile} profile passed on {report['python']} {report['interpreter_platform']}")


def check_clean_profiles(output: Path) -> None:
    """Use the locked installer to restore each profile in a new private venv."""
    require_native_runtime()
    minor = f"py{sys.version_info.major}{sys.version_info.minor}"
    locks = ROOT / "requirements" / minor
    check_profile("dev", output / "dev.json")
    for profile in PROFILES:
        if profile == "dev":
            continue
        with tempfile.TemporaryDirectory(prefix=f"forecast-{profile}-") as directory:
            environment = Path(directory)
            subprocess.run(
                [sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True,
            )
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            pip = [sys.executable, "-m", "pip", "--python", str(python)]
            install = [
                *pip, "install", "--require-hashes", "--only-binary=:all:",
                "--disable-pip-version-check", "-r", str(locks / f"{profile}.txt"),
            ]
            if profile == "spark":
                install.extend(("--no-binary=pyspark", "--build-constraint", str(locks / "dev.txt")))
            subprocess.run(install, check=True)
            subprocess.run([*pip, "check"], check=True)
            subprocess.run(
                [
                    str(python), str(Path(__file__).resolve()), "--profile", profile,
                    "--output", str(output / f"{profile}.json"),
                ],
                check=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.all:
        check_clean_profiles(arguments.output.resolve())
    else:
        check_profile(arguments.profile, arguments.output.resolve())


if __name__ == "__main__":
    main()
