"""
Pipeline manifest — provenance sidecar for every forecast run.

Captures the full lineage of a forecast: input data fingerprint, cleansing
actions, validation results, regressor screening, config hash, champion model,
and output metadata.  Written as a JSON file alongside each forecast Parquet.

Usage
-----
>>> manifest = build_manifest(
...     run_id="abc123",
...     config=config,
...     actuals=actuals_df,
...     series_builder=builder,
...     champion_model_id="lgbm_direct",
...     forecast=forecast_df,
...     forecast_file="forecast_retail_2024-06-01.parquet",
... )
>>> path = write_manifest(manifest, "data/forecasts/forecast_retail_2024-06-01.parquet")
"""

import hashlib
import importlib.metadata
import json
import logging
import math
import numbers
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import parse_qsl, urlsplit

import polars as pl

from ..config.schema import PlatformConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineManifest:
    """Full provenance record for a single forecast run."""

    # Run identity
    run_id: str = ""
    timestamp: str = ""
    lob: str = ""

    # Input data fingerprint
    input_data_hash: str = ""
    input_row_count: int = 0
    input_series_count: int = 0
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None

    # Cleansing summary
    cleansing_applied: bool = False
    outliers_clipped: int = 0
    stockout_periods_imputed: int = 0
    rows_modified: int = 0

    # Validation summary
    validation_applied: bool = False
    validation_passed: bool = True
    validation_warnings: int = 0
    validation_errors: int = 0

    # Regressor screen summary
    regressor_screen_applied: bool = False
    regressors_dropped: List[str] = field(default_factory=list)
    regressor_warnings: List[str] = field(default_factory=list)

    # Config
    config_hash: str = ""

    # Model
    champion_model_id: str = ""
    backtest_wmape: Optional[float] = None

    # Output
    forecast_horizon: int = 0
    forecast_row_count: int = 0
    forecast_file: str = ""

    # Post-pipeline validation (4-layer framework)
    post_validation_grade: Optional[str] = None
    post_validation_score: Optional[int] = None

    # Version zero is a legacy sidecar, not an atomic run commit.
    schema_version: int = 0
    run_type: str = "forecast"
    status: str = "legacy"
    started_at: str = ""
    completed_at: str = ""
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    input_snapshot_stage: str = "unknown"
    prepared_data_hash: str = ""
    input_digest_kind: str = "legacy"
    config_digest_kind: str = "legacy"
    forecast_origin: Optional[str] = None
    models: List[Dict[str, Any]] = field(default_factory=list)
    execution: Dict[str, Any] = field(default_factory=dict)
    source: Dict[str, Any] = field(default_factory=dict)
    reproducibility: Dict[str, Any] = field(default_factory=dict)
    total_seconds: float = 0.0


def build_manifest(
    run_id: str,
    config: PlatformConfig,
    actuals: pl.DataFrame,
    series_builder: Any,
    champion_model_id: str,
    forecast: pl.DataFrame,
    forecast_file: str,
    backtest_wmape: Optional[float] = None,
    post_validation_result: Optional[Dict[str, Any]] = None,
) -> PipelineManifest:
    """
    Collect provenance from existing pipeline components into a manifest.

    Parameters
    ----------
    run_id:
        Unique identifier for this pipeline run.
    config:
        The PlatformConfig that drove the run.
    actuals:
        Raw input actuals DataFrame (before any processing).
    series_builder:
        The SeriesBuilder instance (carries validation/cleansing reports).
    champion_model_id:
        Name of the champion model used for forecasting.
    forecast:
        The output forecast DataFrame.
    forecast_file:
        Filename of the forecast Parquet file.
    backtest_wmape:
        Optional backtest WMAPE of the champion model.
    post_validation_result:
        Optional dict from run_post_validation() with grade/score keys.
    """
    manifest = PipelineManifest(
        run_id=run_id,
        timestamp=datetime.utcnow().isoformat(timespec="seconds"),
        lob=config.lob,
        champion_model_id=champion_model_id,
        backtest_wmape=backtest_wmape,
        forecast_horizon=config.forecast.horizon_weeks,
        forecast_row_count=len(forecast),
        forecast_file=forecast_file,
    )

    # Input data fingerprint
    manifest.input_row_count = len(actuals)
    time_col = config.forecast.time_column
    id_col = config.forecast.series_id_column
    if id_col in actuals.columns:
        manifest.input_series_count = actuals[id_col].n_unique()
    if time_col in actuals.columns:
        mn = actuals[time_col].min()
        mx = actuals[time_col].max()
        if mn is not None:
            manifest.date_range_start = mn.isoformat() if isinstance(mn, (date, datetime)) else str(mn)
        if mx is not None:
            manifest.date_range_end = mx.isoformat() if isinstance(mx, (date, datetime)) else str(mx)

    manifest.input_data_hash = _hash_dataframe(actuals)

    # Config hash
    manifest.config_hash = _hash_config(config)

    # Cleansing report (if available)
    cleansing = getattr(series_builder, "_last_cleansing_report", None)
    if cleansing is not None:
        manifest.cleansing_applied = True
        manifest.outliers_clipped = getattr(cleansing, "total_outliers", 0)
        manifest.stockout_periods_imputed = getattr(cleansing, "total_stockout_periods", 0)
        manifest.rows_modified = getattr(cleansing, "rows_modified", 0)

    # Validation report (if available)
    validation = getattr(series_builder, "_last_validation_report", None)
    if validation is not None:
        manifest.validation_applied = True
        manifest.validation_passed = getattr(validation, "passed", True)
        manifest.validation_warnings = len(getattr(validation, "warnings", []))
        manifest.validation_errors = len(getattr(validation, "errors", []))

    # Regressor screen report (if available)
    screen = getattr(series_builder, "_last_regressor_screen_report", None)
    if screen is not None:
        manifest.regressor_screen_applied = True
        manifest.regressors_dropped = list(getattr(screen, "dropped_columns", []))
        manifest.regressor_warnings = list(getattr(screen, "warnings", []))

    # Post-pipeline validation (4-layer framework)
    if post_validation_result and not post_validation_result.get("skipped"):
        manifest.post_validation_grade = post_validation_result.get("grade")
        manifest.post_validation_score = post_validation_result.get("score")

    return manifest


def write_manifest(manifest: PipelineManifest, forecast_path: str) -> str:
    """
    Write a manifest as a JSON sidecar alongside the forecast file.

    Parameters
    ----------
    manifest:
        The PipelineManifest to write.
    forecast_path:
        Full path to the forecast Parquet file.

    Returns
    -------
    Path to the written manifest JSON file.
    """
    from .run_store import atomic_bytes_create

    if manifest.schema_version != 0:
        raise ValueError("Versioned run manifests must be published through RunStore")
    p = Path(forecast_path)
    manifest_path = p.parent / f"{p.stem}_manifest.json"

    atomic_bytes_create(manifest_path, canonical_json(asdict(manifest)))

    logger.info("Manifest written to %s", manifest_path)
    return str(manifest_path)


def read_manifest(manifest_path: str) -> PipelineManifest:
    """
    Load a PipelineManifest from a JSON file.

    Parameters
    ----------
    manifest_path:
        Path to the manifest JSON file.
    """
    data = json.loads(Path(manifest_path).read_text())
    return PipelineManifest(**{
        k: v for k, v in data.items()
        if k in PipelineManifest.__dataclass_fields__
    })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_dataframe(df: pl.DataFrame) -> str:
    """Hash the normalized in-memory IPC representation, never just its shape.

    Committed runs instead record the checksum of their actual stored Parquet
    bytes. Neither representation claims semantic equality across serializers.
    """
    raw = df.write_ipc(None, compression="uncompressed").getvalue()
    return hashlib.sha256(raw).hexdigest()


def _hash_config(config: PlatformConfig) -> str:
    """Hash the canonical, explicitly safe configuration projection."""
    return hashlib.sha256(canonical_json(configuration_snapshot(config))).hexdigest()


_PRIVATE_KEYS = frozenset({
    "password", "passwd", "secret", "secretkey", "jwtsecretkey", "privatekey",
    "apikey", "accesstoken", "refreshtoken", "token", "authorization",
    "credential", "credentials", "clientsecret", "webhookurl", "connectionstring",
    "auth", "authtoken", "bearer", "sastoken", "headers", "cookies", "apikeyenvvar",
})
_TOKEN_SHAPE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")


def _private_name(name: str) -> bool:
    return re.sub(r"[^a-z0-9]", "", name.lower()) in _PRIVATE_KEYS


def _private_string(value: str) -> bool:
    if "PRIVATE KEY-----" in value or _TOKEN_SHAPE.search(value):
        return True
    if "://" in value:
        try:
            parsed = urlsplit(value)
        except ValueError:
            return True
        if parsed.username is not None or parsed.password is not None:
            return True
        if any(_private_name(key) or key.lower() in {"sig", "signature", "key"} for key, _ in parse_qsl(parsed.query)):
            return True
    return False


def normalized_json_value(value: Any) -> Any:
    """Normalize supported metadata types without object repr or pickle."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return normalized_json_value(value.value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            return {"$float": "nan" if math.isnan(numeric) else ("inf" if numeric > 0 else "-inf")}
        return numeric
    if isinstance(value, (datetime, date)):
        return {"$datetime" if isinstance(value, datetime) else "$date": value.isoformat()}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Metadata mappings must have string keys")
        return {key: normalized_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized_json_value(item) for item in value]
    raise TypeError("Unsupported metadata value type")


def canonical_json(value: Any) -> bytes:
    """Canonical JSON v1: sorted UTF-8 keys, compact form and tagged special values."""
    return json.dumps(
        normalized_json_value(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")


def _safe_settings(value: Any, omitted: List[str], prefix: str = "") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Settings mappings must have string keys")
            label = f"{prefix}.{key}" if prefix else key
            if _private_name(key) or key.endswith(("_path", "_dir")):
                omitted.append(label)
            else:
                result[key] = _safe_settings(item, omitted, label)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_settings(item, omitted, prefix) for item in value]
    if isinstance(value, str) and (_private_string(value) or Path(value).is_absolute()
                                  or re.match(r"^[A-Za-z]:[\\/]", value)):
        omitted.append(prefix)
        return {"$omitted": "private_setting"}
    return normalized_json_value(value)


def configuration_snapshot(
    config: PlatformConfig, execution: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Project scientific settings only, excluding credentials and server locations."""
    omitted = [
        "description", "ai", "analysis", "observability", "output.locations",
        "transition.override_store_path", "forecast.external_regressors.future_features_path",
    ]
    forecast = asdict(config.forecast)
    forecast["external_regressors"].pop("future_features_path", None)
    transition = asdict(config.transition)
    transition.pop("override_store_path", None)
    settings = {
        "lob": config.lob,
        "hierarchies": [asdict(item) for item in config.hierarchies],
        "reconciliation": asdict(config.reconciliation),
        "forecast": forecast,
        "backtest": asdict(config.backtest),
        "transition": transition,
        "data_quality": asdict(config.data_quality),
        "parallelism": asdict(config.parallelism),
        "post_validation": asdict(config.post_validation),
        "metrics": list(config.metrics),
        "output": {"grain": dict(config.output.grain)},
    }
    return {
        "schema_version": 1,
        "normalization": "canonical-json-v1",
        "settings": _safe_settings(settings, omitted),
        "execution": _safe_settings(dict(execution or {}), omitted, "execution"),
        "omitted_settings": sorted(set(omitted)),
    }


def model_metadata(model: Any, configured_frequency: str, horizon: int) -> Dict[str, Any]:
    """Capture declared model parameters without serializing fitted state or secrets."""
    return _model_metadata(model, configured_frequency, horizon, frozenset())


def _model_metadata(
    model: Any, configured_frequency: str, horizon: int, ancestors: frozenset[int],
) -> Dict[str, Any]:
    from ..forecasting.constrained import ConstrainedDemandEstimator
    from ..forecasting.ensemble import WeightedEnsembleForecaster

    if id(model) in ancestors or len(ancestors) >= 16:
        raise ValueError("Composite forecaster metadata requires an acyclic tree of at most 16 levels")
    ancestors = ancestors | {id(model)}
    omitted: List[str] = []
    if not isinstance(model.name, str) or not model.name or _private_string(model.name):
        raise ValueError("Model identity cannot contain credential material")
    if isinstance(model, WeightedEnsembleForecaster):
        components = model._forecasters
    elif isinstance(model, ConstrainedDemandEstimator):
        components = [model.base]
    else:
        components = None
    component_metadata = None if components is None else [
        _model_metadata(component, configured_frequency, horizon, ancestors)
        for component in components
    ]
    # Wrapper getters may flatten child values over their own constraints.
    declared = (
        {"base": model.base.name, **asdict(model.constraints)}
        if isinstance(model, ConstrainedDemandEstimator) else model.get_params()
    )
    if not isinstance(declared, Mapping):
        raise TypeError("Model parameters must be a mapping")
    parameters = _safe_settings(declared, omitted, "parameters")
    result = {
        "name": model.name,
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameters": parameters,
        "configured_frequency": configured_frequency,
        "model_frequency": getattr(model, "frequency", None),
        "horizon_periods": horizon,
        "parameters_omitted": omitted,
        "fitted_model_persisted": False,
    }
    for name in ("num_threads", "n_jobs", "_freq"):
        if hasattr(model, name):
            result[name.lstrip("_")] = normalized_json_value(getattr(model, name))
    module = sys.modules.get(type(model).__module__)
    filename = getattr(module, "__file__", None)
    if filename and Path(filename).is_file():
        result["implementation_sha256"] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    else:
        result["implementation_sha256"] = None
    if component_metadata is not None:
        result["components"] = component_metadata
    return result


def runtime_metadata() -> Dict[str, Any]:
    """Record installed runtime versions; absent optional backends remain explicit."""
    versions: Dict[str, Optional[str]] = {
        "python": ".".join(map(str, sys.version_info[:3])), "polars": pl.__version__,
    }
    for package in ("statsforecast", "mlforecast", "lightgbm", "xgboost", "numpy", "neuralforecast", "nixtla"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    revision = os.environ.get("FORECAST_BUILD_REVISION") or None
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("FORECAST_BUILD_REVISION must be a hexadecimal source revision")
    source_root = Path(__file__).resolve().parents[1]
    source_files = (
        ("pipeline", "forecast.py"), ("pipeline", "backtest.py"),
        ("pipeline", "run_store.py"), ("pipeline", "manifest.py"),
        ("series", "builder.py"), ("config", "schema.py"),
    )
    hashes = {
        ".".join((*parts[:-1], Path(parts[-1]).stem)): hashlib.sha256(
            source_root.joinpath(*parts).read_bytes()
        ).hexdigest()
        for parts in source_files
    }
    return {
        "runtime_versions": versions, "code_revision": revision or "not_stamped",
        "observed_source_sha256": hashes,
        "source_hash_scope": "listed installed source files observed at publication; not a full dependency closure",
    }
