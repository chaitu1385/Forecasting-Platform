"""
YAML configuration loader with LOB inheritance.

Supports a base config that LOB-specific configs can selectively override.
Nested dicts are deep-merged; lists are replaced entirely.
"""

import copy
import math
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, get_args, get_origin

import yaml

from .schema import (
    AIConfig,
    AlertConfig,
    AnalysisConfig,
    BacktestConfig,
    CalibrationConfig,
    CleansingConfig,
    ConstraintConfig,
    DataQualityConfig,
    DataQualityReportConfig,
    ExternalRegressorConfig,
    ForecastConfig,
    HierarchyConfig,
    HorizonBucket,
    ObservabilityConfig,
    OutputConfig,
    ParallelismConfig,
    PlatformConfig,
    PostValidationConfig,
    ReconciliationConfig,
    RegressorScreenConfig,
    StructuralBreakConfig,
    TransitionConfig,
    ValidationConfig,
)

_PERIOD_ALIASES = {
    ForecastConfig: {"horizon_periods": "horizon_weeks"},
    BacktestConfig: {"val_periods": "val_weeks", "gap_periods": "gap_weeks"},
    DataQualityConfig: {"min_series_length_periods": "min_series_length_weeks"},
    TransitionConfig: {"transition_window_periods": "transition_window_weeks"},
}


def _validate_types(value: Any, expected: Any, path: str = "config") -> None:
    """Check known schema fields without coercion or a new unknown-key policy."""
    if expected is Any:
        return
    origin = get_origin(expected)
    args = get_args(expected)
    if origin is Union and type(None) in args:
        if value is None:
            return
        _validate_types(value, next(t for t in args if t is not type(None)), path)
    elif is_dataclass(expected):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping")
        schema_fields = {f.name: f for f in fields(expected)}
        aliases = _PERIOD_ALIASES.get(expected, {})
        for f in schema_fields.values():
            if f.default is MISSING and f.default_factory is MISSING and f.name not in value:
                raise ValueError(f"{path}.{f.name} is required")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            schema_field = schema_fields.get(aliases.get(key, key))
            if schema_field is not None:
                _validate_types(item, schema_field.type, f"{path}.{key}")
    elif origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list")
        for index, item in enumerate(value):
            _validate_types(item, args[0], f"{path}[{index}]")
    elif origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping")
        for key, item in value.items():
            _validate_types(key, args[0], f"{path} key")
            _validate_types(item, args[1], f"{path}.{key}")
    elif expected is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{path} must be a finite number")
    elif type(value) is not expected:
        raise ValueError(f"{path} must be {expected.__name__}")


def _parse_yaml_mapping(content: str) -> Dict[str, Any]:
    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        # Do not echo source snippets, which can contain paths or credentials.
        raise ValueError(f"Invalid config YAML syntax{location}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping; use {} for platform defaults")
    return raw


def parse_config_yaml(content: str) -> PlatformConfig:
    """Parse YAML and validate known configuration fields, preserving period aliases."""
    return _dict_to_config(_parse_yaml_mapping(content))


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base*. Lists are replaced, not appended."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_hierarchy(raw: Dict[str, Any]) -> HierarchyConfig:
    return HierarchyConfig(
        name=raw["name"],
        levels=raw.get("levels", []),
        id_column=raw.get("id_column", ""),
        fixed=raw.get("fixed", False),
        reconciliation_level=raw.get("reconciliation_level"),
    )


def _dict_to_config(d: Dict[str, Any]) -> PlatformConfig:
    """Convert a raw dict (from YAML) to a typed PlatformConfig."""
    _validate_types(d, PlatformConfig)
    hierarchies = [
        _parse_hierarchy(h) for h in d.get("hierarchies", [])
    ]

    recon_raw = d.get("reconciliation", {})
    reconciliation = ReconciliationConfig(
        method=recon_raw.get("method", "bottom_up"),
        product_level=recon_raw.get("product_level"),
        geography_level=recon_raw.get("geography_level"),
    )

    fc_raw = d.get("forecast", {})
    er_raw = fc_raw.get("external_regressors", {})
    screen_raw = er_raw.get("screen", {})
    cal_raw = fc_raw.get("calibration", {})
    constraint_raw = fc_raw.get("constraints", {})
    forecast = ForecastConfig(
        horizon_weeks=fc_raw.get("horizon_periods",
                                 fc_raw.get("horizon_weeks", 39)),
        frequency=fc_raw.get("frequency", "W"),
        target_column=fc_raw.get("target_column", "quantity"),
        time_column=fc_raw.get("time_column", "week"),
        series_id_column=fc_raw.get("series_id_column", "series_id"),
        forecasters=fc_raw.get("forecasters", ["naive_seasonal"]),
        quantiles=fc_raw.get("quantiles", []),
        intermittent_forecasters=fc_raw.get("intermittent_forecasters", []),
        sparse_detection=fc_raw.get("sparse_detection", True),
        sparse_adi_threshold=fc_raw.get("sparse_adi_threshold", 1.32),
        sparse_cv2_threshold=fc_raw.get("sparse_cv2_threshold", 0.49),
        external_regressors=ExternalRegressorConfig(
            enabled=er_raw.get("enabled", False),
            feature_columns=er_raw.get("feature_columns", []),
            future_features_path=er_raw.get("future_features_path"),
            feature_types=er_raw.get("feature_types", {}),
            screen=RegressorScreenConfig(**screen_raw),
        ),
        calibration=CalibrationConfig(**cal_raw),
        constraints=ConstraintConfig(**constraint_raw),
    )

    bt_raw = d.get("backtest", {})
    backtest = BacktestConfig(
        n_folds=bt_raw.get("n_folds", 3),
        val_weeks=bt_raw.get("val_periods",
                             bt_raw.get("val_weeks", 13)),
        gap_weeks=bt_raw.get("gap_periods",
                             bt_raw.get("gap_weeks", 0)),
        champion_granularity=bt_raw.get("champion_granularity", "lob"),
        primary_metric=bt_raw.get("primary_metric", "wmape"),
        secondary_metric=bt_raw.get("secondary_metric", "normalized_bias"),
        selection_strategy=bt_raw.get("selection_strategy", "champion"),
        horizon_buckets=[HorizonBucket(**b) for b in bt_raw.get("horizon_buckets", [])],
    )

    dq_raw = d.get("data_quality", {})
    validation_raw = dq_raw.get("validation", {})
    cleansing_raw = dq_raw.get("cleansing", {})
    breaks_raw = dq_raw.get("structural_breaks", {})
    report_raw = dq_raw.get("report", {})
    data_quality = DataQualityConfig(
        fill_gaps=dq_raw.get("fill_gaps", True),
        fill_value=dq_raw.get("fill_value", 0.0),
        min_series_length_weeks=dq_raw.get(
            "min_series_length_periods",
            dq_raw.get("min_series_length_weeks", 52),
        ),
        drop_zero_series=dq_raw.get("drop_zero_series", False),
        validate_frequency=dq_raw.get("validate_frequency", False),
        validation=ValidationConfig(**validation_raw),
        cleansing=CleansingConfig(**cleansing_raw),
        structural_breaks=StructuralBreakConfig(**breaks_raw),
        report=DataQualityReportConfig(**report_raw),
    )

    tr_raw = d.get("transition", {})
    transition = TransitionConfig(
        transition_window_weeks=tr_raw.get(
            "transition_window_periods",
            tr_raw.get("transition_window_weeks", 13),
        ),
        ramp_shape=tr_raw.get("ramp_shape", "linear"),
        enable_overrides=tr_raw.get("enable_overrides", True),
        override_store_path=tr_raw.get(
            "override_store_path", "data/overrides.duckdb"
        ),
    )

    out_raw = d.get("output", {})
    output = OutputConfig(
        grain=out_raw.get("grain", {}),
        forecast_path=out_raw.get("forecast_path", "data/forecasts/"),
        metrics_path=out_raw.get("metrics_path", "data/metrics/"),
        bi_export_path=out_raw.get("bi_export_path", "data/bi_exports/"),
        format=out_raw.get("format", "parquet"),
    )

    # ── Parallelism config ───────────────────────────────────────────────
    par_raw = d.get("parallelism", {})
    parallelism = ParallelismConfig(
        backend=par_raw.get("backend", "local"),
        n_workers=par_raw.get("n_workers", -1),
        n_jobs_statsforecast=par_raw.get("n_jobs_statsforecast", -1),
        num_threads_mlforecast=par_raw.get("num_threads_mlforecast", -1),
        batch_size=par_raw.get("batch_size", 0),
        gpu=par_raw.get("gpu", False),
    )

    # ── Observability config ──────────────────────────────────────────────
    obs_raw = d.get("observability", {})
    alert_raw = obs_raw.get("alerts", {})
    observability = ObservabilityConfig(
        log_format=obs_raw.get("log_format", "text"),
        log_level=obs_raw.get("log_level", "INFO"),
        metrics_backend=obs_raw.get("metrics_backend", "log"),
        statsd_host=obs_raw.get("statsd_host", "localhost"),
        statsd_port=obs_raw.get("statsd_port", 8125),
        metrics_prefix=obs_raw.get("metrics_prefix", "forecast_platform"),
        cost_per_second=obs_raw.get("cost_per_second", 0.0),
        alerts=AlertConfig(
            channels=alert_raw.get("channels", ["log"]),
            webhook_url=alert_raw.get("webhook_url", ""),
            min_severity=alert_raw.get("min_severity", "warning"),
            webhook_timeout=alert_raw.get("webhook_timeout", 10),
        ),
    )

    # ── Post-validation config ─────────────────────────────────────────
    pv_raw = d.get("post_validation", {})
    post_validation = PostValidationConfig(
        enabled=pv_raw.get("enabled", True),
        structural_checks=pv_raw.get("structural_checks", True),
        logical_checks=pv_raw.get("logical_checks", True),
        business_rules_checks=pv_raw.get("business_rules_checks", True),
        simpsons_paradox_checks=pv_raw.get("simpsons_paradox_checks", True),
        max_yoy_change_pct=pv_raw.get("max_yoy_change_pct", 500.0),
        max_period_change_pct=pv_raw.get("max_period_change_pct", 500.0),
        custom_range_rules=pv_raw.get("custom_range_rules", []),
        simpsons_segment_columns=pv_raw.get("simpsons_segment_columns", []),
        halt_on_blocker=pv_raw.get("halt_on_blocker", False),
        min_grade=pv_raw.get("min_grade", "D"),
    )

    analysis = AnalysisConfig(**d.get("analysis", {}))
    ai = AIConfig(**d.get("ai", {}))

    return PlatformConfig(
        lob=d.get("lob", "default"),
        description=d.get("description", ""),
        hierarchies=hierarchies,
        reconciliation=reconciliation,
        forecast=forecast,
        backtest=backtest,
        transition=transition,
        data_quality=data_quality,
        output=output,
        analysis=analysis,
        parallelism=parallelism,
        observability=observability,
        ai=ai,
        post_validation=post_validation,
        metrics=d.get("metrics", ["wmape", "normalized_bias"]),
    )


def load_config(path: str) -> PlatformConfig:
    """Load a single YAML config file."""
    with open(path, encoding="utf-8") as f:
        return parse_config_yaml(f.read())


def load_config_with_overrides(
    base_path: str,
    override_path: Optional[str] = None,
) -> PlatformConfig:
    """
    Load a base config and optionally deep-merge an LOB override on top.

    Usage::

        # Base only
        cfg = load_config_with_overrides("configs/platform_config.yaml")

        # Base + Surface overrides
        cfg = load_config_with_overrides(
            "configs/platform_config.yaml",
            "configs/lob/surface.yaml",
        )
    """
    with open(base_path, encoding="utf-8") as f:
        base = _parse_yaml_mapping(f.read())

    if override_path and Path(override_path).exists():
        with open(override_path, encoding="utf-8") as f:
            override = _parse_yaml_mapping(f.read())
        merged = _deep_merge(base, override)
    else:
        merged = base

    return _dict_to_config(merged)
