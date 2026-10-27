"""
Post-pipeline validation step.

Runs the 4-layer validation framework against backtest results or forecast
output and produces an A-F confidence grade.

This module bridges the PlatformConfig's PostValidationConfig with the
validation package's ``run_full_validation()`` orchestrator, adapting
forecasting-specific inputs into the config format expected by each layer.
"""

import logging
from typing import Any, Dict, List, Optional

import polars as pl

from ..config.schema import PostValidationConfig
from ..validation import (
    check_simpsons_multi_segment,
    format_confidence_badge,
    run_logical_checks,
    run_structural_checks,
    score_confidence,
    validate_business_rules,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when post-validation encounters a BLOCKER and halt_on_blocker is True."""


def run_post_validation(
    results_df: pl.DataFrame,
    config: PostValidationConfig,
    lob: str,
    forecast_df: Optional[pl.DataFrame] = None,
    *,
    id_col: str = "series_id",
    time_col: Optional[str] = None,
) -> dict:
    """Run 4-layer validation on pipeline output.

    Parameters
    ----------
    results_df
        Backtest results DataFrame (with columns like wmape, forecast,
        actual, series_id, model_id, etc.) or forecast output DataFrame.
    config
        PostValidationConfig controlling which layers run and thresholds.
    lob
        Line of business name (for logging).
    forecast_df
        Optional separate forecast DataFrame to validate (e.g., from
        ForecastPipeline). If None, results_df is used.
    id_col
        Series identifier in the output. Backtest results use the canonical
        ``series_id`` even when input columns are customized.
    time_col
        Output time column. If None, detect ``target_week``, ``week``, or
        ``date``. ForecastPipeline supplies its configured time column.

    Returns
    -------
    dict with ``grade``, ``score``, ``badge``, ``layers``, ``confidence``.

    Raises
    ------
    ValidationError
        A BLOCKER is found and ``halt_on_blocker`` is enabled. Unexpected
        checker exceptions propagate regardless of that policy.
    """
    if not config.enabled:
        logger.info("[%s] Post-validation disabled, skipping.", lob)
        return {
            "grade": "N/A",
            "score": -1,
            "badge": "[N/A] Post-validation disabled",
            "layers": {},
            "confidence": {},
            "skipped": True,
        }

    df = results_df
    target_df = forecast_df if forecast_df is not None else df

    logger.info("[%s] Running post-pipeline validation (%d rows)...", lob, len(df))

    # --- Layer 1: Structural checks ---
    structural_result = None
    if config.structural_checks:
        structural_config = _build_structural_config(df, id_col, time_col)
        structural_result = run_structural_checks(df, config=structural_config)
        logger.info(
            "[%s] Structural: %d/%d checks passed",
            lob,
            structural_result.get("checks_passed", 0),
            structural_result.get("checks_run", 0),
        )

    # --- Layer 2: Logical checks ---
    logical_result = None
    if config.logical_checks:
        logical_config = _build_logical_config(df, time_col)
        logical_result = run_logical_checks(df, config=logical_config)
        logger.info(
            "[%s] Logical: %d/%d checks passed",
            lob,
            logical_result.get("checks_passed", 0),
            logical_result.get("checks_run", 0),
        )

    # --- Layer 3: Business rules ---
    business_result = None
    if config.business_rules_checks:
        business_config = _build_business_config(target_df, config, time_col)
        business_result = validate_business_rules(target_df, config=business_config)
        logger.info(
            "[%s] Business rules: %d/%d checks passed",
            lob,
            business_result.get("checks_passed", 0),
            business_result.get("checks_run", 0),
        )

    # --- Layer 4: Simpson's Paradox ---
    paradox_result = None
    if config.simpsons_paradox_checks:
        paradox_result = _run_simpsons_check(df, config, id_col)
        if paradox_result:
            any_paradox = paradox_result.get("any_paradox", False)
            if paradox_result.get("evaluated") is False:
                logger.warning(
                    "[%s] Simpson's Paradox: %s; status=%s comparisons=%s",
                    lob, "DETECTED with unavailable comparisons" if any_paradox else "unavailable",
                    paradox_result.get("status"), paradox_result.get("comparisons_run", 0),
                )
            else:
                logger.info("[%s] Simpson's Paradox: %s", lob, "DETECTED" if any_paradox else "not detected")

    # --- Confidence scoring ---
    confidence = score_confidence(
        structural=structural_result,
        logical=logical_result,
        business=business_result,
        paradox=paradox_result,
    )
    badge = format_confidence_badge(confidence)

    grade = confidence["grade"]
    score = confidence["score"]
    logger.info("[%s] Validation grade: %s (%d/100)", lob, grade, score)

    result = {
        "grade": grade,
        "score": score,
        "badge": badge,
        "layers": {
            "structural": structural_result,
            "logical": logical_result,
            "business": business_result,
            "paradox": paradox_result,
        },
        "confidence": confidence,
        "skipped": False,
    }

    # --- Halt on blocker ---
    if config.halt_on_blocker:
        caps = confidence.get("caps_applied", {})
        if caps.get("blocker_cap") or (paradox_result and paradox_result.get("severity") == "BLOCKER"):
            raise ValidationError(
                f"[{lob}] Post-validation BLOCKER detected (grade={grade}, "
                f"score={score}). Pipeline halted. "
                f"Details: {confidence.get('recommendations', [])}"
            )

    # --- Grade check ---
    _GRADE_RANK = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    if _GRADE_RANK.get(grade, 4) > _GRADE_RANK.get(config.min_grade, 3):
        logger.warning(
            "[%s] Validation grade %s is below minimum %s. "
            "Recommendations: %s",
            lob, grade, config.min_grade,
            confidence.get("recommendations", []),
        )

    return result


# ---------------------------------------------------------------------------
# Config builders — adapt PostValidationConfig to per-layer config dicts
# ---------------------------------------------------------------------------

def _find_time_column(df: pl.DataFrame, time_col: Optional[str]) -> Optional[str]:
    candidates = [time_col] if time_col is not None else ["target_week", "week", "date"]
    return next((col for col in candidates if col in df.columns), None)


def _build_structural_config(
    df: pl.DataFrame,
    id_col: str = "series_id",
    time_col: Optional[str] = None,
) -> dict:
    """Build structural check config from available columns."""
    config: Dict[str, Any] = {"min_rows": 1}

    # Detect primary key columns
    pk_candidates = []
    for col in [id_col, "model_id", "fold", _find_time_column(df, time_col)]:
        if col is not None and col in df.columns and col not in pk_candidates:
            pk_candidates.append(col)
    if pk_candidates:
        config["primary_key"] = pk_candidates

    return config


def _build_logical_config(df: pl.DataFrame, time_col: Optional[str] = None) -> dict:
    """Build logical check config from available columns."""
    config: Dict[str, Any] = {}

    # Date column detection
    date_col = _find_time_column(df, time_col)
    if date_col is not None:
        config["date_column"] = date_col

    # Value column detection
    for col in ["forecast", "quantity", "actual"]:
        if col in df.columns:
            config["value_column"] = col
            break

    # Group column for aggregation checks
    if "model_id" in df.columns:
        config["group_col"] = "model_id"

    return config


def _build_business_config(
    df: pl.DataFrame,
    pv_config: PostValidationConfig,
    time_col: Optional[str] = None,
) -> dict:
    """Build business rules config from PostValidationConfig."""
    config: Dict[str, Any] = {}

    # Non-negative columns
    non_neg = [c for c in ["forecast", "quantity", "actual"] if c in df.columns]
    if non_neg:
        config["non_negative_columns"] = non_neg

    # Temporal consistency
    date_col = _find_time_column(df, time_col)
    value_cols = [c for c in ["forecast", "quantity"] if c in df.columns]
    if date_col and value_cols:
        config["temporal"] = {
            "date_column": date_col,
            "value_columns": value_cols,
            "max_period_change_pct": pv_config.max_period_change_pct,
        }

    # Custom range rules from config
    if pv_config.custom_range_rules:
        config["range_rules"] = pv_config.custom_range_rules

    return config


def _run_simpsons_check(
    df: pl.DataFrame,
    config: PostValidationConfig,
    id_col: str = "series_id",
) -> Optional[dict]:
    """Run Simpson's Paradox detection."""
    if "competition_scope" in df.columns:
        return _run_competition_simpsons(df, config, id_col)
    # Determine value column for paradox checking
    value_col = None
    for col in ["wmape", "mape", "forecast", "quantity"]:
        if col in df.columns:
            value_col = col
            break
    if value_col is None:
        logger.debug("No suitable value column for Simpson's Paradox check.")
        return None

    # Determine segment columns
    segment_cols: List[str] = list(config.simpsons_segment_columns)
    if not segment_cols:
        # Auto-detect categorical/low-cardinality columns
        candidates = []
        for col in ["model_id", id_col, "channel", "grain_level"]:
            if col in df.columns:
                n_unique = df[col].n_unique()
                if 2 <= n_unique <= 50:
                    candidates.append(col)
        segment_cols = candidates[:5]

    if not segment_cols:
        logger.debug("No segment columns available for Simpson's Paradox check.")
        return None

    return check_simpsons_multi_segment(
        df,
        segment_columns=segment_cols,
        value_col=value_col,
    )


def _run_competition_simpsons(
    df: pl.DataFrame, config: PostValidationConfig, id_col: str,
) -> dict:
    """Compare eligible model arms within their declared observation population."""
    from itertools import combinations

    from ..metrics.aggregation import (
        SCORING_POLICY,
        observation_metrics,
        validate_comparable_records,
    )

    scopes = df["competition_scope"].unique().to_list()
    if len(scopes) != 1 or scopes[0] not in {"global", "lob", "series_id", "product_group"}:
        raise ValueError("Simpson validation requires one supported declared competition scope")
    scope = scopes[0]
    groups = [] if scope == "global" else [scope]
    validate_comparable_records(df, groups)
    populations = df.partition_by(groups, maintain_order=True) if groups else [df]
    comparisons = []
    unavailable = []
    reversals = []
    any_paradox = False

    for population in populations:
        models = sorted(population["model_id"].unique().to_list())
        context = {
            "competition_scope": scope,
            "group_key": str(population[scope][0]) if groups else "global",
            "metric_rows": population.height,
            "n_series": population["series_id"].n_unique(),
            "candidate_models": models,
        }
        if len(models) < 2:
            unavailable.append({**context, "reason": "single_candidate"})
            continue
        requested = list(config.simpsons_segment_columns)
        if requested and (set(requested) - set(population.columns) or "model_id" in requested):
            raise ValueError("Simpson segment columns must exist and cannot be the model comparison arm")
        candidates = requested or [column for column in (id_col, "channel", "grain_level") if column in population.columns]
        segments = [column for column in candidates if 2 <= population[column].n_unique() <= 50]
        if not segments:
            unavailable.append({**context, "reason": "insufficient_segments"})
            continue

        for segment in segments:
            if population[segment].null_count():
                raise ValueError("Simpson comparison segments cannot contain null labels")
            for part in population.partition_by(segment):
                if set(part["model_id"].unique().to_list()) != set(models):
                    raise ValueError("Simpson comparison coverage differs within a declared segment")
            validate_comparable_records(population, [segment])
            arms = observation_metrics(
                population, ["wmape"], ["model_id", segment], preserve_context=False,
            )
            if not arms["wmape"].drop_nulls().is_finite().all():
                raise ValueError("Simpson comparison requires finite aggregate scores")
            if arms["wmape"].null_count():
                unavailable.append({
                    **context, "segment_column": segment, "reason": "undefined_comparison_arm",
                    "arms": arms.select("model_id", segment, "wmape", "total_volume", "n_observations").to_dicts(),
                })
                continue
            for first, second in combinations(models, 2):
                pair = arms.filter(pl.col("model_id").is_in([first, second]))
                checked = check_simpsons_multi_segment(
                    pair, segment_columns=[segment], value_col="wmape", weight_col="total_volume",
                    period_col="model_id", period_a=first, period_b=second,
                )
                result = checked["results"][segment]
                comparisons.append({
                    **context, "segment_column": segment, "model_a": first, "model_b": second,
                    "result": result,
                })
                if checked["any_paradox"]:
                    any_paradox = True
                    reversals.append(f"{context['group_key']}:{segment}:{first}:{second}")

    evaluated = bool(comparisons) and not unavailable
    status = "evaluated" if evaluated else "partial" if comparisons else (
        "not_evaluable" if any(item["reason"] == "undefined_comparison_arm" for item in unavailable)
        else "not_applicable"
    )
    return {
        "any_paradox": any_paradox,
        "evaluated": evaluated,
        "status": status,
        "severity": "BLOCKER" if any_paradox else "WARNING" if unavailable else "PASS",
        "scoring_policy": SCORING_POLICY,
        "metric": "wmape",
        "comparison_axis": "model_id",
        "comparisons_run": len(comparisons),
        "comparisons": comparisons,
        "results": {f"comparison_{index}": item["result"] for index, item in enumerate(comparisons)},
        "unavailable": unavailable,
        "reversal_segments": reversals,
    }
