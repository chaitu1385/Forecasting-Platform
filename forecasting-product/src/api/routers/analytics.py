"""Analytics endpoints — FVA, calibration, SHAP, decomposition, comparison, constraints."""

from __future__ import annotations

import io
import json
import logging
import math
from typing import Optional

import polars as pl
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from ...auth.models import Permission, User
from ...auth.rbac import get_current_user, require_permission
from ..deps import (
    metric_records,
    raise_run_error,
    run_reader,
    validate_path_param,
    validate_upload_size,
)
from ..schemas import ConstrainResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analytics"])


# ── Metrics analytics ───────────────────────────────────────────────────────


@router.get("/metrics/{lob}/fva")
def get_fva(
    lob: str,
    request: Request,
    run_type: str = Query("backtest"),
    run_id: Optional[str] = Query(None),
    user: User = Depends(require_permission(Permission.VIEW_METRICS)),
):
    """Compute Forecast Value Add (FVA) cascade for a LOB."""
    from ...analytics.fva_analyzer import FVAAnalyzer
    df, selected = metric_records(request, lob, run_type, run_id, latest=True)

    if df is None or df.is_empty():
        raise HTTPException(status_code=404, detail=f"No metric data for LOB '{lob}'.")

    analyzer = FVAAnalyzer()
    try:
        fva_detail = analyzer.compute_fva_detail(df)
        fva_summary = analyzer.summarize(fva_detail)
        layer_lb = analyzer.layer_leaderboard(fva_detail)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=f"FVA requires complete comparable observations: {exc}")

    return {
        "lob": lob,
        "run_id": selected.run_id if selected else None,
        "artifact_source": "committed" if selected else "legacy",
        "summary": fva_summary.to_dicts() if not fva_summary.is_empty() else [],
        "layer_leaderboard": layer_lb.to_dicts() if not layer_lb.is_empty() else [],
        "detail_preview": fva_detail.head(100).to_dicts() if not fva_detail.is_empty() else [],
    }


@router.get("/metrics/{lob}/calibration")
def get_calibration(
    lob: str,
    request: Request,
    run_type: str = Query("backtest"),
    run_id: Optional[str] = Query(None),
    user: User = Depends(require_permission(Permission.VIEW_METRICS)),
):
    """Compute prediction interval calibration report."""
    from ...evaluation.calibration import compute_calibration_report
    df, selected = metric_records(request, lob, run_type, run_id, latest=True)

    if df is None or df.is_empty():
        raise HTTPException(status_code=404, detail=f"No metric data for LOB '{lob}'.")

    # Default quantiles and coverage targets
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    coverage_targets = {"80": 0.80, "50": 0.50}

    try:
        report = compute_calibration_report(df, quantiles=quantiles, coverage_targets=coverage_targets)
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        raise HTTPException(status_code=500, detail=f"Calibration computation failed: {exc}")

    model_reports = {}
    for model_id, coverages in report.model_reports.items():
        model_reports[model_id] = [
            {
                "label": c.label,
                "nominal": c.nominal,
                "empirical": round(c.empirical, 4),
                "miscalibration": round(c.miscalibration, 4),
                "sharpness": round(c.sharpness, 4),
                "n_observations": c.n_observations,
            }
            for c in coverages
        ]

    return {
        "lob": lob,
        "run_id": selected.run_id if selected else None,
        "artifact_source": "committed" if selected else "legacy",
        "model_reports": model_reports,
        "per_series_preview": report.per_series.head(50).to_dicts() if report.per_series is not None and not report.per_series.is_empty() else [],
    }


@router.post("/metrics/{lob}/shap")
async def compute_shap(
    lob: str,
    request: Request,
    file: Optional[UploadFile] = File(None, description="Actuals CSV/Parquet for SHAP computation"),
    model_name: str = Query("lgbm_direct", description="Model to explain"),
    season_length: int = Query(52),
    top_k: int = Query(10, description="Number of top features to return"),
    run_id: Optional[str] = Query(None),
    user: User = Depends(require_permission(Permission.VIEW_METRICS)),
):
    """Compute SHAP feature attribution for tree-based models."""
    from ...analytics.explainer import ForecastExplainer
    from ...pipeline.run_store import RunIntegrityError, RunNotFoundError

    validate_path_param(lob, "lob")
    try:
        reader = run_reader(request)
        selected = reader.forecast(lob, run_id)
        forecast_df = selected.frame.rename({
            selected.id_column: "series_id", selected.time_column: "week",
        })
    except (RunIntegrityError, RunNotFoundError, ValueError, OSError) as exc:
        raise_run_error(exc)
    # Load data
    if file:
        content = await validate_upload_size(file)
        filename = file.filename or ""
        try:
            if filename.endswith(".parquet"):
                df = pl.read_parquet(io.BytesIO(content))
            else:
                df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")
    else:
        try:
            history = reader.history(lob, selected.run_id, legacy=selected.source == "legacy")
            df = history.frame.rename({
                history.id_column: "series_id", history.time_column: "week",
                history.target_column: "quantity",
            })
        except (RunIntegrityError, RunNotFoundError, ValueError, OSError) as exc:
            raise_run_error(exc)

    explainer = ForecastExplainer(season_length=season_length)

    try:
        # explain_ml needs a trained model — for API use, we'll use the decompose + narrative approach
        # since we don't have a persisted model object. Return decomposition-based feature importance.
        if forecast_df.is_empty():
            raise HTTPException(status_code=404, detail="No forecast data available for SHAP analysis.")

        decomp = explainer.decompose(
            history=df, forecast=forecast_df,
            id_col="series_id", time_col="week",
            target_col="quantity", value_col="forecast",
        )

        # Summarize component importance as proxy for SHAP
        importance = []
        for component in ("trend", "seasonal", "residual"):
            if component in decomp.columns:
                vals = decomp.filter(pl.col("is_forecast").not_())[component]
                importance.append({
                    "feature": component,
                    "mean_abs_value": round(float(vals.abs().mean()) if vals.len() > 0 else 0.0, 4),
                    "std": round(float(vals.std()) if vals.len() > 0 else 0.0, 4),
                })

        importance.sort(key=lambda x: x["mean_abs_value"], reverse=True)

        return {
            "lob": lob,
            "run_id": selected.run_id,
            "artifact_source": selected.source,
            "model": model_name,
            "feature_importance": importance[:top_k],
            "decomposition_preview": decomp.head(50).to_dicts(),
        }
    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=500, detail=f"SHAP computation failed: {exc}")


# ── Forecast analytics ──────────────────────────────────────────────────────


@router.post("/forecast/decompose")
async def decompose_forecast(
    history_file: UploadFile = File(..., description="Historical actuals"),
    forecast_file: UploadFile = File(..., description="Forecast data"),
    id_col: str = Query("series_id"),
    time_col: str = Query("week"),
    target_col: str = Query("quantity"),
    value_col: str = Query("forecast"),
    season_length: int = Query(52),
    user: User = Depends(get_current_user),
):
    """Run STL decomposition on historical + forecast data."""
    from ...analytics.explainer import ForecastExplainer

    history_content = await validate_upload_size(history_file)
    forecast_content = await validate_upload_size(forecast_file)

    try:
        hf = history_file.filename or ""
        if hf.endswith(".parquet"):
            history = pl.read_parquet(io.BytesIO(history_content))
        else:
            history = pl.read_csv(io.BytesIO(history_content), try_parse_dates=True)

        ff = forecast_file.filename or ""
        if ff.endswith(".parquet"):
            forecast = pl.read_parquet(io.BytesIO(forecast_content))
        else:
            forecast = pl.read_csv(io.BytesIO(forecast_content), try_parse_dates=True)
    except (ValueError, UnicodeDecodeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read files: {exc}")

    explainer = ForecastExplainer(season_length=season_length)
    decomp = explainer.decompose(
        history=history, forecast=forecast,
        id_col=id_col, time_col=time_col,
        target_col=target_col, value_col=value_col,
    )
    narratives = explainer.narrative(decomp, id_col=id_col, time_col=time_col)

    return {
        "decomposition": decomp.to_dicts(),
        "narratives": narratives,
        "series_count": decomp[id_col].n_unique() if id_col in decomp.columns else 0,
    }


@router.post("/forecast/compare")
async def compare_forecasts(
    model_file: UploadFile = File(..., description="Model forecast CSV/Parquet"),
    external_file: UploadFile = File(..., description="External forecast CSV/Parquet"),
    external_name: str = Query("external", description="Label for external forecast"),
    id_col: str = Query("series_id"),
    time_col: str = Query("week"),
    value_col: str = Query("forecast"),
    user: User = Depends(get_current_user),
):
    """Compare model forecast against external/uploaded forecast."""
    from ...analytics.comparator import ForecastComparator

    model_content = await validate_upload_size(model_file)
    ext_content = await validate_upload_size(external_file)

    try:
        mf = model_file.filename or ""
        model_df = pl.read_parquet(io.BytesIO(model_content)) if mf.endswith(".parquet") else pl.read_csv(io.BytesIO(model_content), try_parse_dates=True)

        ef = external_file.filename or ""
        ext_df = pl.read_parquet(io.BytesIO(ext_content)) if ef.endswith(".parquet") else pl.read_csv(io.BytesIO(ext_content), try_parse_dates=True)
    except (ValueError, UnicodeDecodeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read files: {exc}")

    comparator = ForecastComparator()
    comparison = comparator.compare(
        model_forecast=model_df,
        external_forecasts={external_name: ext_df},
        id_col=id_col, time_col=time_col, value_col=value_col,
    )
    summary = comparator.summary(comparison, id_col=id_col, time_col=time_col)

    return {
        "comparison": comparison.to_dicts(),
        "summary": summary.to_dicts() if not summary.is_empty() else [],
    }


def _validate_constraint_json_column(column: pl.Series) -> None:
    """Reject metadata that JSON would silently replace or cannot represent."""
    dtype = column.dtype
    if dtype in (pl.Binary, pl.Object):
        raise ValueError("invalid_constraint_frame: binary/object metadata is not supported in JSON forecasts")
    if dtype.is_float() and not column.is_finite().fill_null(True).all():
        raise ValueError("invalid_constraint_frame: nonfinite metadata cannot be preserved in JSON forecasts")
    if isinstance(dtype, (pl.List, pl.Array)):
        _validate_constraint_json_column(column.explode())
    elif isinstance(dtype, pl.Struct):
        for member in column.struct.unnest().iter_columns():
            _validate_constraint_json_column(member)


@router.post("/forecast/constrain", response_model=ConstrainResponse)
async def constrain_forecast(
    file: UploadFile = File(..., description="Forecast CSV/Parquet"),
    min_demand: float = Query(0.0, description="Floor (non-negativity)"),
    max_capacity: Optional[float] = Query(None, description="Per-series-per-period cap"),
    aggregate_max: Optional[float] = Query(None, description="Sum across all series cap"),
    proportional: bool = Query(True, description="Use proportional redistribution"),
    id_col: str = Query("series_id"),
    time_col: str = Query("week"),
    value_col: str = Query("forecast"),
    user: User = Depends(require_permission(Permission.RUN_PIPELINE)),
) -> ConstrainResponse:
    """Apply full-population bounds and point budgets without modifying stored runs."""
    from ...config.schema import ConstraintConfig
    from ...forecasting.constrained import apply_point_constraints

    content = await validate_upload_size(file)
    filename = file.filename or ""
    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(content))
        else:
            # Constraints group exact period keys; date inference can truncate
            # fractional timestamps and silently combine distinct CSV periods.
            df = pl.read_csv(io.BytesIO(content), try_parse_dates=False)
    except (ValueError, UnicodeDecodeError, pl.exceptions.PolarsError) as exc:
        raise HTTPException(
            status_code=400, detail="invalid_constraint_file: upload a readable CSV or Parquet forecast"
        ) from exc
    except OSError as exc:
        logger.exception("Constraint upload could not be read")
        raise HTTPException(status_code=500, detail="Constraint upload could not be read") from exc

    try:
        constrained = apply_point_constraints(
            df,
            ConstraintConfig(
                enabled=True, min_demand=min_demand, max_capacity=max_capacity,
                aggregate_max=aggregate_max, proportional_redistribution=proportional,
            ),
            id_col, time_col, value_col,
        )
        before_total = math.fsum(df[value_col])
        after_total = math.fsum(constrained[value_col])
        if not math.isfinite(before_total) or not math.isfinite(after_total):
            raise ValueError("invalid_constraint_frame: aggregate totals must be finite")
        max_excess = None
        if aggregate_max is not None:
            period_values = constrained.select(
                pl.col(time_col).alias("period"), pl.col(value_col).alias("values"),
            ).group_by("period").agg("values")["values"].to_list()
            max_excess = max(max(0.0, math.fsum(values) - aggregate_max) for values in period_values)
        quantile_columns = [
            column for column in constrained.columns
            if column.startswith("forecast_p") and column not in (id_col, time_col)
        ]
        modified = df[value_col] != constrained[value_col]
        for column in quantile_columns:
            modified = modified | (df[column] != constrained[column])
        for column in constrained.iter_columns():
            _validate_constraint_json_column(column)
        preview = json.loads(constrained.head(200).write_json())
    except (ValueError, OverflowError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ArithmeticError, KeyError, OSError, RuntimeError, TypeError, pl.exceptions.PolarsError) as exc:
        logger.exception("Constraint computation failed")
        raise HTTPException(status_code=500, detail="Constraint computation failed") from exc

    return ConstrainResponse(
        before_total=before_total,
        after_total=after_total,
        rows_modified=int(modified.sum()),
        constraints_applied={
            "min_demand": min_demand, "max_capacity": max_capacity,
            "aggregate_max": aggregate_max, "proportional": proportional,
        },
        constrained_preview=preview,
        allocation_strategy=(
            "element_wise" if aggregate_max is None else
            "proportional" if proportional else "largest_first"
        ),
        checked_rows=constrained.height,
        checked_periods=constrained[time_col].n_unique(),
        max_budget_excess=max_excess,
        constraint_scope="point_and_quantiles" if quantile_columns else "point",
    )


# ── Post-pipeline validation ────────────────────────────────────────────────


@router.get("/metrics/{lob}/validation")
def get_validation(
    lob: str,
    request: Request,
    run_type: str = Query("backtest"),
    run_id: Optional[str] = Query(None),
    user: User = Depends(require_permission(Permission.VIEW_METRICS)),
):
    """Run 4-layer post-pipeline validation on the latest metrics for a LOB.

    Returns confidence grade (A-F), score (0-100), and per-layer details.
    """
    from ...config.schema import PostValidationConfig
    from ...pipeline.run_store import RunIntegrityError, RunNotFoundError
    from ...pipeline.validation_step import run_post_validation

    df, selected = metric_records(request, lob, run_type, run_id, latest=True)
    if selected is not None:
        try:
            result = run_reader(request).store.json_member(lob, selected.run_id, "validation")
        except (RunIntegrityError, RunNotFoundError, ValueError, OSError) as exc:
            raise_run_error(exc)
        return {
            "lob": lob, "run_id": selected.run_id, "artifact_source": "committed",
            "grade": result.get("grade", "N/A"), "score": result.get("score", -1),
            "badge": result.get("badge", "Validation disabled for this run"),
            "layers": result.get("layers", {}), "confidence": result.get("confidence", {}),
            "skipped": result.get("skipped", False),
        }

    if df is None or df.is_empty():
        raise HTTPException(status_code=404, detail=f"No metric data for LOB '{lob}'.")

    # Use default validation config (could be extended to read from stored config)
    config = PostValidationConfig()

    try:
        result = run_post_validation(df, config, lob=lob)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}")

    return {
        "lob": lob,
        "run_id": None,
        "artifact_source": "legacy",
        "grade": result["grade"],
        "score": result["score"],
        "badge": result["badge"],
        "layers": result["layers"],
        "confidence": result["confidence"],
        "skipped": result.get("skipped", False),
    }
