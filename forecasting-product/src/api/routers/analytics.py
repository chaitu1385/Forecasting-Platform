"""Analytics endpoints — FVA, calibration, decomposition, comparison, constraints."""

from __future__ import annotations

import io
import logging
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
from ..schemas import DecompositionCompatibilityResponse, DecompositionSummaryResponse

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


@router.post("/metrics/{lob}/decomposition", response_model=DecompositionSummaryResponse)
@router.post("/metrics/{lob}/shap", response_model=DecompositionCompatibilityResponse, deprecated=True)
async def compute_decomposition(
    lob: str,
    request: Request,
    file: Optional[UploadFile] = File(None, description="Optional history override; not the committed prepared snapshot"),
    model_name: Optional[str] = Query(None, description="Ignored compatibility parameter; never model provenance", deprecated=True),
    season_length: int = Query(52),
    top_k: int = Query(10, description="Maximum component magnitudes in the deprecated feature_importance field"),
    run_id: Optional[str] = Query(None),
    legacy: bool = Query(False, description="Explicitly read unverified legacy forecasts; incompatible with run_id"),
    id_col: str = Query("series_id", description="Uploaded history identifier column"),
    time_col: str = Query("week", description="Uploaded history time column"),
    target_col: str = Query("quantity", description="Uploaded history target column"),
    user: User = Depends(require_permission(Permission.VIEW_METRICS)),
):
    """Describe moving-average components, not fitted-model feature attribution.

    Exact forecast IDs never fall back to another run. Without an upload, a
    committed forecast uses its own prepared history. The deprecated alias uses
    the identical computation; its old field names do not represent SHAP.
    """
    from ...pipeline.readers import ArtifactRead
    from ...pipeline.run_store import RunIntegrityError, RunNotFoundError
    from ..decomposition import (
        DecompositionInputError,
        UnsupportedDecompositionWindow,
        decompose_artifacts,
        read_decomposition_upload,
    )

    validate_path_param(lob, "lob")
    if top_k < 1 or season_length < 1:
        raise HTTPException(400, detail="season_length and top_k must be positive integers")
    try:
        reader = run_reader(request)
        try:
            selected = reader.forecast(lob, run_id, legacy=legacy)
        except (RunIntegrityError, RunNotFoundError, ValueError, OSError) as exc:
            if not isinstance(exc, (RunNotFoundError, ValueError)):
                logger.exception("Forecast artifact read failed")
            raise_run_error(exc)
        if file:
            content = await validate_upload_size(file)
            frame = read_decomposition_upload(
                content, file.filename or "", target_col, identifier_column=id_col,
                identifier_dtype=selected.frame.schema[selected.id_column],
            )
            history = ArtifactRead(
                frame, None, "uploaded", id_col, time_col, target_col, input_stage="uploaded",
            )
        else:
            try:
                history = reader.history(
                    lob, selected.run_id, stage="prepared", legacy=selected.source == "legacy",
                )
            except (RunIntegrityError, RunNotFoundError, ValueError, OSError) as exc:
                if not isinstance(exc, (RunNotFoundError, ValueError)):
                    logger.exception("Forecast history read failed")
                raise_run_error(exc)
        summary = decompose_artifacts(lob, selected, history, season_length)
        ranked = sorted(summary.components, key=lambda component: component.mean_abs_value, reverse=True)
        return {
            **summary.model_dump(),
            "model": summary.forecast_model,
            "feature_importance": [
                {"feature": component.component, **component.model_dump(exclude={"component"})}
                for component in ranked[:top_k]
            ],
        }
    except HTTPException:
        raise
    except UnsupportedDecompositionWindow as exc:
        raise HTTPException(400, detail={
            "code": "unsupported_decomposition_window", "message": str(exc),
        }) from exc
    except DecompositionInputError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Forecast decomposition failed")
        raise HTTPException(500, detail="Forecast decomposition did not produce a complete finite result") from None


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
    """Run centered-moving-average decomposition on historical + forecast data."""
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


@router.post("/forecast/constrain")
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
):
    """Apply capacity and budget constraints to forecast."""
    content = await validate_upload_size(file)
    filename = file.filename or ""
    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(content))
        else:
            df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
    except (ValueError, UnicodeDecodeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    constrained = df.clone()

    # Apply min demand floor
    if value_col in constrained.columns:
        constrained = constrained.with_columns(
            pl.when(pl.col(value_col) < min_demand)
            .then(min_demand)
            .otherwise(pl.col(value_col))
            .alias(value_col)
        )

    # Apply per-series capacity cap
    if max_capacity is not None and value_col in constrained.columns:
        constrained = constrained.with_columns(
            pl.when(pl.col(value_col) > max_capacity)
            .then(max_capacity)
            .otherwise(pl.col(value_col))
            .alias(value_col)
        )

    # Apply aggregate cap
    if aggregate_max is not None and time_col in constrained.columns and value_col in constrained.columns:
        period_totals = constrained.group_by(time_col).agg(pl.col(value_col).sum().alias("_total"))
        over_cap = period_totals.filter(pl.col("_total") > aggregate_max)

        if not over_cap.is_empty():
            for row in over_cap.iter_rows(named=True):
                period = row[time_col]
                total = row["_total"]
                scale = aggregate_max / total if total > 0 else 1.0
                if proportional:
                    constrained = constrained.with_columns(
                        pl.when(pl.col(time_col) == period)
                        .then(pl.col(value_col) * scale)
                        .otherwise(pl.col(value_col))
                        .alias(value_col)
                    )

    before_total = float(df[value_col].sum()) if value_col in df.columns else 0.0
    after_total = float(constrained[value_col].sum()) if value_col in constrained.columns else 0.0
    rows_modified = 0
    if value_col in df.columns and value_col in constrained.columns:
        rows_modified = int((df[value_col] != constrained[value_col]).sum())

    return {
        "before_total": round(before_total, 2),
        "after_total": round(after_total, 2),
        "rows_modified": rows_modified,
        "constraints_applied": {
            "min_demand": min_demand,
            "max_capacity": max_capacity,
            "aggregate_max": aggregate_max,
            "proportional": proportional,
        },
        "constrained_preview": constrained.head(200).to_dicts(),
    }


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
