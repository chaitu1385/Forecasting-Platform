"""Series analysis endpoints — SBC classification, breaks, cleansing, regressor screening."""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import Literal, Optional

import polars as pl
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from ...auth.models import User
from ...auth.rbac import get_current_user
from ..deps import raise_run_error, run_reader, validate_upload_size

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/series", tags=["series"])


def _load_actuals(data_dir, lob: str, runs_dir=None, run_id: Optional[str] = None) -> pl.DataFrame:
    """Compatibility entrypoint delegating to the authoritative run/history reader."""
    from ...pipeline.readers import RunReader
    from ...pipeline.run_store import RunIntegrityError, RunNotFoundError

    data = Path(data_dir)
    try:
        return RunReader(
            runs_dir if runs_dir is not None else data / "runs",
            data / "forecasts", (data / "history", data / "actuals"),
        ).history(lob, run_id).frame
    except (RunIntegrityError, RunNotFoundError, ValueError, OSError) as exc:
        raise_run_error(exc)


@router.get("/{lob}")
def list_series(
    lob: str,
    request: Request,
    run_id: Optional[str] = Query(None),
    stage: Literal["input", "prepared"] = Query("input"),
    legacy: bool = Query(False),
    user: User = Depends(get_current_user),
):
    """List all series with SBC demand classification (ADI, CV², demand_class)."""
    from ...pipeline.run_store import RunIntegrityError, RunNotFoundError
    from ...series.sparse_detector import SparseDetector
    try:
        selected = run_reader(request).history(lob, run_id, stage=stage, legacy=legacy)
    except (RunIntegrityError, RunNotFoundError, ValueError, OSError) as exc:
        raise_run_error(exc)
    df = selected.frame

    detector = SparseDetector()
    target_col, id_col = selected.target_column, selected.id_column

    classified = detector.classify(df, target_col=target_col, id_col=id_col)

    items = []
    for row in classified.iter_rows(named=True):
        n_obs = df.filter(pl.col(id_col) == row[id_col]).height
        items.append({
            "series_id": str(row[id_col]),
            "adi": round(float(row.get("adi", 0)), 4),
            "cv2": round(float(row.get("cv2", 0)), 4),
            "demand_class": row.get("demand_class", "unknown"),
            "is_sparse": bool(row.get("is_sparse", False)),
            "n_observations": n_obs,
        })

    return {
        "lob": lob, "series_count": len(items), "series": items,
        "run_id": selected.run_id, "artifact_source": selected.source, "input_stage": selected.input_stage,
    }


@router.get("/{lob}/{series_id}/history")
def get_series_history(
    lob: str,
    series_id: str,
    request: Request,
    run_id: Optional[str] = Query(None),
    stage: Literal["input", "prepared"] = Query("input"),
    legacy: bool = Query(False),
    user: User = Depends(get_current_user),
):
    """Return exact-run history, with raw/prepared or unknown legacy stage labelled."""
    from ...pipeline.run_store import RunIntegrityError, RunNotFoundError
    try:
        selected = run_reader(request).history(lob, run_id, stage=stage, legacy=legacy)
    except (RunIntegrityError, RunNotFoundError, ValueError, OSError) as exc:
        raise_run_error(exc)
    df = selected.frame

    id_col, time_col, target_col = selected.id_column, selected.time_column, selected.target_column
    series_df = df.filter(pl.col(id_col).cast(pl.Utf8) == series_id).sort(time_col)

    if series_df.is_empty():
        raise HTTPException(status_code=404, detail=f"Series '{series_id}' not found in LOB '{lob}'.")

    points = [
        {"week": str(row[time_col]),
         "value": float(row[target_col]) if row[target_col] is not None and math.isfinite(float(row[target_col])) else None}
        for row in series_df.iter_rows(named=True)
    ]

    return {
        "series_id": series_id, "lob": lob, "points": points,
        "run_id": selected.run_id, "artifact_source": selected.source, "input_stage": selected.input_stage,
        "input_data_hash": selected.manifest.input_data_hash if selected.manifest else None,
    }


@router.post("/breaks")
async def detect_breaks(
    request: Request,
    file: Optional[UploadFile] = File(None),
    lob: Optional[str] = Query(None, description="LOB to load actuals from data_dir"),
    method: str = Query("cusum", description="Detection method: 'cusum' or 'pelt'"),
    penalty: float = Query(3.0, description="PELT penalty (higher = fewer breaks)"),
    min_segment_length: int = Query(13, description="Min periods between breaks"),
    max_breakpoints: int = Query(5),
    user: User = Depends(get_current_user),
):
    """Detect structural breaks in time series data."""
    from ...config.schema import StructuralBreakConfig
    from ...series.break_detector import StructuralBreakDetector

    df = await _read_upload_or_lob(file, request, lob)

    config = StructuralBreakConfig(
        enabled=True,
        method=method,
        penalty=penalty,
        min_segment_length=min_segment_length,
        max_breakpoints=max_breakpoints,
    )
    detector = StructuralBreakDetector(config)

    id_col, time_col, target_col = _detect_columns(df)
    report = detector.detect(df, target_col=target_col, time_col=time_col, id_col=id_col)

    per_series = []
    if report.per_series is not None and not report.per_series.is_empty():
        per_series = report.per_series.to_dicts()

    return {
        "total_series": report.total_series,
        "series_with_breaks": report.series_with_breaks,
        "total_breaks": report.total_breaks,
        "warnings": report.warnings,
        "per_series": per_series,
    }


@router.post("/cleansing-audit")
async def cleansing_audit(
    request: Request,
    file: Optional[UploadFile] = File(None),
    lob: Optional[str] = Query(None),
    frequency: Literal["D", "W", "M", "Q"] = Query("W", description="Input period frequency; weekly legacy default"),
    outlier_method: str = Query("iqr", description="'iqr' or 'zscore'"),
    iqr_multiplier: float = Query(1.5),
    zscore_threshold: float = Query(3.0),
    outlier_action: str = Query("clip", description="'clip', 'interpolate', or 'flag_only'"),
    stockout_detection: bool = Query(True),
    min_zero_run: int = Query(2),
    user: User = Depends(get_current_user),
):
    """Run demand cleansing and return before/after audit report."""
    from ...config.schema import CleansingConfig
    from ...data.cleanser import DemandCleanser

    df = await _read_upload_or_lob(file, request, lob)

    config = CleansingConfig(
        enabled=True,
        outlier_method=outlier_method,
        iqr_multiplier=iqr_multiplier,
        zscore_threshold=zscore_threshold,
        outlier_action=outlier_action,
        stockout_detection=stockout_detection,
        min_zero_run=min_zero_run,
    )
    cleanser = DemandCleanser(config, frequency=frequency)
    id_col, time_col, target_col = _detect_columns(df)

    result = cleanser.cleanse(df, time_col=time_col, value_col=target_col, sid_col=id_col)

    report = result.report
    per_series_dicts = []
    if report.per_series is not None and not report.per_series.is_empty():
        per_series_dicts = report.per_series.to_dicts()

    return {
        "frequency": frequency,
        "total_series": report.total_series,
        "series_with_outliers": report.series_with_outliers,
        "total_outliers": report.total_outliers,
        "outlier_pct": round(report.outlier_pct, 4),
        "series_with_stockouts": report.series_with_stockouts,
        "total_stockout_periods": report.total_stockout_periods,
        "total_stockout_weeks": report.total_stockout_weeks,
        "excluded_period_weeks": report.excluded_period_weeks,
        "rows_modified": report.rows_modified,
        "per_series": per_series_dicts,
        "cleansed_preview": result.df.head(200).to_dicts(),
    }


@router.post("/regressor-screen")
async def regressor_screen(
    request: Request,
    file: Optional[UploadFile] = File(None),
    lob: Optional[str] = Query(None),
    feature_columns: Optional[str] = Query(None, description="Comma-separated feature column names"),
    target_col: str = Query("quantity"),
    variance_threshold: float = Query(1e-6),
    correlation_threshold: float = Query(0.95),
    mi_enabled: bool = Query(False),
    user: User = Depends(get_current_user),
):
    """Screen regressors for variance, correlation, and mutual information."""
    from ...config.schema import RegressorScreenConfig
    from ...data.regressor_screen import screen_regressors

    df = await _read_upload_or_lob(file, request, lob)

    # Auto-detect feature columns if not provided
    if feature_columns:
        features = [c.strip() for c in feature_columns.split(",")]
    else:
        # Exclude known non-feature columns
        skip = {"series_id", "sku_id", "product_id", "item_id", "week", "date",
                "month", "quarter", target_col}
        features = [c for c in df.columns if c not in skip and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]

    if not features:
        return {
            "screened_columns": [],
            "dropped_columns": [],
            "low_variance_columns": [],
            "high_correlation_pairs": [],
            "low_mi_columns": [],
            "warnings": ["No numeric feature columns found to screen."],
            "per_column_stats": {},
        }

    config = RegressorScreenConfig(
        enabled=True,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        mi_enabled=mi_enabled,
    )
    report = screen_regressors(df, feature_columns=features, target_col=target_col, config=config)

    return {
        "screened_columns": report.screened_columns,
        "dropped_columns": report.dropped_columns,
        "low_variance_columns": report.low_variance_columns,
        "high_correlation_pairs": report.high_correlation_pairs,
        "low_mi_columns": report.low_mi_columns,
        "warnings": report.warnings,
        "per_column_stats": report.per_column_stats,
    }


# ── Helpers ─────────────────────────────────────────────────────────────────


async def _read_upload_or_lob(file, request: Request, lob):
    """Read data from upload or from stored LOB actuals."""
    if file is not None:
        content = await validate_upload_size(file)
        filename = file.filename or ""
        try:
            if filename.endswith(".parquet"):
                return pl.read_parquet(io.BytesIO(content))
            return pl.read_csv(io.BytesIO(content), try_parse_dates=True)
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")
    elif lob:
        return _load_actuals(request.app.state.data_dir, lob, request.app.state.runs_dir)
    else:
        raise HTTPException(status_code=400, detail="Provide either a file upload or lob query parameter.")


def _detect_columns(df: pl.DataFrame):
    """Auto-detect id, time, and target columns."""
    id_col = "series_id"
    for c in ("series_id", "sku_id", "product_id", "item_id"):
        if c in df.columns:
            id_col = c
            break

    time_col = "week"
    for c in ("week", "date", "ds", "timestamp", "month", "quarter"):
        if c in df.columns:
            time_col = c
            break

    target_col = "quantity"
    for c in ("quantity", "target", "value", "sales", "demand", "y"):
        if c in df.columns:
            target_col = c
            break

    return id_col, time_col, target_col
