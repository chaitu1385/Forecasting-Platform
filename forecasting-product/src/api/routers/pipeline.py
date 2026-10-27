"""Pipeline operation endpoints — run backtest/forecast, manifests, costs."""

from __future__ import annotations

import inspect
import io
import json
import logging
from typing import Literal, Optional

import polars as pl
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from ...auth.models import Permission, User
from ...auth.rbac import require_permission
from ...config.loader import parse_config_yaml
from ...config.schema import PlatformConfig
from ...forecasting.registry import registry
from ...metrics.definitions import CONTEXT_METRICS, METRIC_REGISTRY
from ..deps import validate_path_param, validate_upload_size
from ..schemas import EffectivePipelineConfig, PipelineExecutionSettings, PipelineRunResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _can_run_standalone(model_id: Optional[str], frequency: str) -> bool:
    """Check constructor arguments without building/downloading a model."""
    if model_id not in registry.available:
        return False
    signature = inspect.signature(registry.get(model_id))
    kwargs = {"frequency": frequency} if "frequency" in signature.parameters else {}
    try:
        signature.bind(**kwargs)
    except TypeError:
        return False
    return True


async def _load_run_config(
    request: Request, config_file: Optional[UploadFile], lob: str,
) -> PlatformConfig:
    """Resolve and validate a plan before constructing a pipeline or its stores."""
    if config_file is None:
        config = PlatformConfig(lob=lob)
    else:
        content = await validate_upload_size(config_file)
        try:
            config = parse_config_yaml(content.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise HTTPException(400, detail="Invalid config: YAML must be UTF-8") from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, detail=f"Invalid config: {exc}") from exc

    fc = config.forecast
    if not fc.forecasters:
        raise HTTPException(400, detail="Invalid config: forecast.forecasters must not be empty")
    for field in ("forecasters", "intermittent_forecasters"):
        for model in getattr(fc, field):
            if model not in registry.available:
                raise HTTPException(400, detail=f"Invalid config: forecast.{field} has unknown model {model!r}")
            if not _can_run_standalone(model, fc.frequency):
                raise HTTPException(
                    400,
                    detail=f"Invalid config: model {model!r} requires configured components; choose a standalone forecaster",
                )
    if any(not 0 < q < 1 for q in fc.quantiles):
        raise HTTPException(400, detail="Invalid config: forecast.quantiles must be between 0 and 1")
    columns = (fc.series_id_column, fc.time_column, fc.target_column)
    if any(not col.strip() for col in columns) or len(set(columns)) != 3:
        raise HTTPException(400, detail="Invalid config: forecast columns must be non-empty and distinct")
    if config.post_validation.min_grade not in {"A", "B", "C", "D", "F"}:
        raise HTTPException(400, detail="Invalid config: post_validation.min_grade must be A, B, C, D or F")
    bt = config.backtest
    available_metrics = METRIC_REGISTRY.keys() | CONTEXT_METRICS.keys()
    for field, metrics in (
        ("metrics", config.metrics),
        ("backtest.primary_metric", [bt.primary_metric]),
        ("backtest.secondary_metric", [bt.secondary_metric]),
    ):
        for metric in metrics:
            if metric not in available_metrics:
                raise HTTPException(400, detail=f"Invalid config: {field} has unknown metric {metric!r}")
    if bt.primary_metric == bt.secondary_metric:
        raise HTTPException(400, detail="Invalid config: primary_metric and secondary_metric must differ")
    if not {bt.primary_metric, bt.secondary_metric}.issubset(config.metrics):
        raise HTTPException(
            400, detail="Invalid config: metrics must include the backtest primary_metric and secondary_metric",
        )

    config.lob = lob
    config.output.metrics_path = str(request.app.state.data_dir / "metrics")
    config.output.forecast_path = str(request.app.state.data_dir / "forecasts")
    return config


def _execution_settings(
    config: PlatformConfig,
    config_file: Optional[UploadFile],
    model_id: Optional[str],
    model_source: Literal["request", "config", "default", "backtest"],
    horizon_overridden: bool = False,
) -> PipelineExecutionSettings:
    fc, bt, dq, pv = config.forecast, config.backtest, config.data_quality, config.post_validation
    return PipelineExecutionSettings(
        model_id=model_id,
        model_source=model_source,
        horizon_periods=fc.horizon_periods,
        horizon_source="request" if horizon_overridden else ("config" if config_file else "default"),
        config_source="upload" if config_file else "default",
        config=EffectivePipelineConfig(
            frequency=fc.frequency,
            season_length=fc.season_length,
            series_id_column=fc.series_id_column,
            time_column=fc.time_column,
            target_column=fc.target_column,
            forecasters=fc.forecasters,
            intermittent_forecasters=fc.intermittent_forecasters,
            quantiles=fc.quantiles,
            fill_gaps=dq.fill_gaps,
            fill_value=dq.fill_value,
            min_series_length_periods=dq.effective_min_series_length(fc.frequency),
            drop_zero_series=dq.drop_zero_series,
            input_validation_enabled=dq.validation.enabled,
            cleansing_enabled=dq.cleansing.enabled,
            constraints_enabled=fc.constraints.enabled,
            post_validation_enabled=pv.enabled,
            halt_on_blocker=pv.halt_on_blocker,
            min_grade=pv.min_grade,
            backtest_n_folds=bt.n_folds,
            backtest_val_periods=bt.val_periods,
            backtest_gap_periods=bt.gap_periods,
            selection_strategy=bt.selection_strategy,
            champion_granularity=bt.champion_granularity,
            primary_metric=bt.primary_metric,
            secondary_metric=bt.secondary_metric,
        ),
    )


@router.post("/backtest", response_model=PipelineRunResponse, response_model_exclude_unset=True)
async def run_backtest(
    request: Request,
    file: UploadFile = File(..., description="Actuals CSV/Parquet"),
    config_file: Optional[UploadFile] = File(None, description="YAML config file"),
    lob: str = Query("default", description="LOB name"),
    user: User = Depends(require_permission(Permission.RUN_BACKTEST)),
):
    """Run the backtest pipeline on uploaded data."""
    from ...pipeline.backtest import BacktestPipeline

    validate_path_param(lob, "lob")
    config = await _load_run_config(request, config_file, lob)
    content = await validate_upload_size(file)
    filename = file.filename or ""
    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(content))
        else:
            df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
    except (ValueError, UnicodeDecodeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    try:
        pipeline = BacktestPipeline(config=config)
        results = pipeline.run(actuals=df)
    except (ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")

    # Summarize results
    summary = {
        "lob": lob,
        "status": "completed",
    }
    selected_model = None

    if isinstance(results, dict):
        # Pipeline returns a dict with: leaderboard, champions, validation, etc.
        leaderboard = results.get("leaderboard")
        if leaderboard is not None and hasattr(leaderboard, "to_dicts") and not leaderboard.is_empty():
            # Sanitize inf/nan values for JSON serialization
            import math
            rows = leaderboard.to_dicts()
            for row in rows:
                for k, v in row.items():
                    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                        row[k] = None
            summary["leaderboard"] = rows
            # Best WMAPE from first leaderboard entry (sorted ascending).
            # Column can be entirely null when WMAPE is undefined for every
            # model (e.g. all validation windows had zero actuals).
            if "group_key" in leaderboard.columns and leaderboard["group_key"].n_unique() > 1:
                # Separate series/cohort competitions have no single global
                # best score; the per-group leaderboard remains authoritative.
                summary["best_wmape"] = None
            elif "wmape" in leaderboard.columns:
                min_val = leaderboard["wmape"].min()
                if min_val is None:
                    summary["best_wmape"] = None
                else:
                    best = float(min_val)
                    summary["best_wmape"] = best if math.isfinite(best) else None

        champions = results.get("champions")
        if champions is not None and hasattr(champions, "to_dicts") and not champions.is_empty():
            # A multi-model plan cannot be represented by an arbitrary first row.
            if "model_id" in champions.columns and champions["model_id"].n_unique() == 1:
                selected_model = champions["model_id"][0]

        val = results.get("validation")
        if val and not val.get("skipped"):
            summary["validation_grade"] = val.get("grade")
            summary["validation_score"] = val.get("score")
    else:
        # Legacy: object-style return
        if hasattr(results, "leaderboard") and results.leaderboard is not None:
            summary["leaderboard"] = results.leaderboard.to_dicts()
        if hasattr(results, "champion_model_id"):
            selected_model = results.champion_model_id
        if hasattr(results, "wmape"):
            summary["best_wmape"] = float(results.wmape) if results.wmape else None

    if _can_run_standalone(selected_model, config.forecast.frequency):
        summary["champion_model"] = selected_model
    summary["execution"] = _execution_settings(config, config_file, selected_model, "backtest")
    return PipelineRunResponse(**summary)


@router.post("/forecast", response_model=PipelineRunResponse, response_model_exclude_unset=True)
async def run_forecast(
    request: Request,
    file: UploadFile = File(..., description="Actuals CSV/Parquet"),
    config_file: Optional[UploadFile] = File(None, description="YAML config file"),
    lob: str = Query("default", description="LOB name"),
    model_id: Optional[str] = Query(None, description="Specific model to use"),
    horizon: Optional[int] = Query(None, ge=1, description="Override the configured horizon in periods"),
    user: User = Depends(require_permission(Permission.RUN_PIPELINE)),
):
    """Run the forecast pipeline on uploaded data."""
    from ...pipeline.forecast import ForecastPipeline

    validate_path_param(lob, "lob")
    config = await _load_run_config(request, config_file, lob)
    effective_model = model_id if model_id is not None else config.forecast.forecasters[0]
    if effective_model not in registry.available:
        raise HTTPException(400, detail=f"Unknown model_id: {effective_model!r}")
    if not _can_run_standalone(effective_model, config.forecast.frequency):
        raise HTTPException(
            400,
            detail=f"Model {effective_model!r} requires configured components and cannot be selected by this endpoint; choose a standalone model",
        )
    if horizon is not None:
        config.forecast.horizon_weeks = horizon
    execution = _execution_settings(
        config,
        config_file,
        effective_model,
        "request" if model_id is not None else ("config" if config_file else "default"),
        horizon_overridden=horizon is not None,
    )
    content = await validate_upload_size(file)
    filename = file.filename or ""
    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(content))
        else:
            df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
    except (ValueError, UnicodeDecodeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    try:
        pipeline = ForecastPipeline(config=config)
        result = pipeline.run(
            actuals=df,
            champion_model=effective_model,
        )
    except (ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"Forecast pipeline failed: {exc}")

    summary = {
        "lob": lob,
        "status": "completed",
        "model_id": effective_model,
        "horizon": config.forecast.horizon_periods,
        "execution": execution,
    }

    # result is a pl.DataFrame (or dict with "forecast" key for legacy callers)
    forecast_df = None
    if isinstance(result, pl.DataFrame):
        forecast_df = result
    elif isinstance(result, dict) and "forecast" in result:
        forecast_df = result["forecast"]
    elif hasattr(result, "forecast"):
        forecast_df = result.forecast

    if forecast_df is not None and not forecast_df.is_empty():
        fc_cfg = config.forecast
        summary["forecast_rows"] = forecast_df.height
        id_col = fc_cfg.series_id_column if fc_cfg.series_id_column in forecast_df.columns else "series_id"
        summary["series_count"] = forecast_df[id_col].n_unique() if id_col in forecast_df.columns else 0
        summary["forecast_preview"] = forecast_df.head(100).to_dicts()

    return PipelineRunResponse(**summary)


@router.get("/manifests")
def list_manifests(
    request: Request,
    lob: Optional[str] = Query(None, description="Filter by LOB"),
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(require_permission(Permission.VIEW_METRICS)),
):
    """List recent pipeline run manifests."""
    from ...pipeline.manifest import read_manifest

    if lob:
        validate_path_param(lob, "lob")
    manifests_dir = request.app.state.data_dir / "forecasts"
    if not manifests_dir.exists():
        return {"count": 0, "manifests": []}

    manifest_files = []
    search_dirs = [manifests_dir / lob] if lob else list(manifests_dir.iterdir())

    for d in search_dirs:
        if d.is_dir():
            manifest_files.extend(sorted(d.glob("manifest_*.json"), reverse=True))

    manifest_files = sorted(manifest_files, key=lambda f: f.stat().st_mtime, reverse=True)[:limit]

    results = []
    for mf in manifest_files:
        try:
            manifest = read_manifest(str(mf))
            results.append({
                "run_id": manifest.run_id,
                "timestamp": manifest.timestamp,
                "lob": manifest.lob,
                "series_count": manifest.input_series_count,
                "champion_model": manifest.champion_model_id,
                "backtest_wmape": manifest.backtest_wmape,
                "forecast_horizon": manifest.forecast_horizon,
                "forecast_rows": manifest.forecast_row_count,
                "validation_passed": manifest.validation_passed,
                "validation_warnings": manifest.validation_warnings,
                "cleansing_applied": manifest.cleansing_applied,
                "outliers_clipped": manifest.outliers_clipped,
                "post_validation_grade": getattr(manifest, "post_validation_grade", None),
                "post_validation_score": getattr(manifest, "post_validation_score", None),
            })
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning(f"Failed to read manifest {mf}: {exc}")

    return {"count": len(results), "manifests": results}


@router.get("/costs")
def get_costs(
    request: Request,
    lob: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(require_permission(Permission.VIEW_METRICS)),
):
    """Get cost tracking data from pipeline manifests."""
    from ...pipeline.manifest import read_manifest

    if lob:
        validate_path_param(lob, "lob")
    manifests_dir = request.app.state.data_dir / "forecasts"
    if not manifests_dir.exists():
        return {"count": 0, "costs": []}

    manifest_files = []
    search_dirs = [manifests_dir / lob] if lob else list(manifests_dir.iterdir())

    for d in search_dirs:
        if d.is_dir():
            manifest_files.extend(sorted(d.glob("manifest_*.json"), reverse=True))

    manifest_files = sorted(manifest_files, key=lambda f: f.stat().st_mtime, reverse=True)[:limit]

    costs = []
    for mf in manifest_files:
        try:
            manifest = read_manifest(str(mf))
            series_count = manifest.input_series_count or 1
            # Try to read timing from manifest (may not always be present)
            total_seconds = getattr(manifest, "total_seconds", 0.0)
            costs.append({
                "run_id": manifest.run_id,
                "timestamp": manifest.timestamp,
                "lob": manifest.lob,
                "series_count": series_count,
                "champion_model": manifest.champion_model_id,
                "total_seconds": total_seconds,
                "seconds_per_series": round(total_seconds / series_count, 3) if total_seconds else None,
            })
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning(f"Failed to read manifest {mf}: {exc}")

    return {"count": len(costs), "costs": costs}


@router.post("/analyze-multi-file")
async def analyze_multi_file(
    files: list[UploadFile] = File(..., description="Multiple CSV/Parquet files"),
    lob_name: str = Query("analyzed"),
    user: User = Depends(require_permission(Permission.RUN_PIPELINE)),
):
    """Upload multiple files, auto-classify roles, and merge."""
    from ...data.file_classifier import FileClassifier
    from ...data.file_merger import MultiFileMerger

    file_dfs = {}
    for f in files:
        content = await validate_upload_size(f)
        filename = f.filename or f"file_{len(file_dfs)}"
        try:
            if filename.endswith(".parquet"):
                file_dfs[filename] = pl.read_parquet(io.BytesIO(content))
            else:
                file_dfs[filename] = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read '{filename}': {exc}")

    classifier = FileClassifier()
    classification = classifier.classify_files(file_dfs)

    profiles = []
    for p in classification.profiles:
        profiles.append({
            "filename": p.filename,
            "role": p.role,
            "confidence": round(p.confidence, 3),
            "time_column": p.time_column,
            "id_columns": p.id_columns,
            "n_rows": p.n_rows,
            "n_columns": p.n_columns,
            "reasoning": p.reasoning,
        })

    result = {
        "profiles": profiles,
        "primary_file": classification.primary_file.filename if classification.primary_file else None,
        "dimension_files": [f.filename for f in classification.dimension_files],
        "regressor_files": [f.filename for f in classification.regressor_files],
        "warnings": classification.warnings,
    }

    # Attempt merge if there's a primary file and secondary files
    if classification.primary_file and (classification.dimension_files or classification.regressor_files):
        try:
            merger = MultiFileMerger()
            preview = merger.preview_merge(classification)
            result["merge_preview"] = {
                "total_rows": preview.total_rows,
                "total_columns": preview.total_columns,
                "matched_rows": preview.matched_rows,
                "unmatched_primary_keys": preview.unmatched_primary_keys,
                "null_fill_columns": preview.null_fill_columns,
                "warnings": preview.warnings,
                "sample_rows": preview.sample_rows.head(20).to_dicts() if preview.sample_rows is not None else [],
            }
        except (ValueError, KeyError, RuntimeError) as exc:
            result["merge_error"] = str(exc)

    return result
