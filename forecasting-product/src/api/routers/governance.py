"""Governance endpoints — model cards, forecast lineage, BI export."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ...auth.models import Permission, User
from ...auth.rbac import get_current_user, require_permission
from ..deps import metric_records, metric_store, raise_run_error, run_reader, validate_path_param

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/governance", tags=["governance"])


@router.get("/model-cards")
def list_model_cards(
    request: Request,
    user: User = Depends(get_current_user),
):
    """List all registered model cards."""
    from ...analytics.governance import ModelCardRegistry

    base_path = str(request.app.state.data_dir / "model_cards")
    registry = ModelCardRegistry(base_path=base_path)

    try:
        cards_df = registry.all_cards()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read model cards: {exc}")

    if cards_df is None or cards_df.is_empty():
        return {"count": 0, "model_cards": []}

    return {
        "count": cards_df.height,
        "model_cards": cards_df.to_dicts(),
    }


@router.get("/model-cards/{model_name}")
def get_model_card(
    model_name: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    """Get a specific model card by name."""
    from ...analytics.governance import ModelCardRegistry

    validate_path_param(model_name, "model_name")
    base_path = str(request.app.state.data_dir / "model_cards")
    registry = ModelCardRegistry(base_path=base_path)

    card = registry.get(model_name)
    if card is None:
        raise HTTPException(status_code=404, detail=f"Model card '{model_name}' not found.")

    return card.to_dict()


@router.get("/lineage")
def get_lineage(
    request: Request,
    lob: Optional[str] = Query(None),
    model_id: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
    user: User = Depends(get_current_user),
):
    """Get forecast lineage history."""
    from ...analytics.governance import ForecastLineage
    from ...pipeline.run_store import RunIntegrityError, RunNotFoundError

    if lob:
        validate_path_param(lob, "lob")
    base_path = str(request.app.state.data_dir / "lineage")
    lineage = ForecastLineage(base_path=base_path)

    try:
        reader = run_reader(request)
        if run_id is not None:
            if lob is None:
                raise ValueError("An exact run requires its LOB")
            manifests = [reader.store.get(lob, run_id, "forecast")]
        else:
            manifests = reader.store.list(lob=lob, run_type="forecast")
        rows = []
        for manifest in manifests:
            models = {item["name"] for item in manifest.models} | {manifest.champion_model_id}
            if model_id is not None and model_id not in models:
                continue
            rows.append({
                "run_id": manifest.run_id, "timestamp": manifest.started_at,
                "run_date": manifest.started_at[:10], "lob": manifest.lob,
                "model_id": manifest.champion_model_id, "n_series": manifest.input_series_count,
                "horizon_weeks": manifest.forecast_horizon, "selection_strategy": "recorded_run",
                "input_data_hash": manifest.input_data_hash, "config_hash": manifest.config_hash,
                "artifact_source": "committed", "lineage_status": "verified_bundle",
                "user_id": None, "notes": "Immutable run bundle; no model approval inferred",
            })
        if run_id is None:
            history = lineage.history(lob=lob, model_id=model_id)
            for item in history.to_dicts():
                rows.append({**item, "run_id": None, "artifact_source": "legacy", "lineage_status": "unknown",
                             "timestamp": str(item.get("run_date", ""))})
    except (RunIntegrityError, RunNotFoundError, OSError, ValueError) as exc:
        raise_run_error(exc)
    return {"count": len(rows), "lineage": rows}


@router.post("/export/{report_type}")
def bi_export(
    report_type: str,
    request: Request,
    lob: str = Query(..., description="LOB name"),
    run_type: str = Query("backtest"),
    model_id: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
    actuals_run_id: Optional[str] = Query(None, description="Explicit later-observation run for forecast comparison"),
    user: User = Depends(require_permission(Permission.VIEW_METRICS)),
):
    """Export BI report (forecast-actual, leaderboard, or bias-report)."""
    from ...analytics.bi_export import BIExporter
    from ...pipeline.run_store import RunIntegrityError, RunNotFoundError

    validate_path_param(lob, "lob")
    validate_path_param(report_type, "report_type")
    valid_types = ("forecast-actual", "leaderboard", "bias-report")
    if report_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid report type '{report_type}'. Must be one of: {valid_types}",
        )

    base_path = str(request.app.state.data_dir / "bi_exports")
    exporter = BIExporter(base_path=base_path)

    if report_type == "leaderboard":
        from ..deps import leaderboard_records, scoring_provenance

        board, selected = leaderboard_records(request, lob, run_type, run_id)
        try:
            path = exporter.export_leaderboard_frame(
                board, lob=lob, run_id=selected.run_id if selected else None,
            )
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail="BI export could not be confirmed") from exc
        return {
            "report_type": report_type, "export_path": str(path.relative_to(exporter.base_path)), "status": "exported",
            "run_id": selected.run_id if selected else None, "artifact_source": "committed" if selected else "legacy",
            "scoring_provenance": scoring_provenance(board, selected),
        }

    elif report_type == "bias-report":
        _, selected = metric_records(request, lob, run_type, run_id, latest=True)
        store = metric_store(request)
        try:
            path = exporter.export_bias_report(
                metric_store=store, lob=lob, model_id=model_id, run_type=run_type,
                run_id=selected.run_id if selected else None,
            )
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail="BI export could not be confirmed") from exc
        return {
            "report_type": report_type, "export_path": str(path.relative_to(exporter.base_path)), "status": "exported",
            "run_id": selected.run_id if selected else None, "artifact_source": "committed" if selected else "legacy",
        }

    elif report_type == "forecast-actual":
        import polars as pl

        try:
            reader = run_reader(request)
            selected = reader.forecast(lob, run_id)
            history = reader.history(
                lob, actuals_run_id or selected.run_id,
                legacy=selected.source == "legacy" and actuals_run_id is None,
            )
            forecasts = selected.frame.rename({selected.id_column: "series_id", selected.time_column: "week"})
            actuals = history.frame.rename({
                history.id_column: "series_id", history.time_column: "week", history.target_column: "quantity",
            })
            actuals = actuals.with_columns(pl.col("week").cast(forecasts.schema["week"]))
        except (RunIntegrityError, RunNotFoundError, ValueError, OSError) as exc:
            raise_run_error(exc)

        try:
            path = exporter.export_forecast_vs_actual(
                forecasts=forecasts, actuals=actuals, lob=lob,
                run_id=selected.run_id, actuals_run_id=history.run_id,
            )
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail="BI export could not be confirmed") from exc

        return {
            "report_type": report_type, "export_path": str(path.relative_to(exporter.base_path)), "status": "exported",
            "run_id": selected.run_id, "actuals_run_id": history.run_id, "artifact_source": selected.source,
        }
