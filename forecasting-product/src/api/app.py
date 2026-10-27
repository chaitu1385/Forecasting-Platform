"""
Forecasting Product REST API
(Design document §13 / Phase 2)

Exposes forecast results, model leaderboards, drift alerts, series analysis,
hierarchy management, SKU mapping, pipeline operations, analytics, and
governance over HTTP. Built with FastAPI — auto-generates OpenAPI docs at /docs.

Core Endpoints
--------------
GET  /health                         Liveness probe.
GET  /forecast/{lob}                 Latest forecasts for a LOB.
GET  /forecast/{lob}/{series_id}     Latest forecast for a specific series.
GET  /metrics/leaderboard/{lob}      Model leaderboard from metric store.
GET  /metrics/drift/{lob}            Drift alerts for a LOB.
POST /analyze                        Upload CSV → schema + forecastability.

Router Endpoints (see src/api/routers/)
---------------------------------------
/series/*           Series listing, SBC classification, breaks, cleansing, regressors.
/hierarchy/*        Build tree, aggregate, reconcile.
/sku-mapping/*      Phase 1/2 SKU mapping.
/overrides/*        Planner override CRUD.
/pipeline/*         Run backtest/forecast, manifests, costs, multi-file analysis.
/metrics/*/fva      FVA cascade.
/metrics/*/calibration  Prediction interval calibration.
/metrics/*/shap     SHAP feature attribution.
/forecast/decompose Seasonal decomposition.
/forecast/compare   Cross-forecast comparison.
/forecast/constrain Apply capacity/budget constraints.
/governance/*       Model cards, lineage, BI export.

Usage
-----
>>> from src.api.app import create_app
>>> app = create_app(data_dir="data/", metrics_dir="data/metrics/")

Or via scripts/serve.py:
    python scripts/serve.py --port 8000 --data-dir data/

Environment variables
---------------------
API_DATA_DIR      Path to local data directory (default: data/).
API_METRICS_DIR   Path to metrics store (default: data/metrics/).
API_VERSION       Version string embedded in /health (default: "1.0.0").
AUTH_ENABLED      Strict boolean; defaults to true.
JWT_SECRET_KEY    Dedicated production signing key (required outside demo).
API_DEV_MODE      Strict boolean; defaults to false. Explicit demo also requires AUTH_ENABLED=false.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .deps import validate_path_param, validate_upload_size
from .schemas import (
    AnalysisResponse,
    CommentaryRequest,
    CommentaryResponse,
    ConfigRecommendationItem,
    ConfigTuneRequest,
    ConfigTuneResponse,
    DriftAlertItem,
    DriftResponse,
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    KeyMetricItem,
    LeaderboardEntry,
    LeaderboardResponse,
    NLQueryRequest,
    NLQueryResponse,
    TriagedAlertItem,
    TriageRequest,
    TriageResponse,
)

logger = logging.getLogger(__name__)

_API_VERSION = os.environ.get("API_VERSION", "1.0.0")


class _RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter per client IP."""

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Prune all clients, not just the current one, to bound memory under
        # high-cardinality traffic (proxies, scanners, rotating clients).
        self._hits = {
            ip: [t for t in timestamps if t > cutoff]
            for ip, timestamps in self._hits.items()
            if any(t > cutoff for t in timestamps)
        }
        hits = [t for t in self._hits.get(client_ip, []) if t > cutoff]

        if len(hits) >= self.max_requests:
            self._hits[client_ip] = hits
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )

        hits.append(now)
        self._hits[client_ip] = hits
        return await call_next(request)


def _register_routers(app: FastAPI) -> None:
    """Include all domain routers."""
    from .routers import (
        analytics,
        auth,
        governance,
        hierarchy,
        overrides,
        pipeline,
        series,
        sku_mapping,
    )

    for r in (
        auth.router,
        series.router,
        hierarchy.router,
        sku_mapping.router,
        overrides.router,
        pipeline.router,
        analytics.router,
        governance.router,
    ):
        app.include_router(r)


def create_app(
    data_dir: Optional[str] = None,
    metrics_dir: Optional[str] = None,
    title: str = "Forecasting Product API",
    auth_enabled: Optional[bool] = None,
    jwt_secret: Optional[str] = None,
    audit_log_path: Optional[str] = None,
    cors_origins: list[str] | None = None,
    dev_mode: Optional[bool] = None,
) -> FastAPI:
    """
    Factory function — returns a configured FastAPI application.

    Parameters
    ----------
    data_dir:
        Root directory for forecast Parquet files.
        Structure: ``{data_dir}/forecasts/{lob}/forecast_*.parquet``.
    metrics_dir:
        Root directory for the MetricStore Parquet partitions.
    title:
        Application title (appears in /docs).
    """
    from ..auth.settings import AuthSettings

    auth_settings = AuthSettings.from_env(
        auth_enabled=auth_enabled, dev_mode=dev_mode, jwt_secret=jwt_secret,
    )
    data_path = Path(data_dir if data_dir is not None else os.environ.get("API_DATA_DIR", "data")).resolve()
    metrics_path = Path(
        metrics_dir if metrics_dir is not None else os.environ.get("API_METRICS_DIR", str(data_path / "metrics"))
    ).resolve()
    if auth_settings.dev_mode:
        logger.warning("DEMO MODE: identities and roles are simulated; this API is not production-authenticated")

    app = FastAPI(
        title=title,
        description=(
            "REST API for the forecasting product. "
            "Serves forecast results, model leaderboards, and drift alerts."
        ),
        version=_API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS — allow cross-origin requests from frontend (Next.js)
    from fastapi.middleware.cors import CORSMiddleware

    _default_origins = [
        "http://localhost:3000",   # Next.js dev
        "http://localhost:8000",   # API docs (Swagger UI)
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or _default_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate limiting ─────────────────────────────────────────────────────
    app.add_middleware(
        _RateLimitMiddleware,
        max_requests=int(os.environ.get("API_RATE_LIMIT", "100")),
        window_seconds=60,
    )

    # ── Shared state (attached to app for testability) ─────────────────────
    app.state.data_dir = data_path
    app.state.metrics_dir = metrics_path
    app.state.auth_settings = auth_settings
    app.state.auth_enabled = auth_settings.auth_enabled
    app.state.jwt_secret = auth_settings.jwt_secret

    from ..audit.logger import AuditLogger
    app.state.audit_logger = AuditLogger(audit_log_path or str(data_path / "audit_log"))

    from ..auth.models import Permission, User
    from ..auth.rbac import get_current_user, require_permission

    # ── Router registration ────────────────────────────────────────────────
    _register_routers(app)

    # ── Routes ─────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health():
        """Readiness probe — returns dependency health checks."""
        checks = {
            "data_dir": app.state.data_dir.is_dir(),
            "metrics_dir": app.state.metrics_dir.is_dir(),
        }
        status = "ok" if all(checks.values()) else "degraded"
        return HealthResponse(status=status, version=_API_VERSION, checks=checks)

    @app.get("/audit", tags=["audit"])
    def get_audit_log(
        user: User = Depends(require_permission(Permission.VIEW_AUDIT_LOG)),
        action: Optional[str] = Query(None),
        resource_type: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ):
        """Query the audit log. Requires VIEW_AUDIT_LOG permission."""
        results = app.state.audit_logger.query(
            action=action,
            resource_type=resource_type,
            limit=limit,
        )
        return {
            "count": len(results),
            "events": results.to_dicts(),
        }

    @app.get("/forecast/{lob}", response_model=ForecastResponse, tags=["forecasts"])
    def get_forecast(
        lob: str,
        series_id: Optional[str] = Query(None, description="Filter to a single series"),
        horizon: Optional[int] = Query(None, description="Limit output to first N weeks"),
        user: User = Depends(get_current_user),
    ):
        """
        Return the latest forecast Parquet file for a LOB.

        Reads the most recently written ``forecast_*.parquet`` under
        ``{data_dir}/forecasts/{lob}/``.
        """
        validate_path_param(lob, "lob")
        forecast_dir = app.state.data_dir / "forecasts" / lob
        if not forecast_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No forecast data found for LOB '{lob}'. "
                       f"Expected directory: {forecast_dir}",
            )

        parquet_files = sorted(forecast_dir.glob("forecast_*.parquet"))
        if not parquet_files:
            raise HTTPException(
                status_code=404,
                detail=f"No forecast Parquet files in {forecast_dir}.",
            )

        import polars as pl
        df = pl.read_parquet(parquet_files[-1])   # most recent

        if series_id:
            df = df.filter(pl.col("series_id") == series_id)
            if df.is_empty():
                raise HTTPException(
                    status_code=404,
                    detail=f"series_id '{series_id}' not found in LOB '{lob}'.",
                )

        time_column = next(
            (name for name in ("week", "date", "period", "timestamp") if name in df.columns),
            None,
        )
        if time_column is None:
            raise HTTPException(
                status_code=500,
                detail="Forecast artifact has no supported time column.",
            )

        if horizon:
            df = (
                df.sort(["series_id", time_column])
                .group_by("series_id", maintain_order=True)
                .head(horizon)
            )

        has_p10 = "forecast_p10" in df.columns
        has_p50 = "forecast_p50" in df.columns
        has_p90 = "forecast_p90" in df.columns

        points = [
            ForecastPoint(
                series_id=row["series_id"],
                week=row[time_column],
                forecast=float(row["forecast"]),
                model=row.get("model"),
                lob=lob,
                forecast_p10=float(row["forecast_p10"]) if has_p10 and row.get("forecast_p10") is not None else None,
                forecast_p50=float(row["forecast_p50"]) if has_p50 and row.get("forecast_p50") is not None else None,
                forecast_p90=float(row["forecast_p90"]) if has_p90 and row.get("forecast_p90") is not None else None,
            )
            for row in df.iter_rows(named=True)
        ]

        return ForecastResponse(
            lob=lob,
            series_count=df["series_id"].n_unique(),
            forecast_origin=parquet_files[-1].stem.replace(f"forecast_{lob}_", ""),
            points=points,
        )

    @app.get(
        "/forecast/{lob}/{series_id}",
        response_model=ForecastResponse,
        tags=["forecasts"],
    )
    def get_forecast_series(lob: str, series_id: str, user: User = Depends(get_current_user)):
        """Return the latest forecast for a single series within a LOB."""
        return get_forecast(lob=lob, series_id=series_id, horizon=None, user=user)

    @app.get(
        "/metrics/leaderboard/{lob}",
        response_model=LeaderboardResponse,
        tags=["metrics"],
    )
    def get_leaderboard(
        lob: str,
        run_type: str = Query("backtest", description="'backtest' or 'live'"),
        user: User = Depends(get_current_user),
    ):
        """
        Return the model leaderboard for a LOB from the metric store.

        Models are ranked by WMAPE (ascending).
        """
        validate_path_param(lob, "lob")
        from ..metrics.store import MetricStore

        store = MetricStore(str(app.state.metrics_dir))
        try:
            df = store.leaderboard(lob=lob, run_type=run_type)
        except (FileNotFoundError, OSError) as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Leaderboard data not found for LOB '{lob}': {exc}",
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read leaderboard: {exc}",
            )

        if df.is_empty():
            raise HTTPException(
                status_code=404,
                detail=f"No leaderboard data found for LOB '{lob}' (run_type={run_type}).",
            )

        entries = [
            LeaderboardEntry(
                model=row["model_id"],
                wmape=float(row["wmape"]),
                normalized_bias=float(row.get("normalized_bias", 0.0) or 0.0),
                rank=int(row.get("rank", i + 1)),
                n_series=int(row.get("n_series", 0) or 0),
            )
            for i, row in enumerate(df.iter_rows(named=True))
        ]

        return LeaderboardResponse(lob=lob, run_type=run_type, entries=entries)

    @app.get(
        "/metrics/drift/{lob}",
        response_model=DriftResponse,
        tags=["metrics"],
    )
    def get_drift(
        lob: str,
        run_type: str = Query("backtest", description="'backtest' or 'live'"),
        baseline_weeks: int = Query(26, ge=4),
        recent_weeks:   int = Query(8,  ge=2),
        user: User = Depends(get_current_user),
    ):
        """
        Return drift alerts for all series in a LOB.

        Reads the metric store, runs ``ForecastDriftDetector``, and returns
        any accuracy / bias / volume alerts.
        """
        validate_path_param(lob, "lob")
        from ..metrics.drift import DriftConfig, ForecastDriftDetector
        from ..metrics.store import MetricStore

        store = MetricStore(str(app.state.metrics_dir))
        try:
            df = store.read(lob=lob, run_type=run_type)
        except (FileNotFoundError, OSError) as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Metric data not found for LOB '{lob}': {exc}",
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read metric store: {exc}",
            )

        if df is None or df.is_empty():
            raise HTTPException(
                status_code=404,
                detail=f"No metric data found for LOB '{lob}' (run_type={run_type}).",
            )

        cfg = DriftConfig(baseline_weeks=baseline_weeks, recent_weeks=recent_weeks)
        detector = ForecastDriftDetector(cfg)
        alerts = detector.detect(df)

        alert_items = [
            DriftAlertItem(
                series_id=a.series_id,
                metric=a.metric,
                severity=a.severity.value,
                current_value=a.current_value,
                baseline_value=a.baseline_value,
                message=a.message,
            )
            for a in alerts
        ]

        return DriftResponse(
            lob=lob,
            n_critical=sum(1 for a in alerts if a.severity.value == "critical"),
            n_warning=sum(1 for a in alerts if a.severity.value == "warning"),
            alerts=alert_items,
        )

    @app.post("/analyze", response_model=AnalysisResponse, tags=["analytics"])
    async def analyze_data(
        file: UploadFile = File(...),
        lob_name: str = Query("analyzed", description="Name for this analysis"),
        llm_enabled: bool = Query(False, description="Use Claude for interpretation"),
        user: User = Depends(require_permission(Permission.RUN_PIPELINE)),
    ):
        """Upload a CSV or Parquet file for automated analysis and config recommendation.

        Returns schema detection, forecastability signals, hierarchy detection,
        hypotheses, and a ready-to-use PlatformConfig YAML.
        """
        import io
        from dataclasses import asdict

        import polars as pl
        import yaml

        from ..analytics.analyzer import DataAnalyzer
        from ..analytics.llm_analyzer import LLMAnalyzer

        # Read uploaded file with size limit
        content = await validate_upload_size(file)
        filename = file.filename or ""

        try:
            if filename.endswith(".parquet"):
                df = pl.read_parquet(io.BytesIO(content))
            else:
                # Default to CSV
                df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read uploaded file: {exc}",
            )

        if df.is_empty():
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Run analysis
        analyzer = DataAnalyzer(lob_name=lob_name)
        report = analyzer.analyze(df)

        # LLM interpretation (optional)
        llm_narrative = None
        llm_risk_factors = None
        if llm_enabled:
            llm = LLMAnalyzer()
            if llm.available:
                insight = llm.interpret(report)
                llm_narrative = insight.narrative or None
                llm_risk_factors = insight.risk_factors or None

        # Serialize config to YAML
        config_dict = asdict(report.recommended_config)
        config_yaml = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

        schema = report.schema
        fc = report.forecastability

        return AnalysisResponse(
            lob_name=lob_name,
            time_column=schema.time_column,
            target_column=schema.target_column,
            id_columns=schema.id_columns,
            n_series=schema.n_series,
            n_rows=schema.n_rows,
            date_range_start=schema.date_range[0],
            date_range_end=schema.date_range[1],
            frequency=schema.frequency_guess,
            overall_forecastability=round(fc.overall_score, 3),
            forecastability_distribution=fc.score_distribution,
            demand_classes=fc.demand_class_distribution,
            detected_hierarchies=[
                {"name": h.name, "levels": h.levels, "id_column": h.id_column, "fixed": h.fixed}
                for h in report.hierarchy.hierarchies
            ],
            recommended_config_yaml=config_yaml,
            config_reasoning=report.config_reasoning,
            hypotheses=report.hypotheses,
            llm_narrative=llm_narrative,
            llm_risk_factors=llm_risk_factors,
        )

    # ── AI-native endpoints ────────────────────────────────────────────────

    @app.post("/ai/explain", response_model=NLQueryResponse, tags=["ai"])
    async def ai_explain(
        request: NLQueryRequest,
        user: User = Depends(require_permission(Permission.VIEW_FORECASTS)),
    ):
        """Answer a natural-language question about a specific series forecast."""
        import polars as pl

        from ..ai.nl_query import NaturalLanguageQueryEngine

        validate_path_param(request.lob, "lob")
        # Load forecast data for the series
        forecast_dir = app.state.data_dir / "forecasts" / request.lob
        forecast_df = pl.DataFrame()
        if forecast_dir.exists():
            parquet_files = sorted(forecast_dir.glob("forecast_*.parquet"))
            if parquet_files:
                forecast_df = pl.read_parquet(parquet_files[-1])
                forecast_df = forecast_df.filter(pl.col("series_id") == request.series_id)

        # Load metrics data
        metrics_df = pl.DataFrame()
        try:
            from ..metrics.store import MetricStore
            store = MetricStore(str(app.state.metrics_dir))
            metrics_df = store.read(lob=request.lob, run_type="backtest")
        except (FileNotFoundError, OSError, ValueError, KeyError):
            logger.debug("Could not load metrics for NL query (lob=%s)", request.lob, exc_info=True)

        # Load history data if available
        history_df = pl.DataFrame()
        history_dir = app.state.data_dir / "history" / request.lob
        if history_dir.exists():
            parquet_files = sorted(history_dir.glob("*.parquet"))
            if parquet_files:
                try:
                    history_df = pl.read_parquet(parquet_files[-1])
                    history_df = history_df.filter(pl.col("series_id") == request.series_id)
                except (FileNotFoundError, OSError, ValueError) as _exc:
                    logger.debug("Could not load history for NL query (series=%s)", request.series_id, exc_info=True)

        engine = NaturalLanguageQueryEngine()
        result = engine.query(
            series_id=request.series_id,
            question=request.question,
            lob=request.lob,
            history=history_df if not history_df.is_empty() else None,
            forecast=forecast_df if not forecast_df.is_empty() else None,
            metrics_df=metrics_df if not metrics_df.is_empty() else None,
        )

        return NLQueryResponse(
            answer=result.answer,
            supporting_data=result.supporting_data,
            confidence=result.confidence,
            sources_used=result.sources_used,
        )

    @app.post("/ai/triage", response_model=TriageResponse, tags=["ai"])
    async def ai_triage(
        request: TriageRequest,
        user: User = Depends(require_permission(Permission.VIEW_METRICS)),
    ):
        """Triage drift alerts by business impact with suggested actions."""
        from ..ai.anomaly_triage import AnomalyTriageEngine
        from ..metrics.drift import DriftConfig, ForecastDriftDetector
        from ..metrics.store import MetricStore

        validate_path_param(request.lob, "lob")
        store = MetricStore(str(app.state.metrics_dir))
        try:
            df = store.read(lob=request.lob, run_type=request.run_type)
        except (FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=404, detail=f"Metric data not found for LOB '{request.lob}': {exc}")
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read metric store: {exc}")

        if df is None or df.is_empty():
            raise HTTPException(status_code=404, detail=f"No metric data for LOB '{request.lob}'.")

        detector = ForecastDriftDetector(DriftConfig())
        alerts = detector.detect(df)

        if request.severity_filter:
            alerts = [a for a in alerts if a.severity.value == request.severity_filter]

        engine = AnomalyTriageEngine()
        result = engine.query(
            lob=request.lob,
            drift_alerts=alerts,
            max_alerts=request.max_alerts,
        )

        return TriageResponse(
            lob=request.lob,
            executive_summary=result.executive_summary,
            total_alerts=result.total_alerts,
            critical_count=result.critical_count,
            warning_count=result.warning_count,
            ranked_alerts=[
                TriagedAlertItem(
                    series_id=a.series_id,
                    metric=a.metric,
                    severity=a.severity,
                    business_impact_score=a.business_impact_score,
                    suggested_action=a.suggested_action,
                    reasoning=a.reasoning,
                    original_message=a.original_message,
                )
                for a in result.ranked_alerts
            ],
        )

    @app.post("/ai/recommend-config", response_model=ConfigTuneResponse, tags=["ai"])
    async def ai_recommend_config(
        request: ConfigTuneRequest,
        user: User = Depends(require_permission(Permission.RUN_PIPELINE)),
    ):
        """Recommend configuration changes based on backtest performance."""
        from ..ai.config_tuner import ConfigTunerEngine
        from ..config.schema import PlatformConfig
        from ..metrics.store import MetricStore

        validate_path_param(request.lob, "lob")
        store = MetricStore(str(app.state.metrics_dir))

        # Load leaderboard
        leaderboard = None
        try:
            leaderboard = store.leaderboard(lob=request.lob, run_type=request.run_type)
        except (FileNotFoundError, OSError, ValueError, KeyError):
            logger.warning("Could not load leaderboard for config tuner (lob=%s)", request.lob, exc_info=True)

        # Use default config (in production, load from config file)
        config = PlatformConfig(lob=request.lob)

        engine = ConfigTunerEngine()
        result = engine.recommend(
            lob=request.lob,
            current_config=config,
            leaderboard=leaderboard,
        )

        return ConfigTuneResponse(
            lob=request.lob,
            recommendations=[
                ConfigRecommendationItem(
                    field_path=r.field_path,
                    current_value=r.current_value,
                    recommended_value=r.recommended_value,
                    reasoning=r.reasoning,
                    expected_impact=r.expected_impact,
                    risk=r.risk,
                )
                for r in result.recommendations
            ],
            overall_assessment=result.overall_assessment,
            risk_summary=result.risk_summary,
        )

    @app.post("/ai/commentary", response_model=CommentaryResponse, tags=["ai"])
    async def ai_commentary(
        request: CommentaryRequest,
        user: User = Depends(require_permission(Permission.VIEW_METRICS)),
    ):
        """Generate executive forecast commentary for S&OP meetings."""
        from ..ai.commentary import CommentaryEngine
        from ..metrics.drift import DriftConfig, ForecastDriftDetector
        from ..metrics.store import MetricStore

        validate_path_param(request.lob, "lob")
        store = MetricStore(str(app.state.metrics_dir))
        try:
            df = store.read(lob=request.lob, run_type=request.run_type)
        except (FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=404, detail=f"Metric data not found for LOB '{request.lob}': {exc}")
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read metric store: {exc}")

        if df is None or df.is_empty():
            raise HTTPException(status_code=404, detail=f"No metric data for LOB '{request.lob}'.")

        # Get drift alerts
        detector = ForecastDriftDetector(DriftConfig())
        alerts = detector.detect(df)

        # Get leaderboard
        leaderboard = None
        try:
            leaderboard = store.leaderboard(lob=request.lob, run_type=request.run_type)
        except (FileNotFoundError, OSError, ValueError, KeyError):
            logger.warning("Could not load leaderboard for commentary (lob=%s)", request.lob, exc_info=True)

        engine = CommentaryEngine()
        result = engine.generate(
            lob=request.lob,
            metrics_df=df,
            drift_alerts=alerts,
            leaderboard=leaderboard,
            period_start=request.period_start,
            period_end=request.period_end,
        )

        return CommentaryResponse(
            lob=request.lob,
            executive_summary=result.executive_summary,
            key_metrics=[
                KeyMetricItem(
                    name=m.name,
                    value=m.value,
                    unit=m.unit,
                    trend=m.trend,
                )
                for m in result.key_metrics
            ],
            exceptions=result.exceptions,
            action_items=result.action_items,
        )

    return app
