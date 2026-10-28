"""
Pydantic response models for the forecasting product REST API.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' or 'degraded'")
    version: str = Field(description="API version string")
    checks: Optional[Dict[str, bool]] = Field(None, description="Dependency health checks")


class ForecastPoint(BaseModel):
    """A single (series, week, forecast) data point."""
    series_id: str
    week: date
    forecast: Optional[float]
    model: Optional[str] = None
    lob: Optional[str] = None
    forecast_p10: Optional[float] = None
    forecast_p50: Optional[float] = None
    forecast_p90: Optional[float] = None


class ForecastResponse(BaseModel):
    lob: str
    series_count: int
    forecast_origin: Optional[str] = None
    points: List[ForecastPoint]
    run_id: Optional[str] = None
    artifact_source: Literal["committed", "legacy"] = "legacy"
    config_hash: Optional[str] = None
    input_data_hash: Optional[str] = None


class AppliedConstraints(BaseModel):
    """Validated settings used for the complete uploaded forecast."""

    min_demand: float = Field(ge=0, allow_inf_nan=False)
    max_capacity: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    aggregate_max: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    proportional: bool


class ConstrainResponse(BaseModel):
    """Measured full-population constraint results; preview is limited to 200 rows."""

    before_total: float = Field(allow_inf_nan=False)
    after_total: float = Field(allow_inf_nan=False)
    rows_modified: int = Field(
        ge=0, description="Rows with any constrained point or quantile change, counted once per row",
    )
    constraints_applied: AppliedConstraints
    constrained_preview: List[Dict[str, Any]]
    allocation_strategy: Literal["element_wise", "proportional", "largest_first"]
    checked_rows: int = Field(ge=1)
    checked_periods: int = Field(ge=1)
    max_budget_excess: Optional[float] = Field(
        ge=0, allow_inf_nan=False,
        description="Maximum measured positive per-period excess; null when no budget was supplied",
    )
    constraint_scope: Literal["point", "point_and_quantiles"]


class LeaderboardEntry(BaseModel):
    """One model's aggregated performance metrics."""
    model: str
    wmape: Optional[float]
    normalized_bias: Optional[float]
    rank: int
    n_series: int
    scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    primary_score: Optional[float] = None
    group_key: Optional[str] = None
    competition_scope: Optional[str] = None
    validation_grade: Optional[str] = Field(None, description="Post-validation confidence grade (A-F)")
    validation_score: Optional[int] = Field(None, description="Post-validation confidence score (0-100)")


class ScoringProvenance(BaseModel):
    source: Literal["stored", "computed"] = "stored"
    status: Literal["recorded", "unknown"] = "unknown"
    version: Optional[int] = None
    policy: Optional[str] = None
    primary_metric: Optional[str] = None
    secondary_metric: Optional[str] = None
    competition_scope: Optional[str] = None


class LeaderboardResponse(BaseModel):
    lob: str
    run_type: str
    entries: List[LeaderboardEntry]
    run_id: Optional[str] = None
    artifact_source: Literal["committed", "legacy"] = "legacy"
    scoring_provenance: ScoringProvenance = Field(default_factory=ScoringProvenance)


class ValidationLayerResult(BaseModel):
    """Summary of one validation layer's results."""
    checks_run: int = 0
    checks_passed: int = 0
    issues: List[Dict[str, Any]] = []


class ValidationResponse(BaseModel):
    """Response from GET /metrics/{lob}/validation."""
    lob: str
    grade: str = Field(description="Confidence grade: A, B, C, D, or F")
    score: int = Field(description="Confidence score 0-100")
    badge: str = Field(description="Formatted confidence badge")
    layers: Dict[str, Any] = Field(description="Per-layer validation results")
    confidence: Dict[str, Any] = Field(description="Scoring breakdown")
    skipped: bool = False


class HierarchyReconciliationMethod(str, Enum):
    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"
    OLS = "ols"
    WLS = "wls"
    MINT = "mint"


class HierarchyEffectiveMethod(str, Enum):
    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"
    OLS = "ols"
    WLS_STRUCTURAL = "wls_structural"


class HierarchyCoherenceResult(BaseModel):
    """Full-frame checks, not evidence of independent upper-level forecasts."""

    scope: Literal["leaf_derived_parent_totals", "root_forecast_conservation"]
    passed: bool
    checked_rows: int = Field(ge=0)
    checked_periods: int = Field(ge=0)
    checked_groups: int = Field(ge=0)
    value_columns: List[str]
    max_abs_error: float = Field(ge=0, allow_inf_nan=False)


class HierarchyReconcileResponse(BaseModel):
    method: HierarchyReconciliationMethod
    requested_method: HierarchyReconciliationMethod
    effective_method: HierarchyEffectiveMethod
    source_level: str
    output_level: str
    input_rows: int = Field(ge=1)
    output_rows: int = Field(ge=1)
    before_total: float = Field(allow_inf_nan=False)
    after_total: float = Field(allow_inf_nan=False)
    rows: int = Field(ge=1)
    reconciled_preview: List[Dict[str, Any]]
    warnings: List[str]
    coherence: HierarchyCoherenceResult
    validation: Optional[Dict[str, Any]] = None


class DriftAlertItem(BaseModel):
    """One drift alert for a series/metric pair."""
    series_id: str
    metric: str
    severity: str     # "warning" | "critical"
    current_value: float
    baseline_value: float
    message: str


class DriftResponse(BaseModel):
    lob: str
    n_critical: int
    n_warning: int
    alerts: List[DriftAlertItem]


class AnalysisResponse(BaseModel):
    """Response from the /analyze endpoint."""
    lob_name: str
    # Schema detection
    time_column: str
    target_column: str
    id_columns: List[str]
    n_series: int
    n_rows: int
    date_range_start: str
    date_range_end: str
    frequency: str
    # Forecastability
    overall_forecastability: float
    forecastability_distribution: Dict[str, int]
    demand_classes: Dict[str, int]
    # Hierarchy
    detected_hierarchies: List[Dict[str, Any]]
    # Config recommendation
    recommended_config_yaml: str
    config_reasoning: List[str]
    hypotheses: List[str]
    # LLM insights (null if not enabled)
    llm_narrative: Optional[str] = None
    llm_risk_factors: Optional[List[str]] = None


class EffectivePipelineConfig(BaseModel):
    """Allowlisted execution settings, never a serialized PlatformConfig."""

    frequency: Literal["D", "W", "M", "Q"]
    season_length: int
    series_id_column: str
    time_column: str
    target_column: str
    forecasters: List[str]
    intermittent_forecasters: List[str]
    quantiles: List[float]
    fill_gaps: bool
    fill_value: float
    min_series_length_periods: int
    drop_zero_series: bool
    input_validation_enabled: bool
    cleansing_enabled: bool
    constraints_enabled: bool
    post_validation_enabled: bool
    halt_on_blocker: bool
    min_grade: str
    backtest_n_folds: int
    backtest_val_periods: int
    backtest_gap_periods: int
    selection_strategy: str
    champion_granularity: str
    primary_metric: str
    secondary_metric: str


class PipelineExecutionSettings(BaseModel):
    model_id: Optional[str]
    model_source: Literal["request", "config", "default", "backtest"]
    horizon_periods: int = Field(ge=1, description="Production forecast horizon, not a CV window")
    horizon_source: Literal["request", "config", "default"]
    config_source: Literal["upload", "default"]
    config: EffectivePipelineConfig


class RunLinks(BaseModel):
    metadata: str
    config: str
    input_lineage: str
    forecast: Optional[str] = None
    leaderboard: Optional[str] = None


class PipelineRunResponse(BaseModel):
    lob: str
    status: str
    run_id: str
    run_type: Literal["forecast", "backtest"]
    links: RunLinks
    execution: PipelineExecutionSettings
    champion_model: Optional[str] = Field(None, description="Single backtest winner usable by the name-based forecast endpoint")
    best_wmape: Optional[float] = None
    leaderboard: Optional[List[Dict[str, Any]]] = None
    validation_grade: Optional[str] = None
    validation_score: Optional[int] = None
    model_id: Optional[str] = None
    horizon: Optional[int] = None
    forecast_rows: Optional[int] = None
    series_count: Optional[int] = None
    forecast_preview: Optional[List[Dict[str, Any]]] = None



# --------------------------------------------------------------------------- #
#  AI feature request/response models
# --------------------------------------------------------------------------- #

class NLQueryRequest(BaseModel):
    """Request body for POST /ai/explain."""
    series_id: str
    question: str
    lob: str
    run_id: Optional[str] = None
    legacy: bool = False


class NLQueryResponse(BaseModel):
    """Response from POST /ai/explain."""
    answer: str
    supporting_data: Dict[str, Any] = {}
    confidence: str = "low"       # "high" | "medium" | "low"
    sources_used: List[str] = []


class TriageRequest(BaseModel):
    """Request body for POST /ai/triage."""
    lob: str
    run_type: str = "backtest"
    severity_filter: Optional[str] = None   # "warning" | "critical" | None
    max_alerts: int = 50
    run_id: Optional[str] = None


class TriagedAlertItem(BaseModel):
    """A single triaged drift alert."""
    series_id: str
    metric: str
    severity: str
    business_impact_score: float = 0.0
    suggested_action: str = ""
    reasoning: str = ""
    original_message: str = ""


class TriageResponse(BaseModel):
    """Response from POST /ai/triage."""
    lob: str
    executive_summary: str = ""
    total_alerts: int = 0
    critical_count: int = 0
    warning_count: int = 0
    ranked_alerts: List[TriagedAlertItem] = []


class ConfigTuneRequest(BaseModel):
    """Request body for POST /ai/recommend-config."""
    lob: str
    run_type: str = "backtest"


class ConfigRecommendationItem(BaseModel):
    """A single configuration change recommendation."""
    field_path: str
    current_value: Any = None
    recommended_value: Any = None
    reasoning: str = ""
    expected_impact: str = ""
    risk: str = "low"             # "low" | "medium" | "high"


class ConfigTuneResponse(BaseModel):
    """Response from POST /ai/recommend-config."""
    lob: str
    recommendations: List[ConfigRecommendationItem] = []
    overall_assessment: str = ""
    risk_summary: str = ""


class CommentaryRequest(BaseModel):
    """Request body for POST /ai/commentary."""
    lob: str
    run_type: str = "backtest"
    period_start: Optional[date] = None
    period_end: Optional[date] = None


class KeyMetricItem(BaseModel):
    """A key metric with trend direction."""
    name: str
    value: float
    unit: str = ""
    trend: str = "stable"         # "improving" | "stable" | "degrading"


class CommentaryResponse(BaseModel):
    """Response from POST /ai/commentary."""
    lob: str
    executive_summary: str = ""
    key_metrics: List[KeyMetricItem] = []
    exceptions: List[str] = []
    action_items: List[str] = []
