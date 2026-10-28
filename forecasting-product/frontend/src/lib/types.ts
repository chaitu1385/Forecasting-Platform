// ──────────────────────────────────────────────────────────────────────────────
// TypeScript interfaces matching FastAPI response schemas (src/api/schemas.py)
// ──────────────────────────────────────────────────────────────────────────────

// ── Auth ─────────────────────────────────────────────────────────────────────

export type Role = "admin" | "data_scientist" | "planner" | "manager" | "viewer";

export interface User {
  username: string;
  role: Role;
  permissions: string[];
}

export interface AuthStatus {
  mode: "production" | "demo";
  auth_required: boolean;
}

export interface AuthIdentity extends User {
  mode: "production" | "demo";
  email: string;
  expires_at: number | null;
}

// ── Health ────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version: string;
  checks?: Record<string, boolean>;
}

// ── Forecast ─────────────────────────────────────────────────────────────────

export interface ForecastPoint {
  series_id: string;
  week: string;
  forecast: number | null;
  model?: string;
  lob?: string;
  forecast_p10?: number | null;
  forecast_p50?: number | null;
  forecast_p90?: number | null;
}

export interface ForecastResponse {
  lob: string;
  series_count: number;
  forecast_origin?: string;
  points: ForecastPoint[];
  run_id?: string | null;
  artifact_source?: "committed" | "legacy";
  config_hash?: string | null;
  input_data_hash?: string | null;
}

// ── Metrics / Leaderboard ────────────────────────────────────────────────────

export interface LeaderboardEntry {
  model: string;
  wmape: number | null;
  normalized_bias: number | null;
  rank: number;
  n_series: number;
  scores?: Record<string, number | null>;
  primary_score?: number | null;
  group_key?: string | null;
  competition_scope?: string | null;
}

export interface ScoringProvenance {
  source: "stored" | "computed";
  status: "recorded" | "unknown";
  version?: number | null;
  policy?: string | null;
  primary_metric?: string | null;
  secondary_metric?: string | null;
  competition_scope?: string | null;
}

export interface LeaderboardResponse {
  lob: string;
  run_type: string;
  entries: LeaderboardEntry[];
  run_id?: string | null;
  artifact_source?: "committed" | "legacy";
  scoring_provenance?: ScoringProvenance;
}

// ── Drift ────────────────────────────────────────────────────────────────────

export interface DriftAlertItem {
  series_id: string;
  metric: string;
  severity: "warning" | "critical";
  current_value: number;
  baseline_value: number;
  message: string;
}

export interface DriftResponse {
  lob: string;
  n_critical: number;
  n_warning: number;
  alerts: DriftAlertItem[];
}

// ── Analysis ─────────────────────────────────────────────────────────────────

export interface AnalysisResponse {
  lob_name: string;
  time_column: string;
  target_column: string;
  id_columns: string[];
  n_series: number;
  n_rows: number;
  date_range_start: string;
  date_range_end: string;
  frequency: string;
  overall_forecastability: number;
  forecastability_distribution: Record<string, number>;
  demand_classes: Record<string, number>;
  detected_hierarchies: Record<string, unknown>[];
  recommended_config_yaml: string;
  config_reasoning: string[];
  hypotheses: string[];
  llm_narrative?: string;
  llm_risk_factors?: string[];
}

// ── AI: NL Query ─────────────────────────────────────────────────────────────

export interface NLQueryRequest {
  series_id: string;
  question: string;
  run_id?: string;
  legacy?: boolean;
  lob: string;
}

export interface NLQueryResponse {
  answer: string;
  supporting_data: Record<string, unknown>;
  confidence: "high" | "medium" | "low";
  sources_used: string[];
}

// ── AI: Triage ───────────────────────────────────────────────────────────────

export interface TriageRequest {
  lob: string;
  run_type?: string;
  severity_filter?: string;
  max_alerts?: number;
}

export interface TriagedAlertItem {
  series_id: string;
  metric: string;
  severity: string;
  business_impact_score: number;
  suggested_action: string;
  reasoning: string;
  original_message: string;
}

export interface TriageResponse {
  lob: string;
  executive_summary: string;
  total_alerts: number;
  critical_count: number;
  warning_count: number;
  ranked_alerts: TriagedAlertItem[];
}

// ── AI: Config Tuner ─────────────────────────────────────────────────────────

export interface ConfigTuneRequest {
  lob: string;
  run_type?: string;
}

export interface ConfigRecommendationItem {
  field_path: string;
  current_value: unknown;
  recommended_value: unknown;
  reasoning: string;
  expected_impact: string;
  risk: "low" | "medium" | "high";
}

export interface ConfigTuneResponse {
  lob: string;
  recommendations: ConfigRecommendationItem[];
  overall_assessment: string;
  risk_summary: string;
}

// ── AI: Commentary ───────────────────────────────────────────────────────────

export interface CommentaryRequest {
  lob: string;
  run_type?: string;
  period_start?: string;
  period_end?: string;
}

export interface KeyMetricItem {
  name: string;
  value: number;
  unit: string;
  trend: "improving" | "stable" | "degrading";
}

export interface CommentaryResponse {
  lob: string;
  executive_summary: string;
  key_metrics: KeyMetricItem[];
  exceptions: string[];
  action_items: string[];
}

// ── Series ───────────────────────────────────────────────────────────────────

export interface SeriesItem {
  series_id: string;
  adi: number;
  cv2: number;
  demand_class: string;
  is_sparse: boolean;
  n_observations: number;
}

export interface SeriesListResponse {
  lob: string;
  series_count: number;
  series: SeriesItem[];
}

export interface SeriesHistoryPoint {
  week: string;
  value: number | null;
}

export interface SeriesHistoryResponse {
  run_id?: string | null;
  artifact_source?: "committed" | "legacy";
  input_stage?: "input" | "prepared" | "unknown";
  series_id: string;
  lob: string;
  points: SeriesHistoryPoint[];
}

export interface BreakDetectionResponse {
  total_series: number;
  series_with_breaks: number;
  total_breaks: number;
  warnings: string[];
  per_series: Record<string, unknown>[];
}

export interface CleansingAuditResponse {
  total_series: number;
  series_with_outliers: number;
  total_outliers: number;
  outlier_pct: number;
  series_with_stockouts: number;
  total_stockout_periods: number;
  total_stockout_weeks: number;
  excluded_period_weeks: number;
  rows_modified: number;
  per_series: Record<string, unknown>[];
  cleansed_preview: Record<string, unknown>[];
}

export interface RegressorScreenResponse {
  screened_columns: string[];
  dropped_columns: string[];
  low_variance_columns: string[];
  high_correlation_pairs: Record<string, unknown>[];
  low_mi_columns: string[];
  warnings: string[];
  per_column_stats: Record<string, Record<string, unknown>>;
}

// ── Hierarchy ────────────────────────────────────────────────────────────────

export interface HierarchyTreeNode {
  key: string;
  level: string;
  parent: string;
  is_leaf: boolean;
}

export interface HierarchyBuildResponse {
  name: string;
  levels: string[];
  level_stats: { level: string; node_count: number }[];
  total_nodes: number;
  leaf_count: number;
  s_matrix_shape: [number, number];
  s_matrix_sample: Record<string, unknown>[];
  tree_nodes: HierarchyTreeNode[];
}

export interface HierarchyAggregateResponse {
  target_level: string;
  total_rows: number;
  unique_nodes: number;
  top_n_data: Record<string, unknown>[];
}

export interface HierarchyReconcileResponse {
  method: string;
  before_total: number;
  after_total: number;
  rows: number;
  reconciled_preview: Record<string, unknown>[];
}

// ── SKU Mapping ──────────────────────────────────────────────────────────────

export interface SKUMappingResponse {
  phase: number;
  total_mappings: number;
  mappings: Record<string, unknown>[];
}

// ── Overrides ────────────────────────────────────────────────────────────────

export interface OverrideItem {
  override_id: string;
  old_sku: string;
  new_sku: string;
  proportion: number;
  scenario: string;
  ramp_shape: string;
  effective_date?: string | null;
  created_by?: string | null;
  created_at?: string | null;
  notes?: string | null;
  status?: string;
  approved_by?: string | null;
  approved_at?: string | null;
}

export interface OverrideListResponse {
  count: number;
  overrides: OverrideItem[];
}

export interface CreateOverrideRequest {
  old_sku: string;
  new_sku: string;
  proportion: number;
  scenario?: string;
  ramp_shape?: string;
  effective_date?: string | null;
  notes?: string | null;
}

export interface UpdateOverrideRequest {
  proportion?: number;
  scenario?: string;
  ramp_shape?: string;
  effective_date?: string | null;
  notes?: string | null;
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

export interface EffectivePipelineConfig {
  frequency: "D" | "W" | "M" | "Q";
  season_length: number;
  series_id_column: string;
  time_column: string;
  target_column: string;
  forecasters: string[];
  intermittent_forecasters: string[];
  quantiles: number[];
  fill_gaps: boolean;
  fill_value: number;
  min_series_length_periods: number;
  drop_zero_series: boolean;
  input_validation_enabled: boolean;
  cleansing_enabled: boolean;
  constraints_enabled: boolean;
  post_validation_enabled: boolean;
  halt_on_blocker: boolean;
  min_grade: string;
  backtest_n_folds: number;
  backtest_val_periods: number;
  backtest_gap_periods: number;
  selection_strategy: string;
  champion_granularity: string;
  primary_metric: string;
  secondary_metric: string;
}

export interface PipelineExecutionSettings {
  model_id: string | null;
  model_source: "request" | "config" | "default" | "backtest";
  horizon_periods: number;
  horizon_source: "request" | "config" | "default";
  config_source: "upload" | "default";
  config: EffectivePipelineConfig;
}

export interface PipelineRunResponse {
  lob: string;
  status: string;
  run_id: string;
  run_type: "forecast" | "backtest";
  links: RunLinks;
  execution: PipelineExecutionSettings;
  champion_model?: string;
  best_wmape?: number | null;
  leaderboard?: Record<string, unknown>[];
  model_id?: string;
  horizon?: number;
  forecast_rows?: number;
  series_count?: number;
  forecast_preview?: Record<string, unknown>[];
  validation_grade?: string;
  validation_score?: number;
}

export interface ManifestItem {
  run_id: string | null;
  run_type?: "forecast" | "backtest" | "unknown";
  artifact_source?: "committed" | "legacy";
  lineage_status?: "verified_bundle" | "unknown";
  timestamp: string;
  lob: string;
  series_count: number;
  champion_model: string;
  backtest_wmape?: number;
  forecast_horizon: number;
  forecast_rows: number;
  validation_passed: boolean | null;
  validation_warnings: number;
  cleansing_applied: boolean;
  outliers_clipped: number;
}

export interface RunLinks {
  metadata: string;
  config: string;
  input_lineage: string;
  forecast?: string | null;
  leaderboard?: string | null;
}

export interface CommittedRun extends ManifestItem {
  run_id: string;
  run_type: "forecast" | "backtest";
  artifact_source: "committed";
  input_data_hash: string;
  config_hash: string;
  links: RunLinks;
}

export interface RunListResponse {
  count: number;
  runs: CommittedRun[];
}

export interface RunConfigResponse {
  run_id: string;
  lob: string;
  run_type: "forecast" | "backtest";
  config_hash: string;
  configuration: Record<string, unknown>;
}

export interface ManifestListResponse {
  count: number;
  manifests: ManifestItem[];
}

export interface CostItem {
  run_id: string | null;
  timestamp: string;
  lob: string;
  series_count: number;
  champion_model: string;
  total_seconds: number | null;
  seconds_per_series?: number | null;
}

export interface CostListResponse {
  count: number;
  costs: CostItem[];
}

export interface MultiFileProfile {
  filename: string;
  role: string;
  confidence: number;
  time_column?: string;
  id_columns: string[];
  n_rows: number;
  n_columns: number;
  reasoning: string[];
}

export interface MultiFileAnalysisResponse {
  profiles: MultiFileProfile[];
  primary_file?: string;
  dimension_files: string[];
  regressor_files: string[];
  warnings: string[];
  merge_preview?: {
    total_rows: number;
    total_columns: number;
    matched_rows: number;
    unmatched_primary_keys: number;
    null_fill_columns: string[];
    warnings: string[];
    sample_rows: Record<string, unknown>[];
  };
  merge_error?: string;
}

// ── Analytics ────────────────────────────────────────────────────────────────

export interface FVAResponse {
  lob: string;
  summary: Record<string, unknown>[];
  layer_leaderboard: Record<string, unknown>[];
  detail_preview: Record<string, unknown>[];
}

export interface CalibrationCoverage {
  label: string;
  nominal: number;
  empirical: number;
  miscalibration: number;
  sharpness: number;
  n_observations: number;
}

export interface CalibrationResponse {
  lob: string;
  model_reports: Record<string, CalibrationCoverage[]>;
  per_series_preview: Record<string, unknown>[];
}

export interface ShapResponse {
  lob: string;
  model: string;
  feature_importance: { feature: string; mean_abs_value: number; std: number }[];
  decomposition_preview: Record<string, unknown>[];
}

export interface DecomposeResponse {
  decomposition: Record<string, unknown>[];
  narratives: Record<string, string>;
  series_count: number;
}

export interface ForecastCompareResponse {
  comparison: Record<string, unknown>[];
  summary: Record<string, unknown>[];
}

export interface ConstrainResponse {
  before_total: number;
  after_total: number;
  rows_modified: number;
  constraints_applied: {
    min_demand: number;
    max_capacity?: number;
    aggregate_max?: number;
    proportional: boolean;
  };
  constrained_preview: Record<string, unknown>[];
}

// ── Governance ───────────────────────────────────────────────────────────────

export interface ModelCardItem {
  model_name: string;
  lob: string;
  training_start?: string;
  training_end?: string;
  n_series: number;
  n_observations: number;
  backtest_wmape?: number;
  backtest_bias?: number;
  champion_since?: string;
  features: string[];
  config_hash: string;
  notes: string;
}

export interface ModelCardListResponse {
  count: number;
  model_cards: ModelCardItem[];
}

export interface LineageEntry {
  timestamp: string;
  model_id: string;
  lob: string;
  n_series: number;
  horizon_weeks: number;
  selection_strategy: string;
  run_id: string | null;
  notes: string;
  user_id: string | null;
}

export interface LineageResponse {
  count: number;
  lineage: LineageEntry[];
}

export interface BIExportResponse {
  report_type: string;
  export_path: string;
  status: string;
  run_id?: string | null;
  actuals_run_id?: string | null;
  artifact_source?: "committed" | "legacy";
}

// ── Audit ────────────────────────────────────────────────────────────────────

export interface AuditEvent {
  audit_id: string;
  timestamp: string;
  user_id: string;
  user_email: string;
  user_role: string;
  action: string;
  resource_type: string;
  resource_id: string;
  status: string;
  old_value?: string;
  new_value?: string;
  ip_address: string;
  request_id: string;
  error_message?: string;
}

export interface AuditResponse {
  count: number;
  events: AuditEvent[];
}
