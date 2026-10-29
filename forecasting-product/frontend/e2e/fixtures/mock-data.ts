// ──────────────────────────────────────────────────────────────────────────────
// Centralized mock API responses matching TypeScript interfaces from src/lib/types.ts
// ──────────────────────────────────────────────────────────────────────────────

import type {
  HealthResponse,
  ForecastResponse,
  LeaderboardResponse,
  DriftResponse,
  AnalysisResponse,
  SeriesListResponse,
  AuditResponse,
  ManifestListResponse,
  CostListResponse,
  FVAResponse,
  CalibrationResponse,
  DecompositionSummaryResponse,
  ShapResponse,
  ModelCardListResponse,
  LineageResponse,
  OverrideListResponse,
  CommentaryResponse,
  BreakDetectionResponse,
  CleansingAuditResponse,
  RegressorScreenResponse,
} from "@/lib/types";

// ── Health ────────────────────────────────────────────────────────────────────

export const mockHealth: HealthResponse = {
  status: "ok",
  version: "1.0.0",
};

// ── Forecast ─────────────────────────────────────────────────────────────────

function generateForecastPoints() {
  const series = ["sku_001", "sku_002", "sku_003"];
  const models = ["auto_arima", "lgbm_direct", "auto_ets"];
  const points = [];
  for (const sid of series) {
    for (let w = 0; w < 12; w++) {
      const week = `2025-${String(Math.floor(w / 4) + 1).padStart(2, "0")}-${String((w % 4) * 7 + 1).padStart(2, "0")}`;
      points.push({
        series_id: sid,
        week,
        forecast: 100 + Math.round(Math.sin(w) * 20),
        model: models[series.indexOf(sid)],
        lob: "retail",
      });
    }
  }
  return points;
}

export const mockForecast: ForecastResponse = {
  lob: "retail",
  series_count: 3,
  forecast_origin: "2024-12-31",
  points: generateForecastPoints(),
};

// ── Leaderboard ──────────────────────────────────────────────────────────────

export const mockLeaderboard: LeaderboardResponse = {
  lob: "retail",
  run_type: "backtest",
  entries: [
    { model: "lgbm_direct", wmape: 0.082, normalized_bias: -0.012, rank: 1, n_series: 50 },
    { model: "auto_arima", wmape: 0.095, normalized_bias: 0.005, rank: 2, n_series: 50 },
    { model: "auto_ets", wmape: 0.112, normalized_bias: -0.031, rank: 3, n_series: 50 },
    { model: "seasonal_naive", wmape: 0.162, normalized_bias: 0.048, rank: 4, n_series: 50 },
  ],
};

// ── Drift ────────────────────────────────────────────────────────────────────

export const mockDrift: DriftResponse = {
  lob: "retail",
  n_critical: 1,
  n_warning: 2,
  alerts: [
    {
      series_id: "sku_001",
      metric: "wmape",
      severity: "critical",
      current_value: 0.35,
      baseline_value: 0.12,
      message: "WMAPE increased by 192% from baseline",
    },
    {
      series_id: "sku_002",
      metric: "normalized_bias",
      severity: "warning",
      current_value: 0.15,
      baseline_value: 0.05,
      message: "Bias drifted upward significantly",
    },
    {
      series_id: "sku_003",
      metric: "wmape",
      severity: "warning",
      current_value: 0.22,
      baseline_value: 0.14,
      message: "WMAPE increased by 57% from baseline",
    },
  ],
};

// ── Series List ──────────────────────────────────────────────────────────────

export const mockSeriesList: SeriesListResponse = {
  lob: "retail",
  series_count: 5,
  series: [
    { series_id: "sku_001", adi: 1.2, cv2: 0.3, demand_class: "smooth", is_sparse: false, n_observations: 104 },
    { series_id: "sku_002", adi: 1.5, cv2: 0.8, demand_class: "erratic", is_sparse: false, n_observations: 96 },
    { series_id: "sku_003", adi: 2.8, cv2: 0.4, demand_class: "intermittent", is_sparse: true, n_observations: 78 },
    { series_id: "sku_004", adi: 3.1, cv2: 1.2, demand_class: "lumpy", is_sparse: true, n_observations: 65 },
    { series_id: "sku_005", adi: 1.1, cv2: 0.2, demand_class: "smooth", is_sparse: false, n_observations: 110 },
  ],
};

// ── Break Detection ──────────────────────────────────────────────────────────

export const mockBreakDetection: BreakDetectionResponse = {
  total_series: 5,
  series_with_breaks: 2,
  total_breaks: 3,
  warnings: [],
  per_series: [
    { series_id: "sku_001", breaks: [{ index: 52, date: "2024-01-08", statistic: 4.2 }] },
    { series_id: "sku_003", breaks: [{ index: 30, date: "2023-07-17", statistic: 3.8 }, { index: 78, date: "2024-07-01", statistic: 5.1 }] },
  ],
};

// ── Cleansing Audit ──────────────────────────────────────────────────────────

export const mockCleansingAudit: CleansingAuditResponse = {
  total_series: 5,
  series_with_outliers: 2,
  total_outliers: 8,
  outlier_pct: 1.5,
  series_with_stockouts: 1,
  total_stockout_periods: 2,
  total_stockout_weeks: 6,
  excluded_period_weeks: 0,
  rows_modified: 14,
  per_series: [
    { series_id: "sku_001", outliers_detected: 3, stockout_periods: 0 },
    { series_id: "sku_002", outliers_detected: 5, stockout_periods: 2 },
  ],
  cleansed_preview: [
    { series_id: "sku_001", week: "2024-03-04", original: 250, cleansed: 180, action: "outlier_clipped" },
  ],
};

// ── Regressor Screen ─────────────────────────────────────────────────────────

export const mockRegressorScreen: RegressorScreenResponse = {
  screened_columns: ["temperature", "promo_flag", "holiday"],
  dropped_columns: ["wind_speed", "humidity"],
  low_variance_columns: ["wind_speed"],
  high_correlation_pairs: [{ col_a: "temperature", col_b: "humidity", correlation: 0.92 }],
  low_mi_columns: ["humidity"],
  warnings: [],
  per_column_stats: {
    temperature: { variance: 120.5, mi_score: 0.45, correlation_with_target: 0.62 },
    promo_flag: { variance: 0.25, mi_score: 0.78, correlation_with_target: 0.55 },
  },
};

// ── Audit ────────────────────────────────────────────────────────────────────

export const mockAudit: AuditResponse = {
  count: 5,
  events: [
    {
      audit_id: "aud_001",
      timestamp: "2025-01-15T10:30:00Z",
      user_id: "admin",
      user_email: "admin@example.com",
      user_role: "admin",
      action: "run_backtest",
      resource_type: "pipeline",
      resource_id: "run_abc123",
      status: "success",
      ip_address: "192.168.1.1",
      request_id: "req_001",
    },
    {
      audit_id: "aud_002",
      timestamp: "2025-01-15T11:00:00Z",
      user_id: "planner1",
      user_email: "planner@example.com",
      user_role: "planner",
      action: "create_override",
      resource_type: "override",
      resource_id: "ovr_001",
      status: "success",
      ip_address: "192.168.1.2",
      request_id: "req_002",
    },
    {
      audit_id: "aud_003",
      timestamp: "2025-01-15T12:00:00Z",
      user_id: "ds_user",
      user_email: "ds@example.com",
      user_role: "data_scientist",
      action: "run_forecast",
      resource_type: "pipeline",
      resource_id: "run_def456",
      status: "success",
      ip_address: "192.168.1.3",
      request_id: "req_003",
    },
    {
      audit_id: "aud_004",
      timestamp: "2025-01-14T09:00:00Z",
      user_id: "admin",
      user_email: "admin@example.com",
      user_role: "admin",
      action: "promote_model",
      resource_type: "model",
      resource_id: "lgbm_direct",
      status: "success",
      ip_address: "192.168.1.1",
      request_id: "req_004",
    },
    {
      audit_id: "aud_005",
      timestamp: "2025-01-14T14:00:00Z",
      user_id: "manager1",
      user_email: "manager@example.com",
      user_role: "manager",
      action: "approve_override",
      resource_type: "override",
      resource_id: "ovr_002",
      status: "success",
      ip_address: "192.168.1.4",
      request_id: "req_005",
    },
  ],
};

// ── Manifests ────────────────────────────────────────────────────────────────

export const mockManifests: ManifestListResponse = {
  count: 3,
  manifests: [
    {
      run_id: "run_2025_01_15_001",
      timestamp: "2025-01-15T10:30:00Z",
      lob: "retail",
      series_count: 50,
      champion_model: "lgbm_direct",
      backtest_wmape: 0.082,
      forecast_horizon: 12,
      forecast_rows: 600,
      validation_passed: true,
      validation_warnings: 0,
      cleansing_applied: true,
      outliers_clipped: 8,
    },
    {
      run_id: "run_2025_01_14_001",
      timestamp: "2025-01-14T09:00:00Z",
      lob: "retail",
      series_count: 50,
      champion_model: "auto_arima",
      backtest_wmape: 0.095,
      forecast_horizon: 12,
      forecast_rows: 600,
      validation_passed: true,
      validation_warnings: 2,
      cleansing_applied: true,
      outliers_clipped: 5,
    },
    {
      run_id: "run_2025_01_13_001",
      timestamp: "2025-01-13T15:00:00Z",
      lob: "retail",
      series_count: 48,
      champion_model: "lgbm_direct",
      backtest_wmape: 0.088,
      forecast_horizon: 12,
      forecast_rows: 576,
      validation_passed: false,
      validation_warnings: 5,
      cleansing_applied: false,
      outliers_clipped: 0,
    },
  ],
};

// ── Costs ────────────────────────────────────────────────────────────────────

export const mockCosts: CostListResponse = {
  count: 3,
  costs: [
    {
      run_id: "run_2025_01_15_001",
      timestamp: "2025-01-15T10:30:00Z",
      lob: "retail",
      series_count: 50,
      champion_model: "lgbm_direct",
      total_seconds: 145.2,
      seconds_per_series: 2.9,
    },
    {
      run_id: "run_2025_01_14_001",
      timestamp: "2025-01-14T09:00:00Z",
      lob: "retail",
      series_count: 50,
      champion_model: "auto_arima",
      total_seconds: 230.5,
      seconds_per_series: 4.6,
    },
    {
      run_id: "run_2025_01_13_001",
      timestamp: "2025-01-13T15:00:00Z",
      lob: "retail",
      series_count: 48,
      champion_model: "lgbm_direct",
      total_seconds: 138.8,
      seconds_per_series: 2.9,
    },
  ],
};

// ── FVA ──────────────────────────────────────────────────────────────────────

export const mockFVA: FVAResponse = {
  lob: "retail",
  summary: [
    { layer: "naive", avg_wmape: 0.162, series_count: 50 },
    { layer: "statistical", avg_wmape: 0.095, series_count: 50, fva_pct: 41.4 },
    { layer: "ml", avg_wmape: 0.082, series_count: 50, fva_pct: 13.7 },
    { layer: "champion", avg_wmape: 0.079, series_count: 50, fva_pct: 3.7 },
  ],
  layer_leaderboard: [
    { model: "lgbm_direct", layer: "ml", wmape: 0.082, rank: 1 },
    { model: "auto_arima", layer: "statistical", wmape: 0.095, rank: 2 },
    { model: "auto_ets", layer: "statistical", wmape: 0.112, rank: 3 },
  ],
  detail_preview: [
    { series_id: "sku_001", naive_wmape: 0.18, champion_wmape: 0.07, fva_pct: 61.1 },
  ],
};

// ── Calibration ──────────────────────────────────────────────────────────────

export const mockCalibration: CalibrationResponse = {
  lob: "retail",
  model_reports: {
    lgbm_direct: [
      { label: "50%", nominal: 0.5, empirical: 0.48, miscalibration: 0.02, sharpness: 15.3, n_observations: 600 },
      { label: "80%", nominal: 0.8, empirical: 0.76, miscalibration: 0.04, sharpness: 28.7, n_observations: 600 },
      { label: "95%", nominal: 0.95, empirical: 0.93, miscalibration: 0.02, sharpness: 45.1, n_observations: 600 },
    ],
    auto_arima: [
      { label: "50%", nominal: 0.5, empirical: 0.52, miscalibration: 0.02, sharpness: 18.1, n_observations: 600 },
      { label: "80%", nominal: 0.8, empirical: 0.78, miscalibration: 0.02, sharpness: 32.4, n_observations: 600 },
      { label: "95%", nominal: 0.95, empirical: 0.91, miscalibration: 0.04, sharpness: 51.2, n_observations: 600 },
    ],
  },
  per_series_preview: [],
};

// ── Decomposition ────────────────────────────────────────────────────────────

export const mockDecomposition: DecompositionSummaryResponse = {
  lob: "retail", run_id: null, artifact_source: "legacy",
  forecast_model: "lgbm_direct", forecast_model_source: "forecast_frame",
  history_source: "legacy", history_stage: "unknown", history_run_id: null,
  method: "moving_average_decomposition", model_specific: false,
  season_length: 3, trend_window: 3, seasonal_basis: "observation_position",
  statistics_population: "history_rows_for_forecast_series", std_ddof: 1,
  series_count: 1, history_rows: 12, excluded_history_rows: 0, forecast_rows: 3,
  checked_rows: 15, preview_rows: 1, preview_truncated: true,
  components: [
    { component: "trend", mean_abs_value: 155, std: 30, defined_count: 10, undefined_count: 2 },
    { component: "seasonal", mean_abs_value: 2, std: 2.5, defined_count: 12, undefined_count: 0 },
    { component: "residual", mean_abs_value: 0, std: 0, defined_count: 10, undefined_count: 2 },
  ],
  decomposition_preview: [{
    series_id: "sku_001", week: "2025-01-06", value: 97,
    trend: null, seasonal: -3, residual: null, is_forecast: false,
  }],
};

export const mockShap: ShapResponse = {
  ...mockDecomposition,
  model: mockDecomposition.forecast_model,
  feature_importance: mockDecomposition.components.map(({ component, ...stats }) => ({
    feature: component, ...stats,
  })),
};

// ── Model Cards ──────────────────────────────────────────────────────────────

export const mockModelCards: ModelCardListResponse = {
  count: 2,
  model_cards: [
    {
      model_name: "lgbm_direct",
      lob: "retail",
      training_start: "2023-01-02",
      training_end: "2024-12-30",
      n_series: 50,
      n_observations: 5200,
      backtest_wmape: 0.082,
      backtest_bias: -0.012,
      champion_since: "2025-01-10",
      features: ["lag_1", "rolling_mean_4", "month_sin", "promo_flag"],
      config_hash: "abc123def",
      notes: "Champion model selected via walk-forward backtest",
    },
    {
      model_name: "auto_arima",
      lob: "retail",
      training_start: "2023-01-02",
      training_end: "2024-12-30",
      n_series: 50,
      n_observations: 5200,
      backtest_wmape: 0.095,
      backtest_bias: 0.005,
      champion_since: "2024-11-01",
      features: [],
      config_hash: "xyz789abc",
      notes: "Statistical baseline model",
    },
  ],
};

// ── Lineage ──────────────────────────────────────────────────────────────────

export const mockLineage: LineageResponse = {
  count: 3,
  lineage: [
    {
      timestamp: "2025-01-15T10:30:00Z",
      model_id: "lgbm_direct",
      lob: "retail",
      n_series: 50,
      horizon_weeks: 12,
      selection_strategy: "walk_forward",
      run_id: "run_2025_01_15_001",
      notes: "Promoted as champion after backtest",
      user_id: "admin",
    },
    {
      timestamp: "2025-01-10T09:00:00Z",
      model_id: "auto_arima",
      lob: "retail",
      n_series: 50,
      horizon_weeks: 12,
      selection_strategy: "walk_forward",
      run_id: "run_2025_01_10_001",
      notes: "Previous champion",
      user_id: "ds_user",
    },
    {
      timestamp: "2024-12-20T14:00:00Z",
      model_id: "auto_ets",
      lob: "retail",
      n_series: 48,
      horizon_weeks: 12,
      selection_strategy: "walk_forward",
      run_id: "run_2024_12_20_001",
      notes: "Initial deployment",
      user_id: "ds_user",
    },
  ],
};

// ── Overrides ────────────────────────────────────────────────────────────────

export const mockOverrides: OverrideListResponse = {
  count: 2,
  overrides: [
    {
      override_id: "ovr_001",
      old_sku: "SKU-OLD-100",
      new_sku: "SKU-NEW-200",
      proportion: 0.75,
      scenario: "base",
      ramp_shape: "linear",
      effective_date: "2025-02-01",
      created_by: "planner1",
      notes: "Product line refresh",
    },
    {
      override_id: "ovr_002",
      old_sku: "SKU-OLD-300",
      new_sku: "SKU-NEW-400",
      proportion: 1.0,
      scenario: "optimistic",
      ramp_shape: "step",
      effective_date: "2025-03-01",
      created_by: "planner1",
      notes: "Full replacement",
    },
  ],
};

// ── Commentary ───────────────────────────────────────────────────────────────

export const mockCommentary: CommentaryResponse = {
  lob: "retail",
  executive_summary:
    "Overall forecast accuracy improved 3.2% week-over-week. LightGBM remains the champion model across 50 series with 8.2% WMAPE.",
  key_metrics: [
    { name: "Overall WMAPE", value: 0.082, unit: "%", trend: "improving" },
    { name: "Bias", value: -0.012, unit: "%", trend: "stable" },
    { name: "Series at Risk", value: 3, unit: "count", trend: "improving" },
  ],
  exceptions: ["SKU_001 shows significant drift — investigate recent promotion impact"],
  action_items: [
    "Review SKU_001 forecast override",
    "Schedule model retrain for Q2 data refresh",
  ],
};

// ── Analysis ─────────────────────────────────────────────────────────────────

export const mockAnalysis: AnalysisResponse = {
  lob_name: "retail",
  time_column: "week",
  target_column: "quantity",
  id_columns: ["series_id"],
  n_series: 50,
  n_rows: 5200,
  date_range_start: "2023-01-02",
  date_range_end: "2024-12-30",
  frequency: "W",
  overall_forecastability: 0.72,
  forecastability_distribution: { high: 0.4, medium: 0.35, low: 0.25 },
  demand_classes: { smooth: 30, erratic: 10, intermittent: 7, lumpy: 3 },
  detected_hierarchies: [{ levels: ["category", "region"], depth: 2 }],
  recommended_config_yaml:
    "forecast:\n  frequency: W\n  horizon_periods: 12\n  models:\n    - auto_arima\n    - lgbm_direct\n",
  config_reasoning: [
    "Weekly frequency detected from date intervals",
    "12-week horizon recommended for retail S&OP cycles",
    "Mix of smooth and intermittent series suggests ensemble approach",
  ],
  hypotheses: [
    "Promotional events may cause demand spikes in smooth series",
    "Intermittent series may benefit from Croston-family models",
  ],
};
