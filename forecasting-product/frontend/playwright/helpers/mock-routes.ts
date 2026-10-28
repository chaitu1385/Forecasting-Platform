import type { Page } from "@playwright/test";
import { ROLE_PERMISSIONS } from "../../src/lib/constants";
import type { Role } from "../../src/lib/types";

// ──────────────────────────────────────────────────────────────────────────────
// Mock API responses for Playwright tests.
// Uses page.route() to intercept network requests to the FastAPI backend.
// Data shapes mirror e2e/fixtures/mock-data.ts for consistency.
// ──────────────────────────────────────────────────────────────────────────────

const API = "http://localhost:8000";

// ── Fixture data ─────────────────────────────────────────────────────────────

function forecastPoints() {
  const series = ["sku_001", "sku_002", "sku_003"];
  const models = ["auto_arima", "lgbm_direct", "auto_ets"];
  const points: Record<string, unknown>[] = [];
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

const MOCK = {
  health: { status: "ok", version: "1.0.0" },

  token: { access_token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0X3VzZXIiLCJyb2xlIjoiYWRtaW4iLCJleHAiOjk5OTk5OTk5OTl9.mock", token_type: "bearer" },

  forecast: {
    lob: "retail",
    series_count: 3,
    forecast_origin: "2024-12-31",
    points: forecastPoints(),
  },

  leaderboard: {
    lob: "retail",
    run_type: "backtest",
    entries: [
      { model: "lgbm_direct", wmape: 0.082, normalized_bias: -0.012, rank: 1, n_series: 50 },
      { model: "auto_arima", wmape: 0.095, normalized_bias: 0.005, rank: 2, n_series: 50 },
      { model: "auto_ets", wmape: 0.112, normalized_bias: -0.031, rank: 3, n_series: 50 },
      { model: "seasonal_naive", wmape: 0.162, normalized_bias: 0.048, rank: 4, n_series: 50 },
    ],
  },

  drift: {
    lob: "retail",
    n_critical: 1,
    n_warning: 2,
    alerts: [
      { series_id: "sku_001", metric: "wmape", severity: "critical", current_value: 0.35, baseline_value: 0.12, message: "WMAPE increased by 192% from baseline" },
      { series_id: "sku_002", metric: "normalized_bias", severity: "warning", current_value: 0.15, baseline_value: 0.05, message: "Bias drifted upward significantly" },
      { series_id: "sku_003", metric: "wmape", severity: "warning", current_value: 0.22, baseline_value: 0.14, message: "WMAPE increased by 57% from baseline" },
    ],
  },

  seriesList: {
    lob: "retail",
    series_count: 5,
    series: [
      { series_id: "sku_001", adi: 1.2, cv2: 0.3, demand_class: "smooth", is_sparse: false, n_observations: 104 },
      { series_id: "sku_002", adi: 1.5, cv2: 0.8, demand_class: "erratic", is_sparse: false, n_observations: 96 },
      { series_id: "sku_003", adi: 2.8, cv2: 0.4, demand_class: "intermittent", is_sparse: true, n_observations: 78 },
      { series_id: "sku_004", adi: 3.1, cv2: 1.2, demand_class: "lumpy", is_sparse: true, n_observations: 65 },
      { series_id: "sku_005", adi: 1.1, cv2: 0.2, demand_class: "smooth", is_sparse: false, n_observations: 110 },
    ],
  },

  breakDetection: {
    total_series: 5,
    series_with_breaks: 2,
    total_breaks: 3,
    warnings: [],
    per_series: [
      { series_id: "sku_001", breaks: [{ index: 52, date: "2024-01-08", statistic: 4.2 }] },
      { series_id: "sku_003", breaks: [{ index: 30, date: "2023-07-17", statistic: 3.8 }, { index: 78, date: "2024-07-01", statistic: 5.1 }] },
    ],
  },

  cleansingAudit: {
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
  },

  regressorScreen: {
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
  },

  audit: {
    count: 5,
    events: [
      { audit_id: "aud_001", timestamp: "2025-01-15T10:30:00Z", user_id: "admin", user_email: "admin@example.com", user_role: "admin", action: "run_backtest", resource_type: "pipeline", resource_id: "run_abc123", status: "success", ip_address: "192.168.1.1", request_id: "req_001" },
      { audit_id: "aud_002", timestamp: "2025-01-15T11:00:00Z", user_id: "planner1", user_email: "planner@example.com", user_role: "planner", action: "create_override", resource_type: "override", resource_id: "ovr_001", status: "success", ip_address: "192.168.1.2", request_id: "req_002" },
    ],
  },

  manifests: {
    count: 3,
    manifests: [
      { run_id: "run_2025_01_15_001", timestamp: "2025-01-15T10:30:00Z", lob: "retail", series_count: 50, champion_model: "lgbm_direct", backtest_wmape: 0.082, forecast_horizon: 12, forecast_rows: 600, validation_passed: true, validation_warnings: 0, cleansing_applied: true, outliers_clipped: 8 },
      { run_id: "run_2025_01_14_001", timestamp: "2025-01-14T09:00:00Z", lob: "retail", series_count: 50, champion_model: "auto_arima", backtest_wmape: 0.095, forecast_horizon: 12, forecast_rows: 600, validation_passed: true, validation_warnings: 2, cleansing_applied: true, outliers_clipped: 5 },
    ],
  },

  costs: {
    count: 3,
    costs: [
      { run_id: "run_2025_01_15_001", timestamp: "2025-01-15T10:30:00Z", lob: "retail", series_count: 50, champion_model: "lgbm_direct", total_seconds: 145.2, seconds_per_series: 2.9 },
      { run_id: "run_2025_01_14_001", timestamp: "2025-01-14T09:00:00Z", lob: "retail", series_count: 50, champion_model: "auto_arima", total_seconds: 230.5, seconds_per_series: 4.6 },
    ],
  },

  fva: {
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
    ],
    detail_preview: [
      { series_id: "sku_001", naive_wmape: 0.18, champion_wmape: 0.07, fva_pct: 61.1 },
    ],
  },

  calibration: {
    lob: "retail",
    model_reports: {
      lgbm_direct: [
        { label: "50%", nominal: 0.5, empirical: 0.48, miscalibration: 0.02, sharpness: 15.3, n_observations: 600 },
        { label: "80%", nominal: 0.8, empirical: 0.76, miscalibration: 0.04, sharpness: 28.7, n_observations: 600 },
        { label: "95%", nominal: 0.95, empirical: 0.93, miscalibration: 0.02, sharpness: 45.1, n_observations: 600 },
      ],
    },
    per_series_preview: [],
  },

  shap: {
    lob: "retail",
    model: "lgbm_direct",
    feature_importance: [
      { feature: "lag_1", mean_abs_value: 0.45, std: 0.12 },
      { feature: "rolling_mean_4", mean_abs_value: 0.32, std: 0.08 },
      { feature: "month_sin", mean_abs_value: 0.18, std: 0.05 },
      { feature: "promo_flag", mean_abs_value: 0.15, std: 0.04 },
      { feature: "temperature", mean_abs_value: 0.08, std: 0.03 },
    ],
    decomposition_preview: [],
  },

  modelCards: {
    count: 2,
    model_cards: [
      { model_name: "lgbm_direct", lob: "retail", training_start: "2023-01-02", training_end: "2024-12-30", n_series: 50, n_observations: 5200, backtest_wmape: 0.082, backtest_bias: -0.012, champion_since: "2025-01-10", features: ["lag_1", "rolling_mean_4", "month_sin", "promo_flag"], config_hash: "abc123def", notes: "Champion model" },
      { model_name: "auto_arima", lob: "retail", training_start: "2023-01-02", training_end: "2024-12-30", n_series: 50, n_observations: 5200, backtest_wmape: 0.095, backtest_bias: 0.005, champion_since: "2024-11-01", features: [], config_hash: "xyz789abc", notes: "Statistical baseline" },
    ],
  },

  lineage: {
    count: 3,
    lineage: [
      { timestamp: "2025-01-15T10:30:00Z", model_id: "lgbm_direct", lob: "retail", n_series: 50, horizon_weeks: 12, selection_strategy: "walk_forward", run_id: "run_2025_01_15_001", notes: "Promoted as champion", user_id: "admin" },
      { timestamp: "2025-01-10T09:00:00Z", model_id: "auto_arima", lob: "retail", n_series: 50, horizon_weeks: 12, selection_strategy: "walk_forward", run_id: "run_2025_01_10_001", notes: "Previous champion", user_id: "ds_user" },
    ],
  },

  overrides: {
    count: 2,
    overrides: [
      { override_id: "ovr_001", old_sku: "SKU-OLD-100", new_sku: "SKU-NEW-200", proportion: 0.75, scenario: "base", ramp_shape: "linear", effective_date: "2025-02-01", created_by: "planner1", notes: "Product line refresh" },
      { override_id: "ovr_002", old_sku: "SKU-OLD-300", new_sku: "SKU-NEW-400", proportion: 1.0, scenario: "optimistic", ramp_shape: "step", effective_date: "2025-03-01", created_by: "planner1", notes: "Full replacement" },
    ],
  },

  commentary: {
    lob: "retail",
    executive_summary: "Overall forecast accuracy improved 3.2% week-over-week. LightGBM remains the champion model across 50 series with 8.2% WMAPE.",
    key_metrics: [
      { name: "Overall WMAPE", value: 0.082, unit: "%", trend: "improving" },
      { name: "Bias", value: -0.012, unit: "%", trend: "stable" },
      { name: "Series at Risk", value: 3, unit: "count", trend: "improving" },
    ],
    exceptions: ["SKU_001 shows significant drift"],
    action_items: ["Review SKU_001 forecast override", "Schedule model retrain for Q2 data refresh"],
  },

  analysis: {
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
    recommended_config_yaml: "forecast:\n  frequency: W\n  horizon_periods: 12\n",
    config_reasoning: ["Weekly frequency detected"],
    hypotheses: ["Promotional events may cause demand spikes"],
  },

  skuMapping: {
    phase: 1,
    old_sku_count: 10,
    new_sku_count: 8,
    mapped_pairs: 6,
    unmapped_old: 4,
    unmapped_new: 2,
    mappings: [
      { old_sku: "SKU-OLD-100", new_sku: "SKU-NEW-200", confidence: 0.92, method: "name_similarity" },
    ],
  },

  hierarchyBuild: {
    levels: ["category", "region", "sku"],
    n_nodes: 15,
    n_leaf: 10,
    tree: { name: "total", children: [{ name: "electronics", children: [{ name: "east" }, { name: "west" }] }] },
  },

  hierarchyAggregate: {
    target_level: "category",
    n_nodes: 3,
    aggregated_rows: 156,
    preview: [{ category: "electronics", week: "2025-01-06", quantity: 500 }],
  },

  hierarchyReconcile: {
    method: "bottom_up",
    n_series_reconciled: 10,
    coherence_before: 0.85,
    coherence_after: 1.0,
    preview: [{ series_id: "sku_001", week: "2025-01-06", original: 100, reconciled: 105 }],
  },

  // AI endpoints
  aiExplain: {
    question: "Why is WMAPE high?",
    answer: "WMAPE is elevated due to demand volatility in 3 intermittent SKUs.",
    sources: ["drift analysis", "series classification"],
  },

  aiTriage: {
    critical_count: 1,
    warning_count: 2,
    recommendations: [
      { series_id: "sku_001", severity: "critical", recommendation: "Investigate promotional impact" },
    ],
  },

  aiConfigTune: {
    current_wmape: 0.082,
    estimated_wmape: 0.075,
    changes: [{ param: "n_estimators", old: 100, new: 200, impact: -0.007 }],
    config_yaml: "models:\n  lgbm_direct:\n    n_estimators: 200\n",
  },
} as const;

// ── Route handler ────────────────────────────────────────────────────────────

export async function mockApiRoutes(page: Page) {
  // Single route handler — dispatches based on URL pathname.
  // This avoids glob-matching issues with query strings.
  await page.route(`${API}/**`, (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();
    const json = (data: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(data) });

    // Health
    if (path === "/health") return json(MOCK.health);

    // Auth
    if (path === "/auth/status") return json({ mode: "demo", auth_required: false });
    if (path === "/auth/me") {
      const headers = route.request().headers();
      const role = headers["x-demo-role"] as Role;
      if (!Object.prototype.hasOwnProperty.call(ROLE_PERMISSIONS, role)) return json({ detail: "Invalid demo role" }, 401);
      return json({
        mode: "demo", username: decodeURIComponent(headers["x-demo-user"] ?? ""),
        email: "", role, permissions: ROLE_PERMISSIONS[role], expires_at: null,
      });
    }

    // Metrics — leaderboard, drift, FVA, calibration, SHAP
    if (path.includes("/metrics/leaderboard/")) return json(MOCK.leaderboard);
    if (path.includes("/metrics/drift/")) return json(MOCK.drift);
    if (path.includes("/fva")) return json(MOCK.fva);
    if (path.includes("/calibration")) return json(MOCK.calibration);
    if (path.includes("/shap")) return json(MOCK.shap);

    // Forecast
    if (path.startsWith("/forecast/decompose")) return json({ series_id: "sku_001", components: [] });
    if (path.startsWith("/forecast/compare")) return json({ comparison: [] });
    if (path.startsWith("/forecast/constrain")) return json({ constrained: [] });
    if (path.startsWith("/forecast/")) return json(MOCK.forecast);

    // Series
    if (path.includes("/history")) return json({ series_id: "sku_001", observations: [] });
    if (path.includes("/breaks")) return json(MOCK.breakDetection);
    if (path.includes("/cleansing")) return json(MOCK.cleansingAudit);
    if (path.includes("/regressor")) return json(MOCK.regressorScreen);
    if (path.startsWith("/series/")) return json(MOCK.seriesList);

    // Audit
    if (path.startsWith("/audit")) return json(MOCK.audit);

    // Pipeline — specific paths first
    if (path === "/pipeline/runs") return json({ count: 0, runs: [] });
    if (path === "/pipeline/manifests" || path.startsWith("/pipeline/manifests")) return json(MOCK.manifests);
    if (path === "/pipeline/costs" || path.startsWith("/pipeline/costs")) return json(MOCK.costs);
    if (path.includes("/analyze-multi-file")) return json(MOCK.analysis);
    if (path.startsWith("/pipeline/")) return json({ run_id: "run_test_001", status: "completed", lob: "retail" });

    // Governance — model cards, lineage, export
    if (path.startsWith("/governance/model-cards")) return json(MOCK.modelCards);
    if (path.startsWith("/governance/lineage")) return json(MOCK.lineage);
    if (path.startsWith("/governance/export/")) return json({ format: "csv", rows: 100, url: "/downloads/export.csv" });

    // Overrides
    if (path.startsWith("/overrides")) {
      if (method === "POST") return json({ override_id: "ovr_new" }, 201);
      if (method === "DELETE") return json({ deleted: true });
      return json(MOCK.overrides);
    }

    // AI
    if (path === "/ai/explain") return json(MOCK.aiExplain);
    if (path === "/ai/triage") return json(MOCK.aiTriage);
    if (path === "/ai/recommend-config") return json(MOCK.aiConfigTune);
    if (path === "/ai/commentary") return json(MOCK.commentary);

    // Analyze
    if (path.startsWith("/analyze")) return json(MOCK.analysis);

    // SKU mapping
    if (path.startsWith("/sku-mapping/")) return json(MOCK.skuMapping);

    // Hierarchy
    if (path.startsWith("/hierarchy/build")) return json(MOCK.hierarchyBuild);
    if (path.startsWith("/hierarchy/aggregate")) return json(MOCK.hierarchyAggregate);
    if (path.startsWith("/hierarchy/reconcile")) return json(MOCK.hierarchyReconcile);

    // Catch-all
    return json({});
  });
}

export { MOCK };
