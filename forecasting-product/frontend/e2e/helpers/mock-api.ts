// ──────────────────────────────────────────────────────────────────────────────
// Mock the api-client module so all API calls return deterministic test data
// ──────────────────────────────────────────────────────────────────────────────

import { vi } from "vitest";
import {
  mockHealth,
  mockForecast,
  mockLeaderboard,
  mockDrift,
  mockSeriesList,
  mockBreakDetection,
  mockCleansingAudit,
  mockRegressorScreen,
  mockAudit,
  mockManifests,
  mockCosts,
  mockFVA,
  mockCalibration,
  mockShap,
  mockModelCards,
  mockLineage,
  mockOverrides,
  mockCommentary,
  mockAnalysis,
} from "../fixtures/mock-data";

export function setupApiMocks() {
  vi.mock("@/lib/api-client", () => ({
    api: {
      // Health
      health: vi.fn().mockResolvedValue(mockHealth),

      // Auth
      getToken: vi.fn().mockResolvedValue({
        access_token: "mock-jwt-token",
        token_type: "bearer",
      }),

      // Forecasts
      getForecasts: vi.fn().mockResolvedValue(mockForecast),
      getForecastSeries: vi.fn().mockResolvedValue(mockForecast),
      listRuns: vi.fn().mockResolvedValue({ count: 0, runs: [] }),

      // Metrics
      getLeaderboard: vi.fn().mockResolvedValue(mockLeaderboard),
      getDrift: vi.fn().mockResolvedValue(mockDrift),

      // Analysis
      analyze: vi.fn().mockResolvedValue(mockAnalysis),

      // AI
      aiExplain: vi.fn().mockResolvedValue({
        answer: "The forecast shows an upward trend driven by seasonal patterns.",
        supporting_data: {},
        confidence: "high",
        sources_used: ["time_series_analysis"],
      }),
      aiTriage: vi.fn().mockResolvedValue({
        lob: "retail",
        executive_summary: "3 alerts detected, 1 critical requiring immediate attention.",
        total_alerts: 3,
        critical_count: 1,
        warning_count: 2,
        ranked_alerts: [],
      }),
      aiRecommendConfig: vi.fn().mockResolvedValue({
        lob: "retail",
        recommendations: [],
        overall_assessment: "Current configuration is well-optimized.",
        risk_summary: "Low risk across all recommendations.",
      }),
      aiCommentary: vi.fn().mockResolvedValue(mockCommentary),

      // Audit
      getAuditLog: vi.fn().mockResolvedValue(mockAudit),

      // Series
      listSeries: vi.fn().mockResolvedValue(mockSeriesList),
      getSeriesHistory: vi.fn().mockResolvedValue({
        series_id: "sku_001",
        lob: "retail",
        points: [
          { week: "2025-01-01", value: 100 },
          { week: "2025-01-08", value: 112 },
        ],
      }),
      detectBreaks: vi.fn().mockResolvedValue(mockBreakDetection),
      detectBreaksFromLob: vi.fn().mockResolvedValue(mockBreakDetection),
      cleansingAudit: vi.fn().mockResolvedValue(mockCleansingAudit),
      cleansingAuditFromLob: vi.fn().mockResolvedValue(mockCleansingAudit),
      regressorScreen: vi.fn().mockResolvedValue(mockRegressorScreen),
      regressorScreenFromLob: vi.fn().mockResolvedValue(mockRegressorScreen),

      // Hierarchy
      buildHierarchy: vi.fn().mockResolvedValue({
        name: "product",
        levels: ["category", "subcategory"],
        level_stats: [
          { level: "category", node_count: 5 },
          { level: "subcategory", node_count: 15 },
        ],
        total_nodes: 20,
        leaf_count: 15,
        s_matrix_shape: [20, 15],
        s_matrix_sample: [],
      }),
      aggregateHierarchy: vi.fn().mockResolvedValue({
        target_level: "category",
        total_rows: 260,
        unique_nodes: 5,
        top_n_data: [],
      }),
      reconcileHierarchy: vi.fn().mockResolvedValue({
        method: "bottom_up",
        before_total: 10000,
        after_total: 10000,
        rows: 260,
        reconciled_preview: [],
      }),

      // SKU Mapping
      skuMappingPhase1: vi.fn().mockResolvedValue({
        phase: 1,
        total_mappings: 5,
        mappings: [],
      }),
      skuMappingPhase2: vi.fn().mockResolvedValue({
        phase: 2,
        total_mappings: 5,
        mappings: [],
      }),

      // Overrides
      listOverrides: vi.fn().mockResolvedValue(mockOverrides),
      createOverride: vi.fn().mockResolvedValue({ override_id: "ovr_new", status: "created" }),
      updateOverride: vi.fn().mockResolvedValue({ override_id: "ovr_001", status: "updated" }),
      deleteOverride: vi.fn().mockResolvedValue({ override_id: "ovr_001", status: "deleted" }),

      // Pipeline
      runBacktest: vi.fn().mockResolvedValue({
        lob: "retail",
        status: "completed",
        champion_model: "lgbm_direct",
        best_wmape: 0.082,
        leaderboard: [],
      }),
      runForecast: vi.fn().mockResolvedValue({
        lob: "retail",
        status: "completed",
        forecast_rows: 600,
        series_count: 50,
        forecast_preview: [],
      }),
      listManifests: vi.fn().mockResolvedValue(mockManifests),
      getCosts: vi.fn().mockResolvedValue(mockCosts),
      analyzeMultiFile: vi.fn().mockResolvedValue({
        profiles: [],
        dimension_files: [],
        regressor_files: [],
        warnings: [],
      }),

      // Analytics
      getFVA: vi.fn().mockResolvedValue(mockFVA),
      getCalibration: vi.fn().mockResolvedValue(mockCalibration),
      getShap: vi.fn().mockResolvedValue(mockShap),
      decomposeForecast: vi.fn().mockResolvedValue({
        decomposition: [],
        narratives: {},
        series_count: 0,
      }),
      compareForecasts: vi.fn().mockResolvedValue({
        comparison: [],
        summary: [],
      }),
      constrainForecast: vi.fn().mockResolvedValue({
        before_total: 10000,
        after_total: 9500,
        rows_modified: 25,
        constraints_applied: { min_demand: 0, proportional: false },
        constrained_preview: [],
      }),

      // Governance
      listModelCards: vi.fn().mockResolvedValue(mockModelCards),
      getModelCard: vi.fn().mockResolvedValue(mockModelCards.model_cards[0]),
      getLineage: vi.fn().mockResolvedValue(mockLineage),
      biExport: vi.fn().mockResolvedValue({
        report_type: "forecast-actual",
        export_path: "/tmp/export.csv",
        status: "completed",
      }),
    },
    ApiError: class ApiError extends Error {
      status: number;
      constructor(status: number, message: string) {
        super(message);
        this.status = status;
        this.name = "ApiError";
      }
    },
  }));
}
