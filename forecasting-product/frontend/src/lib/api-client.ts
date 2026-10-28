// ──────────────────────────────────────────────────────────────────────────────
// Centralized API client for FastAPI backend
// ──────────────────────────────────────────────────────────────────────────────

import type {
  HealthResponse,
  AuthIdentity,
  AuthStatus,
  Role,
  ForecastResponse,
  LeaderboardResponse,
  DriftResponse,
  AnalysisResponse,
  NLQueryRequest,
  NLQueryResponse,
  TriageRequest,
  TriageResponse,
  ConfigTuneRequest,
  ConfigTuneResponse,
  CommentaryRequest,
  CommentaryResponse,
  AuditResponse,
  SeriesListResponse,
  SeriesHistoryResponse,
  BreakDetectionResponse,
  CleansingAuditResponse,
  RegressorScreenResponse,
  HierarchyBuildResponse,
  HierarchyAggregateResponse,
  HierarchyReconciliationMethod,
  HierarchyReconcileOptions,
  HierarchyReconcileResponse,
  SKUMappingResponse,
  OverrideListResponse,
  CreateOverrideRequest,
  UpdateOverrideRequest,
  PipelineRunResponse,
  RunListResponse,
  RunConfigResponse,
  ManifestListResponse,
  CostListResponse,
  MultiFileAnalysisResponse,
  FVAResponse,
  CalibrationResponse,
  ShapResponse,
  DecomposeResponse,
  ForecastCompareResponse,
  ConstrainResponse,
  ModelCardListResponse,
  ModelCardItem,
  LineageResponse,
  BIExportResponse,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const MAX_RETRIES = 3;
const RETRY_BASE_MS = 1000;
const RETRYABLE_STATUS_CODES = new Set([502, 503, 504, 429]);

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type BearerCredentials = { kind: "bearer"; token: string; expiresAt?: number };
type DemoCredentials = { kind: "demo"; username: string; role: Role };
type RequestCredentials = BearerCredentials | DemoCredentials;
export type ApiSession = (BearerCredentials & { expiresAt: number }) | DemoCredentials;
export const AUTH_INVALIDATED_EVENT = "forecasting:authentication-invalidated";
let currentSession: ApiSession | null = null;

export function setApiSession(session: ApiSession | null): void {
  if (session && typeof window === "undefined") {
    throw new Error("Authentication sessions are browser-only");
  }
  currentSession = session ? { ...session } : null;
}

function invalidateSession(owner: ApiSession | null): void {
  if (!owner || owner !== currentSession) return;
  currentSession = null;
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(AUTH_INVALIDATED_EVENT));
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function request<T>(
  path: string,
  options: RequestInit = {},
  authentication?: RequestCredentials | null,
): Promise<T> {
  const owner = currentSession;
  const canRetry = options.method === undefined || ["GET", "HEAD"].includes(options.method.toUpperCase());
  const credentials = authentication === undefined ? owner : authentication;
  const token = credentials?.kind === "bearer" ? credentials.token : null;
  const headers: Record<string, string> = {};
  new Headers(options.headers).forEach((value, key) => { headers[key] = value; });
  for (const key of ["authorization", "x-demo-user", "x-demo-role"]) delete headers[key];
  if (credentials?.kind === "demo") {
    headers["X-Demo-User"] = encodeURIComponent(credentials.username);
    headers["X-Demo-Role"] = credentials.role;
  }
  const assertCurrentSession = () => {
    if (authentication === undefined && owner !== currentSession) {
      throw new ApiError(401, "Session changed. Sign in again.");
    }
  };

  if (token) {
    if (credentials?.kind === "bearer" && credentials.expiresAt !== undefined
        && Date.now() >= credentials.expiresAt * 1000) {
      if (authentication === undefined) invalidateSession(owner);
      throw new ApiError(401, "Session expired. Please log in again.");
    }
    headers["Authorization"] = `Bearer ${token}`;
  }

  // Don't set Content-Type for FormData (browser sets boundary automatically)
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  let lastError: ApiError | null = null;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    assertCurrentSession();
    let res: Response;
    try {
      res = await fetch(`${BASE_URL}${path}`, {
        ...options,
        headers,
      });
    } catch (err) {
      if (options.signal?.aborted || (err instanceof Error && err.name === "AbortError")) throw err;
      lastError = new ApiError(
        0,
        `Unable to reach the API server at ${BASE_URL}. ` +
          `Please check that the backend is running and NEXT_PUBLIC_API_URL is set correctly. ` +
          `(${err instanceof Error ? err.message : String(err)})`,
      );
      if (canRetry && attempt < MAX_RETRIES) {
        await sleep(RETRY_BASE_MS * 2 ** attempt);
        continue;
      }
      throw lastError;
    }

    assertCurrentSession();
    if (!res.ok) {
      if (res.status === 401) {
        if (authentication === undefined) invalidateSession(owner);
        throw new ApiError(401, "Invalid or expired authentication. Sign in again.");
      }
      const text = await res.text().catch(() => "Unknown error");
      lastError = new ApiError(res.status, text);

      if (canRetry && RETRYABLE_STATUS_CODES.has(res.status) && attempt < MAX_RETRIES) {
        await sleep(RETRY_BASE_MS * 2 ** attempt);
        continue;
      }
      throw lastError;
    }

    const result: T = await res.json();
    assertCurrentSession();
    return result;
  }

  throw lastError ?? new ApiError(0, "Request failed after retries");
}

// ── Endpoints ────────────────────────────────────────────────────────────────

function verifiedForecast(result: ForecastResponse, lob: string, runId?: string, legacy = false): ForecastResponse {
  if (result.lob !== lob || (runId && result.run_id !== runId)
      || (legacy && result.artifact_source !== "legacy")) {
    throw new ApiError(502, "The API returned a different forecast identity. No substitute run was displayed.");
  }
  return result;
}

export const api = {
  // Health
  health: () => request<HealthResponse>("/health", {}, null),

  // Auth
  authStatus: (signal?: AbortSignal) =>
    request<AuthStatus>("/auth/status", { signal }, null),
  getCurrentUser: (token: string, signal?: AbortSignal) =>
    request<AuthIdentity>("/auth/me", { signal }, { kind: "bearer", token }),
  getDemoUser: (username: string, role: Role, signal?: AbortSignal) =>
    request<AuthIdentity>("/auth/me", { signal }, { kind: "demo", username, role }),

  // Forecasts
  getForecasts: (lob: string, params?: { series_id?: string; horizon?: number; run_id?: string; legacy?: boolean }) => {
    const searchParams = new URLSearchParams();
    if (params?.series_id) searchParams.set("series_id", params.series_id);
    if (params?.horizon) searchParams.set("horizon", String(params.horizon));
    if (params?.run_id) searchParams.set("run_id", params.run_id);
    if (params?.legacy) searchParams.set("legacy", "true");
    const qs = searchParams.toString();
    return request<ForecastResponse>(`/forecast/${encodeURIComponent(lob)}${qs ? `?${qs}` : ""}`)
      .then(result => verifiedForecast(result, lob, params?.run_id, params?.legacy));
  },

  getForecastSeries: (lob: string, seriesId: string, runId?: string) =>
    request<ForecastResponse>(
      `/forecast/${encodeURIComponent(lob)}/${encodeURIComponent(seriesId)}${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}`,
    ).then(result => verifiedForecast(result, lob, runId)),

  // Metrics
  getLeaderboard: (lob: string, runType: string = "backtest", runId?: string) =>
    request<LeaderboardResponse>(
      `/metrics/leaderboard/${encodeURIComponent(lob)}?${new URLSearchParams({
        run_type: runType, ...(runId ? { run_id: runId } : {}),
      }).toString()}`,
    ).then(result => {
      if (runId && result.run_id !== runId) throw new ApiError(502, "The API returned a different metric run.");
      return result;
    }),

  getDrift: (
    lob: string,
    params?: { run_type?: string; baseline_weeks?: number; recent_weeks?: number },
  ) => {
    const searchParams = new URLSearchParams();
    searchParams.set("run_type", params?.run_type || "backtest");
    if (params?.baseline_weeks) searchParams.set("baseline_weeks", String(params.baseline_weeks));
    if (params?.recent_weeks) searchParams.set("recent_weeks", String(params.recent_weeks));
    return request<DriftResponse>(
      `/metrics/drift/${encodeURIComponent(lob)}?${searchParams.toString()}`,
    );
  },

  // Analysis (file upload)
  analyze: (file: File, lobName: string = "analyzed", llmEnabled: boolean = false) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<AnalysisResponse>(
      `/analyze?lob_name=${encodeURIComponent(lobName)}&llm_enabled=${llmEnabled}`,
      { method: "POST", body: formData },
    );
  },

  // AI endpoints
  aiExplain: (body: NLQueryRequest) =>
    request<NLQueryResponse>("/ai/explain", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  aiTriage: (body: TriageRequest) =>
    request<TriageResponse>("/ai/triage", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  aiRecommendConfig: (body: ConfigTuneRequest) =>
    request<ConfigTuneResponse>("/ai/recommend-config", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  aiCommentary: (body: CommentaryRequest) =>
    request<CommentaryResponse>("/ai/commentary", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // Audit
  getAuditLog: (params?: { action?: string; resource_type?: string; limit?: number }) => {
    const searchParams = new URLSearchParams();
    if (params?.action) searchParams.set("action", params.action);
    if (params?.resource_type) searchParams.set("resource_type", params.resource_type);
    if (params?.limit) searchParams.set("limit", String(params.limit));
    const qs = searchParams.toString();
    return request<AuditResponse>(`/audit${qs ? `?${qs}` : ""}`);
  },

  // ── Series ──────────────────────────────────────────────────────────────────

  listSeries: (lob: string, runId?: string) =>
    request<SeriesListResponse>(`/series/${encodeURIComponent(lob)}${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}`),

  getSeriesHistory: (lob: string, seriesId: string, runId?: string) =>
    request<SeriesHistoryResponse>(
      `/series/${encodeURIComponent(lob)}/${encodeURIComponent(seriesId)}/history${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}`,
    ).then(result => {
      if (runId && result.run_id !== runId) throw new ApiError(502, "The API returned history for a different run.");
      return result;
    }),

  detectBreaks: (file: File, params?: { lob?: string; method?: string; penalty?: number }) => {
    const formData = new FormData();
    formData.append("file", file);
    const sp = new URLSearchParams();
    if (params?.lob) sp.set("lob", params.lob);
    if (params?.method) sp.set("method", params.method);
    if (params?.penalty) sp.set("penalty", String(params.penalty));
    const qs = sp.toString();
    return request<BreakDetectionResponse>(`/series/breaks${qs ? `?${qs}` : ""}`, {
      method: "POST", body: formData,
    });
  },

  detectBreaksFromLob: (lob: string, method: string = "cusum", penalty: number = 3.0) =>
    request<BreakDetectionResponse>(
      `/series/breaks?lob=${encodeURIComponent(lob)}&method=${encodeURIComponent(method)}&penalty=${penalty}`,
      { method: "POST" },
    ),

  cleansingAudit: (file: File, params?: Record<string, string | number | boolean>) => {
    const formData = new FormData();
    formData.append("file", file);
    const sp = new URLSearchParams();
    if (params) Object.entries(params).forEach(([k, v]) => sp.set(k, String(v)));
    const qs = sp.toString();
    return request<CleansingAuditResponse>(`/series/cleansing-audit${qs ? `?${qs}` : ""}`, {
      method: "POST", body: formData,
    });
  },

  cleansingAuditFromLob: (lob: string) =>
    request<CleansingAuditResponse>(
      `/series/cleansing-audit?lob=${encodeURIComponent(lob)}`,
      { method: "POST" },
    ),

  regressorScreen: (file: File, params?: Record<string, string | number | boolean>) => {
    const formData = new FormData();
    formData.append("file", file);
    const sp = new URLSearchParams();
    if (params) Object.entries(params).forEach(([k, v]) => sp.set(k, String(v)));
    const qs = sp.toString();
    return request<RegressorScreenResponse>(`/series/regressor-screen${qs ? `?${qs}` : ""}`, {
      method: "POST", body: formData,
    });
  },

  regressorScreenFromLob: (lob: string) =>
    request<RegressorScreenResponse>(
      `/series/regressor-screen?lob=${encodeURIComponent(lob)}`,
      { method: "POST" },
    ),

  // ── Hierarchy ───────────────────────────────────────────────────────────────

  buildHierarchy: (file: File, levels: string, idColumn: string = "series_id", name: string = "product") => {
    const formData = new FormData();
    formData.append("file", file);
    return request<HierarchyBuildResponse>(
      `/hierarchy/build?levels=${encodeURIComponent(levels)}&id_column=${encodeURIComponent(idColumn)}&name=${encodeURIComponent(name)}`,
      { method: "POST", body: formData },
    );
  },

  aggregateHierarchy: (file: File, levels: string, targetLevel: string, params?: Record<string, string | number>) => {
    const formData = new FormData();
    formData.append("file", file);
    const sp = new URLSearchParams({ levels, target_level: targetLevel });
    if (params) Object.entries(params).forEach(([k, v]) => sp.set(k, String(v)));
    return request<HierarchyAggregateResponse>(
      `/hierarchy/aggregate?${sp.toString()}`,
      { method: "POST", body: formData },
    );
  },

  reconcileHierarchy: (
    file: File,
    levels: string,
    method: HierarchyReconciliationMethod = "bottom_up",
    params?: HierarchyReconcileOptions,
    actualsFile?: File,
  ) => {
    const formData = new FormData();
    formData.append("file", file);
    if (actualsFile) formData.append("actuals_file", actualsFile);
    const sp = new URLSearchParams({ levels, method });
    if (params) Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined) sp.set(k, String(v));
    });
    return request<HierarchyReconcileResponse>(
      `/hierarchy/reconcile?${sp.toString()}`,
      { method: "POST", body: formData },
    );
  },

  // ── SKU Mapping ─────────────────────────────────────────────────────────────

  skuMappingPhase1: (file: File, params?: { launch_window_days?: number; min_base_similarity?: number; min_confidence?: string }) => {
    const formData = new FormData();
    formData.append("product_master", file);
    const sp = new URLSearchParams();
    if (params?.launch_window_days) sp.set("launch_window_days", String(params.launch_window_days));
    if (params?.min_base_similarity) sp.set("min_base_similarity", String(params.min_base_similarity));
    if (params?.min_confidence) sp.set("min_confidence", params.min_confidence);
    const qs = sp.toString();
    return request<SKUMappingResponse>(`/sku-mapping/phase1${qs ? `?${qs}` : ""}`, {
      method: "POST", body: formData,
    });
  },

  skuMappingPhase2: (productMaster: File, salesHistory?: File, params?: Record<string, string | number>) => {
    const formData = new FormData();
    formData.append("product_master", productMaster);
    if (salesHistory) formData.append("sales_history", salesHistory);
    const sp = new URLSearchParams();
    if (params) Object.entries(params).forEach(([k, v]) => sp.set(k, String(v)));
    const qs = sp.toString();
    return request<SKUMappingResponse>(`/sku-mapping/phase2${qs ? `?${qs}` : ""}`, {
      method: "POST", body: formData,
    });
  },

  // ── Overrides ───────────────────────────────────────────────────────────────

  listOverrides: (params?: { old_sku?: string; new_sku?: string }) => {
    const sp = new URLSearchParams();
    if (params?.old_sku) sp.set("old_sku", params.old_sku);
    if (params?.new_sku) sp.set("new_sku", params.new_sku);
    const qs = sp.toString();
    return request<OverrideListResponse>(`/overrides${qs ? `?${qs}` : ""}`);
  },

  createOverride: (body: CreateOverrideRequest) =>
    request<{ override_id: string; status: string }>("/overrides", {
      method: "POST", body: JSON.stringify(body),
    }),

  updateOverride: async (overrideId: string, body: UpdateOverrideRequest) => {
    const result = await request<{ override_id: string; status: string }>(
      `/overrides/${encodeURIComponent(overrideId)}`,
      { method: "PUT", body: JSON.stringify(body) },
    );
    if (result.override_id !== overrideId) {
      throw new ApiError(502, "The API returned a different override ID. No replacement was displayed.");
    }
    return result;
  },

  deleteOverride: (overrideId: string) =>
    request<{ override_id: string; status: string }>(
      `/overrides/${encodeURIComponent(overrideId)}`,
      { method: "DELETE" },
    ),

  // ── Pipeline ────────────────────────────────────────────────────────────────

  listRuns: (lob: string, runType: "forecast" | "backtest" = "forecast") =>
    request<RunListResponse>(
      `/pipeline/runs?${new URLSearchParams({ lob, run_type: runType }).toString()}`,
    ),

  getRunConfig: (lob: string, runId: string) =>
    request<RunConfigResponse>(
      `/pipeline/runs/${encodeURIComponent(lob)}/${encodeURIComponent(runId)}/config`,
    ).then(result => {
      if (result.lob !== lob || result.run_id !== runId) {
        throw new ApiError(502, "The API returned configuration for a different run.");
      }
      return result;
    }),

  runBacktest: (file: File, lob: string = "default", configFile?: File) => {
    const formData = new FormData();
    formData.append("file", file);
    if (configFile) formData.append("config_file", configFile);
    return request<PipelineRunResponse>(
      `/pipeline/backtest?lob=${encodeURIComponent(lob)}`,
      { method: "POST", body: formData },
    );
  },

  runForecast: (file: File, lob: string = "default", params?: { model_id?: string; horizon?: number; configFile?: File }) => {
    const formData = new FormData();
    formData.append("file", file);
    if (params?.configFile) formData.append("config_file", params.configFile);
    const sp = new URLSearchParams({ lob });
    if (params?.model_id !== undefined) sp.set("model_id", params.model_id);
    if (params?.horizon !== undefined) sp.set("horizon", String(params.horizon));
    return request<PipelineRunResponse>(
      `/pipeline/forecast?${sp.toString()}`,
      { method: "POST", body: formData },
    );
  },

  listManifests: (lob?: string, limit: number = 20) => {
    const sp = new URLSearchParams();
    if (lob) sp.set("lob", lob);
    sp.set("limit", String(limit));
    return request<ManifestListResponse>(`/pipeline/manifests?${sp.toString()}`);
  },

  getCosts: (lob?: string, limit: number = 20) => {
    const sp = new URLSearchParams();
    if (lob) sp.set("lob", lob);
    sp.set("limit", String(limit));
    return request<CostListResponse>(`/pipeline/costs?${sp.toString()}`);
  },

  analyzeMultiFile: (files: File[], lobName: string = "analyzed") => {
    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    return request<MultiFileAnalysisResponse>(
      `/pipeline/analyze-multi-file?lob_name=${encodeURIComponent(lobName)}`,
      { method: "POST", body: formData },
    );
  },

  // ── Analytics ───────────────────────────────────────────────────────────────

  getFVA: (lob: string, runType: string = "backtest", runId?: string) =>
    request<FVAResponse>(
      `/metrics/${encodeURIComponent(lob)}/fva?${new URLSearchParams({
        run_type: runType, ...(runId ? { run_id: runId } : {}),
      }).toString()}`,
    ),

  getCalibration: (lob: string, runType: string = "backtest", runId?: string) =>
    request<CalibrationResponse>(
      `/metrics/${encodeURIComponent(lob)}/calibration?${new URLSearchParams({
        run_type: runType, ...(runId ? { run_id: runId } : {}),
      }).toString()}`,
    ),

  getShap: (lob: string, params?: { model_name?: string; season_length?: number; top_k?: number }) => {
    const sp = new URLSearchParams();
    if (params?.model_name) sp.set("model_name", params.model_name);
    if (params?.season_length) sp.set("season_length", String(params.season_length));
    if (params?.top_k) sp.set("top_k", String(params.top_k));
    const qs = sp.toString();
    return request<ShapResponse>(
      `/metrics/${encodeURIComponent(lob)}/shap${qs ? `?${qs}` : ""}`,
      { method: "POST" },
    );
  },

  decomposeForecast: (historyFile: File, forecastFile: File, params?: Record<string, string | number>) => {
    const formData = new FormData();
    formData.append("history_file", historyFile);
    formData.append("forecast_file", forecastFile);
    const sp = new URLSearchParams();
    if (params) Object.entries(params).forEach(([k, v]) => sp.set(k, String(v)));
    const qs = sp.toString();
    return request<DecomposeResponse>(`/forecast/decompose${qs ? `?${qs}` : ""}`, {
      method: "POST", body: formData,
    });
  },

  compareForecasts: (modelFile: File, externalFile: File, externalName: string = "external") => {
    const formData = new FormData();
    formData.append("model_file", modelFile);
    formData.append("external_file", externalFile);
    return request<ForecastCompareResponse>(
      `/forecast/compare?external_name=${encodeURIComponent(externalName)}`,
      { method: "POST", body: formData },
    );
  },

  constrainForecast: (file: File, params?: { min_demand?: number; max_capacity?: number; aggregate_max?: number; proportional?: boolean }) => {
    const formData = new FormData();
    formData.append("file", file);
    const sp = new URLSearchParams();
    if (params?.min_demand !== undefined) sp.set("min_demand", String(params.min_demand));
    if (params?.max_capacity !== undefined) sp.set("max_capacity", String(params.max_capacity));
    if (params?.aggregate_max !== undefined) sp.set("aggregate_max", String(params.aggregate_max));
    if (params?.proportional !== undefined) sp.set("proportional", String(params.proportional));
    const qs = sp.toString();
    return request<ConstrainResponse>(`/forecast/constrain${qs ? `?${qs}` : ""}`, {
      method: "POST", body: formData,
    });
  },

  // ── Governance ──────────────────────────────────────────────────────────────

  listModelCards: () => request<ModelCardListResponse>("/governance/model-cards"),

  getModelCard: (modelName: string) =>
    request<ModelCardItem>(`/governance/model-cards/${encodeURIComponent(modelName)}`),

  getLineage: (lob?: string, modelId?: string, runId?: string) => {
    const sp = new URLSearchParams();
    if (lob) sp.set("lob", lob);
    if (modelId) sp.set("model_id", modelId);
    if (runId) sp.set("run_id", runId);
    const qs = sp.toString();
    return request<LineageResponse>(`/governance/lineage${qs ? `?${qs}` : ""}`);
  },

  biExport: (
    reportType: string, lob: string, runType: string = "backtest", modelId?: string,
    runs?: { run_id?: string; actuals_run_id?: string },
  ) => {
    const sp = new URLSearchParams({ lob, run_type: runType });
    if (modelId) sp.set("model_id", modelId);
    if (runs?.run_id) sp.set("run_id", runs.run_id);
    if (runs?.actuals_run_id) sp.set("actuals_run_id", runs.actuals_run_id);
    return request<BIExportResponse>(
      `/governance/export/${encodeURIComponent(reportType)}?${sp.toString()}`,
      { method: "POST" },
    );
  },
};

export { ApiError };
