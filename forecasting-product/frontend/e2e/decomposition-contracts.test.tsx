import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import BacktestPage from "@/app/backtest/page";
import { AuthProvider, useAuth } from "@/providers/auth-provider";
import { LobProvider } from "@/providers/lob-provider";
import { api, setApiSession } from "@/lib/api-client";
import type { CommittedRun, DecompositionSummaryResponse, LeaderboardResponse } from "@/lib/types";

vi.mock("@/components/charts/leaderboard-bar", () => ({ LeaderboardBar: () => <div /> }));
vi.mock("@/components/charts/fva-cascade", () => ({ FVACascade: () => <div /> }));
vi.mock("@/components/charts/calibration-plot", () => ({ CalibrationPlot: () => <div /> }));
vi.mock("@/components/ai/config-tuner-panel", () => ({ ConfigTunerPanel: () => <div /> }));

const firstId = "1".repeat(32);
const secondId = "2".repeat(32);
const leaderboardId = "a".repeat(32);

function forecastRun(runId: string): CommittedRun {
  return {
    run_id: runId, run_type: "forecast", artifact_source: "committed",
    timestamp: runId === firstId ? "2026-01-01T00:00:00Z" : "2026-01-02T00:00:00Z",
    lob: "retail", series_count: 1, champion_model: "recorded-seasonal",
    forecast_horizon: 3, forecast_rows: 3, validation_passed: true,
    validation_warnings: 0, cleansing_applied: false, outliers_clipped: 0,
    input_data_hash: "b".repeat(64), config_hash: "c".repeat(64),
    links: { metadata: "/metadata", config: "/config", input_lineage: "/input" },
  };
}

function summary(runId: string | null = secondId, lob = "retail"): DecompositionSummaryResponse {
  return {
    lob, run_id: runId, artifact_source: runId ? "committed" : "legacy",
    forecast_model: runId === firstId ? "recorded-old" : "recorded-new",
    forecast_model_source: runId ? "manifest" : "forecast_frame",
    history_source: runId ? "committed" : "legacy", history_stage: runId ? "prepared" : "unknown",
    history_run_id: runId, method: "moving_average_decomposition", model_specific: false,
    season_length: 3, trend_window: 3, seasonal_basis: "observation_position",
    statistics_population: "history_rows_for_forecast_series", std_ddof: 1,
    series_count: 1, history_rows: 12, excluded_history_rows: 0, forecast_rows: 3,
    checked_rows: 15, preview_rows: 1, preview_truncated: true,
    components: [
      { component: "trend", mean_abs_value: runId === firstId ? 155 : 255, std: 30, defined_count: 10, undefined_count: 2 },
      { component: "seasonal", mean_abs_value: 2, std: 2.5, defined_count: 12, undefined_count: 0 },
      { component: "residual", mean_abs_value: 0, std: 0, defined_count: 10, undefined_count: 2 },
    ],
    decomposition_preview: [{
      series_id: "hand-example", week: "2026-01-05", value: 97,
      trend: null, seasonal: -3, residual: null, is_forecast: false,
    }],
  };
}

function leaderboard(lob = "retail", runType = "backtest"): LeaderboardResponse {
  return { lob, run_type: runType, run_id: leaderboardId, artifact_source: "committed", entries: [] };
}

function AuthenticationControls() {
  const { mode, user, loginWithToken, logout } = useAuth();
  return <div>
    <button disabled={mode !== "production"} onClick={() => void loginWithToken("synthetic-unit-credential")}>
      Replace authentication
    </button>
    <button onClick={logout}>Unit logout</button>
    <p>{user ? "Unit signed in" : "Unit signed out"}</p>
  </div>;
}

async function show() {
  vi.spyOn(api, "authStatus").mockResolvedValue({ mode: "production", auth_required: true });
  vi.spyOn(api, "getCurrentUser").mockImplementation(async () => ({
    username: "same-viewer", email: "", role: "viewer", mode: "production",
    permissions: ["view_metrics", "view_forecasts"], expires_at: Math.floor(Date.now() / 1000) + 3600,
  }));
  vi.spyOn(api, "getLeaderboard").mockImplementation(async (lob, type) => leaderboard(lob, type));
  vi.spyOn(api, "getFVA").mockResolvedValue({ lob: "retail", summary: [], layer_leaderboard: [], detail_preview: [] });
  vi.spyOn(api, "getCalibration").mockResolvedValue({ lob: "retail", model_reports: {}, per_series_preview: [] });
  vi.spyOn(api, "listRuns").mockResolvedValue({ count: 2, runs: [forecastRun(secondId), forecastRun(firstId)] });
  vi.spyOn(api, "getForecastDecomposition").mockImplementation(async (lob, params) =>
    summary(params?.legacy ? null : params?.run_id, lob));
  vi.spyOn(api, "getShap");
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <AuthProvider><LobProvider><AuthenticationControls /><BacktestPage /></LobProvider></AuthProvider>
    </QueryClientProvider>,
  );
  await waitFor(() => expect(screen.getByRole("button", { name: "Replace authentication" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "Replace authentication" }));
  await screen.findByText("Unit signed in");
  await waitFor(() => expect(screen.getByRole("button", { name: "Load decomposition" })).toBeEnabled());
  fireEvent.change(screen.getByLabelText("Season length (observations)"), { target: { value: "3" } });
  return client;
}

function load() {
  fireEvent.click(screen.getByRole("button", { name: "Load decomposition" }));
}

function defer<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

beforeEach(() => {
  localStorage.clear();
  setApiSession(null);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  setApiSession(null);
});

describe("truthful forecast decomposition ownership", () => {
  it("uses a separate actual forecast ID and truthful model, method, population and edge labels", async () => {
    await show();
    load();
    const result = await screen.findByTestId("decomposition-result");
    const section = screen.getByRole("region", { name: "Forecast decomposition" });
    expect(api.getForecastDecomposition).toHaveBeenLastCalledWith("retail", {
      run_id: secondId, legacy: false, season_length: 3,
    });
    expect(api.getShap).not.toHaveBeenCalled();
    expect(screen.getByTestId("decomposition-run-id")).toHaveTextContent(secondId);
    expect(result).not.toHaveTextContent(leaderboardId);
    expect(section).toHaveTextContent("Not model-specific feature attribution");
    expect(section).toHaveTextContent("separate forecast source, not the leaderboard run");
    expect(result).toHaveTextContent("moving_average_decomposition");
    expect(result).toHaveTextContent("recorded-new");
    expect(result).toHaveTextContent("recorded prepared snapshot from this same forecast run");
    expect(result).toHaveTextContent("Component magnitude");
    expect(result).toHaveTextContent("10 defined / 2 undefined");
    expect(result).toHaveTextContent("Checked all 15 rows");
    expect(result).toHaveTextContent("Preview: 1 rows (truncated)");
    expect(section).not.toHaveTextContent(/SHAP|waterfall|feature importance|LOESS|causal/i);
    expect(screen.getByLabelText("Decomposition forecast source")).toHaveValue(secondId);
  });

  it("pins latest once and keeps an explicitly selected older forecast", async () => {
    const client = await show();
    load();
    await screen.findByTestId("decomposition-result");
    act(() => client.setQueryData(["runs", "retail", "forecast"], {
      count: 1, runs: [forecastRun("3".repeat(32))],
    }));
    load();
    await screen.findByTestId("decomposition-result");
    expect(api.getForecastDecomposition).toHaveBeenLastCalledWith("retail", {
      run_id: secondId, legacy: false, season_length: 3,
    });
    act(() => client.setQueryData(["runs", "retail", "forecast"], {
      count: 2, runs: [forecastRun(secondId), forecastRun(firstId)],
    }));
    await waitFor(() => expect(within(screen.getByLabelText("Decomposition forecast source"))
      .getByRole("option", { name: new RegExp(firstId) })).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("Decomposition forecast source"), { target: { value: firstId } });
    expect(screen.getByLabelText("Decomposition forecast source")).toHaveValue(firstId);
    expect(screen.queryByTestId("decomposition-result")).not.toBeInTheDocument();
    load();
    await screen.findByTestId("decomposition-result");
    expect(api.getForecastDecomposition).toHaveBeenLastCalledWith("retail", {
      run_id: firstId, legacy: false, season_length: 3,
    });
    expect(screen.getByTestId("component-trend")).toHaveTextContent("155.0000");
    expect(screen.getByTestId("decomposition-result")).toHaveTextContent("recorded-old");
  });

  it("keeps legacy identity and missing model provenance explicitly unknown", async () => {
    await show();
    vi.mocked(api.getForecastDecomposition).mockResolvedValue({
      ...summary(null), forecast_model: null, forecast_model_source: "unknown",
    });
    fireEvent.change(screen.getByLabelText("Decomposition forecast source"), { target: { value: "legacy" } });
    load();
    const result = await screen.findByTestId("decomposition-result");
    expect(api.getForecastDecomposition).toHaveBeenLastCalledWith("retail", {
      run_id: undefined, legacy: true, season_length: 3,
    });
    expect(result).toHaveTextContent("Legacy / unverified");
    expect(result).toHaveTextContent("Unknown (not recorded)");
    expect(result).toHaveTextContent("legacy history; preparation stage and lineage unknown");
    expect(result).not.toHaveTextContent("recorded prepared snapshot");
  });

  it.each(["source", "season", "lob", "runType", "leaderboard", "refresh"])(
    "clears a completed result when its %s changes", async change => {
      const client = await show();
      load();
      await screen.findByTestId("decomposition-result");
      if (change === "source") fireEvent.change(screen.getByLabelText("Decomposition forecast source"), { target: { value: firstId } });
      if (change === "season") fireEvent.change(screen.getByLabelText("Season length (observations)"), { target: { value: "4" } });
      if (change === "lob") fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "different" } });
      if (change === "runType") fireEvent.change(screen.getAllByRole("combobox")[0], { target: { value: "live" } });
      if (change === "leaderboard") {
        await act(async () => {
          vi.mocked(api.getLeaderboard).mockResolvedValue({ ...leaderboard(), run_id: "b".repeat(32) });
          await client.invalidateQueries({ queryKey: ["leaderboard"] });
        });
      }
      if (change === "refresh") fireEvent.click(screen.getByRole("button", { name: "Refresh forecast sources" }));
      await waitFor(() => expect(screen.queryByTestId("decomposition-result")).not.toBeInTheDocument());
    },
  );

  it.each(["source", "lob", "season", "logout", "replacement"])(
    "cannot repaint a late response after %s changes", async change => {
      await show();
      const pending = defer<DecompositionSummaryResponse>();
      vi.mocked(api.getForecastDecomposition).mockReturnValueOnce(pending.promise);
      load();
      await screen.findByText("Computing forecast decomposition...");
      if (change === "source") fireEvent.change(screen.getByLabelText("Decomposition forecast source"), { target: { value: firstId } });
      if (change === "lob") fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "different" } });
      if (change === "season") fireEvent.change(screen.getByLabelText("Season length (observations)"), { target: { value: "4" } });
      if (change === "logout") fireEvent.click(screen.getByRole("button", { name: "Unit logout" }));
      if (change === "replacement") {
        fireEvent.click(screen.getByRole("button", { name: "Replace authentication" }));
        await waitFor(() => expect(api.getCurrentUser).toHaveBeenCalledTimes(2));
      }
      await act(async () => pending.resolve(summary()));
      expect(screen.queryByTestId("decomposition-result")).not.toBeInTheDocument();
      expect(screen.queryByText("Computing forecast decomposition...")).not.toBeInTheDocument();
    },
  );

  it("clears completed data on same-user authentication replacement and suppresses a late old error", async () => {
    await show();
    load();
    await screen.findByTestId("decomposition-result");
    fireEvent.click(screen.getByRole("button", { name: "Replace authentication" }));
    await waitFor(() => expect(api.getCurrentUser).toHaveBeenCalledTimes(2));
    expect(screen.queryByTestId("decomposition-result")).not.toBeInTheDocument();
    const pending = defer<DecompositionSummaryResponse>();
    vi.mocked(api.getForecastDecomposition).mockReturnValueOnce(pending.promise);
    await waitFor(() => expect(screen.getByRole("button", { name: "Load decomposition" })).toBeEnabled());
    load();
    fireEvent.click(screen.getByRole("button", { name: "Replace authentication" }));
    await waitFor(() => expect(api.getCurrentUser).toHaveBeenCalledTimes(3));
    await act(async () => pending.reject(new Error("old-session-only-error")));
    expect(screen.queryByText("old-session-only-error")).not.toBeInTheDocument();
  });

  it.each([403, 404, 500])("does not preserve prior success or substitute latest after a %s read failure", async status => {
    await show();
    fireEvent.change(screen.getByLabelText("Decomposition forecast source"), { target: { value: firstId } });
    load();
    await screen.findByTestId("decomposition-result");
    vi.mocked(api.getForecastDecomposition).mockRejectedValueOnce(new Error(`${status} exact read failed`));
    load();
    await screen.findByText(`${status} exact read failed`);
    expect(screen.queryByTestId("decomposition-result")).not.toBeInTheDocument();
    expect(api.getForecastDecomposition).toHaveBeenLastCalledWith("retail", {
      run_id: firstId, legacy: false, season_length: 3,
    });
    expect(api.getForecastDecomposition).toHaveBeenCalledTimes(2);
  });

  it("does not substitute zeros for an explicitly undefined component statistic", async () => {
    await show();
    const data = summary();
    data.components[0] = { component: "trend", mean_abs_value: null, std: null, defined_count: 0, undefined_count: 12 };
    vi.mocked(api.getForecastDecomposition).mockResolvedValue(data);
    load();
    const component = await screen.findByTestId("component-trend");
    expect(within(component).getAllByText(/Undefined/)).toHaveLength(2);
    expect(component).not.toHaveTextContent("0.0000");
  });

  it.each(["0", "-1", "1.5", ""])("rejects invalid season input %s without a request", async value => {
    await show();
    fireEvent.change(screen.getByLabelText("Season length (observations)"), { target: { value } });
    expect(screen.getByRole("button", { name: "Load decomposition" })).toBeDisabled();
    expect(api.getForecastDecomposition).not.toHaveBeenCalled();
  });
});

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

describe("decomposition transport identity", () => {
  it("uses the canonical POST with exact selector and preserves optional upload column names", async () => {
    const fetch = vi.fn().mockResolvedValue(jsonResponse({
      ...summary(firstId), history_source: "uploaded", history_stage: "uploaded", history_run_id: null,
    }));
    vi.stubGlobal("fetch", fetch);
    const file = new File(["synthetic"], "history.csv");
    await api.getForecastDecomposition("retail", {
      run_id: firstId, season_length: 3, id_col: "item", time_col: "period", target_col: "units",
    }, file);
    const [url, options] = fetch.mock.calls[0];
    expect(new URL(url).pathname).toBe("/metrics/retail/decomposition");
    expect(new URL(url).searchParams.get("run_id")).toBe(firstId);
    expect(new URL(url).searchParams.get("id_col")).toBe("item");
    expect(options.method).toBe("POST");
    expect(options.body.get("file")).toBe(file);
    expect(options.headers["Content-Type"]).toBeUndefined();
  });

  it("retains the typed alias and compatibility query parameters without treating them as provenance", async () => {
    const body = { ...summary(firstId), model: "recorded-old", feature_importance: [] };
    const fetch = vi.fn().mockResolvedValue(jsonResponse(body));
    vi.stubGlobal("fetch", fetch);
    expect(await api.getShap("retail", { run_id: firstId, model_name: "caller-text", top_k: 1 })).toEqual(body);
    expect(new URL(fetch.mock.calls[0][0]).pathname).toBe("/metrics/retail/shap");
    expect(new URL(fetch.mock.calls[0][0]).searchParams.get("model_name")).toBe("caller-text");
  });

  it.each([
    { run_id: firstId },
    { lob: "another" },
    { method: "shap" },
    { model_specific: true },
    { history_run_id: firstId },
    { history_stage: "input" },
    { artifact_source: "legacy" },
  ])("rejects mismatched provenance %j", async mismatch => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ ...summary(), ...mismatch })));
    await expect(api.getForecastDecomposition("retail", { run_id: secondId })).rejects.toThrow("different decomposition source or method");
  });

  it("never promotes a legacy source to a fabricated committed identity", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(summary())));
    await expect(api.getForecastDecomposition("retail", { legacy: true })).rejects.toThrow("different decomposition source or method");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(summary(null))));
    expect((await api.getForecastDecomposition("retail", { legacy: true })).run_id).toBeNull();
  });

  it.each(["network", "503", "404"])("does not retry or fall back after %s", async failure => {
    const fetch = vi.fn();
    if (failure === "network") fetch.mockRejectedValue(new TypeError("synthetic network loss"));
    else fetch.mockResolvedValue(new Response("synthetic read failure", { status: Number(failure) }));
    vi.stubGlobal("fetch", fetch);
    await expect(api.getForecastDecomposition("retail", { run_id: firstId })).rejects.toThrow();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("rejects an old response after replacement authentication for the same username", async () => {
    const pending = defer<Response>();
    vi.stubGlobal("fetch", vi.fn().mockReturnValue(pending.promise));
    setApiSession({ kind: "demo", username: "same-viewer", role: "viewer" });
    const request = api.getForecastDecomposition("retail", { run_id: secondId });
    const rejected = expect(request).rejects.toThrow("Session changed");
    setApiSession({ kind: "demo", username: "same-viewer", role: "viewer" });
    pending.resolve(jsonResponse(summary()));
    await rejected;
  });
});
