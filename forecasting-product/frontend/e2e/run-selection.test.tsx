import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ForecastPage from "@/app/forecast/page";
import { LobProvider } from "@/providers/lob-provider";
import { api, setApiSession } from "@/lib/api-client";
import type { CommittedRun, ForecastResponse } from "@/lib/types";

vi.mock("@/components/charts/time-series-line", () => ({ TimeSeriesLine: () => <div /> }));
vi.mock("@/components/charts/fan-chart", () => ({ FanChart: () => <div /> }));
vi.mock("@/components/ai/nl-query-panel", () => ({ NLQueryPanel: () => <div /> }));
vi.mock("@/components/forecast/decomposition-panel", () => ({ DecompositionPanel: () => <div /> }));
vi.mock("@/components/forecast/comparison-panel", () => ({ ComparisonPanel: () => <div /> }));
vi.mock("@/components/forecast/constrained-forecast-panel", () => ({ ConstrainedForecastPanel: () => <div /> }));

const firstId = "1".repeat(32);
const secondId = "2".repeat(32);

function run(runId: string): CommittedRun {
  return {
    run_id: runId, run_type: "forecast", artifact_source: "committed",
    timestamp: runId === firstId ? "2026-01-01T00:00:00Z" : "2026-01-02T00:00:00Z",
    lob: "retail", series_count: 1, champion_model: "naive_seasonal",
    forecast_horizon: 2, forecast_rows: 2, validation_passed: true,
    validation_warnings: 0, cleansing_applied: false, outliers_clipped: 0,
    input_data_hash: "a".repeat(64), config_hash: "b".repeat(64),
    links: { metadata: "/metadata", config: "/config", input_lineage: "/input" },
  };
}

function forecast(runId = secondId, lob = "retail"): ForecastResponse {
  return {
    lob, run_id: runId, artifact_source: "committed", series_count: 1,
    input_data_hash: "a".repeat(64), config_hash: "b".repeat(64),
    points: [{
      series_id: "synthetic", week: "2026-01-01",
      forecast: runId === firstId ? 10 : 20, model: "naive_seasonal",
    }],
  };
}

function showPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <LobProvider><ForecastPage /></LobProvider>
    </QueryClientProvider>,
  );
  return client;
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

describe("committed forecast selection", () => {
  beforeEach(() => {
    vi.spyOn(api, "listRuns").mockResolvedValue({ count: 2, runs: [run(secondId), run(firstId)] });
  });

  it("lists and selects an older exact run, its values and its stored configuration", async () => {
    const fetch = vi.spyOn(api, "getForecasts").mockImplementation(async (lob, params) => forecast(params?.run_id, lob));
    const configuration = vi.spyOn(api, "getRunConfig").mockResolvedValue({
      lob: "retail", run_id: firstId, run_type: "forecast", config_hash: "b".repeat(64),
      configuration: { model: "naive_seasonal", marker: "older-config" },
    });
    showPage();
    await waitFor(() => expect(screen.getByTestId("forecast-run-id")).toHaveTextContent(secondId));
    fireEvent.click(screen.getByText("Forecast data", { exact: true }));
    expect(await screen.findByText("20.00", { exact: true })).toBeVisible();
    fireEvent.change(screen.getByLabelText("Forecast run", { exact: true }), { target: { value: firstId } });
    await waitFor(() => expect(screen.getByTestId("forecast-run-id")).toHaveTextContent(firstId));
    expect(screen.getByText("10.00", { exact: true })).toBeVisible();
    expect(screen.queryByText("20.00", { exact: true })).not.toBeInTheDocument();
    expect(fetch).toHaveBeenLastCalledWith("retail", { run_id: firstId, legacy: false });
    fireEvent.click(screen.getByRole("button", { name: "View stored configuration" }));
    await screen.findByText(/older-config/);
    expect(configuration).toHaveBeenCalledWith("retail", firstId);
  });

  it("does not paint an older in-flight selection over the current run", async () => {
    let finish!: (value: ForecastResponse) => void;
    const pending = new Promise<ForecastResponse>(resolve => { finish = resolve; });
    vi.spyOn(api, "getForecasts").mockImplementation((lob, params) =>
      params?.run_id === firstId ? pending : Promise.resolve(forecast(secondId, lob)));
    showPage();
    await waitFor(() => expect(screen.getByTestId("forecast-run-id")).toHaveTextContent(secondId));
    fireEvent.change(screen.getByLabelText("Forecast run"), { target: { value: firstId } });
    fireEvent.change(screen.getByLabelText("Forecast run"), { target: { value: secondId } });
    await waitFor(() => expect(screen.getByTestId("forecast-run-id")).toHaveTextContent(secondId));
    await act(async () => { finish(forecast(firstId)); });
    expect(screen.getByTestId("forecast-run-id")).toHaveTextContent(secondId);
  });

  it("resets a selected run when the LOB changes", async () => {
    const fetch = vi.spyOn(api, "getForecasts").mockImplementation(async (lob, params) => forecast(params?.run_id, lob));
    showPage();
    await waitFor(() => expect(screen.getByTestId("forecast-run-id")).toHaveTextContent(secondId));
    fireEvent.change(screen.getByLabelText("Forecast run"), { target: { value: firstId } });
    await waitFor(() => expect(screen.getByTestId("forecast-run-id")).toHaveTextContent(firstId));
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "different_lob" } });
    await waitFor(() => expect(fetch).toHaveBeenLastCalledWith("different_lob", { run_id: undefined, legacy: false }));
    expect(screen.getByLabelText("Forecast run")).toHaveValue("");
  });
});

describe("run transport boundaries", () => {
  it("rejects a response that substitutes latest for the requested ID", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(forecast(secondId)), {
      status: 200, headers: { "Content-Type": "application/json" },
    })));
    await expect(api.getForecasts("retail", { run_id: firstId })).rejects.toThrow("different forecast identity");
  });

  it.each(["network", "503"])("does not automatically repeat an ambiguous pipeline write (%s)", async failure => {
    const fetch = vi.fn();
    if (failure === "network") fetch.mockRejectedValue(new TypeError("synthetic connection loss"));
    else fetch.mockResolvedValue(new Response("synthetic confirmation unavailable", { status: 503 }));
    vi.stubGlobal("fetch", fetch);
    await expect(api.runForecast(new File(["synthetic"], "input.csv"), "retail")).rejects.toThrow();
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});
