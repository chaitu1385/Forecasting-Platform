import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import BacktestPage from "@/app/backtest/page";
import { LobProvider } from "@/providers/lob-provider";
import { api, setApiSession } from "@/lib/api-client";
import type { FVAResponse, LeaderboardResponse } from "@/lib/types";

vi.mock("@/components/charts/fva-cascade", () => ({ FVACascade: () => <div /> }));
vi.mock("@/components/charts/calibration-plot", () => ({ CalibrationPlot: () => <div /> }));
vi.mock("@/components/ai/config-tuner-panel", () => ({ ConfigTunerPanel: () => <div /> }));
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => <>{children}</>,
  BarChart: ({ data }: { data: unknown[] }) => <div data-testid="leaderboard-chart">{JSON.stringify(data)}</div>,
  Bar: () => null, XAxis: () => null, YAxis: () => null,
  CartesianGrid: () => null, Tooltip: () => null, Cell: () => null,
}));

const runId = "1".repeat(32);

function show(response: LeaderboardResponse) {
  vi.spyOn(api, "getLeaderboard").mockResolvedValue(response);
  vi.spyOn(api, "getFVA").mockResolvedValue({
    lob: "retail", summary: [], layer_leaderboard: [], detail_preview: [],
  });
  vi.spyOn(api, "getCalibration").mockResolvedValue({
    lob: "retail", model_reports: {}, per_series_preview: [],
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><LobProvider><BacktestPage /></LobProvider></QueryClientProvider>);
}

function response(): LeaderboardResponse {
  return {
    lob: "retail", run_type: "backtest", run_id: runId, artifact_source: "committed",
    scoring_provenance: {
      source: "stored", status: "recorded", version: 1, policy: "raw-observations-v1",
      primary_metric: "rmse", secondary_metric: "normalized_bias", competition_scope: "lob",
    },
    entries: [
      { model: "candidate_a", wmape: 0.4, normalized_bias: -0.1, rank: 1, n_series: 2,
        primary_score: 2, scores: { rmse: 2, wmape: 0.4 }, group_key: "retail", competition_scope: "lob" },
      { model: "candidate_b", wmape: 0.1, normalized_bias: -0.2, rank: 2, n_series: 2,
        primary_score: 3, scores: { rmse: 3, wmape: 0.1 }, group_key: "retail", competition_scope: "lob" },
    ],
  };
}

beforeEach(() => {
  localStorage.clear();
  setApiSession(null);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  setApiSession(null);
});

describe("stored leaderboard presentation", () => {
  it("uses the stored criterion and ranking instead of sorting by WMAPE", async () => {
    show(response());
    await screen.findByTestId("leaderboard-chart");
    const best = screen.getByText("Best Model", { exact: true }).parentElement!;
    expect(best).toHaveTextContent("candidate_a");
    const chart = JSON.parse(screen.getByTestId("leaderboard-chart").textContent!);
    expect(chart.map((item: { rawModel: string }) => item.rawModel)).toEqual(["candidate_a", "candidate_b"]);
    expect(chart.map((item: { score: number }) => item.score)).toEqual([2, 3]);
    expect(screen.getAllByText(/RMSE/).length).toBeGreaterThan(0);
    await waitFor(() => expect(api.getFVA).toHaveBeenLastCalledWith("retail", "backtest", runId));
    await waitFor(() => expect(api.getCalibration).toHaveBeenLastCalledWith("retail", "backtest", runId));
  });

  it("labels macro WMAPE and does not drop rows lacking volume WMAPE", async () => {
    const value = response();
    value.scoring_provenance!.primary_metric = "macro_wmape";
    value.entries = value.entries.map((entry, index) => ({
      ...entry, wmape: null, primary_score: index === 0 ? 0.1 : 0.55,
      scores: { macro_wmape: index === 0 ? 0.1 : 0.55 },
    }));
    show(value);
    await screen.findByTestId("leaderboard-chart");
    expect(screen.getAllByText(/Macro WMAPE/).length).toBeGreaterThan(0);
    expect(screen.queryByText("Volume WMAPE", { exact: true })).not.toBeInTheDocument();
    const chart = JSON.parse(screen.getByTestId("leaderboard-chart").textContent!);
    expect(chart.map((item: { score: number }) => item.score)).toEqual([0.1, 0.55]);
  });

  it("keeps cohort identity and order without announcing a global winner", async () => {
    const value = response();
    value.scoring_provenance!.competition_scope = "series_id";
    value.entries = [
      { ...value.entries[0], group_key: "large", competition_scope: "series_id" },
      { ...value.entries[1], group_key: "large", competition_scope: "series_id" },
      { ...value.entries[1], rank: 1, group_key: "small", competition_scope: "series_id" },
      { ...value.entries[0], rank: 2, group_key: "small", competition_scope: "series_id" },
    ];
    show(value);
    await screen.findByTestId("leaderboard-chart");
    const best = screen.getByText("Best Model", { exact: true }).parentElement!;
    expect(best).toHaveTextContent("Per competition");
    const rows = within(screen.getByRole("table", { name: "Stored model leaderboard" })).getAllByRole("row").slice(1);
    expect(rows.map(row => row.getAttribute("data-group"))).toEqual(["large", "large", "small", "small"]);
    expect(rows.map(row => row.getAttribute("data-model"))).toEqual([
      "candidate_a", "candidate_b", "candidate_b", "candidate_a",
    ]);
  });

  it("keeps missing historical scoring metadata unknown", async () => {
    const value = response();
    value.scoring_provenance = {
      source: "stored", status: "unknown", version: null, policy: null,
      primary_metric: null, secondary_metric: null, competition_scope: null,
    };
    value.entries = value.entries.map(entry => ({
      ...entry, primary_score: null, scores: { wmape: entry.wmape },
      group_key: null, competition_scope: null,
    }));
    show(value);
    const table = await screen.findByRole("table", { name: "Stored model leaderboard" });
    expect(screen.getByText(/Scoring policy unknown/)).toBeVisible();
    expect(screen.getByText(/Primary criterion unknown/)).toBeVisible();
    expect(within(table).getByText("Stored WMAPE: 40.0%")).toBeVisible();
    expect(within(table).getByText("Stored WMAPE: 10.0%")).toBeVisible();
    expect(screen.queryByTestId("leaderboard-chart")).not.toBeInTheDocument();
    expect(screen.queryByText("Volume WMAPE", { exact: true })).not.toBeInTheDocument();
    expect(screen.getByText("Best Model", { exact: true }).parentElement).toHaveTextContent("Unknown scope");
  });

  it.each(["rmse", "mae"])("preserves named historical %s values without inventing a primary criterion", async metric => {
    const value = response();
    value.scoring_provenance = {
      source: "stored", status: "unknown", version: null, policy: null,
      primary_metric: null, secondary_metric: null, competition_scope: null,
    };
    value.entries = value.entries.map((entry, index) => ({
      ...entry, wmape: null, normalized_bias: null, primary_score: null,
      scores: { [metric]: index + 2 }, group_key: null, competition_scope: null,
    }));
    show(value);
    const table = await screen.findByRole("table", { name: "Stored model leaderboard" });
    expect(within(table).getByText(`${metric.toUpperCase()}: 2.00`)).toBeVisible();
    expect(within(table).getByText(`${metric.toUpperCase()}: 3.00`)).toBeVisible();
    expect(screen.getByText(/Primary criterion unknown/)).toBeVisible();
    expect(screen.queryByTestId("leaderboard-chart")).not.toBeInTheDocument();
    expect(within(table).queryByText("Stored WMAPE", { exact: true })).not.toBeInTheDocument();
    expect(within(table).getAllByRole("row").slice(1).map(row => row.getAttribute("data-model"))).toEqual([
      "candidate_a", "candidate_b",
    ]);
    expect(screen.getByText("Best Model", { exact: true }).parentElement).toHaveTextContent("Unknown scope");
  });

  it("does not paint an earlier run's derived analysis over a changed LOB", async () => {
    let finish!: (value: FVAResponse) => void;
    const pending = new Promise<FVAResponse>(resolve => { finish = resolve; });
    show(response());
    vi.mocked(api.getLeaderboard).mockImplementation(async lob => ({
      ...response(), lob, run_id: lob === "retail" ? runId : "2".repeat(32),
    }));
    vi.mocked(api.getFVA).mockImplementation(lob => lob === "retail" ? pending : Promise.resolve({
      lob, summary: [], detail_preview: [], layer_leaderboard: [{ marker: "current-analysis" }],
    }));
    await screen.findByTestId("leaderboard-chart");
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "different_lob" } });
    await screen.findByText("current-analysis");
    await act(async () => finish({
      lob: "retail", summary: [], detail_preview: [], layer_leaderboard: [{ marker: "stale-analysis" }],
    }));
    expect(screen.getByText("current-analysis")).toBeVisible();
    expect(screen.queryByText("stale-analysis")).not.toBeInTheDocument();
    expect(api.getFVA).toHaveBeenLastCalledWith("different_lob", "backtest", "2".repeat(32));
  });
});
