import { act, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import DataOnboardingPage from "@/app/data-onboarding/page";
import { api } from "@/lib/api-client";
import type { AnalysisResponse } from "@/lib/types";

vi.mock("@/components/charts/forecastability-gauge", () => ({ ForecastabilityGauge: () => null }));
vi.mock("@/components/charts/demand-class-donut", () => ({ DemandClassDonut: () => null }));

function deferred() {
  let resolve!: (value: AnalysisResponse) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<AnalysisResponse>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

function analysis(yaml = "forecast:\n  horizon_periods: 5"): AnalysisResponse {
  return {
    lob_name: "retail", time_column: "week", target_column: "quantity", id_columns: ["series_id"],
    n_series: 1, n_rows: 60, date_range_start: "2024-01-01", date_range_end: "2025-02-17",
    frequency: "W", overall_forecastability: 0.8, forecastability_distribution: {},
    demand_classes: {}, detected_hierarchies: [], recommended_config_yaml: yaml,
    config_reasoning: [], hypotheses: [],
  };
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false }, queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><DataOnboardingPage /></QueryClientProvider>);
}

function upload(name = "actuals.csv") {
  const container = screen.getByRole("button", { name: /Time Series Data/i });
  const input = container.querySelector("input");
  expect(input).not.toBeNull();
  fireEvent.change(input!, { target: { files: [new File(["synthetic data"], name)] } });
}

afterEach(() => vi.restoreAllMocks());

describe("onboarding plan ownership", () => {
  it("invalidates generated configuration when LOB changes, even if the old LOB is restored", async () => {
    vi.spyOn(api, "analyze").mockResolvedValue(analysis());
    renderPage();
    upload();
    await screen.findByText("Pipeline Execution");
    const lob = screen.getByRole("textbox");
    fireEvent.change(lob, { target: { value: "wholesale" } });
    expect(screen.queryByText("Pipeline Execution")).not.toBeInTheDocument();
    expect(screen.queryByText("Recommended Configuration")).not.toBeInTheDocument();
    fireEvent.change(lob, { target: { value: "retail" } });
    expect(screen.queryByText("Pipeline Execution")).not.toBeInTheDocument();
  });

  it("cannot use analysis that completes for a previous LOB", async () => {
    const pending = deferred();
    vi.spyOn(api, "analyze").mockReturnValue(pending.promise);
    renderPage();
    upload();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "wholesale" } });
    await act(async () => pending.resolve(analysis()));
    expect(screen.queryByText("Pipeline Execution")).not.toBeInTheDocument();
  });

  it("hides the old plan while a different file is being analyzed and after failure", async () => {
    const pending = deferred();
    vi.spyOn(api, "analyze").mockResolvedValueOnce(analysis()).mockReturnValueOnce(pending.promise);
    renderPage();
    upload("first.csv");
    await screen.findByText("Pipeline Execution");
    upload("second.csv");
    expect(screen.queryByText("Pipeline Execution")).not.toBeInTheDocument();
    await act(async () => pending.reject(new Error("Invalid second file")));
    await screen.findByText("Invalid second file");
    expect(screen.queryByText("Pipeline Execution")).not.toBeInTheDocument();
  });

  it("keeps the latest file/config pair when analysis completes out of order", async () => {
    const first = deferred();
    const second = deferred();
    vi.spyOn(api, "analyze").mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    renderPage();
    upload("first.csv");
    upload("second.csv");
    await act(async () => second.resolve(analysis("forecast:\n  horizon_periods: 7")));
    await screen.findByText("Pipeline Execution");
    await act(async () => first.resolve(analysis("forecast:\n  horizon_periods: 5")));
    expect(screen.getByText("second.csv")).toBeInTheDocument();
    expect(screen.getByText(/horizon_periods: 7/)).toBeInTheDocument();
    expect(screen.queryByText(/horizon_periods: 5/)).not.toBeInTheDocument();
  });
});
