import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PipelineExecutionPanel } from "@/components/pipeline/pipeline-execution-panel";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import type { PipelineExecutionSettings, PipelineRunResponse } from "@/lib/types";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

const dataFile = new File(["series_id,week,quantity\nsku,2024-01-01,10"], "actuals.csv");
const configYaml = "forecast:\n  horizon_periods: 5\n  forecasters: [naive_seasonal, croston]\n";

function execution(model = "croston"): PipelineExecutionSettings {
  return {
    model_id: model,
    model_source: "backtest",
    horizon_periods: 5,
    horizon_source: "config",
    config_source: "upload",
    config: {
      frequency: "W", season_length: 52,
      series_id_column: "series_id", time_column: "week", target_column: "quantity",
      forecasters: ["naive_seasonal", "croston"], intermittent_forecasters: [], quantiles: [],
      fill_gaps: false, fill_value: 0, min_series_length_periods: 1, drop_zero_series: false,
      input_validation_enabled: false, cleansing_enabled: false, constraints_enabled: false,
      post_validation_enabled: false, halt_on_blocker: false, min_grade: "D",
      backtest_n_folds: 1, backtest_val_periods: 4, backtest_gap_periods: 0,
      selection_strategy: "champion", champion_granularity: "lob",
      primary_metric: "wmape", secondary_metric: "normalized_bias",
    },
  };
}

function backtestResult(model = "croston"): PipelineRunResponse {
  return {
    lob: "retail", status: "completed", champion_model: model, execution: execution(model),
    run_id: "1".repeat(32), run_type: "backtest",
    links: { metadata: "/run/one", config: "/run/one/config", input_lineage: "/run/one/input" },
  };
}

function forecastResult(): PipelineRunResponse {
  return {
    lob: "retail", status: "completed", model_id: "croston", horizon: 5, forecast_rows: 5,
    series_count: 1, execution: { ...execution(), model_source: "request" },
    run_id: "2".repeat(32), run_type: "forecast",
    links: { metadata: "/run/two", config: "/run/two/config", input_lineage: "/run/two/input" },
  };
}

async function readFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

describe("pipeline execution intent", () => {
  let backtest: ReturnType<typeof vi.spyOn<typeof api, "runBacktest">>;
  let forecast: ReturnType<typeof vi.spyOn<typeof api, "runForecast">>;

  beforeEach(() => {
    backtest = vi.spyOn(api, "runBacktest").mockResolvedValue(backtestResult());
    forecast = vi.spyOn(api, "runForecast").mockResolvedValue(forecastResult());
  });

  afterEach(() => vi.restoreAllMocks());

  it("submits the selected champion, resolved horizon and the same configuration", async () => {
    render(<PipelineExecutionPanel lobName="retail" initialFile={dataFile} configYaml={configYaml} />);
    fireEvent.click(screen.getByRole("button", { name: "Run Backtest" }));
    await screen.findByText("Backtest Result");
    fireEvent.click(screen.getByRole("button", { name: "Run Forecast" }));
    await screen.findByText("Forecast Result");
    expect(forecast).toHaveBeenCalledWith(dataFile, "retail", expect.objectContaining({
      model_id: "croston", horizon: 5, configFile: expect.any(File),
    }));
    const sentBacktestConfig = backtest.mock.calls[0][2];
    const sentForecastConfig = forecast.mock.calls[0][2]?.configFile;
    expect(sentBacktestConfig).toBeDefined();
    expect(sentForecastConfig).toBeDefined();
    expect(await readFile(sentBacktestConfig!)).toBe(configYaml);
    expect(await readFile(sentForecastConfig!)).toBe(configYaml);
    expect(screen.getByText("Effective model")).toBeInTheDocument();
    expect(screen.getByText("Horizon (periods)")).toBeInTheDocument();
    expect(screen.getByText("Frequency")).toBeInTheDocument();
  });

  it("keeps standalone forecasting usable without inventing a champion or horizon", async () => {
    render(<PipelineExecutionPanel lobName="retail" initialFile={dataFile} />);
    fireEvent.click(screen.getByRole("button", { name: "Run Forecast" }));
    await waitFor(() => expect(forecast).toHaveBeenCalledOnce());
    expect(forecast.mock.calls[0][2]?.model_id).toBeUndefined();
    expect(forecast.mock.calls[0][2]?.horizon).toBeUndefined();
    expect(screen.getByText(/No backtest winner selected/i)).toBeInTheDocument();
  });

  it.each(["file", "lob", "config"] as const)("invalidates completed results on a changed %s", async (input) => {
    const props = { lobName: "retail", initialFile: dataFile, configYaml };
    const { rerender } = render(<PipelineExecutionPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Run Backtest" }));
    await screen.findByText("Backtest Result");
    fireEvent.click(screen.getByRole("button", { name: "Run Forecast" }));
    await screen.findByText("Forecast Result");
    const next = {
      ...props,
      ...(input === "file" ? { initialFile: new File(["changed"], "changed.csv") } : {}),
      ...(input === "lob" ? { lobName: "wholesale" } : {}),
      ...(input === "config" ? { configYaml: configYaml.replace("5", "7") } : {}),
    };
    rerender(<PipelineExecutionPanel {...next} />);
    expect(screen.queryByText("Backtest Result")).not.toBeInTheDocument();
    expect(screen.queryByText("Forecast Result")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run Forecast" }));
    await waitFor(() => expect(forecast).toHaveBeenCalledTimes(2));
    expect(forecast.mock.calls[1][2]?.model_id).toBeUndefined();
    expect(forecast.mock.calls[1][2]?.horizon).toBeUndefined();
  });

  it("clears a removed initial file and prevents execution", async () => {
    const { rerender } = render(<PipelineExecutionPanel lobName="retail" initialFile={dataFile} />);
    rerender(<PipelineExecutionPanel lobName="retail" initialFile={null} />);
    expect(screen.getByRole("button", { name: "Run Backtest" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Run Forecast" })).toBeDisabled();
  });

  it("cannot forecast with a previous winner while a new backtest is pending", async () => {
    render(<PipelineExecutionPanel lobName="retail" initialFile={dataFile} configYaml={configYaml} />);
    fireEvent.click(screen.getByRole("button", { name: "Run Backtest" }));
    await screen.findByText("Backtest Result");
    const pending = deferred<ReturnType<typeof backtestResult>>();
    backtest.mockReturnValueOnce(pending.promise);
    fireEvent.click(screen.getByRole("button", { name: "Run Backtest" }));
    expect(screen.queryByText("Backtest Result")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Forecast" })).toBeDisabled();
    await act(async () => pending.resolve(backtestResult("naive_seasonal")));
    fireEvent.click(screen.getByRole("button", { name: "Run Forecast" }));
    await waitFor(() => expect(forecast).toHaveBeenCalledOnce());
    expect(forecast.mock.calls[0][2]?.model_id).toBe("naive_seasonal");
  });

  it.each(["weighted_ensemble", null])("does not submit a non-runnable winner (%s)", async (model) => {
    backtest.mockResolvedValueOnce({
      ...backtestResult(),
      champion_model: undefined,
      execution: { ...execution(), model_id: model },
    });
    render(<PipelineExecutionPanel lobName="retail" initialFile={dataFile} configYaml={configYaml} />);
    fireEvent.click(screen.getByRole("button", { name: "Run Backtest" }));
    await screen.findByText("Backtest Result");
    expect(screen.getByText(/single runnable champion/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Forecast" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Run Forecast" }));
    expect(forecast).not.toHaveBeenCalled();
  });

  it.each(["file", "lob", "config"] as const)("ignores a backtest that finishes after changing %s", async (input) => {
    const pending = deferred<ReturnType<typeof backtestResult>>();
    backtest.mockReturnValueOnce(pending.promise);
    const props = { lobName: "retail", initialFile: dataFile, configYaml };
    const { rerender } = render(<PipelineExecutionPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Run Backtest" }));
    rerender(<PipelineExecutionPanel
      lobName={input === "lob" ? "wholesale" : props.lobName}
      initialFile={input === "file" ? new File(["new"], "new.csv") : props.initialFile}
      configYaml={input === "config" ? "forecast:\n  horizon_periods: 8" : props.configYaml}
    />);
    await act(async () => pending.resolve(backtestResult()));
    expect(screen.queryByText("Backtest Result")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run Forecast" }));
    await waitFor(() => expect(forecast).toHaveBeenCalledOnce());
    expect(forecast.mock.calls[0][2]?.model_id).toBeUndefined();
  });

  it("ignores stale forecast successes and errors after config changes", async () => {
    const pending = deferred<ReturnType<typeof forecastResult>>();
    forecast.mockReturnValueOnce(pending.promise);
    const { rerender } = render(<PipelineExecutionPanel lobName="retail" initialFile={dataFile} configYaml={configYaml} />);
    fireEvent.click(screen.getByRole("button", { name: "Run Forecast" }));
    rerender(<PipelineExecutionPanel lobName="retail" initialFile={dataFile} configYaml="forecast: {}" />);
    await act(async () => pending.resolve(forecastResult()));
    expect(screen.queryByText("Forecast Result")).not.toBeInTheDocument();

    const failed = deferred<ReturnType<typeof forecastResult>>();
    forecast.mockReturnValueOnce(failed.promise);
    fireEvent.click(screen.getByRole("button", { name: "Run Forecast" }));
    rerender(<PipelineExecutionPanel lobName="retail" initialFile={dataFile} configYaml={configYaml} />);
    await act(async () => failed.reject(new Error("stale failure")));
    expect(screen.queryByText("stale failure")).not.toBeInTheDocument();
  });

  it("uses an uploaded YAML file for both operations and invalidates a prior winner", async () => {
    render(<PipelineExecutionPanel lobName="retail" initialFile={dataFile} configYaml={configYaml} />);
    fireEvent.click(screen.getByRole("button", { name: "Run Backtest" }));
    await screen.findByText("Backtest Result");
    const upload = screen.getByRole("button", { name: /Pipeline Configuration/i });
    const input = upload.querySelector("input");
    expect(input).not.toBeNull();
    const selected = new File([configYaml], "selected.yaml", { type: "text/yaml" });
    fireEvent.change(input!, { target: { files: [selected] } });
    expect(screen.queryByText("Backtest Result")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run Backtest" }));
    await screen.findByText("Backtest Result");
    fireEvent.click(screen.getByRole("button", { name: "Run Forecast" }));
    await waitFor(() => expect(forecast).toHaveBeenCalledOnce());
    expect(backtest.mock.calls[1][2]).toBe(selected);
    expect(forecast.mock.calls[0][2]?.configFile).toBe(selected);
  });
});

describe("async result ownership", () => {
  it("keeps the latest operation when completions arrive in reverse order", async () => {
    const { result } = renderHook(() => useAsyncOperation<string>());
    const first = deferred<string>();
    const latest = deferred<string>();
    act(() => { void result.current.run(() => first.promise); });
    act(() => { void result.current.run(() => latest.promise); });
    await act(async () => latest.resolve("latest"));
    await act(async () => first.resolve("stale"));
    expect(result.current.result).toBe("latest");
  });

  it("reset invalidates pending success and does not clear a newer loading state", async () => {
    const { result } = renderHook(() => useAsyncOperation<string>());
    const old = deferred<string>();
    const next = deferred<string>();
    act(() => { void result.current.run(() => old.promise); });
    act(() => result.current.reset());
    act(() => { void result.current.run(() => next.promise); });
    await act(async () => old.resolve("stale"));
    expect(result.current.result).toBeNull();
    expect(result.current.loading).toBe(true);
    await act(async () => next.resolve("next"));
    expect(result.current.result).toBe("next");
  });
});

describe("forecast API request serialization", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("does not drop invalid explicit values before the API can reject them", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(forecastResult())));
    vi.stubGlobal("fetch", fetch);
    await api.runForecast(dataFile, "retail", { horizon: 0, model_id: "" });
    const url = new URL(fetch.mock.calls[0][0]);
    expect(url.searchParams.get("horizon")).toBe("0");
    expect(url.searchParams.get("model_id")).toBe("");
  });
});
