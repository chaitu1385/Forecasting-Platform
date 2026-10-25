"use client";

import { useState, useEffect } from "react";
import { FileUpload } from "@/components/data/file-upload";
import { DataTable } from "@/components/data/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import type { PipelineExecutionSettings, PipelineRunResponse } from "@/lib/types";

function ExecutionDetails({ settings }: { settings: PipelineExecutionSettings }) {
  return (
    <details className="text-sm">
      <summary className="cursor-pointer font-medium">Effective configuration</summary>
      <p className="my-2 text-xs text-muted-foreground">
        Model source: {settings.model_source}. Horizon source: {settings.horizon_source}.
        {" "}Configuration source: {settings.config_source}.
      </p>
      <pre className="overflow-x-auto rounded-md border p-3 text-xs">
        {JSON.stringify(settings.config, null, 2)}
      </pre>
    </details>
  );
}

export function PipelineExecutionPanel({ lobName, initialFile, configYaml }: { lobName: string; initialFile?: File | null; configYaml?: string }) {
  const [pipelineFile, setPipelineFile] = useState<File | null>(initialFile ?? null);
  const [uploadedConfig, setUploadedConfig] = useState<File | null>(null);

  useEffect(() => {
    setPipelineFile(initialFile ?? null);
  }, [initialFile]);
  const backtest = useAsyncOperation<PipelineRunResponse>();
  const forecast = useAsyncOperation<PipelineRunResponse>();
  const { reset: resetBacktest } = backtest;
  const { reset: resetForecast } = forecast;

  useEffect(() => {
    resetBacktest();
    resetForecast();
  }, [pipelineFile, lobName, configYaml, uploadedConfig, resetBacktest, resetForecast]);

  const makeConfigFile = (): File | undefined => {
    if (uploadedConfig) return uploadedConfig;
    if (configYaml === undefined) return undefined;
    return new File([configYaml], "config.yaml", { type: "text/yaml" });
  };

  const handleRunBacktest = () => {
    if (!pipelineFile) return;
    backtest.reset();
    forecast.reset();
    void backtest.run(() => api.runBacktest(pipelineFile, lobName, makeConfigFile()));
  };

  const handleRunForecast = () => {
    if (!pipelineFile) return;
    forecast.reset();
    const configFile = makeConfigFile();
    void forecast.run(() => api.runForecast(pipelineFile, lobName, {
      configFile,
      model_id: backtest.result?.champion_model,
      horizon: backtest.result?.execution.horizon_periods,
    }));
  };

  const combinedError = backtest.error || forecast.error;
  const running = backtest.loading || forecast.loading;
  const noSingleWinner = backtest.result !== null && !backtest.result.champion_model;

  return (
    <section className="space-y-4 rounded-lg border p-6">
      <h2 className="text-lg font-semibold">Pipeline Execution</h2>
      <p className="text-sm text-muted-foreground">
        Run Backtest and Forecast pipelines directly from the UI.
      </p>
      <FileUpload
        accept=".csv"
        onFileSelect={(file) => setPipelineFile(file)}
        label="Pipeline Data File"
      />
      {pipelineFile && (
        <p className="text-xs text-muted-foreground">
          Using: <span className="font-medium">{pipelineFile.name}</span> ({(pipelineFile.size / 1024).toFixed(1)} KB)
        </p>
      )}
      <FileUpload
        accept=".yaml,.yml"
        onFileSelect={setUploadedConfig}
        label="Pipeline Configuration (optional YAML)"
      />
      <p className="text-xs text-muted-foreground">
        {uploadedConfig
          ? `Using uploaded configuration: ${uploadedConfig.name}`
          : configYaml !== undefined ? "Using the recommended configuration." : "Using platform defaults."}
      </p>
      {uploadedConfig && (
        <button
          onClick={() => setUploadedConfig(null)}
          className="rounded-md border px-3 py-1.5 text-sm hover:bg-muted"
        >
          {configYaml !== undefined ? "Use recommended configuration" : "Use platform defaults"}
        </button>
      )}
      {!backtest.result && (
        <p className="text-xs text-muted-foreground">
          No backtest winner selected. Standalone forecasts use the first configured model,
          or seasonal naive with the platform defaults. The configured horizon is preserved.
        </p>
      )}
      {noSingleWinner && (
        <p role="status" className="text-sm text-muted-foreground">
          This backtest has no single runnable champion. Composite or multi-model results
          cannot be handed off by model name alone. Select a single-model plan and rerun
          the backtest; no winner will be silently substituted.
        </p>
      )}
      <div className="flex flex-wrap gap-3">
        <button
          onClick={handleRunBacktest}
          disabled={!pipelineFile || running}
          className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {backtest.loading ? "Running Backtest..." : "Run Backtest"}
        </button>
        <button
          onClick={handleRunForecast}
          disabled={!pipelineFile || running || noSingleWinner}
          className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {forecast.loading ? "Running Forecast..." : "Run Forecast"}
        </button>
      </div>
      {(backtest.loading || forecast.loading) && <ChartSkeleton height="h-24" />}
      {combinedError && <ErrorDisplay message={combinedError} onRetry={() => {
        backtest.setError(null);
        forecast.setError(null);
      }} />}
      {backtest.result && (
        <div className="space-y-2 rounded-md bg-green-50 dark:bg-green-950 p-4">
          <h3 className="text-sm font-medium">Backtest Result</h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard label="Status" value={backtest.result.status} />
            <MetricCard label="Champion" value={backtest.result.execution.model_id || "N/A"} />
            <MetricCard label="Best WMAPE" value={backtest.result.best_wmape != null ? `${(backtest.result.best_wmape * 100).toFixed(1)}%` : "N/A"} />
            <MetricCard label="Series" value={backtest.result.series_count ?? "N/A"} />
          </div>
          <ExecutionDetails settings={backtest.result.execution} />
          {backtest.result.leaderboard && backtest.result.leaderboard.length > 0 && (
            <DataTable
              columns={Object.keys(backtest.result.leaderboard[0]).map(k => ({ key: k, label: k, sortable: true }))}
              data={backtest.result.leaderboard}
              pageSize={10}
            />
          )}
        </div>
      )}
      {forecast.result && (
        <div className="space-y-2 rounded-md bg-blue-50 dark:bg-blue-950 p-4">
          <h3 className="text-sm font-medium">Forecast Result</h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard label="Status" value={forecast.result.status} />
            <MetricCard label="Effective model" value={forecast.result.execution.model_id ?? "N/A"} />
            <MetricCard label="Horizon (periods)" value={forecast.result.execution.horizon_periods} />
            <MetricCard label="Frequency" value={forecast.result.execution.config.frequency} />
            <MetricCard label="Rows" value={forecast.result.forecast_rows ?? "N/A"} />
            <MetricCard label="Series" value={forecast.result.series_count ?? "N/A"} />
            <MetricCard label="LOB" value={forecast.result.lob} />
          </div>
          <ExecutionDetails settings={forecast.result.execution} />
          {forecast.result.forecast_preview && forecast.result.forecast_preview.length > 0 && (
            <DataTable
              columns={Object.keys(forecast.result.forecast_preview[0]).map(k => ({ key: k, label: k, sortable: true }))}
              data={forecast.result.forecast_preview}
              pageSize={10}
            />
          )}
        </div>
      )}
    </section>
  );
}
