"use client";

import { useState, useMemo, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLob } from "@/providers/lob-provider";
import { TimeSeriesLine } from "@/components/charts/time-series-line";
import { FanChart } from "@/components/charts/fan-chart";
import { NLQueryPanel } from "@/components/ai/nl-query-panel";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { NoDataGuide } from "@/components/shared/no-data-guide";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { DecompositionPanel } from "@/components/forecast/decomposition-panel";
import { ComparisonPanel } from "@/components/forecast/comparison-panel";
import { ConstrainedForecastPanel } from "@/components/forecast/constrained-forecast-panel";
import { useForecast } from "@/hooks/use-forecast";
import { formatNumber } from "@/lib/utils";
import { COLORS } from "@/lib/constants";
import { WorkflowNav } from "@/components/shared/workflow-nav";
import { DataTable } from "@/components/data/data-table";
import { api } from "@/lib/api-client";

export default function ForecastPage() {
  const { lob, setLob } = useLob();
  const [selectedSeries, setSelectedSeries] = useState("");
  const [selection, setSelection] = useState({ lob, id: "" });
  const [showConfig, setShowConfig] = useState(false);
  const [showData, setShowData] = useState(false);
  const selectedRun = selection.lob === lob ? selection.id : "";
  const runs = useQuery({
    queryKey: ["runs", lob, "forecast"],
    queryFn: () => api.listRuns(lob, "forecast"),
    enabled: !!lob,
  });
  const { data, isLoading, error, refetch } = useForecast(lob, {
    run_id: selectedRun && selectedRun !== "legacy" ? selectedRun : undefined,
    legacy: selectedRun === "legacy",
  });
  const currentRunId = data?.run_id;
  const savedConfig = useQuery({
    queryKey: ["run-config", lob, currentRunId],
    queryFn: () => {
      if (!currentRunId) throw new Error("No committed run is selected");
      return api.getRunConfig(lob, currentRunId);
    },
    enabled: showConfig && !!currentRunId,
  });
  useEffect(() => {
    setSelection({ lob, id: "" });
    setSelectedSeries("");
    setShowConfig(false);
  }, [lob]);

  // Extract unique series IDs
  const seriesIds = useMemo(() => {
    if (!data) return [];
    return [...new Set(data.points.map((p) => p.series_id))];
  }, [data]);

  // Filter points for selected series
  const seriesPoints = useMemo(() => {
    if (!data || !selectedSeries) return [];
    return data.points
      .filter((p) => p.series_id === selectedSeries)
      .sort((a, b) => a.week.localeCompare(b.week))
      .map((p) => ({
        week: p.week,
        forecast: p.forecast,
        model: p.model || "unknown",
        forecast_p10: p.forecast_p10,
        forecast_p50: p.forecast_p50,
        forecast_p90: p.forecast_p90,
      }));
  }, [data, selectedSeries]);

  // Check if quantile data is available
  const hasQuantiles = seriesPoints.some((p) => p.forecast_p10 != null && p.forecast_p90 != null);

  // Auto-select first series
  if (seriesIds.length > 0 && !selectedSeries) {
    setSelectedSeries(seriesIds[0]);
  }

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Forecast Viewer</h1>
        <p className="text-sm text-muted-foreground">
          View forecasts with fan charts, seasonal decomposition, AI Q&amp;A, and constrained forecasting.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-1">
          <label className="text-xs font-medium">LOB</label>
          <input
            type="text"
            value={lob}
            onChange={(e) => setLob(e.target.value)}
            className="rounded-md border bg-background px-3 py-1.5 text-sm w-32"
          />
        </div>
        <div className="space-y-1">
          <label htmlFor="forecast-run" className="text-xs font-medium">Forecast run</label>
          <select
            id="forecast-run" value={selectedRun}
            onChange={(event) => {
              setSelection({ lob, id: event.target.value });
              setSelectedSeries("");
              setShowConfig(false);
            }}
            disabled={runs.isLoading}
            className="max-w-md rounded-md border bg-background px-3 py-1.5 text-sm"
          >
            <option value="">Latest available</option>
            {runs.data?.runs.map(run => (
              <option key={run.run_id} value={run.run_id}>
                {run.timestamp} | {run.champion_model} | {run.run_id.slice(0, 8)}
              </option>
            ))}
            <option value="legacy">Legacy files (unverified lineage)</option>
          </select>
        </div>
        {seriesIds.length > 0 && (
          <div className="space-y-1">
            <label className="text-xs font-medium">Series</label>
            <select
              value={selectedSeries}
              onChange={(e) => setSelectedSeries(e.target.value)}
              className="rounded-md border bg-background px-3 py-1.5 text-sm max-w-xs"
            >
              {seriesIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </div>
        )}
        <button
          onClick={() => refetch()}
          className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          Load Forecasts
        </button>
      </div>

      {isLoading && <ChartSkeleton />}
      {runs.error && <ErrorDisplay message={runs.error.message} onRetry={() => runs.refetch()} />}
      {error && (error.message.includes("404") ? (
        <NoDataGuide lob={lob} dataType="forecast" />
      ) : (
        <ErrorDisplay message={error.message} onRetry={() => refetch()} />
      ))}

      {data && data.points.length === 0 && (
        <NoDataGuide lob={lob} dataType="forecast" />
      )}

      {data && data.points.length > 0 && (
        <>
          <section aria-label="Run provenance" className="space-y-2 rounded-lg border p-4 text-sm">
            <p>
              Run: <code data-testid="forecast-run-id">{data.run_id ?? "Legacy / unknown"}</code>
            </p>
            {data.run_id ? (
              <>
                <p className="break-all text-xs text-muted-foreground">Input SHA256: {data.input_data_hash}</p>
                <p className="break-all text-xs text-muted-foreground">Configuration SHA256: {data.config_hash}</p>
                <button
                  type="button" onClick={() => setShowConfig(value => !value)}
                  className="rounded-md border px-3 py-1.5 text-sm"
                >
                  {showConfig ? "Hide stored configuration" : "View stored configuration"}
                </button>
                {showConfig && savedConfig.isLoading && <p role="status">Loading stored configuration...</p>}
                {showConfig && savedConfig.error && <ErrorDisplay message={savedConfig.error.message} onRetry={() => savedConfig.refetch()} />}
                {showConfig && savedConfig.data && (
                  <pre className="max-h-80 overflow-auto rounded-md bg-muted p-3 text-xs">
                    {JSON.stringify(savedConfig.data.configuration, null, 2)}
                  </pre>
                )}
              </>
            ) : (
              <p className="text-muted-foreground">Legacy artifact: original input/configuration lineage is unknown.</p>
            )}
          </section>
          {/* Summary */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard label="Series Count" value={formatNumber(data.series_count)} />
            <MetricCard label="Forecast Origin" value={data.forecast_origin || "N/A"} />
            <MetricCard label="Points" value={formatNumber(data.points.length)} />
            <MetricCard
              label="Horizon"
              value={selectedSeries ? `${seriesPoints.length} periods` : "N/A"}
            />
          </div>

          {/* Forecast Chart */}
          {seriesPoints.length > 0 && (
            <section className="space-y-3">
              <h2 className="text-lg font-semibold">
                Forecast: {selectedSeries}
              </h2>
              <div className="rounded-lg border p-4">
                <TimeSeriesLine
                  data={seriesPoints}
                  xKey="week"
                  lines={[
                    {
                      dataKey: "forecast",
                      name: "Forecast",
                      color: COLORS.primary,
                    },
                  ]}
                  height={350}
                />
                <p className="mt-2 text-xs text-muted-foreground">
                  Model: {seriesPoints[0]?.model}
                </p>
              </div>
            </section>
          )}
          {seriesPoints.length > 0 && (
            <details
              className="rounded-lg border p-4"
              open={showData}
              onToggle={event => setShowData(event.currentTarget.open)}
            >
              <summary className="cursor-pointer text-sm font-medium">Forecast data</summary>
              <DataTable
                columns={[
                  { key: "week", label: "Period" },
                  { key: "forecast", label: "Forecast", render: value => typeof value === "number" ? formatNumber(value, 2) : "N/A" },
                  { key: "model", label: "Model" },
                ]}
                data={seriesPoints}
                pageSize={10}
              />
            </details>
          )}

          {/* Fan Chart */}
          {hasQuantiles && seriesPoints.length > 0 && (
            <section className="space-y-3">
              <h2 className="text-lg font-semibold">Fan Chart (P10–P90)</h2>
              <div className="rounded-lg border p-4">
                <FanChart data={seriesPoints} height={380} />
              </div>
            </section>
          )}
          {!hasQuantiles && (
            <section className="rounded-lg border p-6">
              <h2 className="text-lg font-semibold">Fan Chart (P10–P90)</h2>
              <p className="text-sm text-muted-foreground mt-2">
                Fan chart requires quantile columns (forecast_p10, forecast_p50, forecast_p90) in forecast data.
                Enable prediction intervals in config to generate these.
              </p>
            </section>
          )}

          {/* Decomposition */}
          <DecompositionPanel />

          {/* AI Q&A — live */}
          {selectedSeries && (
            <section>
              <h2 className="text-lg font-semibold mb-3">Ask About This Forecast</h2>
              <NLQueryPanel
                key={`${lob}:${data.run_id ?? "legacy"}:${selectedSeries}`}
                seriesId={selectedSeries}
                lob={lob}
                runId={data.run_id ?? undefined}
                legacy={data.artifact_source === "legacy"}
                suggestions={[
                  "Why this trend?",
                  "What drives seasonality?",
                  "How does this compare to last year?",
                  "Is this forecast reliable?",
                ]}
              />
            </section>
          )}

          {/* Comparison & Constraints */}
          <ComparisonPanel />
          <ConstrainedForecastPanel />
        </>
      )}

      <WorkflowNav currentStep="/forecast" />
    </div>
  );
}
