"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/providers/auth-provider";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import { DataTable } from "@/components/data/data-table";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { MetricCard } from "@/components/shared/metric-card";
import type { DecompositionSummaryResponse, User } from "@/lib/types";

interface Props {
  lob: string;
  runType: string;
  leaderboardRunId?: string | null;
}

interface RequestOwner {
  user: User;
  lob: string;
  runType: string;
  leaderboardRunId?: string | null;
  source: string;
  season: string;
}

interface OwnedSummary extends RequestOwner {
  summary: DecompositionSummaryResponse;
}

export function ForecastDecompositionPanel({ lob, runType, leaderboardRunId }: Props) {
  const { user } = useAuth();
  const canView = !!user?.permissions.includes("view_metrics");
  const [selection, setSelection] = useState({ user, id: "" });
  const [season, setSeason] = useState("52");
  const [requestOwner, setRequestOwner] = useState<RequestOwner | null>(null);
  const selectedRun = selection.user === user ? selection.id : "";
  const { result, loading, error, run, reset } = useAsyncOperation<OwnedSummary>();
  const runs = useQuery({
    queryKey: ["runs", lob, "forecast"],
    queryFn: () => api.listRuns(lob, "forecast"),
    enabled: canView && !!lob,
  });

  useEffect(() => { reset(); }, [lob, runType, leaderboardRunId, user, reset]);

  const isCurrent = (owner: RequestOwner | null) => !!owner
    && owner.user === user && owner.source === selectedRun && owner.season === season
    && owner.lob === lob && owner.runType === runType && owner.leaderboardRunId === leaderboardRunId;
  const summary = isCurrent(result) ? result?.summary : null;
  const ownsStatus = isCurrent(requestOwner);
  const validSeason = /^\d+$/.test(season) && Number.isSafeInteger(Number(season)) && Number(season) > 0;
  const load = () => {
    if (!user || !canView || !validSeason || (!selectedRun && (!runs.data || runs.isFetching || runs.isError))) return;
    const source = selectedRun || runs.data?.runs[0]?.run_id || "legacy";
    const owner = { user, lob, runType, leaderboardRunId, source, season };
    reset();
    // Pin the listed forecast ID before the POST; never send the leaderboard ID.
    setSelection({ user, id: source });
    setRequestOwner(owner);
    void run(async () => ({
      ...owner,
      summary: await api.getForecastDecomposition(lob, {
        run_id: source === "legacy" ? undefined : source,
        legacy: source === "legacy", season_length: Number(season),
      }),
    }));
  };
  const maximum = Math.max(0, ...(summary?.components.map(item => item.mean_abs_value ?? 0) ?? []));

  return (
    <section aria-label="Forecast decomposition" className="space-y-3">
      <h2 className="text-lg font-semibold">Forecast decomposition</h2>
      <p className="text-sm text-muted-foreground">
        Centered moving-average trend, seasonal-position averages and residuals.
        {" "}Not model-specific feature attribution.
        {" "}This uses a separate forecast source, not the leaderboard run.
      </p>
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1">
          <label htmlFor="decomposition-run" className="text-xs font-medium">Decomposition forecast source</label>
          <select
            id="decomposition-run" value={selectedRun}
            onChange={event => { reset(); setSelection({ user, id: event.target.value }); }}
            disabled={!canView || runs.isLoading}
            className="block max-w-md rounded-md border bg-background px-3 py-1.5 text-sm"
          >
            <option value="">Latest listed forecast (separate source)</option>
            {canView && runs.data?.runs.map(item => (
              <option key={item.run_id} value={item.run_id}>
                {item.timestamp} | {item.champion_model || "Unknown model"} | {item.run_id}
              </option>
            ))}
            {canView && selectedRun && selectedRun !== "legacy" && !runs.data?.runs.some(item => item.run_id === selectedRun)
              && <option value={selectedRun}>Pinned forecast | {selectedRun}</option>}
            <option value="legacy">Legacy forecast files (unverified identity)</option>
          </select>
        </div>
        <div className="space-y-1">
          <label htmlFor="decomposition-season" className="text-xs font-medium">Season length (observations)</label>
          <input
            id="decomposition-season" type="number" min={1} step={1} value={season}
            onChange={event => { reset(); setSeason(event.target.value); }}
            className="block w-28 rounded-md border bg-background px-3 py-1.5 text-sm"
          />
        </div>
        <button
          type="button" onClick={load}
          disabled={!canView || !validSeason || (ownsStatus && loading) || (!selectedRun && (!runs.data || runs.isFetching || runs.isError))}
          className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50"
        >
          Load decomposition
        </button>
        <button
          type="button"
          onClick={() => { reset(); setSelection({ user, id: "" }); void runs.refetch(); }}
          disabled={!canView || runs.isFetching}
          className="rounded-md border px-3 py-1.5 text-sm disabled:opacity-50"
        >
          Refresh forecast sources
        </button>
      </div>
      {!canView && <p className="text-sm">Sign in with permission to view metrics.</p>}
      {!validSeason && <p role="alert" className="text-sm">Enter a positive whole-number season length.</p>}
      {canView && runs.error && <ErrorDisplay message={runs.error.message} onRetry={() => runs.refetch()} />}
      {ownsStatus && loading && <p role="status">Computing forecast decomposition...</p>}
      {ownsStatus && error && <ErrorDisplay message={error} onRetry={load} />}
      {summary && (
        <div className="space-y-3" data-testid="decomposition-result">
          <div className="rounded-lg border p-4 text-sm space-y-1">
            <p>Method: <code>{summary.method}</code></p>
            <p>Forecast source: <code data-testid="decomposition-run-id">{summary.run_id ?? "Legacy / unverified"}</code></p>
            {summary.artifact_source === "committed"
              ? <p>This immutable forecast ID is pinned until you change or refresh the source.</p>
              : <p>Legacy files have no verified immutable run identity; a later read can see different files.</p>}
            <p>
              History: {summary.history_source === "committed"
                ? "recorded prepared snapshot from this same forecast run"
                : summary.history_source === "uploaded"
                  ? "uploaded alternative; not the committed prepared snapshot"
                  : "legacy history; preparation stage and lineage unknown"}.
            </p>
            <p>Model evidence: {summary.forecast_model_source === "manifest"
              ? "recorded run manifest" : summary.forecast_model_source === "forecast_frame"
                ? "recorded forecast rows" : "not recorded"}; no fitted model is loaded.</p>
            <p>Centered window: {summary.trend_window} observations; seasonal period: {summary.season_length} observations.</p>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <MetricCard label="Recorded forecast model" value={summary.forecast_model ?? "Unknown (not recorded)"} />
            <MetricCard label="Forecast series" value={summary.series_count} />
            <MetricCard label="LOB" value={summary.lob} />
          </div>
          <div className="rounded-lg border p-4">
            <h3 className="text-sm font-medium mb-2">Component magnitude</h3>
            <p className="text-xs text-muted-foreground mb-3">
              Mean absolute component over defined historical observations of forecast series.
              {" "}Sample standard deviation uses signed values. Undefined centered-window edges are excluded, not zero.
            </p>
            <div className="space-y-3">
              {summary.components.map(component => (
                <div key={component.component} data-testid={`component-${component.component}`}>
                  <div className="flex items-center gap-3 text-sm">
                    <span className="w-20">{component.component}</span>
                    <div className="flex-1 h-4 bg-muted rounded-full overflow-hidden" aria-hidden="true">
                      {component.mean_abs_value !== null && <div
                        className="h-full bg-primary rounded-full"
                        style={{ width: `${maximum > 0 ? component.mean_abs_value / maximum * 100 : 0}%` }}
                      />}
                    </div>
                    <span>{component.mean_abs_value === null ? "Undefined" : component.mean_abs_value.toFixed(4)}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {component.defined_count} defined / {component.undefined_count} undefined;
                    {" "}sample standard deviation: {component.std === null ? "Undefined" : component.std.toFixed(4)}.
                  </p>
                </div>
              ))}
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Checked all {summary.checked_rows} rows: {summary.history_rows} history and {summary.forecast_rows} forecast.
            {" "}{summary.excluded_history_rows} history-only rows excluded from statistics.
            {" "}Preview: {summary.preview_rows} rows{summary.preview_truncated ? " (truncated)" : ""}.
          </p>
          {summary.decomposition_preview.length > 0 && <DataTable
            columns={Object.keys(summary.decomposition_preview[0]).map(key => ({ key, label: key, sortable: true }))}
            data={summary.decomposition_preview}
            pageSize={10}
          />}
        </div>
      )}
    </section>
  );
}
