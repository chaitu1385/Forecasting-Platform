"use client";

import { Suspense, useState, useEffect, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import { useLob } from "@/providers/lob-provider";
import { NLQueryPanel } from "@/components/ai/nl-query-panel";
import { MetricCard } from "@/components/shared/metric-card";
import { DataTable } from "@/components/data/data-table";
import { Skeleton } from "@/components/shared/loading-skeleton";
import { DEMAND_CLASS_COLORS } from "@/lib/constants";
import { BreakTimeline } from "@/components/charts/break-timeline";
import { CleansingOverlay } from "@/components/charts/cleansing-overlay";
import { NoDataGuide } from "@/components/shared/no-data-guide";
import { WorkflowNav } from "@/components/shared/workflow-nav";
import { api } from "@/lib/api-client";
import type {
  SeriesItem,
  SeriesHistoryPoint,
  BreakDetectionResponse,
  CleansingAuditResponse,
  RegressorScreenResponse,
} from "@/lib/types";

// ── Helpers ─────────────────────────────────────────────────────────────────

function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
      {message}
    </div>
  );
}

function PanelLoading() {
  return (
    <div className="space-y-3 rounded-lg border p-4">
      <Skeleton className="h-5 w-48" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-3/4" />
      <Skeleton className="h-32 w-full" />
    </div>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function SeriesExplorerPage() {
  return (
    <Suspense fallback={<PanelLoading />}>
      <SeriesExplorerContent />
    </Suspense>
  );
}

function SeriesExplorerContent() {
  const searchParams = useSearchParams();
  const { lob: sharedLob, setLob: setSharedLob } = useLob();
  const [lob, setLobLocal] = useState(searchParams.get("lob") || sharedLob);

  // Sync local changes to shared context
  const setLob = useCallback((value: string) => {
    setLobLocal(value);
    setSharedLob(value);
  }, [setSharedLob]);

  // Series list state
  const [seriesList, setSeriesList] = useState<SeriesItem[]>([]);
  const [seriesLoading, setSeriesLoading] = useState(true);
  const [seriesError, setSeriesError] = useState<string | null>(null);
  const [selectedSeries, setSelectedSeries] = useState<string>("");

  // Structural break detection state
  const [breaks, setBreaks] = useState<BreakDetectionResponse | null>(null);
  const [breaksLoading, setBreaksLoading] = useState(true);
  const [breaksError, setBreaksError] = useState<string | null>(null);

  // Cleansing audit state
  const [cleansing, setCleansing] = useState<CleansingAuditResponse | null>(null);
  const [cleansingLoading, setCleansingLoading] = useState(true);
  const [cleansingError, setCleansingError] = useState<string | null>(null);

  // Regressor screening state
  const [regressors, setRegressors] = useState<RegressorScreenResponse | null>(null);
  const [regressorsLoading, setRegressorsLoading] = useState(true);
  const [regressorsError, setRegressorsError] = useState<string | null>(null);

  // Series history (for charts)
  const [seriesHistory, setSeriesHistory] = useState<SeriesHistoryPoint[]>([]);

  // Fetch series list
  const fetchSeries = useCallback(async () => {
    setSeriesLoading(true);
    setSeriesError(null);
    try {
      const res = await api.listSeries(lob);
      setSeriesList(res.series);
      if (res.series.length > 0 && !selectedSeries) {
        setSelectedSeries(res.series[0].series_id);
      }
    } catch (err) {
      setSeriesError(err instanceof Error ? err.message : "Failed to load series");
    } finally {
      setSeriesLoading(false);
    }
  }, [lob, selectedSeries]);

  // Fetch structural breaks
  const fetchBreaks = useCallback(async () => {
    setBreaksLoading(true);
    setBreaksError(null);
    try {
      const res = await api.detectBreaksFromLob(lob);
      setBreaks(res);
    } catch (err) {
      setBreaksError(err instanceof Error ? err.message : "Failed to detect breaks");
    } finally {
      setBreaksLoading(false);
    }
  }, [lob]);

  // Fetch cleansing audit
  const fetchCleansing = useCallback(async () => {
    setCleansingLoading(true);
    setCleansingError(null);
    try {
      const res = await api.cleansingAuditFromLob(lob);
      setCleansing(res);
    } catch (err) {
      setCleansingError(err instanceof Error ? err.message : "Failed to load cleansing audit");
    } finally {
      setCleansingLoading(false);
    }
  }, [lob]);

  // Fetch regressor screening
  const fetchRegressors = useCallback(async () => {
    setRegressorsLoading(true);
    setRegressorsError(null);
    try {
      const res = await api.regressorScreenFromLob(lob);
      setRegressors(res);
    } catch (err) {
      setRegressorsError(err instanceof Error ? err.message : "Failed to load regressor screening");
    } finally {
      setRegressorsLoading(false);
    }
  }, [lob]);

  // Fire all fetches when LOB changes
  useEffect(() => {
    fetchSeries();
    fetchBreaks();
    fetchCleansing();
    fetchRegressors();
  }, [fetchSeries, fetchBreaks, fetchCleansing, fetchRegressors]);

  // Fetch series history when selected series changes
  useEffect(() => {
    if (!selectedSeries || !lob) return;
    api.getSeriesHistory(lob, selectedSeries)
      .then((res) => setSeriesHistory(res.points))
      .catch(() => setSeriesHistory([]));
  }, [lob, selectedSeries]);

  // Derived: selected series details
  const currentSeries = seriesList.find((s) => s.series_id === selectedSeries);

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Series Explorer</h1>
        <p className="text-sm text-muted-foreground">
          Deep-dive into series-level demand patterns, structural breaks, quality, and cleansing.
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
          <label className="text-xs font-medium">Series</label>
          {seriesLoading ? (
            <Skeleton className="h-8 w-48" />
          ) : seriesError ? (
            <span className="text-xs text-red-500">Error loading series</span>
          ) : (
            <select
              value={selectedSeries}
              onChange={(e) => setSelectedSeries(e.target.value)}
              className="rounded-md border bg-background px-3 py-1.5 text-sm"
            >
              {seriesList.map((s) => (
                <option key={s.series_id} value={s.series_id}>
                  {s.series_id} ({s.demand_class})
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* Series Overview Cards */}
      <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {seriesLoading ? (
          <>
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </>
        ) : seriesError ? (
          <div className="col-span-4">
            {seriesError.includes("404") ? (
              <NoDataGuide lob={lob} dataType="series" />
            ) : (
              <ErrorMessage message={seriesError} />
            )}
          </div>
        ) : (
          <>
            <MetricCard
              label="Series Count"
              value={seriesList.length}
            />
            <MetricCard
              label="Observations"
              value={currentSeries?.n_observations ?? "-"}
              unit="periods"
            />
            <MetricCard
              label="Sparse"
              value={currentSeries?.is_sparse ? "Yes" : "No"}
            />
            <MetricCard
              label="Demand Class"
              value={currentSeries?.demand_class ?? "-"}
            />
          </>
        )}
      </section>

      {/* SBC Scatter */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">SBC Demand Classification</h2>
        {seriesLoading ? (
          <Skeleton className="h-64 w-full" />
        ) : seriesError ? (
          <ErrorMessage message={seriesError} />
        ) : (
          <div className="rounded-lg border p-4">
            <div className="relative h-64 w-full">
              <svg viewBox="0 0 400 250" className="h-full w-full">
                {/* Grid lines */}
                <line x1="50" y1="0" x2="50" y2="220" stroke="hsl(var(--border))" strokeDasharray="2" />
                <line x1="50" y1="220" x2="380" y2="220" stroke="hsl(var(--border))" strokeDasharray="2" />
                {/* Quadrant lines at ADI=1.32 and CV2=0.49 */}
                <line x1={50 + (1.32 / 3) * 330} y1="0" x2={50 + (1.32 / 3) * 330} y2="220" stroke="hsl(var(--muted-foreground))" strokeDasharray="4" opacity={0.5} />
                <line x1="50" y1={220 - (0.49 / 1) * 220} x2="380" y2={220 - (0.49 / 1) * 220} stroke="hsl(var(--muted-foreground))" strokeDasharray="4" opacity={0.5} />
                {/* Axis labels */}
                <text x="215" y="245" textAnchor="middle" className="fill-muted-foreground text-[10px]">ADI</text>
                <text x="15" y="110" textAnchor="middle" transform="rotate(-90,15,110)" className="fill-muted-foreground text-[10px]">CV2</text>
                {/* Quadrant labels */}
                <text x="120" y="30" className="fill-muted-foreground text-[9px]" opacity={0.6}>Smooth</text>
                <text x="300" y="30" className="fill-muted-foreground text-[9px]" opacity={0.6}>Intermittent</text>
                <text x="120" y="200" className="fill-muted-foreground text-[9px]" opacity={0.6}>Erratic</text>
                <text x="300" y="200" className="fill-muted-foreground text-[9px]" opacity={0.6}>Lumpy</text>
                {/* Points */}
                {seriesList.map((s) => (
                  <circle
                    key={s.series_id}
                    cx={50 + (Math.min(s.adi, 3) / 3) * 330}
                    cy={220 - (Math.min(s.cv2, 1) / 1) * 220}
                    r={s.series_id === selectedSeries ? 8 : 6}
                    fill={DEMAND_CLASS_COLORS[s.demand_class] ?? "#888"}
                    stroke={s.series_id === selectedSeries ? "hsl(var(--foreground))" : "none"}
                    strokeWidth={2}
                    opacity={0.8}
                    className="cursor-pointer"
                    onClick={() => setSelectedSeries(s.series_id)}
                  >
                    <title>{s.series_id}: ADI={s.adi.toFixed(2)}, CV2={s.cv2.toFixed(2)}</title>
                  </circle>
                ))}
              </svg>
            </div>
            <p className="mt-2 text-xs text-muted-foreground text-center">
              {seriesList.length} series loaded from LOB &quot;{lob}&quot;
            </p>
          </div>
        )}
      </section>

      {/* Structural Break Detection + Demand Cleansing Audit */}
      <div className="grid gap-4 sm:grid-cols-2">
        {/* Structural Break Detection */}
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Structural Break Detection</h2>
          {breaksLoading ? (
            <PanelLoading />
          ) : breaksError ? (
            <ErrorMessage message={breaksError} />
          ) : breaks ? (
            <div className="space-y-3 rounded-lg border p-4">
              <div className="grid grid-cols-3 gap-2">
                <MetricCard label="Total Series" value={breaks.total_series} />
                <MetricCard label="With Breaks" value={breaks.series_with_breaks} />
                <MetricCard label="Total Breaks" value={breaks.total_breaks} />
              </div>
              {breaks.warnings.length > 0 && (
                <div className="rounded border border-yellow-200 bg-yellow-50 p-2 text-xs text-yellow-800 dark:border-yellow-800 dark:bg-yellow-950 dark:text-yellow-300">
                  {breaks.warnings.map((w, i) => (
                    <p key={i}>{w}</p>
                  ))}
                </div>
              )}
              {/* Break timeline chart for selected series */}
              {selectedSeries && seriesHistory.length > 0 && (() => {
                const seriesBreaks = breaks.per_series.find(
                  (s) => (s as Record<string, unknown>).series_id === selectedSeries
                ) as Record<string, unknown> | undefined;
                const breakDates = Array.isArray(seriesBreaks?.break_dates)
                  ? (seriesBreaks.break_dates as string[])
                  : [];
                return breakDates.length > 0 ? (
                  <div className="rounded-lg border p-3">
                    <p className="text-xs font-medium mb-1">{selectedSeries} — Breaks</p>
                    <BreakTimeline data={seriesHistory} breakDates={breakDates} height={220} />
                  </div>
                ) : null;
              })()}
              {breaks.per_series.length > 0 && (
                <DataTable
                  columns={[
                    { key: "series_id", label: "Series", sortable: true },
                    { key: "n_breaks", label: "Breaks", sortable: true },
                    { key: "break_dates", label: "Break Dates", render: (v) => Array.isArray(v) ? v.join(", ") : String(v ?? "") },
                    { key: "method", label: "Method" },
                  ]}
                  data={breaks.per_series}
                  pageSize={5}
                />
              )}
            </div>
          ) : null}
        </section>

        {/* Demand Cleansing Audit */}
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Demand Cleansing Audit</h2>
          {cleansingLoading ? (
            <PanelLoading />
          ) : cleansingError ? (
            <ErrorMessage message={cleansingError} />
          ) : cleansing ? (
            <div className="space-y-3 rounded-lg border p-4">
              <div className="grid grid-cols-2 gap-2">
                <MetricCard label="Series w/ Outliers" value={cleansing.series_with_outliers} />
                <MetricCard label="Total Outliers" value={cleansing.total_outliers} />
                <MetricCard label="Outlier %" value={`${(cleansing.outlier_pct * 100).toFixed(1)}%`} />
                <MetricCard label="Rows Modified" value={cleansing.rows_modified} />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <MetricCard label="Stockout Series" value={cleansing.series_with_stockouts} />
                <MetricCard label="Stockout Periods" value={cleansing.total_stockout_periods} />
              </div>
              {/* Cleansing before/after chart for selected series */}
              {selectedSeries && seriesHistory.length > 0 && cleansing.cleansed_preview.length > 0 && (() => {
                // Build overlay data: match original history with cleansed preview
                const cleanedRows = cleansing.cleansed_preview.filter(
                  (r) => (r as Record<string, unknown>).series_id === selectedSeries
                ) as Record<string, unknown>[];
                if (cleanedRows.length === 0) return null;
                // Build a week→cleaned map
                const cleanedMap = new Map<string, number>();
                for (const r of cleanedRows) {
                  const w = String(r.week ?? r.date ?? "");
                  const v = Number(r.quantity ?? r.target ?? r.value ?? r.demand ?? 0);
                  if (w) cleanedMap.set(w, v);
                }
                const overlayData = seriesHistory.map((h) => {
                  const cleaned = cleanedMap.get(h.week) ?? h.value;
                  return {
                    week: h.week, original: h.value, cleaned,
                    is_outlier: h.value != null && cleaned != null && Math.abs(cleaned - h.value) > 0.01,
                  };
                });
                const hasChanges = overlayData.some((d) => d.is_outlier);
                return hasChanges ? (
                  <div className="rounded-lg border p-3">
                    <p className="text-xs font-medium mb-1">{selectedSeries} — Before/After Cleansing</p>
                    <CleansingOverlay data={overlayData} height={220} />
                  </div>
                ) : null;
              })()}
              {cleansing.per_series.length > 0 && (
                <DataTable
                  columns={[
                    { key: "series_id", label: "Series", sortable: true },
                    { key: "n_outliers", label: "Outliers", sortable: true },
                    { key: "n_stockouts", label: "Stockouts", sortable: true },
                    { key: "rows_modified", label: "Modified", sortable: true },
                  ]}
                  data={cleansing.per_series}
                  pageSize={5}
                />
              )}
            </div>
          ) : null}
        </section>
      </div>

      {/* Regressor Screening */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Regressor Screening</h2>
        {regressorsLoading ? (
          <PanelLoading />
        ) : regressorsError ? (
          <ErrorMessage message={regressorsError} />
        ) : regressors ? (
          <div className="space-y-4 rounded-lg border p-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricCard label="Screened" value={regressors.screened_columns.length} />
              <MetricCard label="Kept" value={regressors.screened_columns.length - regressors.dropped_columns.length} />
              <MetricCard label="Dropped" value={regressors.dropped_columns.length} />
              <MetricCard label="Low Variance" value={regressors.low_variance_columns.length} />
            </div>

            {/* Kept vs Dropped columns */}
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <h3 className="text-sm font-medium mb-1">Kept Columns</h3>
                <div className="flex flex-wrap gap-1">
                  {regressors.screened_columns
                    .filter((c) => !regressors.dropped_columns.includes(c))
                    .map((col) => (
                      <span
                        key={col}
                        className="rounded bg-green-100 px-2 py-0.5 text-xs text-green-800 dark:bg-green-900 dark:text-green-200"
                      >
                        {col}
                      </span>
                    ))}
                  {regressors.screened_columns.filter(
                    (c) => !regressors.dropped_columns.includes(c)
                  ).length === 0 && (
                    <span className="text-xs text-muted-foreground">None</span>
                  )}
                </div>
              </div>
              <div>
                <h3 className="text-sm font-medium mb-1">Dropped Columns</h3>
                <div className="flex flex-wrap gap-1">
                  {regressors.dropped_columns.map((col) => (
                    <span
                      key={col}
                      className="rounded bg-red-100 px-2 py-0.5 text-xs text-red-800 dark:bg-red-900 dark:text-red-200"
                    >
                      {col}
                    </span>
                  ))}
                  {regressors.dropped_columns.length === 0 && (
                    <span className="text-xs text-muted-foreground">None</span>
                  )}
                </div>
              </div>
            </div>

            {/* High correlation pairs */}
            {regressors.high_correlation_pairs.length > 0 && (
              <div>
                <h3 className="text-sm font-medium mb-1">High Correlation Pairs</h3>
                <DataTable
                  columns={[
                    { key: "col_a", label: "Column A", sortable: true },
                    { key: "col_b", label: "Column B", sortable: true },
                    { key: "correlation", label: "Correlation", sortable: true,
                      render: (v) => typeof v === "number" ? v.toFixed(3) : String(v ?? "") },
                  ]}
                  data={regressors.high_correlation_pairs}
                  pageSize={5}
                />
              </div>
            )}

            {/* Per-column stats */}
            {Object.keys(regressors.per_column_stats).length > 0 && (
              <div>
                <h3 className="text-sm font-medium mb-1">Per-Column Statistics</h3>
                <DataTable
                  columns={[
                    { key: "column", label: "Column", sortable: true },
                    { key: "variance", label: "Variance", sortable: true,
                      render: (v) => typeof v === "number" ? v.toFixed(4) : String(v ?? "") },
                    { key: "mi_score", label: "MI Score", sortable: true,
                      render: (v) => typeof v === "number" ? v.toFixed(4) : String(v ?? "") },
                    { key: "kept", label: "Status",
                      render: (v) =>
                        v ? (
                          <span className="text-green-600 dark:text-green-400">Kept</span>
                        ) : (
                          <span className="text-red-600 dark:text-red-400">Dropped</span>
                        ),
                    },
                  ]}
                  data={Object.entries(regressors.per_column_stats).map(
                    ([col, stats]) => ({
                      column: col,
                      ...stats,
                      kept: !regressors.dropped_columns.includes(col),
                    })
                  )}
                  pageSize={10}
                />
              </div>
            )}

            {/* Warnings */}
            {regressors.warnings.length > 0 && (
              <div className="rounded border border-yellow-200 bg-yellow-50 p-2 text-xs text-yellow-800 dark:border-yellow-800 dark:bg-yellow-950 dark:text-yellow-300">
                {regressors.warnings.map((w, i) => (
                  <p key={i}>{w}</p>
                ))}
              </div>
            )}
          </div>
        ) : null}
      </section>

      {/* AI Q&A — this is live (uses /ai/explain endpoint) */}
      <section>
        <h2 className="text-lg font-semibold mb-3">AI Series Analysis</h2>
        <NLQueryPanel
          seriesId={selectedSeries}
          lob={lob}
          suggestions={[
            "Why is demand volatile?",
            "Is there a trend?",
            "Any anomalies?",
            "What seasonal pattern exists?",
          ]}
        />
      </section>

      <WorkflowNav currentStep="/series-explorer" />
    </div>
  );
}
