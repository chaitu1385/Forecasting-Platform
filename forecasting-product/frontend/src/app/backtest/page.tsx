"use client";

import { useState, useEffect } from "react";
import { useLob } from "@/providers/lob-provider";
import { LeaderboardBar } from "@/components/charts/leaderboard-bar";
import { FVACascade } from "@/components/charts/fva-cascade";
import { CalibrationPlot } from "@/components/charts/calibration-plot";
import { ConfigTunerPanel } from "@/components/ai/config-tuner-panel";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { NoDataGuide } from "@/components/shared/no-data-guide";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { DataTable } from "@/components/data/data-table";
import { useLeaderboard } from "@/hooks/use-leaderboard";
import { api } from "@/lib/api-client";
import { formatPct, hasFiniteWmape } from "@/lib/utils";
import { WorkflowNav } from "@/components/shared/workflow-nav";
import type { FVAResponse, CalibrationResponse, ShapResponse } from "@/lib/types";

export default function BacktestPage() {
  const { lob, setLob } = useLob();
  const [runType, setRunType] = useState("backtest");
  const { data, isLoading, error, refetch } = useLeaderboard(lob, runType);

  // FVA state
  const [fvaData, setFvaData] = useState<FVAResponse | null>(null);
  const [fvaLoading, setFvaLoading] = useState(false);
  const [fvaError, setFvaError] = useState<string | null>(null);

  // Calibration state
  const [calData, setCalData] = useState<CalibrationResponse | null>(null);
  const [calLoading, setCalLoading] = useState(false);
  const [calError, setCalError] = useState<string | null>(null);

  // SHAP state
  const [shapData, setShapData] = useState<ShapResponse | null>(null);
  const [shapLoading, setShapLoading] = useState(false);
  const [shapError, setShapError] = useState<string | null>(null);

  // Fetch FVA and Calibration when lob/runType changes
  useEffect(() => {
    setFvaLoading(true);
    setFvaError(null);
    api.getFVA(lob, runType)
      .then(setFvaData)
      .catch((err) => setFvaError(err instanceof Error ? err.message : "Failed to load FVA data"))
      .finally(() => setFvaLoading(false));
  }, [lob, runType]);

  useEffect(() => {
    setCalLoading(true);
    setCalError(null);
    api.getCalibration(lob, runType)
      .then(setCalData)
      .catch((err) => setCalError(err instanceof Error ? err.message : "Failed to load calibration data"))
      .finally(() => setCalLoading(false));
  }, [lob, runType]);

  // Fetch SHAP on demand
  const loadShap = () => {
    setShapLoading(true);
    setShapError(null);
    api.getShap(lob)
      .then(setShapData)
      .catch((err) => setShapError(err instanceof Error ? err.message : "Failed to load SHAP data"))
      .finally(() => setShapLoading(false));
  };

  // Transform FVA summary data for FVACascade chart
  const fvaChartData = fvaData?.summary?.map((row) => ({
    layer: String(row.layer || row.model || ""),
    wmape: Number(row.wmape || 0),
  })) ?? [];

  // Transform calibration data for CalibrationPlot
  const calChartData: { nominal: number; empirical: number; model?: string }[] = [];
  if (calData?.model_reports) {
    for (const [model, coverages] of Object.entries(calData.model_reports)) {
      for (const cov of coverages) {
        calChartData.push({ nominal: cov.nominal, empirical: cov.empirical, model });
      }
    }
  }

  const rankedEntries = data?.entries.filter(hasFiniteWmape).sort((a, b) => a.wmape - b.wmape) ?? [];

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Backtest Results</h1>
        <p className="text-sm text-muted-foreground">
          Model leaderboard, FVA analysis, calibration, feature attribution, and AI config tuning.
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
          <label className="text-xs font-medium">Run Type</label>
          <select
            value={runType}
            onChange={(e) => setRunType(e.target.value)}
            className="rounded-md border bg-background px-3 py-1.5 text-sm"
          >
            <option value="backtest">Backtest</option>
            <option value="live">Live</option>
          </select>
        </div>
        <button
          onClick={() => refetch()}
          className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          Load
        </button>
      </div>

      {/* Leaderboard */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Model Leaderboard</h2>
        {isLoading && <ChartSkeleton />}
        {error && (error.message.includes("404") ? (
          <NoDataGuide lob={lob} dataType="backtest" />
        ) : (
          <ErrorDisplay message={error.message} onRetry={() => refetch()} />
        ))}
        {data && (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricCard label="Models" value={data.entries.length} />
              <MetricCard
                label="Best WMAPE"
                value={rankedEntries.length > 0 ? formatPct(rankedEntries[0].wmape) : "N/A"}
              />
              <MetricCard
                label="Best Model"
                value={rankedEntries[0]?.model ?? "N/A"}
              />
              <MetricCard label="Run Type" value={data.run_type} />
            </div>
            <div className="rounded-lg border p-4">
              <LeaderboardBar entries={data.entries} />
            </div>

            {/* Leaderboard table */}
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50 text-left text-xs text-muted-foreground">
                    <th className="px-3 py-2">Rank</th>
                    <th className="px-3 py-2">Model</th>
                    <th className="px-3 py-2">WMAPE</th>
                    <th className="px-3 py-2">Bias</th>
                    <th className="px-3 py-2">Series</th>
                  </tr>
                </thead>
                <tbody>
                  {[...data.entries]
                    .sort((a, b) => a.rank - b.rank)
                    .map((e) => (
                      <tr key={e.model} className="border-b last:border-0">
                        <td className="px-3 py-2 font-medium">{e.rank}</td>
                        <td className="px-3 py-2">{e.model}</td>
                        <td className="px-3 py-2">{formatPct(e.wmape)}</td>
                        <td className="px-3 py-2">{formatPct(e.normalized_bias)}</td>
                        <td className="px-3 py-2">{e.n_series}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>

      {/* FVA Cascade (live) */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Forecast Value Added (FVA)</h2>
        {fvaLoading && <ChartSkeleton />}
        {fvaError && <ErrorDisplay message={fvaError} onRetry={() => {
          setFvaLoading(true);
          setFvaError(null);
          api.getFVA(lob, runType).then(setFvaData).catch((err) => setFvaError(err instanceof Error ? err.message : "Failed")).finally(() => setFvaLoading(false));
        }} />}
        {fvaData && (
          <div className="rounded-lg border p-4">
            {fvaChartData.length > 0 ? (
              <FVACascade data={fvaChartData} />
            ) : (
              <p className="text-sm text-muted-foreground text-center py-4">No FVA data available for this LOB.</p>
            )}
            {fvaData.layer_leaderboard && fvaData.layer_leaderboard.length > 0 && (
              <div className="mt-4">
                <h3 className="text-sm font-medium mb-2">Layer Leaderboard</h3>
                <DataTable
                  columns={Object.keys(fvaData.layer_leaderboard[0]).map(k => ({ key: k, label: k, sortable: true }))}
                  data={fvaData.layer_leaderboard}
                  pageSize={10}
                />
              </div>
            )}
          </div>
        )}
      </section>

      {/* Calibration (live) */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Prediction Interval Calibration</h2>
        {calLoading && <ChartSkeleton />}
        {calError && <ErrorDisplay message={calError} onRetry={() => {
          setCalLoading(true);
          setCalError(null);
          api.getCalibration(lob, runType).then(setCalData).catch((err) => setCalError(err instanceof Error ? err.message : "Failed")).finally(() => setCalLoading(false));
        }} />}
        {calData && (
          <div className="rounded-lg border p-4">
            {calChartData.length > 0 ? (
              <CalibrationPlot data={calChartData} />
            ) : (
              <p className="text-sm text-muted-foreground text-center py-4">No calibration data available for this LOB.</p>
            )}
          </div>
        )}
      </section>

      {/* SHAP Feature Attribution */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">SHAP Feature Attribution</h2>
        {!shapData && !shapLoading && (
          <div className="rounded-lg border p-6 text-center">
            <p className="text-sm text-muted-foreground mb-3">
              Top features by mean |SHAP| for LightGBM/XGBoost models with per-series waterfall.
            </p>
            <button
              onClick={loadShap}
              className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Load SHAP Analysis
            </button>
          </div>
        )}
        {shapLoading && <ChartSkeleton />}
        {shapError && <ErrorDisplay message={shapError} onRetry={loadShap} />}
        {shapData && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              <MetricCard label="Model" value={shapData.model} />
              <MetricCard label="Features" value={shapData.feature_importance.length} />
              <MetricCard label="LOB" value={shapData.lob} />
            </div>
            <div className="rounded-lg border p-4">
              <h3 className="text-sm font-medium mb-2">Feature Importance (Mean |SHAP|)</h3>
              <div className="space-y-2">
                {shapData.feature_importance.map((fi) => {
                  const maxVal = Math.max(...shapData.feature_importance.map(f => f.mean_abs_value));
                  const pct = maxVal > 0 ? (fi.mean_abs_value / maxVal) * 100 : 0;
                  return (
                    <div key={fi.feature} className="flex items-center gap-3">
                      <span className="text-xs font-mono w-32 truncate" title={fi.feature}>{fi.feature}</span>
                      <div className="flex-1 h-4 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary rounded-full transition-all"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-16 text-right">{fi.mean_abs_value.toFixed(4)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
            {shapData.decomposition_preview.length > 0 && (
              <div className="mt-4">
                <h3 className="text-sm font-medium mb-2">Decomposition Preview</h3>
                <DataTable
                  columns={Object.keys(shapData.decomposition_preview[0]).map(k => ({ key: k, label: k, sortable: true }))}
                  data={shapData.decomposition_preview}
                  pageSize={10}
                />
              </div>
            )}
          </div>
        )}
      </section>

      {/* AI Config Tuner — live endpoint */}
      <section>
        <h2 className="text-lg font-semibold mb-3">AI Configuration Tuner</h2>
        <ConfigTunerPanel lob={lob} runType={runType} />
      </section>

      <WorkflowNav currentStep="/backtest" />
    </div>
  );
}
