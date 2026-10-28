"use client";

import { useState, useEffect } from "react";
import { useLob } from "@/providers/lob-provider";
import { LeaderboardBar } from "@/components/charts/leaderboard-bar";
import { FVACascade } from "@/components/charts/fva-cascade";
import { CalibrationPlot } from "@/components/charts/calibration-plot";
import { ConfigTunerPanel } from "@/components/ai/config-tuner-panel";
import { ForecastDecompositionPanel } from "@/components/backtest/forecast-decomposition-panel";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { NoDataGuide } from "@/components/shared/no-data-guide";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { DataTable } from "@/components/data/data-table";
import { useLeaderboard } from "@/hooks/use-leaderboard";
import { api } from "@/lib/api-client";
import { formatPct, formatMetric, leaderboardScore, metricLabel, storedMetricScores } from "@/lib/utils";
import { WorkflowNav } from "@/components/shared/workflow-nav";
import type { FVAResponse, CalibrationResponse, LeaderboardEntry } from "@/lib/types";

function StoredMetricValues({ entry }: { entry: LeaderboardEntry }) {
  const scores = Object.entries(storedMetricScores(entry));
  if (scores.length === 0) return <span>No stored metric values</span>;
  return <div className="space-y-1">
    {scores.map(([metric, value]) => (
      <div key={metric}>{metricLabel(metric)}: {formatMetric(metric, value)}</div>
    ))}
  </div>;
}

export default function BacktestPage() {
  const { lob, setLob } = useLob();
  const [runType, setRunType] = useState("backtest");
  const { data, isLoading, error, refetch } = useLeaderboard(lob, runType);

  // FVA state
  const [fvaData, setFvaData] = useState<FVAResponse | null>(null);
  const [fvaLoading, setFvaLoading] = useState(false);
  const [fvaError, setFvaError] = useState<string | null>(null);
  const [fvaReload, setFvaReload] = useState(0);

  // Calibration state
  const [calData, setCalData] = useState<CalibrationResponse | null>(null);
  const [calLoading, setCalLoading] = useState(false);
  const [calError, setCalError] = useState<string | null>(null);
  const [calReload, setCalReload] = useState(0);

  useEffect(() => {
    let active = true;
    setFvaData(null);
    setFvaLoading(false);
    setFvaError(null);
    if (!data) return;
    setFvaLoading(true);
    setFvaError(null);
    api.getFVA(lob, runType, data.run_id ?? undefined)
      .then(result => { if (active) setFvaData(result); })
      .catch((err) => { if (active) setFvaError(err instanceof Error ? err.message : "Failed to load FVA data"); })
      .finally(() => { if (active) setFvaLoading(false); });
    return () => { active = false; };
  }, [lob, runType, data, fvaReload]);

  useEffect(() => {
    let active = true;
    setCalData(null);
    setCalLoading(false);
    setCalError(null);
    if (!data) return;
    setCalLoading(true);
    setCalError(null);
    api.getCalibration(lob, runType, data.run_id ?? undefined)
      .then(result => { if (active) setCalData(result); })
      .catch((err) => { if (active) setCalError(err instanceof Error ? err.message : "Failed to load calibration data"); })
      .finally(() => { if (active) setCalLoading(false); });
    return () => { active = false; };
  }, [lob, runType, data, calReload]);

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

  const provenance = data?.scoring_provenance;
  const primaryMetric = provenance?.status === "recorded" ? provenance.primary_metric ?? null : null;
  const scoreLabel = primaryMetric ? metricLabel(primaryMetric, provenance?.policy) : "Stored metric values";
  const groups = new Set(data?.entries.map(entry =>
    `${entry.competition_scope ?? provenance?.competition_scope ?? "unknown"}:${entry.group_key ?? lob}`));
  const knownScope = provenance?.status === "recorded" && !!provenance.competition_scope && primaryMetric !== null;
  const singleCompetition = knownScope && groups.size === 1;
  const leading = singleCompetition ? data?.entries.find(entry => entry.rank === 1) : undefined;

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Backtest Results</h1>
        <p className="text-sm text-muted-foreground">
          Model leaderboard, FVA analysis, calibration, forecast decomposition, and AI config tuning.
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
            <p className="text-sm text-muted-foreground" data-testid="scoring-provenance">
              {provenance?.status === "recorded"
                ? `${provenance.source === "stored" ? "Stored ranking" : "Computed from raw observations"} | ${provenance.policy} | ${scoreLabel} | Scope: ${provenance.competition_scope ?? "unknown"}`
                : "Scoring policy unknown: historical values and ordering are shown unchanged."}
            </p>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricCard label="Models" value={new Set(data.entries.map(entry => entry.model)).size} />
              <MetricCard
                label={knownScope ? `Best ${scoreLabel}` : "Primary criterion"}
                value={!primaryMetric ? "Unknown" : leading
                  ? formatMetric(primaryMetric, leaderboardScore(leading, primaryMetric)) : "N/A"}
              />
              <MetricCard
                label="Best Model"
                value={!knownScope ? "Unknown scope" : groups.size > 1 ? "Per competition" : leading?.model ?? "N/A"}
              />
              <MetricCard label="Run Type" value={data.run_type} />
            </div>
            <div className="rounded-lg border p-4">
              {primaryMetric ? (
                <LeaderboardBar entries={data.entries} primaryMetric={primaryMetric} scoringPolicy={provenance?.policy} />
              ) : (
                <p className="text-sm text-muted-foreground">
                  Primary criterion unknown; available named stored scores are shown below without re-ranking.
                </p>
              )}
            </div>

            {/* Leaderboard table */}
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-sm" aria-label="Stored model leaderboard">
                <thead>
                  <tr className="border-b bg-muted/50 text-left text-xs text-muted-foreground">
                    <th className="px-3 py-2">Rank</th>
                    <th className="px-3 py-2">Competition</th>
                    <th className="px-3 py-2">Model</th>
                    <th className="px-3 py-2">{scoreLabel}</th>
                    <th className="px-3 py-2">Bias</th>
                    <th className="px-3 py-2">Series</th>
                  </tr>
                </thead>
                <tbody>
                  {data.entries.map((e) => (
                      <tr key={`${e.competition_scope ?? "unknown"}:${e.group_key ?? "unknown"}:${e.model}`}
                        data-model={e.model} data-group={e.group_key ?? "unknown"} className="border-b last:border-0">
                        <td className="px-3 py-2 font-medium">{e.rank}</td>
                        <td className="px-3 py-2">{e.group_key ?? "Unknown"}</td>
                        <td className="px-3 py-2">{e.model}</td>
                        <td className="px-3 py-2">{primaryMetric
                          ? formatMetric(primaryMetric, leaderboardScore(e, primaryMetric))
                          : <StoredMetricValues entry={e} />}</td>
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
        <p className="text-xs text-muted-foreground">Current raw-observation analysis for this run, separate from its stored ranking.</p>
        {fvaLoading && <ChartSkeleton />}
        {fvaError && <ErrorDisplay message={fvaError} onRetry={() => setFvaReload(value => value + 1)} />}
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
        {calError && <ErrorDisplay message={calError} onRetry={() => setCalReload(value => value + 1)} />}
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

      <ForecastDecompositionPanel
        key={`${lob}:${runType}:${data?.run_id ?? "unknown"}:${data?.artifact_source ?? "unresolved"}`}
        lob={lob} runType={runType} leaderboardRunId={data?.run_id}
      />

      {/* AI Config Tuner — live endpoint */}
      <section>
        <h2 className="text-lg font-semibold mb-3">AI Configuration Tuner</h2>
        <ConfigTunerPanel lob={lob} runType={runType} />
      </section>

      <WorkflowNav currentStep="/backtest" />
    </div>
  );
}
