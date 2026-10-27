"use client";

import { useState, useEffect } from "react";
import { useLob } from "@/providers/lob-provider";
import { NoDataGuide } from "@/components/shared/no-data-guide";
import { DriftHistogram } from "@/components/charts/drift-histogram";
import { TriagePanel } from "@/components/ai/triage-panel";
import { DataTable } from "@/components/data/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useDrift } from "@/hooks/use-drift";
import { useAudit } from "@/hooks/use-audit";
import { api } from "@/lib/api-client";
import { SEVERITY_COLORS } from "@/lib/constants";
import type { ManifestListResponse, CostListResponse } from "@/lib/types";
import { WorkflowNav } from "@/components/shared/workflow-nav";

function ManifestsTabContent({ lob }: { lob: string }) {
  const [manifests, setManifests] = useState<ManifestListResponse | null>(null);
  const [manifestsLoading, setManifestsLoading] = useState(false);
  const [manifestsError, setManifestsError] = useState<string | null>(null);

  const [costs, setCosts] = useState<CostListResponse | null>(null);
  const [costsLoading, setCostsLoading] = useState(false);
  const [costsError, setCostsError] = useState<string | null>(null);

  useEffect(() => {
    setManifestsLoading(true);
    setManifestsError(null);
    api.listManifests(lob)
      .then(setManifests)
      .catch((err) => setManifestsError(err instanceof Error ? err.message : "Failed to load manifests"))
      .finally(() => setManifestsLoading(false));

    setCostsLoading(true);
    setCostsError(null);
    api.getCosts(lob)
      .then(setCosts)
      .catch((err) => setCostsError(err instanceof Error ? err.message : "Failed to load costs"))
      .finally(() => setCostsLoading(false));
  }, [lob]);

  return (
    <div className="space-y-6">
      {/* Pipeline Manifests */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Pipeline Manifests</h2>
        {manifestsLoading && <ChartSkeleton />}
        {manifestsError && (
          <ErrorDisplay
            message={manifestsError}
            onRetry={() => {
              setManifestsLoading(true);
              setManifestsError(null);
              api.listManifests(lob)
                .then(setManifests)
                .catch((err) => setManifestsError(err instanceof Error ? err.message : "Failed"))
                .finally(() => setManifestsLoading(false));
            }}
          />
        )}
        {manifests && (
          <>
            <MetricCard label="Total Manifests" value={manifests.count} className="w-fit" />
            {manifests.manifests.length > 0 ? (
              <DataTable
                columns={[
                  { key: "run_id", label: "Run ID", sortable: true },
                  { key: "timestamp", label: "Timestamp", sortable: true },
                  { key: "lob", label: "LOB", sortable: true },
                  { key: "series_count", label: "Series", sortable: true },
                  { key: "champion_model", label: "Champion", sortable: true },
                  { key: "backtest_wmape", label: "WMAPE", sortable: true,
                    render: (v) => v != null ? `${((v as number) * 100).toFixed(1)}%` : "N/A",
                  },
                  { key: "forecast_horizon", label: "Horizon", sortable: true },
                  { key: "forecast_rows", label: "Rows", sortable: true },
                  { key: "validation_passed", label: "Valid", sortable: true,
                    render: (v) => (
                      <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                        v ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                          : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                      }`}>
                        {v ? "Yes" : "No"}
                      </span>
                    ),
                  },
                  { key: "validation_warnings", label: "Warnings", sortable: true },
                ]}
                data={manifests.manifests as unknown as Record<string, unknown>[]}
                pageSize={10}
              />
            ) : (
              <p className="text-sm text-muted-foreground text-center py-4">No manifests found for this LOB.</p>
            )}
          </>
        )}
      </section>

      {/* Cost Tracking */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Cost Tracking</h2>
        {costsLoading && <ChartSkeleton />}
        {costsError && (
          <ErrorDisplay
            message={costsError}
            onRetry={() => {
              setCostsLoading(true);
              setCostsError(null);
              api.getCosts(lob)
                .then(setCosts)
                .catch((err) => setCostsError(err instanceof Error ? err.message : "Failed"))
                .finally(() => setCostsLoading(false));
            }}
          />
        )}
        {costs && (
          <>
            <MetricCard label="Cost Records" value={costs.count} className="w-fit" />
            {costs.costs.length > 0 ? (
              <DataTable
                columns={[
                  { key: "run_id", label: "Run ID", sortable: true },
                  { key: "timestamp", label: "Timestamp", sortable: true },
                  { key: "lob", label: "LOB", sortable: true },
                  { key: "series_count", label: "Series", sortable: true },
                  { key: "champion_model", label: "Model", sortable: true },
                  { key: "total_seconds", label: "Total (s)", sortable: true,
                    render: (v) => typeof v === "number" ? v.toFixed(1) : "N/A",
                  },
                  { key: "seconds_per_series", label: "Per Series (s)", sortable: true,
                    render: (v) => v != null ? (v as number).toFixed(3) : "N/A",
                  },
                ]}
                data={costs.costs as unknown as Record<string, unknown>[]}
                pageSize={10}
              />
            ) : (
              <p className="text-sm text-muted-foreground text-center py-4">No cost data found for this LOB.</p>
            )}
          </>
        )}
      </section>
    </div>
  );
}

export default function HealthPage() {
  const { lob, setLob } = useLob();
  const [activeTab, setActiveTab] = useState<"drift" | "audit" | "manifests">("drift");
  const [auditAction, setAuditAction] = useState("");
  const [auditLimit, setAuditLimit] = useState(100);

  const drift = useDrift(lob);
  const audit = useAudit({
    action: auditAction || undefined,
    limit: auditLimit,
  });

  const tabs = [
    { id: "drift" as const, label: "Drift & Triage" },
    { id: "audit" as const, label: "Audit Log" },
    { id: "manifests" as const, label: "Pipeline Manifests" },
  ];

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Platform Health</h1>
        <p className="text-sm text-muted-foreground">
          Monitor drift, triage anomalies, review audit log, and track pipeline manifests.
        </p>
      </div>

      {/* LOB Control */}
      <div className="flex items-end gap-4">
        <div className="space-y-1">
          <label className="text-xs font-medium">LOB</label>
          <input
            type="text"
            value={lob}
            onChange={(e) => setLob(e.target.value)}
            className="rounded-md border bg-background px-3 py-1.5 text-sm w-32"
          />
        </div>
      </div>

      {/* Tab navigation */}
      <div className="flex gap-1 border-b">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Drift & Triage Tab */}
      {activeTab === "drift" && (
        <div className="space-y-6">
          {drift.isLoading && <ChartSkeleton />}
          {drift.error && (drift.error.message.includes("404") ? (
            <NoDataGuide lob={lob} dataType="drift" />
          ) : (
            <ErrorDisplay message={drift.error.message} onRetry={() => drift.refetch()} />
          ))}
          {drift.data && (
            <>
              <div className="grid grid-cols-3 gap-3">
                <MetricCard label="Total Alerts" value={drift.data.alerts.length} />
                <MetricCard label="Critical" value={drift.data.n_critical} />
                <MetricCard label="Warning" value={drift.data.n_warning} />
              </div>

              {/* Alert histogram */}
              {drift.data.alerts.length > 0 && (
                <section className="space-y-3">
                  <h2 className="text-lg font-semibold">Alert Distribution</h2>
                  <div className="rounded-lg border p-4">
                    <DriftHistogram alerts={drift.data.alerts} />
                  </div>
                </section>
              )}

              {/* Alert table */}
              {drift.data.alerts.length > 0 && (
                <section className="space-y-3">
                  <h2 className="text-lg font-semibold">Drift Alerts</h2>
                  <div className="overflow-x-auto rounded-md border">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b bg-muted/50 text-left text-xs text-muted-foreground">
                          <th className="px-3 py-2">Series</th>
                          <th className="px-3 py-2">Metric</th>
                          <th className="px-3 py-2">Severity</th>
                          <th className="px-3 py-2">Current</th>
                          <th className="px-3 py-2">Baseline</th>
                          <th className="px-3 py-2">Message</th>
                        </tr>
                      </thead>
                      <tbody>
                        {drift.data.alerts.map((a, i) => (
                          <tr key={i} className="border-b last:border-0">
                            <td className="px-3 py-2 font-medium">{a.series_id}</td>
                            <td className="px-3 py-2">{a.metric}</td>
                            <td className="px-3 py-2">
                              <span
                                className="rounded px-1.5 py-0.5 text-xs font-medium text-white"
                                style={{ backgroundColor: SEVERITY_COLORS[a.severity] }}
                              >
                                {a.severity}
                              </span>
                            </td>
                            <td className="px-3 py-2">{a.current_value.toFixed(3)}</td>
                            <td className="px-3 py-2">{a.baseline_value.toFixed(3)}</td>
                            <td className="px-3 py-2 text-xs text-muted-foreground max-w-xs truncate">
                              {a.message}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}

              {/* AI Triage */}
              <section>
                <h2 className="text-lg font-semibold mb-3">AI Anomaly Triage</h2>
                <TriagePanel lob={lob} />
              </section>
            </>
          )}

          {drift.data && drift.data.alerts.length === 0 && (
            <div className="rounded-lg border p-8 text-center">
              <p className="text-sm text-muted-foreground">No drift alerts detected for this LOB.</p>
            </div>
          )}
        </div>
      )}

      {/* Audit Log Tab */}
      {activeTab === "audit" && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-1">
              <label className="text-xs font-medium">Action Filter</label>
              <input
                type="text"
                value={auditAction}
                onChange={(e) => setAuditAction(e.target.value)}
                placeholder="e.g. create_override"
                className="rounded-md border bg-background px-3 py-1.5 text-sm w-48"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">Max Entries</label>
              <input
                type="number"
                value={auditLimit}
                onChange={(e) => setAuditLimit(Number(e.target.value))}
                min={10}
                max={1000}
                className="rounded-md border bg-background px-3 py-1.5 text-sm w-24"
              />
            </div>
          </div>

          {audit.isLoading && <ChartSkeleton />}
          {audit.error && (
            <ErrorDisplay message={audit.error.message} onRetry={() => audit.refetch()} />
          )}
          {audit.data && (
            <>
              <MetricCard label="Events Returned" value={audit.data.count} className="w-fit" />
              <DataTable
                columns={[
                  { key: "timestamp", label: "Timestamp", sortable: true },
                  { key: "user_id", label: "User", sortable: true },
                  { key: "action", label: "Action", sortable: true },
                  { key: "resource_type", label: "Resource", sortable: true },
                  { key: "status", label: "Status", sortable: true,
                    render: (v) => (
                      <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                        v === "success"
                          ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                          : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                      }`}>
                        {String(v)}
                      </span>
                    ),
                  },
                ]}
                data={audit.data.events as unknown as Record<string, unknown>[]}
                pageSize={20}
              />
            </>
          )}
        </div>
      )}

      {/* Manifests Tab */}
      {activeTab === "manifests" && (
        <ManifestsTabContent lob={lob} />
      )}

      <WorkflowNav currentStep="/health" />
    </div>
  );
}
