"use client";

import { useState, useRef } from "react";
import { api } from "@/lib/api-client";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { MetricCard } from "@/components/shared/metric-card";
import { HierarchySunburst } from "@/components/charts/hierarchy-sunburst";
import { WorkflowNav } from "@/components/shared/workflow-nav";
import type {
  HierarchyBuildResponse,
  HierarchyAggregateResponse,
  HierarchyEffectiveMethod,
  HierarchyReconciliationMethod,
  HierarchyReconcileResponse,
} from "@/lib/types";

const RECONCILIATION_METHODS: { value: HierarchyReconciliationMethod; label: string; description: string }[] = [
  { value: "bottom_up", label: "Bottom-Up", description: "Preserve leaf forecasts and derive parent totals" },
  { value: "top_down", label: "Top-Down", description: "Split root forecasts using positive volumes from earlier leaf history" },
  { value: "ols", label: "OLS", description: "Ordinary Least Squares with upper forecasts derived from the supplied leaves" },
  { value: "wls", label: "WLS (structural)", description: "Structural weights, not estimated residual variances" },
  { value: "mint", label: "MinT (structural fallback)", description: "Uses structural WLS because this interface does not supply residual covariance" },
];

const EFFECTIVE_METHOD_LABELS: Record<HierarchyEffectiveMethod, string> = {
  bottom_up: "Bottom-Up",
  top_down: "Top-Down",
  ols: "OLS",
  wls_structural: "Structural WLS",
};

function GenericTable({ data }: { data: Record<string, unknown>[] }) {
  if (data.length === 0) return <p className="text-sm text-muted-foreground">No data.</p>;
  const cols = Object.keys(data[0]);
  return (
    <div className="overflow-x-auto rounded-md border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-muted/50 text-left text-xs text-muted-foreground">
            {cols.map((c) => (
              <th key={c} className="px-3 py-2">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="border-b last:border-0">
              {cols.map((c) => (
                <td key={c} className="px-3 py-2">{String(row[c] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function HierarchyPage() {
  // Shared state
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [levels, setLevels] = useState("");

  const {
    result: buildResult, loading: buildLoading, error: buildError,
    run: runBuild, reset: resetBuild,
  } = useAsyncOperation<HierarchyBuildResponse>();

  // Aggregation state
  const [targetLevel, setTargetLevel] = useState("");
  const {
    result: aggResult, loading: aggLoading, error: aggError,
    run: runAggregate, reset: resetAggregate,
  } = useAsyncOperation<HierarchyAggregateResponse>();

  // Reconciliation state
  const [reconMethod, setReconMethod] = useState<HierarchyReconciliationMethod>("bottom_up");
  const [actualsFile, setActualsFile] = useState<File | null>(null);
  const [historyValueColumn, setHistoryValueColumn] = useState("quantity");
  const {
    result: reconResult, loading: reconLoading, error: reconError,
    run: runReconcile, reset: resetReconcile,
  } = useAsyncOperation<HierarchyReconcileResponse>();
  const levelList = levels.split(",").map(level => level.trim());
  const leafLevel = levelList[levelList.length - 1];
  const topDown = reconMethod === "top_down";

  const resetInputs = () => {
    resetBuild();
    resetAggregate();
    resetReconcile();
    setTargetLevel("");
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] ?? null;
    resetInputs();
    setFile(selected);
  };

  const handleBuild = async () => {
    if (!file || !levels.trim()) return;
    resetBuild();
    const result = await runBuild(() => api.buildHierarchy(file, levels.trim(), leafLevel));
    if (result && result.levels.length > 0 && !targetLevel) {
      setTargetLevel(result.levels[0]);
    }
  };

  const handleAggregate = () => {
    if (!file || !levels.trim() || !targetLevel) return;
    resetAggregate();
    void runAggregate(() => api.aggregateHierarchy(file, levels.trim(), targetLevel, { id_column: leafLevel }));
  };

  const handleReconcile = () => {
    if (!file || !levels.trim() || (topDown && (!actualsFile || !historyValueColumn.trim()))) return;
    resetReconcile();
    void runReconcile(() => api.reconcileHierarchy(
      file, levels.trim(), reconMethod,
      { id_column: leafLevel, ...(topDown ? { history_value_column: historyValueColumn.trim() } : {}) },
      topDown && actualsFile ? actualsFile : undefined,
    ));
  };

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Hierarchy Manager</h1>
        <p className="text-sm text-muted-foreground">
          Visualize hierarchy structure, aggregate data across levels, and run reconciliation.
        </p>
      </div>

      {/* Overview metrics */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard label="Levels" value={buildResult ? String(buildResult.levels.length) : "--"} />
        <MetricCard label="Total Nodes" value={buildResult ? String(buildResult.total_nodes) : "--"} />
        <MetricCard label="Leaf Nodes" value={buildResult ? String(buildResult.leaf_count) : "--"} />
        <MetricCard label="S-Matrix" value={buildResult ? `${buildResult.s_matrix_shape[0]}x${buildResult.s_matrix_shape[1]}` : "--"} />
      </div>

      {/* Shared file + levels input */}
      <section className="space-y-3 rounded-lg border p-4">
        <h2 className="text-lg font-semibold">Data Input</h2>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <div className="flex-1">
            <label htmlFor="hierarchy-file" className="mb-1 block text-sm font-medium">Upload CSV or Parquet</label>
            <input
              id="hierarchy-file"
              ref={fileRef}
              type="file"
              accept=".csv,.parquet"
              onChange={handleFileChange}
              className="block w-full rounded-md border px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-primary/10 file:px-3 file:py-1 file:text-sm file:font-medium"
            />
          </div>
          <div className="flex-1">
            <label htmlFor="hierarchy-levels" className="mb-1 block text-sm font-medium">Hierarchy Levels (comma-separated)</label>
            <input
              id="hierarchy-levels"
              type="text"
              value={levels}
              onChange={(e) => { resetInputs(); setLevels(e.target.value); }}
              placeholder="e.g. category,subcategory,brand"
              className="block w-full rounded-md border px-3 py-2 text-sm"
            />
          </div>
          <p className="text-xs text-muted-foreground">
            {topDown
              ? "For top-down reconciliation, upload root forecasts here and earlier leaf history below."
              : "Reconciliation requires every leaf in every period, with all hierarchy columns."}
            {" "}Forecast values use the forecast column and dates use week; the last hierarchy level identifies leaves.
          </p>
        </div>
      </section>

      {/* Build Hierarchy Panel */}
      <section className="space-y-3 rounded-lg border p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Build Hierarchy</h2>
          <button
            onClick={handleBuild}
            disabled={buildLoading || !file || !levels.trim()}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
          >
            {buildLoading ? "Building..." : "Build Hierarchy"}
          </button>
        </div>
        {buildError && (
          <div className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">{buildError}</div>
        )}
        {buildResult && (
          <div className="space-y-3">
            <h3 className="text-sm font-semibold">Level Stats</h3>
            <GenericTable
              data={buildResult.level_stats.map((ls) => ({
                Level: ls.level,
                "Node Count": ls.node_count,
              }))}
            />
            <div className="grid grid-cols-3 gap-3 text-sm">
              <div className="rounded-md border p-3">
                <span className="text-muted-foreground">Total Nodes</span>
                <p className="text-lg font-semibold">{buildResult.total_nodes}</p>
              </div>
              <div className="rounded-md border p-3">
                <span className="text-muted-foreground">Leaf Count</span>
                <p className="text-lg font-semibold">{buildResult.leaf_count}</p>
              </div>
              <div className="rounded-md border p-3">
                <span className="text-muted-foreground">S-Matrix Shape</span>
                <p className="text-lg font-semibold">{buildResult.s_matrix_shape[0]} x {buildResult.s_matrix_shape[1]}</p>
              </div>
            </div>
            {/* Sunburst visualization */}
            {buildResult.tree_nodes && buildResult.tree_nodes.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold mb-2">Hierarchy Tree</h3>
                <div className="rounded-lg border p-2">
                  <HierarchySunburst nodes={buildResult.tree_nodes} height={400} />
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Reconciliation Methods Reference */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Reconciliation Methods</h2>
        <p className="text-xs text-muted-foreground">Middle-out is unavailable: this interface does not accept middle-level forecasts.</p>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {RECONCILIATION_METHODS.map((m) => (
            <div key={m.value} className="rounded-lg border p-4">
              <h3 className="font-semibold text-sm">{m.label}</h3>
              <p className="mt-1 text-xs text-muted-foreground">{m.description}</p>
              <code className="mt-2 block text-xs text-primary">{m.value}</code>
            </div>
          ))}
        </div>
      </section>

      {/* Aggregation + Reconciliation panels side by side */}
      <div className="grid gap-4 sm:grid-cols-2">
        {/* Aggregation Panel */}
        <section className="space-y-3 rounded-lg border p-4">
          <h2 className="text-lg font-semibold">Aggregation Explorer</h2>
          <div className="space-y-2">
            <label htmlFor="hierarchy-target" className="block text-sm font-medium">Target Level</label>
            <select
              id="hierarchy-target"
              value={targetLevel}
              onChange={(e) => { resetAggregate(); setTargetLevel(e.target.value); }}
              className="block w-full rounded-md border px-3 py-2 text-sm"
            >
              <option value="" disabled>
                {buildResult ? "Select a level" : "Build hierarchy first"}
              </option>
              {(buildResult?.levels ?? []).map((lvl) => (
                <option key={lvl} value={lvl}>{lvl}</option>
              ))}
            </select>
          </div>
          <button
            onClick={handleAggregate}
            disabled={aggLoading || !file || !levels.trim() || !targetLevel}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
          >
            {aggLoading ? "Aggregating..." : "Aggregate"}
          </button>
          {aggError && (
            <div className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">{aggError}</div>
          )}
          {aggResult && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">
                Level: <strong>{aggResult.target_level}</strong> | Rows: {aggResult.total_rows} | Unique Nodes: {aggResult.unique_nodes}
              </p>
              <GenericTable data={aggResult.top_n_data} />
            </div>
          )}
        </section>

        {/* Reconciliation Panel */}
        <section className="space-y-3 rounded-lg border p-4">
          <h2 className="text-lg font-semibold">Reconciliation Execution</h2>
          <div className="space-y-2">
            <label htmlFor="hierarchy-method" className="block text-sm font-medium">Method</label>
            <select
              id="hierarchy-method"
              value={reconMethod}
              onChange={(e) => {
                const selected = RECONCILIATION_METHODS.find(method => method.value === e.target.value);
                if (selected) { resetReconcile(); setActualsFile(null); setReconMethod(selected.value); }
              }}
              className="block w-full rounded-md border px-3 py-2 text-sm"
            >
              {RECONCILIATION_METHODS.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </div>
          {topDown && (
            <div className="space-y-3">
              <div>
                <label htmlFor="hierarchy-actuals-file" className="mb-1 block text-sm font-medium">Historical actuals</label>
                <input
                  id="hierarchy-actuals-file"
                  type="file"
                  accept=".csv,.parquet"
                  onChange={(e) => { resetReconcile(); setActualsFile(e.target.files?.[0] ?? null); }}
                  className="block w-full rounded-md border px-3 py-2 text-sm"
                />
              </div>
              <div>
                <label htmlFor="hierarchy-history-value" className="mb-1 block text-sm font-medium">History value column</label>
                <input
                  id="hierarchy-history-value"
                  type="text"
                  value={historyValueColumn}
                  onChange={(e) => { resetReconcile(); setHistoryValueColumn(e.target.value); }}
                  className="block w-full rounded-md border px-3 py-2 text-sm"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                History must include all hierarchy levels and complete leaf/period coverage, precede the forecast window,
                and have positive total volume for every leaf. Missing groups never receive equal shares.
              </p>
            </div>
          )}
          <button
            onClick={handleReconcile}
            disabled={reconLoading || !file || !levels.trim() || (topDown && (!actualsFile || !historyValueColumn.trim()))}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
          >
            {reconLoading ? "Reconciling..." : "Reconcile"}
          </button>
          {reconError && (
            <div role="alert" className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">{reconError}</div>
          )}
          {reconResult && (
            <div className="space-y-2" data-testid="reconciliation-result">
              <p className="text-sm">
                Effective method: <strong data-testid="reconciliation-effective-method">{EFFECTIVE_METHOD_LABELS[reconResult.effective_method]}</strong>
                {" "}({reconResult.effective_method}); requested: {reconResult.requested_method}
              </p>
              <p className="text-xs text-muted-foreground">
                {reconResult.source_level} to {reconResult.output_level} | Input rows: {reconResult.input_rows} | Output rows: {reconResult.output_rows}
              </p>
              <div className="flex gap-3 text-sm">
                <div className="rounded-md border p-2 flex-1">
                  <span className="text-muted-foreground">Before Total</span>
                  <p className="font-semibold">{reconResult.before_total.toLocaleString()}</p>
                </div>
                <div className="rounded-md border p-2 flex-1">
                  <span className="text-muted-foreground">After Total</span>
                  <p className="font-semibold">{reconResult.after_total.toLocaleString()}</p>
                </div>
                <div className="rounded-md border p-2 flex-1">
                  <span className="text-muted-foreground">Rows</span>
                  <p className="font-semibold">{reconResult.rows}</p>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                {reconResult.coherence.scope === "root_forecast_conservation" ? "Root conservation" : "Leaf-derived parent totals"}
                {": "}{reconResult.coherence.checked_rows} rows, {reconResult.coherence.checked_periods} periods,
                {" "}{reconResult.coherence.checked_groups} parent/period groups checked.
                {" "}Maximum absolute error: {reconResult.coherence.max_abs_error}.
                {" "}Preview shows {reconResult.reconciled_preview.length} of {reconResult.output_rows} output rows.
              </p>
              {reconResult.warnings.length > 0 && (
                <ul className="list-disc space-y-1 pl-4 text-xs text-muted-foreground">
                  {reconResult.warnings.map(warning => <li key={warning}>{warning}</li>)}
                </ul>
              )}
              <GenericTable data={reconResult.reconciled_preview} />
            </div>
          )}
        </section>
      </div>

      <WorkflowNav currentStep="/hierarchy" />
    </div>
  );
}
