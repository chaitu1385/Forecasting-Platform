"use client";

import { useId, useState } from "react";
import { FileUpload } from "@/components/data/file-upload";
import { DataTable } from "@/components/data/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import { formatNumber } from "@/lib/utils";
import type { ConstrainParams, ConstrainResponse } from "@/lib/types";

const STRATEGY_LABELS: Record<ConstrainResponse["allocation_strategy"], string> = {
  element_wise: "Element-wise bounds",
  proportional: "Floor-preserving proportional",
  largest_first: "Largest-first",
};

export function ConstrainedForecastPanel() {
  const inputId = useId();
  const [file, setFile] = useState<File | null>(null);
  const [minDemand, setMinDemand] = useState<string>("0");
  const [maxCapacity, setMaxCapacity] = useState<string>("");
  const [aggregateMax, setAggregateMax] = useState<string>("");
  const [proportional, setProportional] = useState(false);
  const { result, loading, error, run, reset } = useAsyncOperation<ConstrainResponse>();

  const handleConstrain = () => {
    if (!file) return;
    const params: ConstrainParams = {};
    if (minDemand) params.min_demand = Number(minDemand);
    if (maxCapacity) params.max_capacity = Number(maxCapacity);
    if (aggregateMax) params.aggregate_max = Number(aggregateMax);
    params.proportional = proportional;
    reset();
    void run(() => api.constrainForecast(file, params));
  };

  return (
    <section aria-labelledby={`${inputId}-heading`} className="space-y-4 rounded-lg border p-6">
      <h2 id={`${inputId}-heading`} className="text-lg font-semibold">Constrained Forecast</h2>
      <p className="text-sm text-muted-foreground">
        Apply capacity and per-period budget constraints to the complete uploaded forecast.
        Aggregate budgets support point forecasts only. Quantiles can use element-wise bounds without a budget.
      </p>
      <FileUpload
        accept=".csv,.parquet"
        onFileSelect={(f) => { reset(); setFile(f); }}
        label="Forecast File"
      />
      {file && <p className="text-sm text-muted-foreground">Selected file: {file.name}</p>}
      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-1">
          <label htmlFor={`${inputId}-floor`} className="text-xs font-medium">Min Demand</label>
          <input
            id={`${inputId}-floor`}
            type="number"
            value={minDemand}
            onChange={(e) => { reset(); setMinDemand(e.target.value); }}
            className="rounded-md border bg-background px-3 py-1.5 text-sm w-28"
          />
        </div>
        <div className="space-y-1">
          <label htmlFor={`${inputId}-capacity`} className="text-xs font-medium">Max Capacity</label>
          <input
            id={`${inputId}-capacity`}
            type="number"
            value={maxCapacity}
            onChange={(e) => { reset(); setMaxCapacity(e.target.value); }}
            placeholder="Optional"
            className="rounded-md border bg-background px-3 py-1.5 text-sm w-28"
          />
        </div>
        <div className="space-y-1">
          <label htmlFor={`${inputId}-budget`} className="text-xs font-medium">Aggregate Max (per period)</label>
          <input
            id={`${inputId}-budget`}
            type="number"
            value={aggregateMax}
            onChange={(e) => { reset(); setAggregateMax(e.target.value); }}
            placeholder="Optional"
            className="rounded-md border bg-background px-3 py-1.5 text-sm w-28"
          />
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={proportional}
            onChange={(e) => { reset(); setProportional(e.target.checked); }}
            className="rounded"
          />
          Proportional
        </label>
      </div>
      <p className="text-xs text-muted-foreground">
        Unchecked: largest-first, breaking ties by series ID. Checked: proportional allocation of headroom above the floor.
      </p>
      <button
        onClick={handleConstrain}
        disabled={!file || loading}
        className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
      >
        {loading ? "Applying Constraints..." : "Apply Constraints"}
      </button>
      {loading && <ChartSkeleton />}
      {error && <div role="alert"><ErrorDisplay message={error} onRetry={handleConstrain} /></div>}
      {result && (
        <div role="region" aria-label="Constraint result" className="space-y-3">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard label="Before Total" value={String(result.before_total)} />
            <MetricCard label="After Total" value={String(result.after_total)} />
            <MetricCard label="Rows Modified" value={formatNumber(result.rows_modified)} />
            <MetricCard
              label="Change"
              value={result.before_total > 0 ? `${(((result.after_total - result.before_total) / result.before_total) * 100).toFixed(1)}%` : "N/A"}
            />
          </div>
          <p className="text-sm">
            Strategy: <strong>{STRATEGY_LABELS[result.allocation_strategy]}</strong>.
            {" "}Scope: {result.constraint_scope === "point" ? "Point forecasts" : "Point and marginal quantile bounds"}.
          </p>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <MetricCard label="Rows Checked" value={formatNumber(result.checked_rows)} />
            <MetricCard label="Periods Checked" value={formatNumber(result.checked_periods)} />
            <MetricCard
              label="Maximum Budget Excess"
              value={result.max_budget_excess === null ? "No budget supplied" : String(result.max_budget_excess)}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Totals and checks cover all rows and periods. Preview: {result.constrained_preview.length} of {result.checked_rows} rows.
            {" "}Stored forecasts are unchanged.
          </p>
          {result.constrained_preview.length > 0 && (
            <DataTable
              columns={Object.keys(result.constrained_preview[0]).map((k) => ({
                key: k,
                label: k,
                sortable: true,
              }))}
              data={result.constrained_preview}
              pageSize={10}
            />
          )}
        </div>
      )}
    </section>
  );
}
