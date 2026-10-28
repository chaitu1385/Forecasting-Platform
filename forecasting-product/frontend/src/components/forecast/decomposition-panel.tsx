"use client";

import { useState } from "react";
import { FileUpload } from "@/components/data/file-upload";
import { DataTable } from "@/components/data/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import type { DecomposeResponse } from "@/lib/types";

export function DecompositionPanel() {
  const [historyFile, setHistoryFile] = useState<File | null>(null);
  const [forecastFile, setForecastFile] = useState<File | null>(null);
  const { result, loading, error, run, setError } = useAsyncOperation<DecomposeResponse>();

  const handleDecompose = () => {
    if (!historyFile || !forecastFile) return;
    run(() => api.decomposeForecast(historyFile, forecastFile));
  };

  return (
    <section className="space-y-4 rounded-lg border p-6">
      <h2 className="text-lg font-semibold">Seasonal Decomposition</h2>
      <p className="text-sm text-muted-foreground">
        Centered moving-average decomposition showing trend, seasonal, and residual components.
      </p>
      <div className="grid gap-4 sm:grid-cols-2">
        <FileUpload
          accept=".csv"
          onFileSelect={(file) => setHistoryFile(file)}
          label="History File"
        />
        <FileUpload
          accept=".csv"
          onFileSelect={(file) => setForecastFile(file)}
          label="Forecast File"
        />
      </div>
      <button
        onClick={handleDecompose}
        disabled={!historyFile || !forecastFile || loading}
        className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
      >
        {loading ? "Decomposing..." : "Run Decomposition"}
      </button>
      {loading && <ChartSkeleton />}
      {error && <ErrorDisplay message={error} onRetry={() => setError(null)} />}
      {result && (
        <div className="space-y-3">
          <MetricCard label="Series Count" value={result.series_count} />
          {result.decomposition.length > 0 && (
            <DataTable
              columns={Object.keys(result.decomposition[0]).map((k) => ({
                key: k,
                label: k,
                sortable: true,
              }))}
              data={result.decomposition}
              pageSize={10}
            />
          )}
          {Object.keys(result.narratives).length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Narratives</h3>
              {Object.entries(result.narratives).map(([series, narrative]) => (
                <div key={series} className="rounded-md bg-muted/50 p-3">
                  <p className="text-xs font-medium font-mono">{series}</p>
                  <p className="text-sm text-muted-foreground mt-1">{narrative}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
