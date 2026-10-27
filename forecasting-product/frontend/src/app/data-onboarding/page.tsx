"use client";

import { useState } from "react";
import { FileUpload } from "@/components/data/file-upload";
import { ConfigViewer } from "@/components/data/config-viewer";
import { ForecastabilityGauge } from "@/components/charts/forecastability-gauge";
import { DemandClassDonut } from "@/components/charts/demand-class-donut";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { MultiFilePanel } from "@/components/pipeline/multi-file-panel";
import { PipelineExecutionPanel } from "@/components/pipeline/pipeline-execution-panel";
import { useAnalyze } from "@/hooks/use-analyze";
import { formatNumber } from "@/lib/utils";
import { WorkflowNav } from "@/components/shared/workflow-nav";
import type { AnalysisResponse } from "@/lib/types";

export default function DataOnboardingPage() {
  const [lobName, setLobName] = useState("retail");
  const [llmEnabled, setLlmEnabled] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const analyzeMutation = useAnalyze();
  const analyzedInput = analyzeMutation.variables;
  const result: AnalysisResponse | undefined =
    analyzedInput?.file === uploadedFile && analyzedInput?.lobName === lobName
      && analyzedInput?.llmEnabled === llmEnabled ? analyzeMutation.data : undefined;

  const handleFileSelect = (file: File) => {
    setUploadedFile(file);
    analyzeMutation.mutate({ file, lobName, llmEnabled });
  };

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Data Onboarding</h1>
        <p className="text-sm text-muted-foreground">
          Upload your time series CSV for automated analysis, schema detection, and config generation.
        </p>
      </div>

      {/* Upload Section */}
      <section className="space-y-4 rounded-lg border p-6">
        <h2 className="text-lg font-semibold">Upload Data</h2>

        <div className="flex flex-wrap items-end gap-4">
          <div className="space-y-1">
            <label htmlFor="onboarding-lob" className="text-xs font-medium">LOB Name</label>
            <input
              id="onboarding-lob"
              type="text"
              value={lobName}
              onChange={(e) => {
                setLobName(e.target.value);
                analyzeMutation.reset();
              }}
              className="rounded-md border bg-background px-3 py-1.5 text-sm w-40"
            />
          </div>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={llmEnabled}
              onChange={(e) => {
                setLlmEnabled(e.target.checked);
                analyzeMutation.reset();
              }}
              className="rounded"
            />
            Enable AI Interpretation
          </label>
        </div>

        <FileUpload
          accept=".csv"
          onFileSelect={handleFileSelect}
          label="Time Series Data"
        />

        {analyzeMutation.isPending && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              Analyzing data...
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <ChartSkeleton height="h-24" />
              <ChartSkeleton height="h-24" />
              <ChartSkeleton height="h-24" />
              <ChartSkeleton height="h-24" />
            </div>
          </div>
        )}

        {analyzeMutation.isError && (
          <ErrorDisplay
            message={analyzeMutation.error.message}
            onRetry={() => analyzeMutation.reset()}
          />
        )}
      </section>

      {/* Results */}
      {result && (
        <>
          {/* Schema Detection */}
          <section className="space-y-4">
            <h2 className="text-lg font-semibold">Schema Detection</h2>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricCard label="Series" value={formatNumber(result.n_series)} />
              <MetricCard label="Rows" value={formatNumber(result.n_rows)} />
              <MetricCard label="Frequency" value={result.frequency} />
              <MetricCard
                label="Date Range"
                value={`${result.date_range_start} \u2192 ${result.date_range_end}`}
              />
            </div>
            <div className="grid grid-cols-3 gap-3">
              <div className="rounded-md bg-muted/50 p-3">
                <p className="text-xs text-muted-foreground">Time Column</p>
                <p className="font-mono text-sm font-medium">{result.time_column}</p>
              </div>
              <div className="rounded-md bg-muted/50 p-3">
                <p className="text-xs text-muted-foreground">Target Column</p>
                <p className="font-mono text-sm font-medium">{result.target_column}</p>
              </div>
              <div className="rounded-md bg-muted/50 p-3">
                <p className="text-xs text-muted-foreground">ID Columns</p>
                <p className="font-mono text-sm font-medium">{result.id_columns.join(", ")}</p>
              </div>
            </div>
          </section>

          {/* Forecastability Assessment */}
          <section className="space-y-4">
            <h2 className="text-lg font-semibold">Forecastability Assessment</h2>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="flex flex-col items-center rounded-lg border p-4">
                <ForecastabilityGauge value={result.overall_forecastability} />
              </div>
              <div className="rounded-lg border p-4">
                <h3 className="mb-2 text-sm font-medium">Demand Classification</h3>
                <DemandClassDonut data={result.demand_classes} />
              </div>
            </div>
          </section>

          {/* Config Reasoning */}
          {result.config_reasoning.length > 0 && (
            <section className="space-y-3">
              <h2 className="text-lg font-semibold">Configuration Reasoning</h2>
              <ul className="space-y-1">
                {result.config_reasoning.map((r, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <span className="mt-0.5 text-primary">&bull;</span> {r}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Hypotheses */}
          {result.hypotheses.length > 0 && (
            <section className="space-y-3">
              <h2 className="text-lg font-semibold">Hypotheses</h2>
              <div className="grid gap-2 sm:grid-cols-2">
                {result.hypotheses.map((h, i) => (
                  <div key={i} className="rounded-md bg-blue-50 dark:bg-blue-950 p-3 text-sm">
                    {h}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* AI Narrative */}
          {result.llm_narrative && (
            <section className="space-y-3">
              <h2 className="text-lg font-semibold">AI Interpretation</h2>
              <div className="rounded-md bg-purple-50 dark:bg-purple-950 p-4 text-sm leading-relaxed">
                {result.llm_narrative}
              </div>
              {result.llm_risk_factors && result.llm_risk_factors.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-destructive">Risk Factors</h3>
                  <ul className="mt-1 space-y-1">
                    {result.llm_risk_factors.map((r, i) => (
                      <li key={i} className="text-sm text-muted-foreground">
                        &bull; {r}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </section>
          )}

          {/* Config YAML */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold">Recommended Configuration</h2>
            <ConfigViewer yaml={result.recommended_config_yaml} />
            <button
              onClick={() => {
                const blob = new Blob([result.recommended_config_yaml], { type: "text/yaml" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `${lobName}_config.yaml`;
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="rounded-md border bg-background px-4 py-1.5 text-sm font-medium hover:bg-muted transition-colors"
            >
              Download YAML
            </button>
          </section>

          {/* Multi-File Classification & Merge */}
          <MultiFilePanel lobName={lobName} />

          {/* Pipeline Execution */}
          <PipelineExecutionPanel lobName={lobName} initialFile={uploadedFile} configYaml={result.recommended_config_yaml} />
        </>
      )}

      <WorkflowNav currentStep="/data-onboarding" />
    </div>
  );
}
