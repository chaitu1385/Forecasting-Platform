"use client";

import Link from "next/link";
import { CommentaryPanel } from "@/components/ai/commentary-panel";
import { useAuth } from "@/providers/auth-provider";
import { useLob } from "@/providers/lob-provider";
import { MetricCard } from "@/components/shared/metric-card";
import { CrossRunComparisonPanel } from "@/components/forecast/cross-run-comparison-panel";
import { ModelCardsPanel } from "@/components/governance/model-cards-panel";
import { LineagePanel } from "@/components/governance/lineage-panel";
import { BIExportPanel } from "@/components/governance/bi-export-panel";
import { WorkflowNav } from "@/components/shared/workflow-nav";

export default function SOPPage() {
  const { lob, setLob } = useLob();
  const { mode } = useAuth();

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">S&amp;OP Meeting Prep</h1>
        <p className="text-sm text-muted-foreground">
          AI executive commentary, cross-run comparison, model governance, and BI exports.
        </p>
      </div>

      {/* LOB control */}
      <div className="space-y-1">
        <label className="text-xs font-medium">LOB</label>
        <input
          type="text"
          value={lob}
          onChange={(e) => setLob(e.target.value)}
          className="rounded-md border bg-background px-3 py-1.5 text-sm w-32"
        />
      </div>

      {mode === "demo" ? (
        <section className="space-y-3" aria-label="Illustrative demo metrics">
          <h2 className="text-lg font-semibold">Illustrative demo metrics</h2>
          <p className="text-sm text-muted-foreground">
            These sample values and trends are not calculated from your uploaded data or runs.
          </p>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard label="Overall WMAPE" value="16.2%" trend="improving" />
            <MetricCard label="Bias" value="-2.1%" trend="stable" />
            <MetricCard label="Series at Risk" value="12" trend="degrading" />
            <MetricCard label="Champion Models" value="5" />
          </div>
        </section>
      ) : (
        <p className="rounded-lg border p-4 text-sm text-muted-foreground">
          Summary metrics are not connected to run data on this page. Review the stored
          scores and their scoring scope in{" "}
          <Link href="/backtest" className="text-primary underline">Backtest Results</Link>.
        </p>
      )}

      {/* AI Commentary — live endpoint */}
      <section>
        <h2 className="text-lg font-semibold mb-3">Executive Commentary</h2>
        <CommentaryPanel lob={lob} />
      </section>

      {/* Cross-Run Comparison */}
      <CrossRunComparisonPanel />

      {/* Model Governance */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Model Governance</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          <ModelCardsPanel />
          <LineagePanel lob={lob} />
        </div>
      </section>

      {/* BI Export */}
      <BIExportPanel lob={lob} />

      <WorkflowNav currentStep="/sop" />
    </div>
  );
}
