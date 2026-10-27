"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { MODEL_LAYER_COLORS, COLORS } from "@/lib/constants";
import { modelDisplayName, formatMetric, leaderboardScore, metricLabel } from "@/lib/utils";
import type { LeaderboardEntry } from "@/lib/types";

interface Props {
  entries: LeaderboardEntry[];
  primaryMetric?: string;
  scoringPolicy?: string | null;
}

function getModelColor(model: string): string {
  for (const [layer, color] of Object.entries(MODEL_LAYER_COLORS)) {
    if (model.toLowerCase().includes(layer)) return color;
  }
  return COLORS.primary;
}

export function LeaderboardBar({ entries, primaryMetric = "wmape", scoringPolicy }: Props) {
  const data = entries
    .map((e) => ({
      model: `${modelDisplayName(e.model)}${e.group_key ? ` (${e.group_key})` : ""}`,
      rawModel: e.model,
      key: `${e.competition_scope ?? "unknown"}:${e.group_key ?? "unknown"}:${e.model}`,
      score: leaderboardScore(e, primaryMetric),
      bias: e.normalized_bias,
      series: e.n_series,
    }))
    .filter(entry => entry.score !== null);

  return (
    <div style={{ width: '100%', height: 300 }}>
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} layout="vertical" margin={{ left: 100, right: 20, top: 5, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis
          type="number"
          tickFormatter={(v: number) => formatMetric(primaryMetric, v)}
          domain={primaryMetric === "normalized_bias" ? ["auto", "auto"] : [0, "auto"]}
        />
        <YAxis type="category" dataKey="model" width={90} tick={{ fontSize: 12 }} />
        <Tooltip
          formatter={(value: number) => formatMetric(primaryMetric, value)}
          labelFormatter={(label: string) => `Model: ${label}`}
        />
        <Bar dataKey="score" name={metricLabel(primaryMetric, scoringPolicy)} radius={[0, 4, 4, 0]}>
          {data.map((entry) => (
            <Cell key={entry.key} fill={getModelColor(entry.rawModel)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
    </div>
  );
}
