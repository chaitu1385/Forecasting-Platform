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
import { modelDisplayName, formatPct, hasFiniteWmape } from "@/lib/utils";
import type { LeaderboardEntry } from "@/lib/types";

interface Props {
  entries: LeaderboardEntry[];
}

function getModelColor(model: string): string {
  for (const [layer, color] of Object.entries(MODEL_LAYER_COLORS)) {
    if (model.toLowerCase().includes(layer)) return color;
  }
  return COLORS.primary;
}

export function LeaderboardBar({ entries }: Props) {
  const data = entries
    .filter(hasFiniteWmape)
    .sort((a, b) => a.wmape - b.wmape)
    .map((e) => ({
      model: modelDisplayName(e.model),
      rawModel: e.model,
      wmape: e.wmape,
      bias: e.normalized_bias,
      series: e.n_series,
    }));

  return (
    <div style={{ width: '100%', height: 300 }}>
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} layout="vertical" margin={{ left: 100, right: 20, top: 5, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis
          type="number"
          tickFormatter={(v: number) => formatPct(v)}
          domain={[0, "auto"]}
        />
        <YAxis type="category" dataKey="model" width={90} tick={{ fontSize: 12 }} />
        <Tooltip
          formatter={(value: number) => formatPct(value)}
          labelFormatter={(label: string) => `Model: ${label}`}
        />
        <Bar dataKey="wmape" name="WMAPE" radius={[0, 4, 4, 0]}>
          {data.map((entry) => (
            <Cell key={entry.rawModel} fill={getModelColor(entry.rawModel)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
    </div>
  );
}
