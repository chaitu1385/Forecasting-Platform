"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Scatter,
  ComposedChart,
} from "recharts";
import { COLORS, SEVERITY_COLORS } from "@/lib/constants";

interface CleansingPoint {
  week: string;
  original: number | null;
  cleaned: number | null;
  is_outlier?: boolean;
}

interface Props {
  data: CleansingPoint[];
  height?: number;
}

export function CleansingOverlay({ data, height = 280 }: Props) {
  if (data.length === 0) return null;

  // Add outlier scatter field (only populated for outlier points)
  const chartData = data.map((d) => ({
    ...d,
    outlier_marker: d.is_outlier ? d.original : undefined,
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="week" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip />
        <Legend />
        <Line
          type="monotone"
          dataKey="original"
          stroke={COLORS.neutral}
          strokeDasharray="5 5"
          dot={false}
          strokeWidth={1.5}
          name="Original"
        />
        <Line
          type="monotone"
          dataKey="cleaned"
          stroke={COLORS.primary}
          dot={false}
          strokeWidth={2}
          name="Cleaned"
        />
        <Scatter
          dataKey="outlier_marker"
          fill={SEVERITY_COLORS.critical}
          name="Outliers"
          shape="cross"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
