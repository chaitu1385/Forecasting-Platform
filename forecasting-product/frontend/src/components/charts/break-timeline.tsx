"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { COLORS, SEVERITY_COLORS } from "@/lib/constants";

interface DataPoint {
  week: string;
  value: number | null;
}

interface Props {
  /** Time series data for the selected series */
  data: DataPoint[];
  /** Break dates to mark as vertical lines */
  breakDates: string[];
  /** Chart height in px */
  height?: number;
}

export function BreakTimeline({ data, breakDates, height = 280 }: Props) {
  if (data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 10, right: 20, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="week" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip />
        <Line
          type="monotone"
          dataKey="value"
          stroke={COLORS.primary}
          dot={false}
          strokeWidth={2}
          name="Demand"
        />
        {breakDates.map((d, i) => (
          <ReferenceLine
            key={`${d}-${i}`}
            x={d}
            stroke={SEVERITY_COLORS.critical}
            strokeDasharray="4 4"
            strokeWidth={1.5}
            label={{ value: "Break", position: "top", fontSize: 10, fill: SEVERITY_COLORS.critical }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
