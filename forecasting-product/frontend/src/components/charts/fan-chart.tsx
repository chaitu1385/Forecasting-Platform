"use client";

import dynamic from "next/dynamic";
import { COLORS } from "@/lib/constants";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface FanChartPoint {
  week: string;
  forecast: number | null;
  forecast_p10?: number | null;
  forecast_p50?: number | null;
  forecast_p90?: number | null;
  actual?: number;
}

interface Props {
  data: FanChartPoint[];
  height?: number;
}

export function FanChart({ data, height = 380 }: Props) {
  if (data.length === 0) return null;

  const hasQuantiles = data.some((d) => d.forecast_p10 != null && d.forecast_p90 != null);
  const hasActuals = data.some((d) => d.actual != null);

  const weeks = data.map((d) => d.week);

  const traces: Plotly.Data[] = [];

  // P10-P90 shaded band
  if (hasQuantiles) {
    const p10 = data.map((d) => d.forecast_p10 ?? d.forecast);
    const p90 = data.map((d) => d.forecast_p90 ?? d.forecast);

    traces.push({
      x: [...weeks, ...weeks.slice().reverse()],
      y: [...p90, ...p10.slice().reverse()],
      fill: "toself",
      fillcolor: "rgba(31,119,180,0.15)",
      line: { color: "transparent" },
      type: "scatter",
      mode: "lines",
      name: "P10–P90",
      showlegend: true,
      hoverinfo: "skip",
    });

    // P50 median line
    const hasP50 = data.some((d) => d.forecast_p50 != null);
    if (hasP50) {
      traces.push({
        x: weeks,
        y: data.map((d) => d.forecast_p50 ?? d.forecast),
        type: "scatter",
        mode: "lines",
        name: "P50 (median)",
        line: { color: COLORS.secondary, dash: "dash", width: 1.5 },
      });
    }
  }

  // Point forecast
  traces.push({
    x: weeks,
    y: data.map((d) => d.forecast),
    type: "scatter",
    mode: "lines",
    name: "Forecast",
    line: { color: COLORS.primary, width: 2.5 },
  });

  // Actuals overlay
  if (hasActuals) {
    traces.push({
      x: weeks,
      y: data.map((d) => d.actual ?? null),
      type: "scatter",
      mode: "lines+markers",
      name: "Actuals",
      line: { color: COLORS.neutral, width: 1.5 },
      marker: { size: 4 },
    });
  }

  return (
    <Plot
      data={traces}
      layout={{
        height,
        margin: { t: 30, r: 20, b: 40, l: 50 },
        xaxis: { tickfont: { size: 11 } },
        yaxis: { title: { text: "Demand" }, tickfont: { size: 11 } },
        legend: { orientation: "h", y: -0.15 },
        hovermode: "x unified",
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: "100%" }}
    />
  );
}
