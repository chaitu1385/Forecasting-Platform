import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { LeaderboardEntry } from "./types";

/** Merge Tailwind classes safely */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format a number as percentage string */
export function formatPct(value: number | null | undefined, decimals: number = 1): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  return `${(value * 100).toFixed(decimals)}%`;
}

export function hasFiniteWmape(entry: LeaderboardEntry): entry is LeaderboardEntry & { wmape: number } {
  return entry.wmape != null && Number.isFinite(entry.wmape);
}

export function leaderboardScore(entry: LeaderboardEntry, metric: string): number | null {
  const value = entry.scores && metric in entry.scores
    ? entry.scores[metric]
    : metric === "wmape" ? entry.wmape
    : metric === "normalized_bias" ? entry.normalized_bias
    : entry.primary_score;
  return value != null && Number.isFinite(value) ? value : null;
}

export function metricLabel(metric: string, policy?: string | null): string {
  if (metric === "wmape") return policy === "raw-observations-v1" ? "Volume WMAPE" : "Stored WMAPE";
  if (metric === "macro_wmape") return "Macro WMAPE";
  if (metric === "normalized_bias") return "Normalized bias";
  return metric.toUpperCase();
}

export function formatMetric(metric: string, value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  return ["wmape", "macro_wmape", "normalized_bias", "mape"].includes(metric)
    ? formatPct(value) : formatNumber(value, 2);
}

/** Format a number with locale-aware separators */
export function formatNumber(value: number, decimals: number = 0): string {
  return value.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/** Format seconds into human-readable duration */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

/** Human-friendly model name */
export function modelDisplayName(model: string): string {
  const names: Record<string, string> = {
    seasonal_naive: "Seasonal Naive",
    auto_arima: "AutoARIMA",
    auto_ets: "AutoETS",
    auto_theta: "AutoTheta",
    mstl: "MSTL",
    lgbm_direct: "LightGBM",
    xgboost_direct: "XGBoost",
    nbeats: "N-BEATS",
    nhits: "NHITS",
    tft: "TFT",
    chronos: "Chronos",
    timegpt: "TimeGPT",
    croston: "Croston",
    croston_sba: "Croston SBA",
    tsb: "TSB",
    weighted_ensemble: "Ensemble",
  };
  return names[model] || model.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Truncate string with ellipsis */
export function truncate(str: string, maxLen: number): string {
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen - 1) + "\u2026";
}
