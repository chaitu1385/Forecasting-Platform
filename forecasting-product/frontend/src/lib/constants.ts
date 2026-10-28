// ──────────────────────────────────────────────────────────────────────────────
// Platform constants: colors, roles, permissions, navigation
// ──────────────────────────────────────────────────────────────────────────────

import type { Role } from "./types";

// ── Colors ──────────────────────────────────────────────────────────────────

export const COLORS = {
  primary: "#1f77b4",
  secondary: "#ff7f0e",
  accent: "#9467bd",
  success: "#2ca02c",
  warning: "#ff7f0e",
  danger: "#d62728",
  neutral: "#7f7f7f",
  bgLight: "#f7f7f7",
} as const;

export const SEVERITY_COLORS = {
  critical: "#d62728",
  warning: "#ff7f0e",
  info: "#1f77b4",
} as const;

export const FVA_COLORS = {
  ADDS_VALUE: "#2ca02c",
  NEUTRAL: "#7f7f7f",
  DESTROYS_VALUE: "#d62728",
  BASELINE: "#1f77b4",
} as const;

export const MODEL_LAYER_COLORS: Record<string, string> = {
  naive: "#8c564b",
  statistical: "#1f77b4",
  ml: "#2ca02c",
  neural: "#9467bd",
  foundation: "#ff7f0e",
  intermittent: "#e377c2",
  ensemble: "#17becf",
  override: "#bcbd22",
};

export const DEMAND_CLASS_COLORS: Record<string, string> = {
  Smooth: "#2ca02c",
  Intermittent: "#ff7f0e",
  Erratic: "#9467bd",
  Lumpy: "#d62728",
  insufficient_data: "#7f7f7f",
};

export const CONFIDENCE_COLORS: Record<string, string> = {
  high: "#2ca02c",
  medium: "#ff7f0e",
  low: "#d62728",
};

export const RISK_COLORS: Record<string, string> = {
  low: "#2ca02c",
  medium: "#ff7f0e",
  high: "#d62728",
};

export const TREND_ICONS: Record<string, string> = {
  improving: "\u2191",
  stable: "\u2192",
  degrading: "\u2193",
};

// ── RBAC ─────────────────────────────────────────────────────────────────────

export const PERMISSIONS = {
  VIEW_FORECASTS: "view_forecasts",
  VIEW_METRICS: "view_metrics",
  VIEW_AUDIT_LOG: "view_audit_log",
  CREATE_OVERRIDE: "create_override",
  DELETE_OVERRIDE: "delete_override",
  APPROVE_OVERRIDE: "approve_override",
  RUN_BACKTEST: "run_backtest",
  RUN_PIPELINE: "run_pipeline",
  PROMOTE_MODEL: "promote_model",
  MODIFY_CONFIG: "modify_config",
  MANAGE_USERS: "manage_users",
} as const;

export const ROLE_PERMISSIONS: Record<Role, string[]> = {
  admin: Object.values(PERMISSIONS),
  data_scientist: [
    PERMISSIONS.VIEW_FORECASTS,
    PERMISSIONS.VIEW_METRICS,
    PERMISSIONS.VIEW_AUDIT_LOG,
    PERMISSIONS.CREATE_OVERRIDE,
    PERMISSIONS.DELETE_OVERRIDE,
    PERMISSIONS.RUN_BACKTEST,
    PERMISSIONS.RUN_PIPELINE,
    PERMISSIONS.PROMOTE_MODEL,
    PERMISSIONS.MODIFY_CONFIG,
  ],
  planner: [
    PERMISSIONS.VIEW_FORECASTS,
    PERMISSIONS.VIEW_METRICS,
    PERMISSIONS.CREATE_OVERRIDE,
  ],
  manager: [
    PERMISSIONS.VIEW_FORECASTS,
    PERMISSIONS.VIEW_METRICS,
    PERMISSIONS.VIEW_AUDIT_LOG,
    PERMISSIONS.APPROVE_OVERRIDE,
  ],
  viewer: [PERMISSIONS.VIEW_FORECASTS, PERMISSIONS.VIEW_METRICS],
};

// ── Navigation ───────────────────────────────────────────────────────────────

export interface NavItem {
  title: string;
  href: string;
  icon: string; // lucide icon name
  description: string;
}

export const NAV_ITEMS: NavItem[] = [
  {
    title: "Data Onboarding",
    href: "/data-onboarding",
    icon: "Upload",
    description: "Upload CSV, analyze data, generate config",
  },
  {
    title: "Series Explorer",
    href: "/series-explorer",
    icon: "Search",
    description: "SBC classification, breaks, quality, AI Q&A",
  },
  {
    title: "SKU Transitions",
    href: "/sku-transitions",
    icon: "ArrowLeftRight",
    description: "SKU mapping, planner overrides",
  },
  {
    title: "Hierarchy",
    href: "/hierarchy",
    icon: "GitBranch",
    description: "Tree visualization, reconciliation",
  },
  {
    title: "Backtest Results",
    href: "/backtest",
    icon: "Trophy",
    description: "Leaderboard, FVA, decomposition, AI config tuner",
  },
  {
    title: "Forecast Viewer",
    href: "/forecast",
    icon: "TrendingUp",
    description: "Fan chart, decomposition, NL query",
  },
  {
    title: "Platform Health",
    href: "/health",
    icon: "Activity",
    description: "Drift, triage, audit log, cost tracking",
  },
  {
    title: "S&OP Meeting",
    href: "/sop",
    icon: "Presentation",
    description: "AI commentary, governance, BI export",
  },
];

// ── Glossary ─────────────────────────────────────────────────────────────────

export const GLOSSARY: { term: string; definition: string }[] = [
  { term: "WMAPE", definition: "Weighted Mean Absolute Percentage Error — accuracy metric that weights errors by actual volume." },
  { term: "FVA", definition: "Forecast Value Added — measures whether each layer of sophistication actually improves forecast accuracy." },
  { term: "SBC Matrix", definition: "Syntetos-Boylan Classification — classifies demand patterns into Smooth, Intermittent, Erratic, or Lumpy." },
  { term: "Reconciliation", definition: "Adjusting forecasts so they are coherent across hierarchy levels (parent = sum of children)." },
  { term: "SHAP", definition: "SHapley Additive exPlanations — shows which features drive ML model predictions." },
  { term: "Drift", definition: "When forecast accuracy degrades over time compared to a baseline period." },
  { term: "Champion Model", definition: "The best-performing model for each series, selected via backtesting." },
  { term: "Horizon", definition: "Number of periods into the future being forecast." },
  { term: "Backtest", definition: "Walk-forward validation that simulates real forecasting conditions on historical data." },
  { term: "Structural Break", definition: "A sudden shift in a time series' statistical properties (mean, variance, trend)." },
  { term: "Conformal Prediction", definition: "Distribution-free method for generating calibrated prediction intervals." },
  { term: "Croston", definition: "Specialized method for intermittent demand (many zero-demand periods)." },
  { term: "MinT Reconciliation", definition: "Minimum Trace reconciliation — optimal method using forecast error covariance." },
  { term: "LOB", definition: "Line of Business — organizational grouping of series (e.g., retail, wholesale)." },
];
