// ──────────────────────────────────────────────────────────────────────────────
// E2E Page Rendering Tests
// Verify every page in the Next.js app renders without errors, displays data
// correctly, and has no broken components or missing props.
// ──────────────────────────────────────────────────────────────────────────────

import { describe, test, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { useAuth } from "@/providers/auth-provider";

// Must be called before importing components that use @/lib/api-client
import { setupApiMocks } from "./helpers/mock-api";
setupApiMocks();

// ── Test Utilities ──────────────────────────────────────────────────────────

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: 0,
        gcTime: 0,
      },
    },
  });
}

// Minimal AuthProvider that provides an authenticated admin user
function MockAuthProvider({ children }: { children: React.ReactNode }) {
  const ctx = {
    user: {
      username: "test-admin",
      role: "admin" as const,
      permissions: [
        "view_forecasts",
        "view_metrics",
        "view_audit_log",
        "create_override",
        "delete_override",
        "approve_override",
        "run_backtest",
        "run_pipeline",
        "promote_model",
        "modify_config",
        "manage_users",
      ],
    },
    isAuthenticated: true,
    mode: "demo",
    loading: false,
    modeError: null,
    sessionMessage: null,
    loginDemo: vi.fn(),
    loginWithToken: vi.fn(),
    refreshMode: vi.fn(),
    logout: vi.fn(),
    hasPermission: () => true,
  };

  const AuthContext = React.createContext(ctx);
  return <AuthContext.Provider value={ctx}>{children}</AuthContext.Provider>;
}

// We need to mock the auth provider module so useAuth() returns our mock
vi.mock("@/providers/auth-provider", () => ({
  AuthProvider: ({ children }: { children: React.ReactNode }) => children,
  useAuth: vi.fn().mockReturnValue({
    user: {
      username: "test-admin",
      role: "admin",
      permissions: [
        "view_forecasts",
        "view_metrics",
        "view_audit_log",
        "create_override",
        "delete_override",
        "approve_override",
        "run_backtest",
        "run_pipeline",
        "promote_model",
        "modify_config",
        "manage_users",
      ],
    },
    isAuthenticated: true,
    mode: "demo",
    loading: false,
    modeError: null,
    sessionMessage: null,
    loginDemo: vi.fn().mockResolvedValue(undefined),
    loginWithToken: vi.fn().mockResolvedValue(undefined),
    refreshMode: vi.fn(),
    logout: vi.fn(),
    hasPermission: () => true,
  }),
}));

// Mock Recharts components to avoid SVG rendering issues in jsdom
vi.mock("recharts", () => {
  const React = require("react");
  const createMockChart =
    (name: string) =>
    ({ children, ...props }: any) =>
      React.createElement("div", { "data-testid": `mock-${name}`, ...props }, children);
  return {
    ResponsiveContainer: ({ children }: any) =>
      React.createElement("div", { "data-testid": "responsive-container" }, children),
    LineChart: createMockChart("line-chart"),
    BarChart: createMockChart("bar-chart"),
    ComposedChart: createMockChart("composed-chart"),
    ScatterChart: createMockChart("scatter-chart"),
    PieChart: createMockChart("pie-chart"),
    AreaChart: createMockChart("area-chart"),
    Line: () => null,
    Bar: () => null,
    XAxis: () => null,
    YAxis: () => null,
    CartesianGrid: () => null,
    Tooltip: () => null,
    Legend: () => null,
    ReferenceLine: () => null,
    ReferenceArea: () => null,
    Scatter: () => null,
    Cell: () => null,
    Pie: () => null,
    Area: () => null,
    Label: () => null,
    Brush: () => null,
  };
});

// Mock react-plotly.js
vi.mock("react-plotly.js", () => {
  const React = require("react");
  return {
    default: (props: any) =>
      React.createElement("div", { "data-testid": "mock-plotly", ...props }),
  };
});

function renderPage(PageComponent: React.ComponentType) {
  const queryClient = createTestQueryClient();
  const result = render(
    <QueryClientProvider client={queryClient}>
      <PageComponent />
    </QueryClientProvider>,
  );
  return { ...result, queryClient };
}

// ── Page Tests ──────────────────────────────────────────────────────────────

describe("E2E Page Rendering Tests", () => {
  const consoleErrors: string[] = [];
  const originalConsoleError = console.error;

  beforeEach(() => {
    consoleErrors.length = 0;
    console.error = (...args: unknown[]) => {
      const msg = typeof args[0] === "string" ? args[0] : String(args[0]);
      // Filter out expected React warnings
      if (msg.includes("act(") || msg.includes("not wrapped in act")) return;
      if (msg.includes("ReactDOM.render")) return;
      consoleErrors.push(msg);
    };
    // Reset localStorage
    localStorage.clear();
  });

  // ── 1. Landing Page (/) ─────────────────────────────────────────────────

  test("Landing Page — renders persona cards and navigation", async () => {
    const { default: LandingPage } = await import("@/app/page");
    renderPage(LandingPage);

    // Page heading
    expect(screen.getByText("Sales Forecasting Product")).toBeInTheDocument();

    // 4 persona cards
    expect(screen.getByText("Data Scientist")).toBeInTheDocument();
    expect(screen.getByText("Demand Planner")).toBeInTheDocument();

    // Workflow section
    expect(screen.getByText("Quick Start by Role")).toBeInTheDocument();
    expect(screen.getByText("Workflow")).toBeInTheDocument();
    expect(screen.getByText("Glossary")).toBeInTheDocument();

    // No error boundaries
    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // ── 2. Login Page (/login) ──────────────────────────────────────────────

  test("Login Page — renders form with role selector", async () => {
    const { default: LoginPage } = await import("@/app/login/page");
    renderPage(LoginPage);

    // Heading
    expect(screen.getByText("Forecasting Product")).toBeInTheDocument();

    // Form elements
    const usernameInput = screen.getByPlaceholderText(/enter your username/i);
    expect(usernameInput).toBeInTheDocument();

    // Role selector is present (options contain "Admin — ...")
    const roleSelect = screen.getByLabelText("Role");
    expect(roleSelect).toBeInTheDocument();
    expect(roleSelect.querySelectorAll("option").length).toBe(5);

    // Submit button
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();

    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // ── 3. Data Onboarding Page (/data-onboarding) ─────────────────────────

  test("Data Onboarding Page — renders upload zone and heading", async () => {
    const { default: DataOnboardingPage } = await import("@/app/data-onboarding/page");
    renderPage(DataOnboardingPage);

    // Heading
    expect(screen.getByText("Data Onboarding")).toBeInTheDocument();

    // Upload section
    expect(screen.getByText("Upload Data")).toBeInTheDocument();

    // LOB input
    const lobInput = screen.getByDisplayValue("retail");
    expect(lobInput).toBeInTheDocument();

    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // ── 4. Series Explorer Page (/series-explorer) ─────────────────────────

  test("Series Explorer Page — renders series list and SBC classification", async () => {
    const { default: SeriesExplorerPage } = await import("@/app/series-explorer/page");
    renderPage(SeriesExplorerPage);

    // Heading
    expect(screen.getByText("Series Explorer")).toBeInTheDocument();

    // Wait for data to load (API mocks resolve immediately)
    await waitFor(
      () => {
        expect(screen.getByText("SBC Demand Classification")).toBeInTheDocument();
      },
      { timeout: 5000 },
    );

    // Metric cards should show series count
    await waitFor(() => {
      expect(screen.getByText("Series Count")).toBeInTheDocument();
    });

    // Section headings
    expect(screen.getByText("Structural Break Detection")).toBeInTheDocument();

    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // ── 5. SKU Transitions Page (/sku-transitions) ─────────────────────────

  test("SKU Transitions Page — renders mapping and override panels", async () => {
    const { default: SKUTransitionsPage } = await import("@/app/sku-transitions/page");
    renderPage(SKUTransitionsPage);

    // Heading
    expect(screen.getByText("SKU Transitions")).toBeInTheDocument();

    // Metric cards (hardcoded values on this page)
    expect(screen.getByText("Total SKUs")).toBeInTheDocument();
    expect(screen.getByText("Active Transitions")).toBeInTheDocument();
    expect(screen.getByText("Pending Overrides")).toBeInTheDocument();

    // Ramp shape preview section
    expect(screen.getByText("Transition Ramp Shape Preview")).toBeInTheDocument();

    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // ── 6. Hierarchy Page (/hierarchy) ─────────────────────────────────────

  test("Hierarchy Page — renders build interface and method cards", async () => {
    const { default: HierarchyPage } = await import("@/app/hierarchy/page");
    renderPage(HierarchyPage);

    // Heading
    expect(screen.getByText("Hierarchy Manager")).toBeInTheDocument();

    // Sections
    expect(screen.getByText("Data Input")).toBeInTheDocument();
    expect(screen.getByText("Reconciliation Methods")).toBeInTheDocument();

    // Build button
    expect(screen.getByRole("button", { name: /build hierarchy/i })).toBeInTheDocument();

    // Method cards (may appear multiple times — heading + dropdown option)
    expect(screen.getAllByText("Bottom-Up").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Top-Down").length).toBeGreaterThan(0);
    expect(screen.getAllByText("OLS").length).toBeGreaterThan(0);

    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // ── 7. Backtest Results Page (/backtest) ───────────────────────────────

  test("Backtest Results Page — renders leaderboard and analytics", async () => {
    const { default: BacktestPage } = await import("@/app/backtest/page");
    renderPage(BacktestPage);

    // Heading
    expect(screen.getByText("Backtest Results")).toBeInTheDocument();

    // Wait for leaderboard data
    await waitFor(
      () => {
        expect(screen.getByText("Model Leaderboard")).toBeInTheDocument();
      },
      { timeout: 5000 },
    );

    // Section headings
    await waitFor(() => {
      expect(screen.getByText("Forecast Value Added (FVA)")).toBeInTheDocument();
    });

    expect(screen.getByText("Prediction Interval Calibration")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Forecast decomposition" })).toBeInTheDocument();

    // Metric cards
    await waitFor(() => {
      expect(screen.getByText("Models")).toBeInTheDocument();
    });

    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // ── 8. Forecast Viewer Page (/forecast) ────────────────────────────────

  test("Forecast Viewer Page — renders chart and series selector", async () => {
    const { default: ForecastPage } = await import("@/app/forecast/page");
    renderPage(ForecastPage);

    // Heading
    expect(screen.getByText("Forecast Viewer")).toBeInTheDocument();

    // LOB input
    expect(screen.getByDisplayValue("retail")).toBeInTheDocument();

    // Wait for data to load and series selector to appear
    await waitFor(
      () => {
        // Should have a series selector with our mock data
        const selects = document.querySelectorAll("select");
        expect(selects.length).toBeGreaterThan(0);
      },
      { timeout: 5000 },
    );

    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // ── 9. Platform Health Page (/health) ──────────────────────────────────

  test("Platform Health Page — renders drift, audit, manifests, and costs", async () => {
    const { default: HealthPage } = await import("@/app/health/page");
    renderPage(HealthPage);

    // Heading
    expect(screen.getByText("Platform Health")).toBeInTheDocument();

    // Wait for drift data
    await waitFor(
      () => {
        // Should show critical/warning counts from mock data
        expect(screen.getByText("Critical")).toBeInTheDocument();
      },
      { timeout: 5000 },
    );

    await waitFor(() => {
      expect(screen.getByText("Warning")).toBeInTheDocument();
    });

    // Tab or section for manifests (may appear multiple times)
    await waitFor(() => {
      expect(screen.getAllByText(/Manifests/i).length).toBeGreaterThan(0);
    });

    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // ── 10. S&OP Meeting Page (/sop) ───────────────────────────────────────

  test("S&OP Meeting Page — renders commentary and governance panels", async () => {
    const { default: SOPPage } = await import("@/app/sop/page");
    renderPage(SOPPage);

    // Heading
    expect(screen.getByText("S&OP Meeting Prep")).toBeInTheDocument();

    // Section headings
    expect(screen.getByText("Executive Commentary")).toBeInTheDocument();
    expect(screen.getByText("Model Governance")).toBeInTheDocument();

    expect(screen.getByText("Illustrative demo metrics")).toBeInTheDocument();
    expect(screen.getByText("Overall WMAPE")).toBeInTheDocument();
    expect(screen.getByText("Bias")).toBeInTheDocument();

    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  test.each([
    { label: "production", mode: "production", loading: false, modeError: null },
    { label: "loading", mode: null, loading: true, modeError: null },
    { label: "unavailable mode", mode: null, loading: false, modeError: "Unavailable" },
  ] as const)("S&OP $label does not present sample KPIs as measured facts", async ({ mode, loading, modeError }) => {
    const original = useAuth();
    vi.mocked(useAuth).mockReturnValue({ ...original, mode, loading, modeError });
    try {
      const { default: SOPPage } = await import("@/app/sop/page");
      renderPage(SOPPage);
      expect(screen.queryByText("Illustrative demo metrics")).not.toBeInTheDocument();
      expect(screen.queryByText("16.2%")).not.toBeInTheDocument();
      expect(screen.queryByText("-2.1%")).not.toBeInTheDocument();
      expect(screen.queryByText("Series at Risk")).not.toBeInTheDocument();
      expect(screen.queryByText("Champion Models")).not.toBeInTheDocument();
      expect(screen.getByText(/Summary metrics are not connected/)).toBeInTheDocument();
      expect(screen.getByRole("link", { name: "Backtest Results", exact: true })).toHaveAttribute("href", "/backtest");
      expect(screen.getByText("BI Export", { exact: true })).toBeInTheDocument();
    } finally {
      vi.mocked(useAuth).mockReturnValue(original);
    }
  });
});
