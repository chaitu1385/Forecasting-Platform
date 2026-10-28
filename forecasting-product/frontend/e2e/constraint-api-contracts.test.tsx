import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConstrainedForecastPanel } from "@/components/forecast/constrained-forecast-panel";
import { AuthBoundary } from "@/components/layout/auth-boundary";
import { api, setApiSession } from "@/lib/api-client";
import { ROLE_PERMISSIONS } from "@/lib/constants";
import type { ConstrainResponse } from "@/lib/types";
import { AuthProvider, useAuth } from "@/providers/auth-provider";

const navigation = vi.hoisted(() => ({ replace: vi.fn() }));
vi.mock("next/navigation", () => ({
  useRouter: () => navigation,
  usePathname: () => "/forecast",
}));

const pointFile = new File(
  ["series_id,week,forecast\na,2026-01-05,100\nb,2026-01-05,100\n"],
  "points.csv", { type: "text/csv" },
);
const fetchMock = vi.fn<typeof fetch>();

function result(overrides: Partial<ConstrainResponse> = {}): ConstrainResponse {
  return {
    before_total: 200, after_total: 50, rows_modified: 2,
    constraints_applied: { min_demand: 0, max_capacity: null, aggregate_max: 50, proportional: false },
    constrained_preview: [
      { series_id: "a", week: "2026-01-05", forecast: 0 },
      { series_id: "b", week: "2026-01-05", forecast: 50 },
    ],
    allocation_strategy: "largest_first", checked_rows: 2, checked_periods: 1,
    max_budget_excess: 0, constraint_scope: "point",
    ...overrides,
  };
}

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

function selectFile(file = pointFile) {
  const input = screen.getByRole("button", { name: /Forecast File/ }).querySelector<HTMLInputElement>("input");
  if (!input) throw new Error("Forecast file input is missing");
  fireEvent.change(input, { target: { files: [file] } });
}

function setLimit(label: string, value: string) {
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
}

function submit() {
  fireEvent.click(screen.getByRole("button", { name: "Apply Constraints", exact: true }));
}

function changeOwner(input: "file" | "floor" | "capacity" | "budget" | "strategy") {
  if (input === "file") selectFile(new File(["changed"], "replacement.parquet"));
  if (input === "floor") setLimit("Min Demand", "1");
  if (input === "capacity") setLimit("Max Capacity", "80");
  if (input === "budget") setLimit("Aggregate Max (per period)", "60");
  if (input === "strategy") fireEvent.click(screen.getByRole("checkbox", { name: "Proportional", exact: true }));
}

beforeEach(() => {
  fetchMock.mockReset().mockResolvedValue(response(result()));
  vi.stubGlobal("fetch", fetchMock);
  setApiSession(null);
  navigation.replace.mockClear();
});

afterEach(() => {
  cleanup();
  setApiSession(null);
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("constraint input/output contract", () => {
  it("sends the existing explicit false strategy, zero limits and the selected file through the real client", async () => {
    render(<ConstrainedForecastPanel />);
    expect(screen.getByRole("button", { name: "Apply Constraints" })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "Proportional", exact: true })).not.toBeChecked();
    selectFile();
    setLimit("Max Capacity", "0");
    setLimit("Aggregate Max (per period)", "0");
    submit();
    await screen.findByRole("region", { name: "Constraint result" });
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, options] = fetchMock.mock.calls[0];
    expect(new URL(String(url)).pathname).toBe("/forecast/constrain");
    expect(Object.fromEntries(new URL(String(url)).searchParams)).toEqual({
      min_demand: "0", max_capacity: "0", aggregate_max: "0", proportional: "false",
    });
    expect(options?.method).toBe("POST");
    expect(options?.body).toBeInstanceOf(FormData);
    if (!(options?.body instanceof FormData)) throw new Error("Multipart upload is required");
    expect(options.body.get("file")).toBe(pointFile);
    expect(screen.getByText("Selected file: points.csv")).toBeInTheDocument();
  });

  it("omits blank optional bounds and sends an explicit true proportional choice", async () => {
    fetchMock.mockResolvedValue(response(result({ allocation_strategy: "proportional" })));
    render(<ConstrainedForecastPanel />);
    selectFile();
    fireEvent.click(screen.getByRole("checkbox", { name: "Proportional", exact: true }));
    submit();
    await screen.findByText("Floor-preserving proportional", { exact: true });
    expect(Object.fromEntries(new URL(String(fetchMock.mock.calls[0][0])).searchParams)).toEqual({
      min_demand: "0", proportional: "true",
    });
  });

  it("shows measured full-population totals, actual strategy and unrounded budget excess", async () => {
    fetchMock.mockResolvedValue(response(result({
      before_total: 1000.123456789, after_total: 123.4567890123, checked_rows: 402,
      checked_periods: 2, max_budget_excess: 7.105427357601002e-15,
      allocation_strategy: "proportional",
    })));
    render(<ConstrainedForecastPanel />);
    selectFile();
    submit();
    const output = await screen.findByRole("region", { name: "Constraint result" });
    expect(within(output).getByText("1000.123456789", { exact: true })).toBeInTheDocument();
    expect(within(output).getByText("123.4567890123", { exact: true })).toBeInTheDocument();
    expect(within(output).getByText("7.105427357601002e-15", { exact: true })).toBeInTheDocument();
    expect(within(output).getByText("Floor-preserving proportional", { exact: true })).toBeInTheDocument();
    expect(within(output).getByText(/Preview: 2 of 402 rows/)).toBeInTheDocument();
    expect(within(output).getByText(/Scope: Point forecasts/)).toBeInTheDocument();
    expect(within(output).getByText("Rows Checked")).toBeInTheDocument();
    expect(within(output).getByText("Periods Checked")).toBeInTheDocument();
    expect(within(output).getByText("Rows Modified")).toBeInTheDocument();
  });

  it("discloses marginal-quantile scope without claiming a checked aggregate budget", async () => {
    fetchMock.mockResolvedValue(response(result({
      allocation_strategy: "element_wise", constraint_scope: "point_and_quantiles",
      max_budget_excess: null,
    })));
    render(<ConstrainedForecastPanel />);
    selectFile();
    submit();
    const output = await screen.findByRole("region", { name: "Constraint result" });
    expect(within(output).getByText("Element-wise bounds", { exact: true })).toBeInTheDocument();
    expect(within(output).getByText(/Point and marginal quantile bounds/)).toBeInTheDocument();
    expect(within(output).getByText("No budget supplied", { exact: true })).toBeInTheDocument();
    expect(screen.getByText(/Aggregate budgets support point forecasts only/)).toBeInTheDocument();
  });

  it.each(["infeasible_constraints", "unsupported_aggregate_quantiles"])("shows %s without retrying or retaining an applied result", async (code) => {
    render(<ConstrainedForecastPanel />);
    selectFile();
    submit();
    await screen.findByRole("region", { name: "Constraint result" });
    fetchMock.mockResolvedValue(response({ detail: `${code}: rejected request` }, 400));
    submit();
    expect(screen.queryByRole("region", { name: "Constraint result" })).not.toBeInTheDocument();
    expect(await screen.findByRole("alert")).toHaveTextContent(code);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("region", { name: "Constraint result" })).not.toBeInTheDocument();
  });

  it("only retries on the user's explicit Retry action", async () => {
    fetchMock.mockResolvedValueOnce(response({ detail: "infeasible_constraints" }, 400));
    render(<ConstrainedForecastPanel />);
    selectFile();
    submit();
    await screen.findByRole("alert");
    expect(fetchMock).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "Retry", exact: true }));
    await screen.findByRole("region", { name: "Constraint result" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("preserves custom column parameters in the existing multipart client", async () => {
    await api.constrainForecast(pointFile, {
      id_col: "sku & item", time_col: "month start", value_col: "_total", proportional: false,
    });
    expect(Object.fromEntries(new URL(String(fetchMock.mock.calls[0][0])).searchParams)).toEqual({
      id_col: "sku & item", time_col: "month start", value_col: "_total", proportional: "false",
    });
  });
});

describe("constraint request ownership", () => {
  const inputs = ["file", "floor", "capacity", "budget", "strategy"] as const;

  it.each(inputs)("clears a completed result immediately when %s changes", async input => {
    render(<ConstrainedForecastPanel />);
    selectFile();
    setLimit("Aggregate Max (per period)", "50");
    submit();
    await screen.findByRole("region", { name: "Constraint result" });
    changeOwner(input);
    expect(screen.queryByRole("region", { name: "Constraint result" })).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  for (const outcome of ["success", "failure"] as const) {
    it.each(inputs)(`ignores old ${outcome} after changing %s and completing a newer request`, async input => {
      const pending = deferred<Response>();
      fetchMock.mockReturnValueOnce(pending.promise);
      render(<ConstrainedForecastPanel />);
      selectFile();
      setLimit("Aggregate Max (per period)", "50");
      submit();
      expect(screen.getByRole("button", { name: "Applying Constraints..." })).toBeDisabled();
      changeOwner(input);
      expect(screen.getByRole("button", { name: "Apply Constraints" })).toBeEnabled();
      fetchMock.mockResolvedValueOnce(response(result({ after_total: 30, allocation_strategy: "proportional" })));
      submit();
      const output = await screen.findByRole("region", { name: "Constraint result" });
      expect(within(output).getByText("30", { exact: true })).toBeInTheDocument();
      await act(async () => {
        if (outcome === "success") pending.resolve(response(result({ after_total: 49 })));
        else pending.reject(new Error("obsolete constraint failure"));
        await pending.promise.catch(() => undefined);
      });
      expect(within(output).getByText("30", { exact: true })).toBeInTheDocument();
      expect(screen.queryByText("49", { exact: true })).not.toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Apply Constraints" })).toBeEnabled();
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });
  }

  it("does not repaint a new mount after an old panel is unmounted", async () => {
    const pending = deferred<Response>();
    fetchMock.mockReturnValueOnce(pending.promise);
    const old = render(<ConstrainedForecastPanel />);
    selectFile();
    submit();
    old.unmount();
    render(<ConstrainedForecastPanel />);
    await act(async () => { pending.resolve(response(result())); await pending.promise; });
    expect(screen.queryByRole("region", { name: "Constraint result" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Apply Constraints" })).toBeDisabled();
  });
});

function PrivateAuthHarness() {
  const { mode, loginWithToken, logout, user } = useAuth();
  const [verified, setVerified] = useState(0);
  const signIn = async (token: string) => {
    await loginWithToken(token);
    setVerified(count => count + 1);
  };
  return <>
    <button disabled={mode !== "production"} onClick={() => void signIn("synthetic-first")}>First session</button>
    <button disabled={mode !== "production"} onClick={() => void signIn("synthetic-replacement")}>Replacement session</button>
    <button onClick={logout}>End session</button>
    <p data-testid="verified-sessions">{verified}</p>
    <p data-testid="session-user">{user?.username ?? "signed out"}</p>
    <AuthBoundary><ConstrainedForecastPanel /></AuthBoundary>
  </>;
}

describe("direct protected-session replacement without logout", () => {
  for (const username of ["first", "replacement"]) {
    for (const timing of ["immediate", "deferred"] as const) {
      it.each(["completed", "error", "pending-success", "pending-failure"] as const)(
        `${username} / ${timing} identity cannot retain %s constraint state`,
        async previous => {
          const pending = deferred<Response>();
          const identity = deferred<Response>();
          let identityCalls = 0;
          const verifiedIdentity = (name: string) => response({
            mode: "production", username: name, email: "", role: "data_scientist",
            permissions: ROLE_PERMISSIONS.data_scientist,
            expires_at: Math.floor(Date.now() / 1000) + 3600,
          });
          fetchMock.mockImplementation(async url => {
            const path = new URL(String(url)).pathname;
            if (path === "/auth/status") return response({ mode: "production", auth_required: true });
            if (path === "/auth/me") {
              identityCalls += 1;
              if (identityCalls === 1) return verifiedIdentity("first");
              return timing === "immediate" ? verifiedIdentity(username) : identity.promise;
            }
            if (path === "/forecast/constrain") {
              if (previous === "completed") return response(result());
              if (previous === "error") return response({ detail: "previous constraint failure" }, 400);
              return pending.promise;
            }
            throw new Error("Unexpected test endpoint");
          });
          const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } });
          render(<QueryClientProvider client={client}><AuthProvider><PrivateAuthHarness /></AuthProvider></QueryClientProvider>);
          await waitFor(() => expect(screen.getByRole("button", { name: "First session" })).toBeEnabled());
          fireEvent.click(screen.getByRole("button", { name: "First session" }));
          await waitFor(() => expect(screen.getByTestId("verified-sessions")).toHaveTextContent("1"));
          selectFile();
          setLimit("Aggregate Max (per period)", "50");
          submit();
          if (previous === "completed") await screen.findByRole("region", { name: "Constraint result" });
          else if (previous === "error") await screen.findByRole("alert");
          else expect(screen.getByRole("button", { name: "Applying Constraints..." })).toBeDisabled();
          client.setQueryData(["private-constraint"], { owner: "first" });

          fireEvent.click(screen.getByRole("button", { name: "Replacement session" }));
          if (timing === "deferred") {
            expect(screen.queryByRole("region", { name: "Constrained Forecast" })).not.toBeInTheDocument();
            await act(async () => { identity.resolve(verifiedIdentity(username)); await identity.promise; });
          }
          await waitFor(() => expect(screen.getByTestId("verified-sessions")).toHaveTextContent("2"));
          expect(identityCalls).toBe(2);
          expect(screen.getByTestId("session-user")).toHaveTextContent(username);
          if (previous.startsWith("pending")) {
            await act(async () => {
              if (previous === "pending-success") pending.resolve(response(result({ after_total: 49 })));
              else pending.reject(new Error("previous delayed constraint failure"));
              await pending.promise.catch(() => undefined);
            });
          }
          expect(screen.queryByRole("region", { name: "Constraint result" })).not.toBeInTheDocument();
          expect(screen.queryByRole("alert")).not.toBeInTheDocument();
          expect(screen.queryByText("Selected file: points.csv")).not.toBeInTheDocument();
          expect(screen.getByRole("button", { name: "Apply Constraints" })).toBeDisabled();
          expect(screen.getByLabelText("Aggregate Max (per period)")).toHaveValue(null);
          expect(client.getQueryCache().getAll()).toHaveLength(0);
          client.clear();
        },
      );
    }
  }
});

it("clears completed/pending constraint state and protected cache on logout and replacement login", async () => {
  const pending = deferred<Response>();
  let constraintCalls = 0;
  fetchMock.mockImplementation(async (url, options) => {
    const path = new URL(String(url)).pathname;
    if (path === "/auth/status") return response({ mode: "production", auth_required: true });
    if (path === "/auth/me") {
      const replacement = new Headers(options?.headers).get("authorization")?.endsWith("synthetic-replacement");
      return response({
        mode: "production", username: replacement ? "replacement" : "first", email: "",
        role: "data_scientist", permissions: ROLE_PERMISSIONS.data_scientist,
        expires_at: Math.floor(Date.now() / 1000) + 3600,
      });
    }
    if (path === "/forecast/constrain") {
      constraintCalls += 1;
      return constraintCalls === 1 ? response(result()) : pending.promise;
    }
    throw new Error("Unexpected test endpoint");
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } });
  render(<QueryClientProvider client={client}><AuthProvider><PrivateAuthHarness /></AuthProvider></QueryClientProvider>);
  await waitFor(() => expect(screen.getByRole("button", { name: "First session" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "First session" }));
  await screen.findByRole("region", { name: "Constrained Forecast" });
  selectFile();
  submit();
  await screen.findByRole("region", { name: "Constraint result" });
  client.setQueryData(["private-constraint"], { before_total: 200 });
  submit();
  fireEvent.click(screen.getByRole("button", { name: "End session" }));
  expect(screen.queryByRole("region", { name: "Constraint result" })).not.toBeInTheDocument();
  expect(screen.queryByRole("region", { name: "Constrained Forecast" })).not.toBeInTheDocument();
  expect(client.getQueryData(["private-constraint"])).toBeUndefined();
  fireEvent.click(screen.getByRole("button", { name: "Replacement session" }));
  await screen.findByRole("region", { name: "Constrained Forecast" });
  await act(async () => { pending.resolve(response(result({ after_total: 49 }))); await pending.promise; });
  expect(screen.queryByRole("region", { name: "Constraint result" })).not.toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Apply Constraints" })).toBeDisabled();
  expect(client.getQueryCache().getAll()).toHaveLength(0);
  client.clear();
});
