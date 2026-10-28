import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import HierarchyPage from "@/app/hierarchy/page";
import { api, setApiSession } from "@/lib/api-client";
import type { HierarchyReconcileResponse } from "@/lib/types";

vi.mock("@/components/charts/hierarchy-sunburst", () => ({ HierarchySunburst: () => <div /> }));

const leafFile = new File([
  "department,item,week,forecast\nfood,a,2026-01-05,10\nfood,b,2026-01-05,20\nfood,a,2026-01-12,4\nfood,b,2026-01-12,6\n",
], "leaves.csv", { type: "text/csv" });
const rootFile = new File(["department,week,forecast\nfood,2026-01-05,120\n"], "roots.csv", { type: "text/csv" });
const historyFile = new File([
  "department,item,week,sales\nfood,a,2025-12-01,10\nfood,b,2025-12-01,30\n",
], "history.csv", { type: "text/csv" });

function result(topDown = false): HierarchyReconcileResponse {
  const method = topDown ? "top_down" : "bottom_up";
  return {
    method, requested_method: method, effective_method: method,
    source_level: topDown ? "department" : "item", output_level: "item",
    input_rows: topDown ? 1 : 4, output_rows: topDown ? 2 : 4, rows: topDown ? 2 : 4,
    before_total: topDown ? 120 : 40, after_total: topDown ? 120 : 40,
    reconciled_preview: topDown ? [
      { item: "a", week: "2026-01-05", forecast: 30 },
      { item: "b", week: "2026-01-05", forecast: 90 },
    ] : [
      { item: "a", week: "2026-01-05", forecast: 10 },
      { item: "b", week: "2026-01-05", forecast: 20 },
      { item: "a", week: "2026-01-12", forecast: 4 },
      { item: "b", week: "2026-01-12", forecast: 6 },
    ],
    warnings: topDown ? [] : ["Parent totals are derived; no independent upper forecasts were supplied."],
    coherence: {
      scope: topDown ? "root_forecast_conservation" : "leaf_derived_parent_totals", passed: true,
      checked_rows: topDown ? 2 : 4, checked_periods: topDown ? 1 : 2,
      checked_groups: topDown ? 1 : 2, value_columns: ["forecast"], max_abs_error: 0,
    },
    validation: null,
  };
}

function populate(topDown = false) {
  fireEvent.change(screen.getByLabelText("Upload CSV or Parquet"), { target: { files: [topDown ? rootFile : leafFile] } });
  fireEvent.change(screen.getByLabelText("Hierarchy Levels (comma-separated)"), { target: { value: "department,item" } });
  if (topDown) {
    fireEvent.change(screen.getByLabelText("Method"), { target: { value: "top_down" } });
    fireEvent.change(screen.getByLabelText("Historical actuals"), { target: { files: [historyFile] } });
    fireEvent.change(screen.getByLabelText("History value column"), { target: { value: "sales" } });
  }
}

function reconcile() {
  fireEvent.click(screen.getByRole("button", { name: "Reconcile", exact: true }));
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  setApiSession(null);
});

describe("TEST-005 hierarchy numerical result and method-specific inputs", () => {
  it("submits the real leaf level and displays all four exact hand-fixture rows", async () => {
    const request = vi.spyOn(api, "reconcileHierarchy").mockResolvedValue(result());
    render(<HierarchyPage />);
    populate();
    expect(screen.queryByLabelText("Historical actuals")).not.toBeInTheDocument();
    reconcile();
    const preview = await screen.findByTestId("reconciliation-result");
    expect(request).toHaveBeenCalledWith(leafFile, "department,item", "bottom_up", { id_column: "item" }, undefined);
    expect(within(preview).getAllByRole("row")).toHaveLength(5);
    for (const value of ["10", "20", "4", "6"]) {
      expect(within(preview).getByRole("cell", { name: value, exact: true })).toBeVisible();
    }
    expect(preview).toHaveTextContent("4 rows, 2 periods, 2 parent/period groups checked");
    expect(preview).toHaveTextContent("Preview shows 4 of 4 output rows");
    expect(screen.getByTestId("reconciliation-effective-method")).toHaveTextContent("Bottom-Up");
  });

  it("requires earlier history for top-down and forwards its chosen value column", async () => {
    const request = vi.spyOn(api, "reconcileHierarchy").mockResolvedValue(result(true));
    render(<HierarchyPage />);
    populate();
    fireEvent.change(screen.getByLabelText("Method"), { target: { value: "top_down" } });
    const button = screen.getByRole("button", { name: "Reconcile", exact: true });
    expect(button).toBeDisabled();
    expect(screen.getByLabelText("History value column")).toHaveValue("quantity");
    fireEvent.change(screen.getByLabelText("Upload CSV or Parquet"), { target: { files: [rootFile] } });
    fireEvent.change(screen.getByLabelText("Historical actuals"), { target: { files: [historyFile] } });
    fireEvent.change(screen.getByLabelText("History value column"), { target: { value: "" } });
    expect(button).toBeDisabled();
    fireEvent.change(screen.getByLabelText("History value column"), { target: { value: "sales" } });
    reconcile();
    const preview = await screen.findByTestId("reconciliation-result");
    expect(request).toHaveBeenCalledWith(
      rootFile, "department,item", "top_down", { id_column: "item", history_value_column: "sales" }, historyFile,
    );
    expect(within(preview).getByRole("cell", { name: "30", exact: true })).toBeVisible();
    expect(within(preview).getByRole("cell", { name: "90", exact: true })).toBeVisible();
    expect(preview).toHaveTextContent("Root conservation: 2 rows, 1 periods, 1 parent/period groups checked");
  });

  it("shows structural MinT fallback, not an invented residual-covariance calculation", async () => {
    const response: HierarchyReconcileResponse = {
      ...result(), method: "mint", requested_method: "mint", effective_method: "wls_structural",
      warnings: ["Residuals were not supplied; MinT used structural WLS, not residual-covariance MinT."],
    };
    vi.spyOn(api, "reconcileHierarchy").mockResolvedValue(response);
    render(<HierarchyPage />);
    populate();
    fireEvent.change(screen.getByLabelText("Method"), { target: { value: "mint" } });
    expect(screen.queryByRole("option", { name: /middle-out/i })).not.toBeInTheDocument();
    expect(screen.getByText(/Middle-out is unavailable/)).toBeVisible();
    reconcile();
    const preview = await screen.findByTestId("reconciliation-result");
    expect(screen.getByTestId("reconciliation-effective-method")).toHaveTextContent("Structural WLS");
    expect(preview).toHaveTextContent("wls_structural");
    expect(preview).toHaveTextContent("not residual-covariance MinT");
  });

  it("does not reuse an invisible old history upload after leaving and reentering top-down", () => {
    render(<HierarchyPage />);
    populate(true);
    expect(screen.getByRole("button", { name: "Reconcile", exact: true })).toBeEnabled();
    fireEvent.change(screen.getByLabelText("Method"), { target: { value: "bottom_up" } });
    fireEvent.change(screen.getByLabelText("Method"), { target: { value: "top_down" } });
    expect(screen.getByRole("button", { name: "Reconcile", exact: true })).toBeDisabled();
  });

  it("shows an explicit API error, retains the inputs and clears it on correction", async () => {
    vi.spyOn(api, "reconcileHierarchy").mockRejectedValue(new Error("invalid_hierarchy_input: Duplicate node/time observations"));
    render(<HierarchyPage />);
    populate();
    reconcile();
    expect(await screen.findByRole("alert")).toHaveTextContent("Duplicate node/time");
    expect(screen.getByLabelText("Hierarchy Levels (comma-separated)")).toHaveValue("department,item");
    fireEvent.change(screen.getByLabelText("Upload CSV or Parquet"), { target: { files: [leafFile] } });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByTestId("reconciliation-result")).not.toBeInTheDocument();
  });
});

describe("TEST-005 hierarchy request ownership", () => {
  const changes = ["file", "levels", "method", "actuals", "history-value"] as const;
  for (const completed of [false, true]) {
    it.each(changes)(`clears ${completed ? "completed" : "pending"} results when %s changes`, async change => {
      let finish!: (value: HierarchyReconcileResponse) => void;
      const pending = new Promise<HierarchyReconcileResponse>(resolve => { finish = resolve; });
      vi.spyOn(api, "reconcileHierarchy").mockReturnValue(completed ? Promise.resolve(result()) : pending);
      render(<HierarchyPage />);
      populate(change === "actuals" || change === "history-value");
      reconcile();
      if (completed) await screen.findByTestId("reconciliation-result");
      else expect(screen.getByRole("button", { name: "Reconciling..." })).toBeDisabled();

      if (change === "file") {
        fireEvent.change(screen.getByLabelText("Upload CSV or Parquet"), { target: { files: [new File(["replacement"], "new.csv")] } });
      } else if (change === "levels") {
        fireEvent.change(screen.getByLabelText("Hierarchy Levels (comma-separated)"), { target: { value: "category,sku" } });
      } else if (change === "method") {
        fireEvent.change(screen.getByLabelText("Method"), { target: { value: "wls" } });
      } else if (change === "actuals") {
        fireEvent.change(screen.getByLabelText("Historical actuals"), { target: { files: [new File(["replacement"], "new-history.csv")] } });
      } else {
        fireEvent.change(screen.getByLabelText("History value column"), { target: { value: "units" } });
      }
      expect(screen.queryByTestId("reconciliation-result")).not.toBeInTheDocument();
      if (!completed) await act(async () => { finish(result()); });
      expect(screen.queryByTestId("reconciliation-result")).not.toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Reconcile", exact: true })).toBeEnabled();
    });
  }

  it("keeps a newer successful request after the replaced request finally completes", async () => {
    let finish!: (value: HierarchyReconcileResponse) => void;
    const old = new Promise<HierarchyReconcileResponse>(resolve => { finish = resolve; });
    const current = result();
    const previous = { ...result(), reconciled_preview: [{ item: "OLD_ONLY", week: "2026-01-05", forecast: 777 }] };
    vi.spyOn(api, "reconcileHierarchy").mockReturnValueOnce(old).mockResolvedValueOnce(current);
    render(<HierarchyPage />);
    populate();
    reconcile();
    fireEvent.change(screen.getByLabelText("Upload CSV or Parquet"), { target: { files: [leafFile] } });
    reconcile();
    const preview = await screen.findByTestId("reconciliation-result");
    await act(async () => { finish(previous); });
    expect(within(preview).queryByRole("cell", { name: "OLD_ONLY" })).not.toBeInTheDocument();
    expect(within(preview).getByRole("cell", { name: "20", exact: true })).toBeVisible();
  });

  it("ignores a replaced request's late error", async () => {
    let fail!: (error: Error) => void;
    vi.spyOn(api, "reconcileHierarchy").mockReturnValue(new Promise((_, reject) => { fail = reject; }));
    render(<HierarchyPage />);
    populate();
    reconcile();
    fireEvent.change(screen.getByLabelText("Method"), { target: { value: "ols" } });
    await act(async () => { fail(new Error("OLD_ONLY")); });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reconcile", exact: true })).toBeEnabled();
  });

  it("does not repaint a remounted page with a prior session's pending result", async () => {
    let finish!: (value: HierarchyReconcileResponse) => void;
    vi.spyOn(api, "reconcileHierarchy").mockReturnValue(new Promise(resolve => { finish = resolve; }));
    const first = render(<HierarchyPage />);
    populate();
    reconcile();
    first.unmount();
    render(<HierarchyPage />);
    await act(async () => { finish(result()); });
    expect(screen.queryByTestId("reconciliation-result")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Hierarchy Levels (comma-separated)")).toHaveValue("");
  });
});

describe("TEST-022 real hierarchy client transport", () => {
  it("sends both multipart uploads and typed query fields without rewriting the files", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify(result(true)), {
      status: 200, headers: { "Content-Type": "application/json" },
    }));
    vi.stubGlobal("fetch", fetcher);
    const response = await api.reconcileHierarchy(rootFile, "department,item", "top_down", {
      id_column: "item", history_value_column: "sales", time_column: "week",
      value_columns: "forecast", validate: true,
    }, historyFile);
    expect(response).toEqual(result(true));
    expect(fetcher).toHaveBeenCalledTimes(1);
    const [url, request] = fetcher.mock.calls[0];
    const parsed = new URL(String(url));
    expect(parsed.pathname).toBe("/hierarchy/reconcile");
    expect(Object.fromEntries(parsed.searchParams)).toEqual({
      levels: "department,item", method: "top_down", id_column: "item",
      history_value_column: "sales", time_column: "week", value_columns: "forecast", validate: "true",
    });
    expect(request?.method).toBe("POST");
    const form = request?.body;
    if (!(form instanceof FormData)) throw new Error("Expected multipart uploads");
    const actuals = form.get("actuals_file");
    const forecast = form.get("file");
    if (!(actuals instanceof File) || !(forecast instanceof File)) throw new Error("Expected both files");
    expect(actuals.name).toBe("history.csv");
    expect(actuals.size).toBe(historyFile.size);
    expect(forecast.name).toBe("roots.csv");
    expect(forecast.size).toBe(rootFile.size);
    expect(new Headers(request?.headers).has("Content-Type")).toBe(false);
  });

  it("does not send a stale response into a replacement authentication session", async () => {
    let finish!: (response: Response) => void;
    const fetcher = vi.fn<typeof fetch>().mockReturnValue(new Promise(resolve => { finish = resolve; }));
    vi.stubGlobal("fetch", fetcher);
    setApiSession({ kind: "demo", username: "old-session", role: "admin" });
    const request = api.reconcileHierarchy(leafFile, "department,item");
    const rejected = expect(request).rejects.toThrow("Session changed");
    setApiSession({ kind: "demo", username: "new-session", role: "viewer" });
    finish(new Response(JSON.stringify(result()), { status: 200 }));
    await rejected;
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it.each(["network", "503"])("never automatically retries a hierarchy POST after %s failure", async failure => {
    const fetcher = vi.fn<typeof fetch>();
    if (failure === "network") fetcher.mockRejectedValue(new TypeError("synthetic network failure"));
    else fetcher.mockResolvedValue(new Response("unavailable", { status: 503 }));
    vi.stubGlobal("fetch", fetcher);
    await expect(api.reconcileHierarchy(leafFile, "department,item")).rejects.toThrow();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});
