import { expect, test, type Page } from "@playwright/test";
import { createHash, randomUUID } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import type {
  ConstrainResponse,
  DecompositionSummaryResponse,
  ForecastResponse,
  HierarchyReconcileResponse,
  OverrideItem,
  OverrideListResponse,
} from "../../src/lib/types";
import { submitPrivately } from "../fixtures/authentication";
import { navigateWithinApp } from "../helpers/auth";

interface FixtureMarker {
  api_url: string;
  frontend_origin: string;
  mode: string;
  forecast_runs: { run_id: string; value: number }[];
}

function runHashes(directory: string): Record<string, string> {
  const hashes: Record<string, string> = {};
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const member = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      for (const [name, hash] of Object.entries(runHashes(member))) {
        hashes[path.join(entry.name, name)] = hash;
      }
    } else if (entry.isFile()) {
      hashes[entry.name] = createHash("sha256").update(readFileSync(member)).digest("hex");
    } else {
      throw new Error("Unexpected nonregular immutable fixture member");
    }
  }
  return hashes;
}

function apiResponse(page: Page, pathname: string, method: string) {
  return page.waitForResponse(response =>
    new URL(response.url()).pathname === pathname && response.request().method() === method);
}

async function removeOwnedOverride(page: Page, apiUrl: string, token: string, id: string) {
  try {
    return await page.evaluate(async ({ url, credential, overrideId }) => {
      const headers = { Authorization: `Bearer ${credential}` };
      const removed = await fetch(`${url}/overrides/${overrideId}`, { method: "DELETE", headers });
      const read = await fetch(`${url}/overrides`, { headers });
      const records: OverrideListResponse = await read.json();
      return { deleteStatus: removed.status, readStatus: read.status, records };
    }, { url: apiUrl, credential: token, overrideId: id });
  } catch {
    throw new Error("Private F8 override cleanup failed");
  }
}

function sortedOverrides(records: OverrideItem[]) {
  return [...records].sort((a, b) => a.override_id.localeCompare(b.override_id));
}

test("F8 workflows retain real outcomes and original run provenance across one authenticated app session", async ({ page }, testInfo) => {
  expect(path.basename(testInfo.config.configFile ?? "")).toBe("playwright.auth.config.ts");
  expect(process.env.AUTH_E2E_FRONTEND_MODE).toBe("production");
  for (const option of ["trace", "screenshot", "video"] as const) {
    expect(testInfo.project.use[option]).toBe("off");
  }
  expect(Boolean(process.env.DEBUG || process.env.PWDEBUG)).toBe(false);
  const root = process.env.AUTH_E2E_ROOT;
  if (!root || !path.isAbsolute(root)) throw new Error("A private absolute authentication fixture root is required");
  const marker: FixtureMarker = JSON.parse(readFileSync(path.join(root, "data", ".authentication-fixture.json"), "utf8"));
  expect(marker.frontend_origin).toBe(testInfo.project.use.baseURL);
  expect(marker.mode).toBe("production");
  expect(new URL(marker.api_url).hostname).toBe("127.0.0.1");
  expect(new URL(marker.frontend_origin).hostname).toBe("127.0.0.1");
  expect(marker.forecast_runs).toHaveLength(2);
  const [original, latest] = marker.forecast_runs;
  expect(original.run_id).toMatch(/^[0-9a-f]{32}$/);
  expect(original.run_id).not.toBe(latest.run_id);
  const runs = path.join(root, "data", "runs");
  const hashesBefore = runHashes(runs);
  expect(Object.keys(hashesBefore).length).toBeGreaterThan(0);
  const admin = readFileSync(path.join(root, "credentials", "admin.jwt"), "utf8").trim();
  let exposed = false;
  let identities = 0;
  let documents = 0;
  let creates = 0;
  let updates = 0;
  page.on("console", message => { exposed ||= message.text().includes(admin); });
  page.on("pageerror", error => { exposed ||= error.message.includes(admin); });
  page.on("request", request => {
    const url = new URL(request.url());
    exposed ||= url.href.includes(admin);
    if (url.pathname === "/auth/me") identities += 1;
    if (request.isNavigationRequest() && request.frame() === page.mainFrame()) documents += 1;
    if (url.pathname === "/overrides" && request.method() === "POST") creates += 1;
    if (url.pathname.startsWith("/overrides/") && request.method() === "PUT") updates += 1;
  });
  await page.route("**/*", route =>
    [marker.frontend_origin, marker.api_url].includes(new URL(route.request().url()).origin)
      ? route.continue() : route.abort());

  await page.goto("/login");
  await expect(page.getByLabel("Access token", { exact: true })).toHaveAttribute("type", "password");
  await expect(page.locator("#role")).toHaveCount(0);
  await submitPrivately(page, admin);
  await page.waitForURL(`${marker.frontend_origin}/`);
  await expect(page.locator("header").getByText("browser-admin", { exact: true })).toBeVisible();

  await navigateWithinApp(page, "/hierarchy");
  await page.getByLabel("Upload CSV or Parquet").setInputFiles({
    name: "f8-leaves.csv", mimeType: "text/csv",
    buffer: Buffer.from("department,item,week,forecast\nfood,a,2026-01-05,10\nfood,b,2026-01-05,20\nfood,a,2026-01-12,4\nfood,b,2026-01-12,6\n"),
  });
  await page.getByLabel("Hierarchy Levels (comma-separated)").fill("department,item");
  const reconciliation = apiResponse(page, "/hierarchy/reconcile", "POST");
  await page.getByRole("button", { name: "Reconcile", exact: true }).click();
  const reconciled = await reconciliation;
  expect(reconciled.status()).toBe(200);
  const hierarchy: HierarchyReconcileResponse = await reconciled.json();
  expect(hierarchy.reconciled_preview).toEqual([
    { item: "a", week: "2026-01-05", forecast: 10 }, { item: "b", week: "2026-01-05", forecast: 20 },
    { item: "a", week: "2026-01-12", forecast: 4 }, { item: "b", week: "2026-01-12", forecast: 6 },
  ]);
  expect(["2026-01-05", "2026-01-12"].map(week =>
    hierarchy.reconciled_preview.filter(row => row.week === week)
      .reduce((total, row) => total + Number(row.forecast), 0))).toEqual([30, 10]);
  expect(hierarchy.coherence.checked_rows).toBe(4);
  const hierarchyOutput = page.getByTestId("reconciliation-result");
  await expect(hierarchyOutput.locator("tbody tr")).toHaveCount(4);
  for (const value of ["10", "20", "4", "6"]) {
    await expect(hierarchyOutput.getByRole("cell", { name: value, exact: true })).toBeVisible();
  }

  const initialRead = apiResponse(page, "/overrides", "GET");
  await navigateWithinApp(page, "/sku-transitions");
  const initialResponse = await initialRead;
  expect(initialResponse.status()).toBe(200);
  const initial: OverrideListResponse = await initialResponse.json();
  let createdId: string | undefined;
  let edited: OverrideItem | undefined;
  let selectedForecast: ForecastResponse | undefined;
  let constraints: ConstrainResponse | undefined;
  let decomposition: DecompositionSummaryResponse | undefined;
  try {
    const panel = page.getByRole("region", { name: "Planner Override Management" });
    const identity = randomUUID();
    await panel.getByRole("button", { name: "Add Override", exact: true }).click();
    await panel.getByLabel("Old SKU *", { exact: true }).fill(`F8_OLD_${identity}`);
    await panel.getByLabel("New SKU *", { exact: true }).fill(`F8_NEW_${identity}`);
    await panel.getByLabel("Proportion", { exact: true }).fill("0.75");
    await panel.getByLabel("Notes", { exact: true }).fill("F8 combined before");
    const creation = apiResponse(page, "/overrides", "POST");
    const createdRead = apiResponse(page, "/overrides", "GET");
    await panel.getByRole("button", { name: "Create Override", exact: true }).click();
    const created = await creation;
    const creationBody: { override_id: string; status: string } = await created.json();
    createdId = creationBody.override_id;
    expect(created.status()).toBe(200);
    expect(createdId).toMatch(/^OVR-[A-F0-9]{8}$/);
    const createdList: OverrideListResponse = await (await createdRead).json();
    const before = createdList.overrides.find(row => row.override_id === createdId);
    if (!before) throw new Error("The newly created F8 override was not read back");
    expect(createdList.count).toBe(initial.count + 1);
    const row = panel.locator(`[data-override-id="${createdId}"]`);
    await expect(row.getByText("F8 combined before", { exact: true })).toBeVisible();
    await row.getByRole("button", { name: "Edit", exact: true }).click();
    await expect(panel.getByLabel("Old SKU *", { exact: true })).toHaveAttribute("readonly", "");
    await expect(panel.getByLabel("New SKU *", { exact: true })).toHaveAttribute("readonly", "");
    await panel.getByLabel("Proportion", { exact: true }).fill("0.25");
    await panel.getByLabel("Notes", { exact: true }).fill("F8 combined after");
    const update = apiResponse(page, `/overrides/${createdId}`, "PUT");
    const updatedRead = apiResponse(page, "/overrides", "GET");
    await panel.getByRole("button", { name: "Save", exact: true }).click();
    const updated = await update;
    expect(updated.status()).toBe(200);
    expect(updated.request().postDataJSON()).toEqual({ proportion: 0.25, notes: "F8 combined after" });
    expect(await updated.json()).toEqual({ override_id: createdId, status: "updated" });
    const updatedList: OverrideListResponse = await (await updatedRead).json();
    edited = { ...before, proportion: 0.25, notes: "F8 combined after" };
    expect(updatedList.overrides.find(item => item.override_id === createdId)).toEqual(edited);
    expect(updatedList.count).toBe(initial.count + 1);
    expect(sortedOverrides(updatedList.overrides.filter(item => item.override_id !== createdId)))
      .toEqual(sortedOverrides(initial.overrides));
    await expect(row.getByText("F8 combined after", { exact: true })).toBeVisible();
    await expect(row.getByRole("cell").nth(2)).toHaveText("0.25");

    await navigateWithinApp(page, "/forecast");
    await expect(page.getByTestId("forecast-run-id")).toHaveText(latest.run_id);
    const originalRead = page.waitForResponse(response =>
      new URL(response.url()).pathname === "/forecast/retail"
      && new URL(response.url()).searchParams.get("run_id") === original.run_id);
    await page.getByLabel("Forecast run", { exact: true }).selectOption(original.run_id);
    const originalResponse = await originalRead;
    expect(originalResponse.status()).toBe(200);
    selectedForecast = await originalResponse.json();
    expect(selectedForecast).toMatchObject({ run_id: original.run_id, artifact_source: "committed" });
    expect(selectedForecast?.points.map(point => point.forecast)).toEqual([10, 10, 10, 10]);
    expect(selectedForecast?.points.every(point => point.model === "naive_seasonal")).toBe(true);
    await expect(page.getByTestId("forecast-run-id")).toHaveText(original.run_id);
    const constraintPanel = page.getByRole("region", { name: "Constrained Forecast", exact: true });
    await constraintPanel.locator("input[type=file]").setInputFiles({
      name: "f8-budget.csv", mimeType: "text/csv",
      buffer: Buffer.from("series_id,week,forecast\na,2026-01-05,100\nb,2026-01-05,100\n"),
    });
    await constraintPanel.getByLabel("Aggregate Max (per period)", { exact: true }).fill("50");
    await expect(constraintPanel.getByRole("checkbox", { name: "Proportional", exact: true })).not.toBeChecked();
    const allocation = apiResponse(page, "/forecast/constrain", "POST");
    await constraintPanel.getByRole("button", { name: "Apply Constraints", exact: true }).click();
    const allocated = await allocation;
    expect(allocated.status()).toBe(200);
    expect(new URL(allocated.url()).searchParams.get("proportional")).toBe("false");
    constraints = await allocated.json();
    expect(constraints).toMatchObject({
      before_total: 200, after_total: 50, rows_modified: 2, allocation_strategy: "largest_first",
      checked_rows: 2, checked_periods: 1, max_budget_excess: 0, constraint_scope: "point",
    });
    expect(constraints?.constrained_preview.map(point => [point.series_id, point.forecast]))
      .toEqual([["a", 0], ["b", 50]]);
    const constrainedOutput = constraintPanel.getByRole("region", { name: "Constraint result", exact: true });
    await expect(constrainedOutput.getByText("Largest-first", { exact: true })).toBeVisible();
    await expect(constrainedOutput.getByRole("row").nth(1).getByRole("cell").nth(2)).toHaveText("0");
    await expect(constrainedOutput.getByRole("row").nth(2).getByRole("cell").nth(2)).toHaveText("50");
    await expect(page.getByTestId("forecast-run-id")).toHaveText(original.run_id);

    await navigateWithinApp(page, "/backtest");
    const componentPanel = page.getByRole("region", { name: "Forecast decomposition", exact: true });
    await page.getByLabel("Decomposition forecast source").selectOption(original.run_id);
    await page.getByLabel("Season length (observations)").fill("3");
    const components = apiResponse(page, "/metrics/retail/decomposition", "POST");
    await page.getByRole("button", { name: "Load decomposition", exact: true }).click();
    const componentResponse = await components;
    expect(componentResponse.status()).toBe(200);
    expect(new URL(componentResponse.url()).searchParams.get("run_id")).toBe(original.run_id);
    decomposition = await componentResponse.json();
    expect(decomposition).toMatchObject({
      run_id: original.run_id, artifact_source: "committed", history_run_id: original.run_id,
      history_source: "committed", history_stage: "prepared", forecast_model: "naive_seasonal",
      forecast_model_source: "manifest", method: "moving_average_decomposition", model_specific: false,
      history_rows: 64, forecast_rows: 4, checked_rows: 68, preview_rows: 50, preview_truncated: true,
    });
    expect(decomposition?.components).toEqual([
      { component: "trend", mean_abs_value: 10, std: 0, defined_count: 62, undefined_count: 2 },
      { component: "seasonal", mean_abs_value: 0, std: 0, defined_count: 64, undefined_count: 0 },
      { component: "residual", mean_abs_value: 0, std: 0, defined_count: 62, undefined_count: 2 },
    ]);
    await expect(page.getByTestId("decomposition-run-id")).toHaveText(original.run_id);
    await expect(page.getByTestId("component-trend")).toContainText("10.0000");
    await expect(componentPanel).toContainText("moving_average_decomposition");
    await expect(componentPanel).toContainText("naive_seasonal");
    await expect(componentPanel).toContainText("Not model-specific feature attribution");
    await expect(componentPanel).toContainText("separate forecast source, not the leaderboard run");
    await expect(componentPanel).toContainText("recorded prepared snapshot from this same forecast run");
    await expect(componentPanel).not.toContainText(/SHAP|waterfall|feature importance|LOESS|causal/i);

    const revisit = apiResponse(page, "/overrides", "GET");
    await navigateWithinApp(page, "/sku-transitions");
    const retained: OverrideListResponse = await (await revisit).json();
    expect(retained.overrides.find(item => item.override_id === createdId)).toEqual(edited);
    await expect(panel.locator(`[data-override-id="${createdId}"]`).getByText("F8 combined after", { exact: true })).toBeVisible();
    await expect(page.locator("header").getByText("browser-admin", { exact: true })).toBeVisible();
    expect({ identities, documents, creates, updates }).toEqual({ identities: 1, documents: 1, creates: 1, updates: 1 });
  } finally {
    if (createdId) {
      const cleanup = await removeOwnedOverride(page, marker.api_url, admin, createdId);
      expect(cleanup.deleteStatus).toBe(200);
      expect(cleanup.readStatus).toBe(200);
      expect(cleanup.records.count).toBe(initial.count);
      expect(sortedOverrides(cleanup.records.overrides)).toEqual(sortedOverrides(initial.overrides));
    }
    expect(runHashes(runs)).toEqual(hashesBefore);
  }
  const persisted = await page.evaluate(() =>
    JSON.stringify([Object.entries(localStorage), Object.entries(sessionStorage), document.cookie]));
  expect(persisted.includes(admin)).toBe(false);
  expect(exposed).toBe(false);
  writeFileSync(testInfo.outputPath("f8-integration-evidence.json"), JSON.stringify({
    source_sha: process.env.GITHUB_SHA ?? null, mode: "production", captures: "off",
    authenticated_sessions: identities, main_document_requests: documents,
    hierarchy, selected_forecast: selectedForecast, constraints, decomposition,
    edited_override: edited, initial_override_count: initial.count,
    owned_override_removed: true, preexisting_overrides_preserved: true,
    stored_hashes_before: hashesBefore, stored_hashes_after: runHashes(runs),
    private_values_exposed: false,
  }, null, 2));
});
