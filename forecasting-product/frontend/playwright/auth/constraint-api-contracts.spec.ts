import { expect, test, type Locator, type Page, type Route, type TestInfo } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import type { ConstrainResponse } from "../../src/lib/types";
import { submitPrivately } from "../helpers/private-auth";

const points = "series_id,week,forecast\na,2026-01-05,100\nb,2026-01-05,100\n";
const quantiles = "series_id,week,forecast,forecast_p10,forecast_p50,forecast_p90\na,2026-01-05,10,-5,10,25\n";

function runHashes(root: string): Record<string, string> {
  const hashes: Record<string, string> = {};
  function visit(directory: string) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(file);
      else if (entry.isFile()) hashes[path.relative(root, file)] = createHash("sha256").update(readFileSync(file)).digest("hex");
    }
  }
  visit(root);
  return hashes;
}

async function openPrivateConstraints(page: Page, testInfo: TestInfo) {
  expect(path.basename(testInfo.config.configFile ?? "")).toBe("playwright.auth.config.ts");
  expect(process.env.AUTH_E2E_FRONTEND_MODE).toBe("production");
  for (const option of ["trace", "screenshot", "video"] as const) {
    expect(testInfo.project.use[option]).toBe("off");
  }
  expect(Boolean(process.env.DEBUG || process.env.PWDEBUG)).toBe(false);
  const root = process.env.AUTH_E2E_ROOT;
  if (!root) throw new Error("A fresh private authentication fixture is required");
  const marker: { api_url: string; frontend_origin: string; mode: string } = JSON.parse(
    readFileSync(path.join(root, "data", ".authentication-fixture.json"), "utf8"),
  );
  const frontend = testInfo.project.use.baseURL;
  if (!frontend) throw new Error("The production frontend URL is required");
  expect(marker.frontend_origin).toBe(frontend);
  expect(marker.mode).toBe("production");
  expect(new URL(frontend).hostname).toBe("127.0.0.1");
  expect(new URL(marker.api_url).hostname).toBe("127.0.0.1");
  const admin = readFileSync(path.join(root, "credentials", "admin.jwt"), "utf8").trim();
  let exposed = false;
  let constraintRequests = 0;
  page.on("console", message => { exposed ||= message.text().includes(admin); });
  page.on("pageerror", error => { exposed ||= error.message.includes(admin); });
  page.on("request", request => {
    exposed ||= request.url().includes(admin);
    if (new URL(request.url()).pathname === "/forecast/constrain") constraintRequests += 1;
  });
  await page.route("**/*", route =>
    [frontend, marker.api_url].includes(new URL(route.request().url()).origin) ? route.continue() : route.abort());
  await page.goto("/login");
  await expect(page.getByLabel("Access token", { exact: true })).toHaveAttribute("type", "password");
  await expect(page.locator("#role")).toHaveCount(0);
  await submitPrivately(page, admin);
  await page.waitForURL(frontend + "/");
  await page.locator("aside").getByRole("link", { name: "6 Forecast Viewer", exact: true }).click();
  const panel = page.getByRole("region", { name: "Constrained Forecast", exact: true });
  await expect(panel).toBeVisible();
  return {
    panel,
    runs: path.join(root, "data", "runs"),
    requestCount: () => constraintRequests,
    assertPrivate: async () => {
      const persisted = await page.evaluate(() =>
        JSON.stringify([Object.entries(localStorage), Object.entries(sessionStorage), document.cookie]));
      expect(persisted.includes(admin)).toBe(false);
      expect(exposed).toBe(false);
    },
  };
}

async function upload(panel: Locator, name: string, content: string) {
  await panel.locator("input[type=file]").setInputFiles({ name, mimeType: "text/csv", buffer: Buffer.from(content) });
}

async function apply(page: Page, panel: Locator) {
  const pending = page.waitForResponse(response =>
    new URL(response.url()).pathname === "/forecast/constrain" && response.request().method() === "POST");
  await panel.getByRole("button", { name: "Apply Constraints", exact: true }).click();
  return pending;
}

function metric(output: Locator, label: string) {
  return output.getByText(label, { exact: true }).locator("..").locator("span").first();
}

test("production constraints show both real strategies, infeasible/quantile errors and measured changes", async ({ page }, testInfo) => {
  const { panel, runs, requestCount, assertPrivate } = await openPrivateConstraints(page, testInfo);
  const before = runHashes(runs);
  expect(Object.keys(before).length).toBeGreaterThan(0);
  await expect(panel.getByRole("checkbox", { name: "Proportional", exact: true })).not.toBeChecked();
  await expect(panel.locator("input[type=file]")).toHaveAttribute("accept", ".csv,.parquet");
  await expect(panel.getByText(/Aggregate budgets support point forecasts only/)).toBeVisible();
  await upload(panel, "points.csv", points);
  await panel.getByLabel("Aggregate Max (per period)", { exact: true }).fill("50");
  const largest = await apply(page, panel);
  expect(largest.status()).toBe(200);
  expect(new URL(largest.url()).searchParams.get("proportional")).toBe("false");
  const largestData: ConstrainResponse = await largest.json();
  expect(largestData.constrained_preview.map(row => row.forecast)).toEqual([0, 50]);
  expect(largestData).toMatchObject({
    before_total: 200, after_total: 50, allocation_strategy: "largest_first",
    checked_rows: 2, checked_periods: 1, max_budget_excess: 0, constraint_scope: "point",
  });
  const output = panel.getByRole("region", { name: "Constraint result", exact: true });
  await expect(metric(output, "Before Total")).toHaveText("200");
  await expect(metric(output, "After Total")).toHaveText("50");
  await expect(output.getByText("Largest-first", { exact: true })).toBeVisible();
  await expect(output.getByRole("row").nth(1).getByRole("cell").nth(2)).toHaveText("0");
  await expect(output.getByRole("row").nth(2).getByRole("cell").nth(2)).toHaveText("50");

  await panel.getByRole("checkbox", { name: "Proportional", exact: true }).check();
  await expect(output).toHaveCount(0);
  const proportional = await apply(page, panel);
  expect(proportional.status()).toBe(200);
  expect(new URL(proportional.url()).searchParams.get("proportional")).toBe("true");
  const proportionalData: ConstrainResponse = await proportional.json();
  expect(proportionalData.constrained_preview.map(row => row.forecast)).toEqual([25, 25]);
  expect(proportionalData.after_total).toBe(50);
  await expect(output.getByText("Floor-preserving proportional", { exact: true })).toBeVisible();
  await expect(output.getByRole("cell", { name: "25", exact: true })).toHaveCount(2);

  await panel.getByLabel("Min Demand", { exact: true }).fill("40");
  await expect(output).toHaveCount(0);
  const infeasible = await apply(page, panel);
  expect(infeasible.status()).toBe(400);
  expect((await infeasible.json()).detail).toContain("infeasible_constraints");
  await expect(panel.getByRole("alert")).toContainText("infeasible_constraints");
  await expect(output).toHaveCount(0);

  await panel.getByLabel("Min Demand", { exact: true }).fill("0");
  await upload(panel, "quantiles.csv", quantiles);
  await expect(panel.getByRole("alert")).toHaveCount(0);
  const unsupported = await apply(page, panel);
  expect(unsupported.status()).toBe(400);
  expect((await unsupported.json()).detail).toContain("unsupported_aggregate_quantiles");
  await expect(panel.getByRole("alert")).toContainText("unsupported_aggregate_quantiles");
  await expect(output).toHaveCount(0);

  await panel.getByLabel("Aggregate Max (per period)", { exact: true }).fill("");
  await panel.getByLabel("Max Capacity", { exact: true }).fill("20");
  await expect(panel.getByRole("alert")).toHaveCount(0);
  const bounded = await apply(page, panel);
  expect(bounded.status()).toBe(200);
  const boundedData: ConstrainResponse = await bounded.json();
  expect(boundedData).toMatchObject({
    before_total: 10, after_total: 10, rows_modified: 1, constraint_scope: "point_and_quantiles",
    allocation_strategy: "element_wise", max_budget_excess: null,
  });
  expect(boundedData.constrained_preview[0]).toMatchObject({
    forecast: 10, forecast_p10: 0, forecast_p50: 10, forecast_p90: 20,
  });
  await expect(metric(output, "Rows Modified")).toHaveText("1");
  await expect(metric(output, "Maximum Budget Excess")).toHaveText("No budget supplied");
  await expect(output.getByText(/Point and marginal quantile bounds/)).toBeVisible();
  expect(requestCount()).toBe(5);
  expect(runHashes(runs)).toEqual(before);
  await assertPrivate();
});

test("a delayed real constraint response cannot replace the newer production result", async ({ page }, testInfo) => {
  const { panel, runs, requestCount, assertPrivate } = await openPrivateConstraints(page, testInfo);
  const before = runHashes(runs);
  await upload(panel, "points.csv", points);
  await panel.getByLabel("Aggregate Max (per period)", { exact: true }).fill("50");
  let release!: () => void;
  let captured!: (value: { status: number; payload: ConstrainResponse } | null) => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  const actualResponse = new Promise<{ status: number; payload: ConstrainResponse } | null>(resolve => { captured = resolve; });
  const holdResponse = async (route: Route) => {
    try {
      const actual = await route.fetch({ maxRetries: 0 });
      captured({ status: actual.status(), payload: await actual.json() });
      await held;
      await route.fulfill({ response: actual });
    } catch {
      captured(null);
      await route.abort().catch(() => undefined);
    }
  };
  await page.route("**/forecast/constrain?**", holdResponse);
  const old = page.waitForResponse(response =>
    new URL(response.url()).pathname === "/forecast/constrain" &&
    new URL(response.url()).searchParams.get("aggregate_max") === "50");
  try {
    await panel.getByRole("button", { name: "Apply Constraints", exact: true }).click();
    const observed = await actualResponse;
    if (observed === null) throw new Error("Private delayed request did not complete");
    expect(observed.status).toBe(200);
    expect(observed.payload.constrained_preview.map(row => row.forecast)).toEqual([0, 50]);
    await page.unroute("**/forecast/constrain?**", holdResponse);
    await panel.getByLabel("Aggregate Max (per period)", { exact: true }).fill("60");
    await expect(panel.getByRole("button", { name: "Apply Constraints", exact: true })).toBeEnabled();
    const current = await apply(page, panel);
    expect(current.status()).toBe(200);
    const data: ConstrainResponse = await current.json();
    expect(data.constrained_preview.map(row => row.forecast)).toEqual([0, 60]);
    const output = panel.getByRole("region", { name: "Constraint result", exact: true });
    await expect(metric(output, "After Total")).toHaveText("60");
    release();
    await (await old).finished();
    await page.evaluate(() => new Promise<void>(resolve => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    }));
    await expect(metric(output, "After Total")).toHaveText("60");
    await expect(output.getByRole("row").nth(2).getByRole("cell").nth(2)).toHaveText("60");
    await expect(panel.getByRole("alert")).toHaveCount(0);
    expect(requestCount()).toBe(2);
    expect(runHashes(runs)).toEqual(before);
    await assertPrivate();
  } finally {
    release();
    await page.unroute("**/forecast/constrain?**", holdResponse);
  }
});
