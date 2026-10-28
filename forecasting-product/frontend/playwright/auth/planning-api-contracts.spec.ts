import { expect, test, type Page, type Response, type Route } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import type { HierarchyReconcileResponse, HierarchyReconciliationMethod } from "../../src/lib/types";
import { submitPrivately } from "../fixtures/authentication";

const leafCsv = "department,item,week,forecast\nfood,a,2026-01-05,10\nfood,b,2026-01-05,20\nfood,a,2026-01-12,4\nfood,b,2026-01-12,6\n";
const rootCsv = "department,week,forecast\nfood,2026-01-05,120\n";
const historyCsv = "department,item,week,sales\nfood,a,2025-12-01,10\nfood,b,2025-12-01,30\n";
const upload = (name: string, content: string) => ({ name, mimeType: "text/csv", buffer: Buffer.from(content) });

function runHashes(directory: string): Record<string, string> {
  const hashes: Record<string, string> = {};
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      for (const [file, hash] of Object.entries(runHashes(target))) {
        hashes[path.join(entry.name, file)] = hash;
      }
    } else if (entry.isFile()) {
      hashes[entry.name] = createHash("sha256").update(readFileSync(target)).digest("hex");
    }
  }
  return hashes;
}

async function openHierarchy(page: Page) {
  await page.locator("aside").getByRole("link", { name: /Hierarchy/ }).click();
  await expect(page.getByRole("heading", { name: "Hierarchy Manager", exact: true })).toBeVisible();
}

async function reconcile(page: Page, method: HierarchyReconciliationMethod): Promise<Response> {
  const response = page.waitForResponse(response => {
    const url = new URL(response.url());
    return url.pathname === "/hierarchy/reconcile"
      && url.searchParams.get("method") === method && response.request().method() === "POST";
  });
  await page.getByRole("button", { name: "Reconcile", exact: true }).click();
  return response;
}

test("TEST-005/022 real hierarchy numbers, history admission, late responses and production roles", async ({ page }, testInfo) => {
  expect(path.basename(testInfo.config.configFile ?? "")).toBe("playwright.auth.config.ts");
  for (const option of ["trace", "screenshot", "video"] as const) {
    expect(testInfo.project.use[option]).toBe("off");
  }
  expect(Boolean(process.env.DEBUG || process.env.PWDEBUG)).toBe(false);
  const root = process.env.AUTH_E2E_ROOT;
  if (!root) throw new Error("The isolated authenticated fixture is required");
  const marker = JSON.parse(readFileSync(path.join(root, "data", ".authentication-fixture.json"), "utf8"));
  const frontend = testInfo.project.use.baseURL!;
  const apiUrl: string = marker.api_url;
  expect(marker.frontend_origin).toBe(frontend);
  expect(marker.mode).toBe("production");
  expect(new URL(apiUrl).hostname).toBe("127.0.0.1");
  const runs = path.join(root, "data", "runs");
  const hashes = runHashes(runs);
  expect(Object.keys(hashes).length).toBeGreaterThan(0);
  const admin = readFileSync(path.join(root, "credentials", "admin.jwt"), "utf8").trim();
  const viewer = readFileSync(path.join(root, "credentials", "viewer.jwt"), "utf8").trim();
  const privateValues = [admin, viewer];
  let exposed = false;
  page.on("console", message => { exposed ||= privateValues.some(value => message.text().includes(value)); });
  page.on("pageerror", error => { exposed ||= privateValues.some(value => error.message.includes(value)); });
  await page.route("**/*", route => {
    const url = route.request().url();
    exposed ||= privateValues.some(value => url.includes(value));
    return [frontend, apiUrl].includes(new URL(url).origin) ? route.continue() : route.abort();
  });

  await page.goto("/login");
  await expect(page.getByLabel("Access token", { exact: true })).toHaveAttribute("type", "password");
  await submitPrivately(page, admin);
  await page.waitForURL(`${frontend}/`);
  await openHierarchy(page);
  await page.getByLabel("Upload CSV or Parquet").setInputFiles(upload("leaves.csv", leafCsv));
  await page.getByLabel("Hierarchy Levels (comma-separated)").fill("department,item");
  const bottomResponse = await reconcile(page, "bottom_up");
  expect(bottomResponse.status()).toBe(200);
  const bottom: HierarchyReconcileResponse = await bottomResponse.json();
  expect(bottom.reconciled_preview).toEqual([
    { item: "a", week: "2026-01-05", forecast: 10 }, { item: "b", week: "2026-01-05", forecast: 20 },
    { item: "a", week: "2026-01-12", forecast: 4 }, { item: "b", week: "2026-01-12", forecast: 6 },
  ]);
  const totals = ["2026-01-05", "2026-01-12"].map(week =>
    bottom.reconciled_preview.filter(row => row.week === week).reduce((sum, row) => sum + Number(row.forecast), 0));
  expect(totals).toEqual([30, 10]);
  const preview = page.getByTestId("reconciliation-result");
  const panel = page.locator("section").filter({
    has: page.getByRole("heading", { name: "Reconciliation Execution", exact: true }),
  });
  await expect(preview.locator("tbody tr")).toHaveCount(4);
  for (const value of ["10", "20", "4", "6"]) {
    await expect(preview.getByRole("cell", { name: value, exact: true })).toBeVisible();
  }
  expect(bottom.coherence.checked_rows).toBe(4);

  await page.getByLabel("Method", { exact: true }).selectOption("top_down");
  await expect(preview).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Reconcile", exact: true })).toBeDisabled();
  await page.getByLabel("Upload CSV or Parquet").setInputFiles(upload("roots.csv", rootCsv));
  await page.getByLabel("Historical actuals").setInputFiles(upload("history.csv", historyCsv));
  await page.getByLabel("History value column").fill("sales");
  const topResponse = await reconcile(page, "top_down");
  expect(topResponse.status()).toBe(200);
  const top: HierarchyReconcileResponse = await topResponse.json();
  expect(Object.fromEntries(top.reconciled_preview.map(row => [row.item, row.forecast]))).toEqual({ a: 30, b: 90 });
  expect(top.coherence.scope).toBe("root_forecast_conservation");
  expect(top.coherence.max_abs_error).toBe(0);
  expect(top.before_total).toBe(120);
  expect(top.after_total).toBe(120);
  await expect(preview.getByRole("cell", { name: "30", exact: true })).toBeVisible();
  await expect(preview.getByRole("cell", { name: "90", exact: true })).toBeVisible();
  await expect(page.getByTestId("reconciliation-effective-method")).toHaveText("Top-Down");

  await page.getByLabel("Historical actuals").setInputFiles(upload("zero-history.csv", historyCsv.replace(",10\n", ",0\n")));
  await expect(preview).toHaveCount(0);
  expect((await reconcile(page, "top_down")).status()).toBe(400);
  await expect(panel.getByRole("alert")).toContainText("positive total volume");

  await page.getByLabel("Method", { exact: true }).selectOption("bottom_up");
  await page.getByLabel("Upload CSV or Parquet").setInputFiles(upload("leaves.csv", leafCsv));
  let release!: () => void;
  let received!: () => void;
  const released = new Promise<void>(resolve => { release = resolve; });
  const fetched = new Promise<void>(resolve => { received = resolve; });
  let hold = true;
  const delayed = async (route: Route) => {
    if (!hold) return route.continue();
    hold = false;
    // Delay delivery of one actual API response without replacing its contents.
    const response = await route.fetch();
    received();
    await released;
    await route.fulfill({ response });
  };
  await page.route("**/hierarchy/reconcile?**", delayed);
  try {
    const previous = reconcile(page, "bottom_up");
    await fetched;
    await page.getByLabel("Upload CSV or Parquet").setInputFiles(upload(
      "replacement.csv", "department,item,week,forecast\nfood,a,2026-01-05,7\nfood,b,2026-01-05,11\n",
    ));
    await page.getByLabel("Method", { exact: true }).selectOption("wls");
    const currentResponse = await reconcile(page, "wls");
    expect(currentResponse.status()).toBe(200);
    const current: HierarchyReconcileResponse = await currentResponse.json();
    expect(current.effective_method).toBe("wls_structural");
    expect(current.after_total).toBe(18);
    for (const row of current.reconciled_preview) {
      expect(Number(row.forecast)).toBeCloseTo(row.item === "a" ? 7 : 11, 6);
    }
    release();
    const late = await previous;
    expect(late.status()).toBe(200);
    expect((await late.json()).after_total).toBe(40);
    await late.finished();
    await page.evaluate(() => new Promise<void>(resolve =>
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    await expect(page.getByTestId("reconciliation-effective-method")).toHaveText("Structural WLS");
    await expect(preview).toContainText("Input rows: 2 | Output rows: 2");
    await expect(preview.getByRole("cell", { name: "20", exact: true })).toHaveCount(0);
  } finally {
    release();
    await page.unroute("**/hierarchy/reconcile?**", delayed);
  }

  await page.getByTitle("Logout", { exact: true }).click();
  await page.waitForURL(`${frontend}/login`);
  await submitPrivately(page, viewer);
  await page.waitForURL(`${frontend}/`);
  await openHierarchy(page);
  await expect(preview).toHaveCount(0);
  await expect(page.getByLabel("Hierarchy Levels (comma-separated)")).toHaveValue("");
  await page.getByLabel("Upload CSV or Parquet").setInputFiles(upload("leaves.csv", leafCsv));
  await page.getByLabel("Hierarchy Levels (comma-separated)").fill("department,item");
  expect((await reconcile(page, "bottom_up")).status()).toBe(403);
  await expect(panel.getByRole("alert")).toBeVisible();
  expect(runHashes(runs)).toEqual(hashes);
  const persisted = await page.evaluate(() =>
    JSON.stringify([Object.entries(localStorage), Object.entries(sessionStorage), document.cookie]));
  expect(privateValues.some(value => persisted.includes(value))).toBe(false);
  expect(exposed).toBe(false);
  await page.getByTitle("Logout", { exact: true }).click();
});
