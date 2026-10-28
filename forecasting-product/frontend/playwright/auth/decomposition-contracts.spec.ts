import { expect, test, type Page, type TestInfo } from "@playwright/test";
import { createHash, createHmac } from "node:crypto";
import { readFileSync, readdirSync, renameSync, writeFileSync } from "node:fs";
import path from "node:path";
import { submitPrivately } from "../fixtures/authentication";
import type { DecompositionSummaryResponse } from "../../src/lib/types";

interface FixtureMarker {
  api_url: string;
  frontend_origin: string;
  mode: string;
  forecast_runs: { run_id: string; value: number }[];
}

function fixture(testInfo: TestInfo) {
  expect(path.basename(testInfo.config.configFile ?? "")).toBe("playwright.auth.config.ts");
  for (const option of ["trace", "screenshot", "video"] as const) {
    expect(testInfo.project.use[option]).toBe("off");
  }
  expect(Boolean(process.env.DEBUG || process.env.PWDEBUG)).toBe(false);
  expect(process.env.AUTH_E2E_FRONTEND_MODE).toBe("production");
  const root = process.env.AUTH_E2E_ROOT;
  if (!root || !path.isAbsolute(root)) throw new Error("A private absolute authentication fixture root is required");
  const marker: FixtureMarker = JSON.parse(readFileSync(path.join(root, "data", ".authentication-fixture.json"), "utf8"));
  expect(marker.frontend_origin).toBe(testInfo.project.use.baseURL);
  expect(marker.mode).toBe("production");
  expect(new URL(marker.api_url).hostname).toBe("127.0.0.1");
  expect(new URL(marker.frontend_origin).hostname).toBe("127.0.0.1");
  expect(marker.forecast_runs).toHaveLength(2);
  for (const run of marker.forecast_runs) expect(run.run_id).toMatch(/^[0-9a-f]{32}$/);
  return { root, marker };
}

function storedHashes(directory: string): Record<string, string> {
  const hashes: Record<string, string> = {};
  for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    const child = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      for (const [name, hash] of Object.entries(storedHashes(child))) hashes[path.join(entry.name, name)] = hash;
    } else if (entry.isFile()) {
      hashes[entry.name] = createHash("sha256").update(readFileSync(child)).digest("hex");
    } else {
      throw new Error("Unexpected nonregular fixture member");
    }
  }
  return hashes;
}

async function openPrivately(page: Page, marker: FixtureMarker, viewer: string, privateValues: string[]) {
  let exposed = false;
  page.on("console", message => { exposed ||= privateValues.some(value => message.text().includes(value)); });
  page.on("pageerror", error => { exposed ||= privateValues.some(value => error.message.includes(value)); });
  await page.route("**/*", route => {
    const url = new URL(route.request().url());
    exposed ||= privateValues.some(value => url.href.includes(value));
    return [marker.frontend_origin, marker.api_url].includes(url.origin) ? route.continue() : route.abort();
  });
  await page.goto("/login");
  await expect(page.getByLabel("Access token", { exact: true })).toHaveAttribute("type", "password");
  await submitPrivately(page, viewer);
  await page.waitForURL(`${marker.frontend_origin}/`);
  await page.locator('aside a[href="/backtest"]').click();
  await expect(page.getByRole("heading", { name: "Backtest Results", exact: true })).toBeVisible();
  await page.getByLabel("Season length (observations)").fill("3");
  return () => expect(exposed).toBe(false);
}

async function loadDecomposition(page: Page, expectedStatus = 200) {
  const response = page.waitForResponse(item =>
    new URL(item.url()).pathname === "/metrics/retail/decomposition" && item.request().method() === "POST");
  await page.getByRole("button", { name: "Load decomposition", exact: true }).click();
  const actual = await response;
  expect(actual.status()).toBe(expectedStatus);
  return actual;
}

test("production decomposition pins actual forecast provenance and preserves every stored member", async ({ page }, testInfo) => {
  const { root, marker } = fixture(testInfo);
  const viewer = readFileSync(path.join(root, "credentials", "viewer.jwt"), "utf8").trim();
  const privateValues = [viewer];
  const safe = await openPrivately(page, marker, viewer, privateValues);
  const runsRoot = path.join(root, "data", "runs");
  const before = storedHashes(runsRoot);
  const [older, latest] = marker.forecast_runs;
  const result: DecompositionSummaryResponse = await (await loadDecomposition(page)).json();
  expect(result.run_id).toBe(latest.run_id);
  expect(result.history_run_id).toBe(latest.run_id);
  expect(result.artifact_source).toBe("committed");
  expect(result.forecast_model).toBe("naive_seasonal");
  expect(result.forecast_model_source).toBe("manifest");
  expect(result.method).toBe("moving_average_decomposition");
  expect(result.model_specific).toBe(false);
  expect(result.history_source).toBe("committed");
  expect(result.history_stage).toBe("prepared");
  expect(result.checked_rows).toBe(68);
  expect(result.preview_rows).toBe(50);
  expect(result.preview_truncated).toBe(true);
  expect(result.components).toEqual([
    { component: "trend", mean_abs_value: 20, std: 0, defined_count: 62, undefined_count: 2 },
    { component: "seasonal", mean_abs_value: 0, std: 0, defined_count: 64, undefined_count: 0 },
    { component: "residual", mean_abs_value: 0, std: 0, defined_count: 62, undefined_count: 2 },
  ]);
  const section = page.getByRole("region", { name: "Forecast decomposition" });
  await expect(page.getByTestId("decomposition-run-id")).toHaveText(latest.run_id);
  await expect(page.getByLabel("Decomposition forecast source")).toHaveValue(latest.run_id);
  await expect(section).toContainText("separate forecast source, not the leaderboard run");
  await expect(section).toContainText("Not model-specific feature attribution");
  await expect(section).toContainText("recorded prepared snapshot from this same forecast run");
  await expect(section).toContainText("naive_seasonal");
  await expect(section).toContainText("moving_average_decomposition");
  await expect(section.getByRole("heading", { name: "Component magnitude" })).toBeVisible();
  await expect(section).not.toContainText(/SHAP|waterfall|feature importance|LOESS|causal/i);

  await page.getByLabel("Decomposition forecast source").selectOption(older.run_id);
  await expect(page.getByTestId("decomposition-result")).toHaveCount(0);
  const previous: DecompositionSummaryResponse = await (await loadDecomposition(page)).json();
  expect(previous.run_id).toBe(older.run_id);
  expect(previous.history_run_id).toBe(older.run_id);
  expect(previous.components[0].mean_abs_value).toBe(10);
  await expect(page.getByTestId("decomposition-run-id")).toHaveText(older.run_id);
  await expect(page.getByTestId("component-trend")).toContainText("10.0000");
  await page.getByLabel("Season length (observations)").fill("7");
  await expect(page.getByTestId("decomposition-result")).toHaveCount(0);
  const unsupported = await loadDecomposition(page, 400);
  expect((await unsupported.json()).detail.code).toBe("unsupported_decomposition_window");
  await expect(section).toContainText("unsupported_decomposition_window");
  await expect(page.getByTestId("decomposition-result")).toHaveCount(0);
  expect(storedHashes(runsRoot)).toEqual(before);
  safe();
  writeFileSync(testInfo.outputPath("decomposition-evidence.json"), JSON.stringify({
    source_sha: process.env.GITHUB_SHA ?? null, mode: "production", captures: "off",
    latest: result, older: previous, unsupported_window_status: 400, stored_hashes_before: before,
    stored_hashes_after: storedHashes(runsRoot), private_values_exposed: false,
  }, null, 2));
});

test("production decomposition denies inactive reads and reports damaged exact runs without stale success", async ({ page }, testInfo) => {
  const { root, marker } = fixture(testInfo);
  const viewer = readFileSync(path.join(root, "credentials", "viewer.jwt"), "utf8").trim();
  const key = readFileSync(path.join(root, "credentials", "signing-key"), "utf8").trim();
  const [header, encoded] = viewer.split(".");
  const claims = JSON.parse(Buffer.from(encoded, "base64url").toString("utf8"));
  claims.is_active = false;
  const signed = `${header}.${Buffer.from(JSON.stringify(claims)).toString("base64url")}`;
  const inactive = `${signed}.${createHmac("sha256", key).update(signed).digest("base64url")}`;
  const safe = await openPrivately(page, marker, viewer, [viewer, key, inactive]);
  const runsRoot = path.join(root, "data", "runs");
  const before = storedHashes(runsRoot);
  const older = marker.forecast_runs[0].run_id;
  await page.getByLabel("Decomposition forecast source").selectOption(older);
  await loadDecomposition(page);
  await expect(page.getByTestId("decomposition-run-id")).toHaveText(older);
  let requests = 0;
  page.on("request", request => {
    if (new URL(request.url()).pathname === "/metrics/retail/decomposition") requests += 1;
  });
  const endpoint = `${marker.api_url}/metrics/retail/decomposition**`;
  // Forward a signed inactive identity to the real JWT boundary, never mock the response.
  await page.route(endpoint, async route => {
    try {
      await route.continue({ headers: { ...route.request().headers(), authorization: `Bearer ${inactive}` } });
    } catch {
      throw new Error("Private denial forwarding failed");
    }
  });
  const denied = await loadDecomposition(page, 403);
  expect((await denied.json()).detail).toBe("User account is deactivated.");
  await expect(page.getByTestId("decomposition-result")).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Forecast decomposition" })).toContainText("deactivated");
  expect(requests).toBe(1);
  expect(storedHashes(runsRoot)).toEqual(before);
  await page.unroute(endpoint);

  const member = path.join(runsRoot, "retail", older, "prepared.parquet");
  const backup = `${member}.owned-test-backup`;
  renameSync(member, backup);
  try {
    const failure = await loadDecomposition(page, 500);
    expect(new URL(failure.url()).searchParams.get("run_id")).toBe(older);
    expect((await failure.json()).detail.code).toBe("run_integrity_error");
    await expect(page.getByTestId("decomposition-result")).toHaveCount(0);
    await expect(page.getByRole("region", { name: "Forecast decomposition" })).toContainText("run_integrity_error");
    expect(requests).toBe(2);
  } finally {
    renameSync(backup, member);
  }
  expect(storedHashes(runsRoot)).toEqual(before);
  await loadDecomposition(page);
  await expect(page.getByTestId("decomposition-run-id")).toHaveText(older);
  await expect(page.getByTestId("component-trend")).toContainText("10.0000");
  await page.getByTitle("Logout", { exact: true }).click();
  await page.waitForURL(`${marker.frontend_origin}/login`);
  await expect(page.getByTestId("decomposition-result")).toHaveCount(0);
  await submitPrivately(page, viewer);
  await page.waitForURL(`${marker.frontend_origin}/`);
  await page.locator('aside a[href="/backtest"]').click();
  await expect(page.getByRole("heading", { name: "Backtest Results", exact: true })).toBeVisible();
  await expect(page.getByTestId("decomposition-result")).toHaveCount(0);
  await expect(page.getByLabel("Decomposition forecast source")).toHaveValue("");
  expect(storedHashes(runsRoot)).toEqual(before);
  safe();
  writeFileSync(testInfo.outputPath("decomposition-evidence.json"), JSON.stringify({
    source_sha: process.env.GITHUB_SHA ?? null, mode: "production", captures: "off",
    exact_run: older, inactive_jwt_status: 403, missing_member_status: 500,
    original_member_restored: true, no_read_retries: true, same_user_login_cleared_result: true,
    stored_hashes_before: before, stored_hashes_after: storedHashes(runsRoot),
    private_values_exposed: false,
  }, null, 2));
});
