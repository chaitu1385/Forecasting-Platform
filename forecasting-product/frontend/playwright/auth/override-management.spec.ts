import { expect, test, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { submitPrivately } from "../fixtures/authentication";
import { navigateWithinApp } from "../helpers/auth";
import type { OverrideItem, OverrideListResponse } from "../../src/lib/types";

function overrideResponse(page: Page, method: string, id?: string) {
  return page.waitForResponse(response =>
    new URL(response.url()).pathname === (id ? `/overrides/${id}` : "/overrides")
    && response.request().method() === method);
}

function artifactHashes(root: string): Record<string, string> {
  const hashes: Record<string, string> = {};
  function visit(directory: string) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(file);
      else if (entry.isFile()) {
        hashes[path.relative(root, file)] = createHash("sha256").update(readFileSync(file)).digest("hex");
      }
    }
  }
  visit(root);
  return hashes;
}

async function deniedMutationStatus(page: Page, apiUrl: string, token: string | null, id: string) {
  try {
    return await page.evaluate(async ({ url, credential }) => {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (credential) headers.Authorization = "Bearer " + credential;
      const response = await fetch(url, {
        method: "PUT", headers, body: JSON.stringify({ notes: "forbidden-browser-edit" }),
      });
      return response.status;
    }, { url: `${apiUrl}/overrides/${id}`, credential: token });
  } catch {
    throw new Error("Private override permission probe failed");
  }
}

test("production override edit preserves identity through reload, storage failure and denied writes", async ({ page }, testInfo) => {
  expect(path.basename(testInfo.config.configFile ?? "")).toBe("playwright.auth.config.ts");
  for (const option of ["trace", "screenshot", "video"] as const) {
    expect(testInfo.project.use[option]).toBe("off");
  }
  expect(Boolean(process.env.DEBUG || process.env.PWDEBUG)).toBe(false);
  const root = process.env.AUTH_E2E_ROOT;
  if (!root) throw new Error("The isolated authentication fixture is required");
  const marker = JSON.parse(readFileSync(path.join(root, "data", ".authentication-fixture.json"), "utf8"));
  const frontend = testInfo.project.use.baseURL!;
  const apiUrl: string = marker.api_url;
  expect(marker.frontend_origin).toBe(frontend);
  expect(marker.mode).toBe("production");
  expect(new URL(apiUrl).hostname).toBe("127.0.0.1");
  expect(new URL(frontend).hostname).toBe("127.0.0.1");
  const admin = readFileSync(path.join(root, "credentials", "admin.jwt"), "utf8").trim();
  const viewer = readFileSync(path.join(root, "credentials", "viewer.jwt"), "utf8").trim();
  const key = readFileSync(path.join(root, "credentials", "signing-key"), "utf8").trim();
  const privateValues = [admin, viewer, key];
  const runs = path.join(root, "data", "runs");
  const beforeHashes = artifactHashes(runs);
  expect(Object.keys(beforeHashes).length).toBeGreaterThan(0);
  let exposed = false;
  let creates = 0;
  let updates = 0;
  page.on("console", message => { exposed ||= privateValues.some(value => message.text().includes(value)); });
  page.on("pageerror", error => { exposed ||= privateValues.some(value => error.message.includes(value)); });
  page.on("request", request => {
    const url = new URL(request.url());
    if (url.pathname === "/overrides" && request.method() === "POST") creates += 1;
    if (url.pathname.startsWith("/overrides/") && request.method() === "PUT") updates += 1;
  });
  await page.route("**/*", route => {
    const url = route.request().url();
    exposed ||= privateValues.some(value => url.includes(value));
    return [frontend, apiUrl].includes(new URL(url).origin) ? route.continue() : route.abort();
  });

  await page.goto("/login");
  await expect(page.getByLabel("Access token", { exact: true })).toHaveAttribute("type", "password");
  await submitPrivately(page, admin);
  await page.waitForURL(`${frontend}/`);
  const initialRead = overrideResponse(page, "GET");
  await navigateWithinApp(page, "/sku-transitions");
  const initial: OverrideListResponse = await (await initialRead).json();
  const panel = page.getByRole("region", { name: "Planner Override Management" });
  await expect(panel.getByText(`${initial.count} override(s)`, { exact: true })).toBeVisible();
  await panel.getByRole("button", { name: "Add Override" }).click();
  await panel.getByLabel("Old SKU *", { exact: true }).fill("OVERRIDE_BROWSER_OLD");
  await panel.getByLabel("New SKU *", { exact: true }).fill("OVERRIDE_BROWSER_NEW");
  await panel.getByLabel("Proportion", { exact: true }).fill("0.75");
  await panel.getByLabel("Scenario", { exact: true }).fill("browser-custom");
  await panel.getByLabel("Effective Date", { exact: true }).fill("2026-01-02");
  await panel.getByLabel("Notes", { exact: true }).fill("browser-before");
  const creation = overrideResponse(page, "POST");
  const createdRead = overrideResponse(page, "GET");
  await panel.getByRole("button", { name: "Create Override", exact: true }).click();
  const created = await creation;
  expect(created.status()).toBe(200);
  const id: string = (await created.json()).override_id;
  expect(id).toMatch(/^OVR-[A-F0-9]{8}$/);
  const createdList: OverrideListResponse = await (await createdRead).json();
  const before: OverrideItem | undefined = createdList.overrides.find(row => row.override_id === id);
  expect(before).toBeDefined();
  const row = panel.locator(`[data-override-id="${id}"]`);
  await expect(row.getByText("browser-before", { exact: true })).toBeVisible();
  await expect(panel.getByText(`${initial.count + 1} override(s)`, { exact: true })).toBeVisible();

  await row.getByRole("button", { name: "Edit", exact: true }).click();
  await expect(panel.getByLabel("Old SKU *", { exact: true })).toHaveAttribute("readonly", "");
  await expect(panel.getByLabel("New SKU *", { exact: true })).toHaveAttribute("readonly", "");
  await expect(panel.getByText(id, { exact: true })).toBeVisible();
  await panel.getByLabel("Proportion", { exact: true }).fill("0");
  await panel.getByLabel("Effective Date", { exact: true }).fill("");
  await panel.getByLabel("Notes", { exact: true }).fill("");
  const updated = overrideResponse(page, "PUT", id);
  const updatedRead = overrideResponse(page, "GET");
  await panel.getByRole("button", { name: "Save", exact: true }).click();
  const update = await updated;
  expect(update.status()).toBe(200);
  expect(update.request().postDataJSON()).toEqual({ proportion: 0, effective_date: null, notes: null });
  expect(await update.json()).toEqual({ override_id: id, status: "updated" });
  const expected = { ...before, proportion: 0, effective_date: null, notes: null };
  const updatedList: OverrideListResponse = await (await updatedRead).json();
  expect(updatedList.count).toBe(initial.count + 1);
  expect(updatedList.overrides.find(item => item.override_id === id)).toEqual(expected);
  await expect(panel.getByRole("heading", { name: "Edit Override", exact: true })).toHaveCount(0);
  await expect(row.getByRole("cell").nth(2)).toHaveText("0");
  expect(creates).toBe(1);

  await page.reload();
  await page.waitForURL(`${frontend}/login`);
  await expect(page.locator(`[data-override-id="${id}"]`)).toHaveCount(0);
  await submitPrivately(page, admin);
  await page.waitForURL(`${frontend}/`);
  const reloadedRead = overrideResponse(page, "GET");
  await navigateWithinApp(page, "/sku-transitions");
  const reloaded: OverrideListResponse = await (await reloadedRead).json();
  expect(reloaded.overrides.find(item => item.override_id === id)).toEqual(expected);
  expect(reloaded.count).toBe(initial.count + 1);

  await row.getByRole("button", { name: "Edit", exact: true }).click();
  await panel.getByLabel("Notes", { exact: true }).fill("OVERRIDE_FIXTURE_REJECTED_NOTES");
  const failed = overrideResponse(page, "PUT", id);
  await panel.getByRole("button", { name: "Save", exact: true }).click();
  expect((await failed).status()).toBe(500);
  await expect(panel.getByRole("alert")).toContainText("Failed to update overrides");
  await expect(panel.getByLabel("Notes", { exact: true })).toHaveValue("OVERRIDE_FIXTURE_REJECTED_NOTES");
  await expect(row.getByRole("cell").nth(6)).toHaveText("");
  expect(updates).toBe(2);
  expect(creates).toBe(1);
  const failedRead = overrideResponse(page, "GET");
  await panel.getByRole("button", { name: "Refresh", exact: true }).click();
  const unchanged: OverrideListResponse = await (await failedRead).json();
  expect(unchanged.overrides.find(item => item.override_id === id)).toEqual(expected);
  expect(unchanged.count).toBe(initial.count + 1);
  await expect(panel.getByLabel("Notes", { exact: true })).toHaveValue("OVERRIDE_FIXTURE_REJECTED_NOTES");

  await panel.getByLabel("Notes", { exact: true }).fill("browser-after-retry");
  const retryRead = overrideResponse(page, "GET");
  await panel.getByRole("button", { name: "Save", exact: true }).click();
  const retried: OverrideListResponse = await (await retryRead).json();
  const finalRow = { ...expected, notes: "browser-after-retry" };
  expect(retried.overrides.find(item => item.override_id === id)).toEqual(finalRow);
  expect(retried.count).toBe(initial.count + 1);
  expect(updates).toBe(3);
  expect(creates).toBe(1);

  await page.getByTitle("Logout", { exact: true }).click();
  await page.waitForURL(`${frontend}/login`);
  await submitPrivately(page, viewer);
  await page.waitForURL(`${frontend}/`);
  await navigateWithinApp(page, "/sku-transitions");
  await expect(row.getByText("browser-after-retry", { exact: true })).toBeVisible();
  await expect(panel.getByRole("button", { name: "Add Override" })).toHaveCount(0);
  await expect(row.getByRole("button", { name: "Edit" })).toHaveCount(0);
  await expect(row.getByRole("button", { name: "Delete" })).toHaveCount(0);
  expect(await deniedMutationStatus(page, apiUrl, viewer, id)).toBe(403);
  expect(await deniedMutationStatus(page, apiUrl, null, id)).toBe(401);
  const deniedRead = overrideResponse(page, "GET");
  await panel.getByRole("button", { name: "Refresh", exact: true }).click();
  const denied: OverrideListResponse = await (await deniedRead).json();
  expect(denied.overrides.find(item => item.override_id === id)).toEqual(finalRow);
  expect(denied.count).toBe(initial.count + 1);
  const persisted = await page.evaluate(() =>
    JSON.stringify([Object.entries(localStorage), Object.entries(sessionStorage), document.cookie]));
  expect(privateValues.some(value => persisted.includes(value))).toBe(false);
  expect(exposed).toBe(false);
  const afterHashes = artifactHashes(runs);
  expect(afterHashes).toEqual(beforeHashes);
  writeFileSync(path.join(root, "override-contract-evidence.json"), JSON.stringify({
    source: process.env.GITHUB_SHA ?? "local",
    override_id: id, initial_count: initial.count, final_count: denied.count,
    identity_and_metadata_preserved: true, failed_save_status: 500,
    viewer_put_status: 403, anonymous_put_status: 401,
    run_artifacts_before: beforeHashes, run_artifacts_after: afterHashes,
  }, null, 2));
});
