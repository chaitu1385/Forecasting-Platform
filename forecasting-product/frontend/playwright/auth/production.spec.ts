import { expect, test, type Page } from "@playwright/test";
import { createHmac } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";

// Evaluate rather than fill: Playwright action diagnostics must never contain a token argument.
async function submitPrivately(page: Page, token: string) {
  try {
    await page.evaluate(value => {
      const input = document.querySelector("#access-token");
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      if (!(input instanceof HTMLInputElement) || input.type !== "password" || !setter) {
        throw new Error("Private credential input is unavailable");
      }
      setter.call(input, value);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }, token);
    await page.getByRole("button", { name: "Sign In", exact: true }).click();
  } catch {
    await page.evaluate(() => document.querySelector("#access-token")?.remove());
    throw new Error("Private credential submission failed");
  }
}

async function protectedStatus(page: Page, apiUrl: string, token: string, method: "GET" | "POST", endpoint: string) {
  try {
    return await page.evaluate(async ({ token, method, url }) => {
      const response = await fetch(url, {
        method,
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        ...(method === "POST" ? { body: JSON.stringify({
          old_sku: "AUTH_SYNTHETIC_OLD", new_sku: "AUTH_SYNTHETIC_NEW", proportion: 0.5,
        }) } : {}),
      });
      return response.status;
    }, { token, method, url: apiUrl + endpoint });
  } catch {
    throw new Error("Private authenticated request failed");
  }
}

test("production operator tokens enforce identity, roles and memory-only lifetime", async ({ page }, testInfo) => {
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
  const retimedToken = (seconds: number) => {
    const [header, encoded] = admin.split(".");
    const claims = JSON.parse(Buffer.from(encoded, "base64url").toString("utf8"));
    claims.iat = Math.floor(Date.now() / 1000) - 100;
    claims.exp = Math.floor(Date.now() / 1000) + seconds;
    const body = `${header}.${Buffer.from(JSON.stringify(claims)).toString("base64url")}`;
    const token = `${body}.${createHmac("sha256", key).update(body).digest("base64url")}`;
    privateValues.push(token);
    return token;
  };
  let exposed = false;
  let identities = 0;
  page.on("console", message => { exposed ||= privateValues.some(value => message.text().includes(value)); });
  page.on("pageerror", error => { exposed ||= privateValues.some(value => error.message.includes(value)); });
  await page.route("**/*", route => {
    const url = route.request().url();
    exposed ||= privateValues.some(value => url.includes(value));
    if (new URL(url).pathname === "/auth/me") identities += 1;
    return [frontend, apiUrl].includes(new URL(url).origin) ? route.continue() : route.abort();
  });

  await page.goto("/login");
  await expect(page.getByLabel("Access token", { exact: true })).toHaveAttribute("type", "password");
  await expect(page.locator("#role")).toHaveCount(0);
  await expect(page.getByText(/Password and SSO sign-in are not configured/)).toBeVisible();
  await submitPrivately(page, retimedToken(-1));
  await expect(page.getByText("The access token is invalid or expired.", { exact: true }).first()).toBeVisible();
  expect(identities).toBe(1);
  expect(await page.locator("#access-token").evaluate(input => (input as HTMLInputElement).value === "")).toBe(true);

  await submitPrivately(page, admin);
  await page.waitForURL(`${frontend}/`);
  await expect(page.locator("header").getByText("browser-admin", { exact: true })).toBeVisible();
  await page.locator("aside").getByRole("link", { name: "6 Forecast Viewer", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Forecast: SYNTHETIC_AUTH_SERIES", exact: true })).toBeVisible();
  expect(await protectedStatus(page, apiUrl, admin, "POST", "/overrides")).toBe(200);
  const persisted = await page.evaluate(() =>
    JSON.stringify([Object.entries(localStorage), Object.entries(sessionStorage), document.cookie]));
  expect(privateValues.some(value => persisted.includes(value))).toBe(false);
  expect(await page.evaluate(() => localStorage.getItem("access_token") === null && localStorage.getItem("user") === null)).toBe(true);

  await page.getByTitle("Logout", { exact: true }).click();
  await page.waitForURL(`${frontend}/login`);
  await expect(page.getByText("SYNTHETIC_AUTH_SERIES", { exact: true })).toHaveCount(0);
  await submitPrivately(page, viewer);
  await page.waitForURL(`${frontend}/`);
  await expect(page.locator("header").getByText("browser-viewer", { exact: true })).toBeVisible();
  expect(await protectedStatus(page, apiUrl, viewer, "GET", "/forecast/retail")).toBe(200);
  expect(await protectedStatus(page, apiUrl, viewer, "POST", "/overrides")).toBe(403);

  await page.reload();
  await page.waitForURL(`${frontend}/login`);
  await expect(page.locator("header").getByText("browser-viewer", { exact: true })).toHaveCount(0);
  await submitPrivately(page, retimedToken(10));
  await page.waitForURL(`${frontend}/`);
  await expect(page.locator("header").getByText("browser-admin", { exact: true })).toBeVisible();
  await page.waitForURL(`${frontend}/login`);
  await expect(page.getByText("Session expired. Sign in again.", { exact: true })).toBeVisible();
  expect(exposed).toBe(false);
});
