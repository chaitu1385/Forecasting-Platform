import { test, expect } from "@playwright/test";
import { mockApiRoutes } from "../helpers/mock-routes";
import { clearAuth, loginViaUI } from "../helpers/auth";

test.describe("Explicit demo login", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page);
  });

  for (const role of ["admin", "data_scientist", "planner"] as const) {
    test(`verified demo ${role} redirects to home`, async ({ page }) => {
      await loginViaUI(page, `${role}_user`, role);
      await expect(page.locator("main h1")).toContainText("Sales Forecasting Product");
    });
  }

  test("empty username shows an error", async ({ page }) => {
    await page.goto("/login");
    await page.getByRole("button", { name: "Sign In to Demo", exact: true }).click();
    await expect(page.getByText("Username is required", { exact: true })).toBeVisible();
    expect(new URL(page.url()).pathname).toBe("/login");
  });

  test("button shows loading while the API verifies the demo identity", async ({ page }) => {
    await page.route("http://localhost:8000/auth/me", async route => {
      await new Promise(resolve => setTimeout(resolve, 500));
      await route.fallback();
    });
    await page.goto("/login");
    await page.getByLabel("Demo username", { exact: true }).fill("test_user");
    await page.getByRole("button", { name: "Sign In to Demo", exact: true }).click();
    await expect(page.locator("button[type='submit']")).toContainText("Signing in");
    await page.waitForURL("/");
  });

  test("all five simulated roles are available only in the explicit demo UI", async ({ page }) => {
    await page.goto("/login");
    await expect(page.getByText("Explicit demo mode:", { exact: false })).toBeVisible();
    const options = page.locator("#role option");
    await expect(options).toHaveCount(5);
    expect(await options.evaluateAll(els => els.map(el => (el as HTMLOptionElement).value)))
      .toEqual(["admin", "data_scientist", "planner", "manager", "viewer"]);
    await expect(page.locator("#access-token")).toHaveCount(0);
  });

  test("demo identity is memory-only and reload signs out", async ({ page }) => {
    await loginViaUI(page, "memory_user", "admin");
    expect(await page.evaluate(() => localStorage.getItem("access_token") === null && localStorage.getItem("user") === null)).toBe(true);
    await page.reload();
    await page.waitForURL("/login");
    await expect(page.locator("header").getByText("memory_user", { exact: true })).toHaveCount(0);
  });

  test("logout clears the current identity", async ({ page }) => {
    await loginViaUI(page, "logout_user", "admin");
    await clearAuth(page);
    await expect(page.locator("header").getByText("logout_user", { exact: true })).toHaveCount(0);
    expect(await page.evaluate(() => localStorage.getItem("access_token") === null)).toBe(true);
  });
});
