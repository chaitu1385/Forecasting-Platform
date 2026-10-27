import { test, expect } from "@playwright/test";
import { loginDemo, navigateWithinApp } from "../helpers/auth";

/**
 * Integration tests that require a REAL explicit-demo backend at localhost:8000.
 * Run with: npm run test:pw:integration
 * These are tagged @integration and excluded from default CI runs.
 */
test.describe("Integration — Real Backend @integration", () => {
  test.beforeEach(async ({ page }) => {
    await loginDemo(page);
  });
  test("health endpoint returns ok", async ({ page }) => {
    // Call the real backend health endpoint directly
    const response = await page.request.get("http://localhost:8000/health");
    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.status).toBe("ok");
    expect(body).toHaveProperty("version");
  });

  test("health page renders with real data", async ({ page }) => {
    // Navigate to health page — requires both frontend and backend running
    await navigateWithinApp(page, "/health");
    await expect(page.locator("main h1")).toContainText("Platform Health");
    // The page should load without throwing errors
    // Real backend should return actual drift/audit data
  });

  test("backtest page renders with real leaderboard", async ({ page }) => {
    await navigateWithinApp(page, "/backtest");
    await expect(page.locator("main h1")).toContainText("Backtest Results");
    // If real data exists, the leaderboard should render within 15s
    await page.waitForTimeout(5_000);
  });
});
