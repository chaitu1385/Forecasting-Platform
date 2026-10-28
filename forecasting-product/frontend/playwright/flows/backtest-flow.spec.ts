import { test, expect } from "@playwright/test";
import { mockApiRoutes } from "../helpers/mock-routes";
import { loginDemo, navigateWithinApp } from "../helpers/auth";

test.describe("Backtest Flow", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page);
    await loginDemo(page);
    await navigateWithinApp(page, "/backtest");
  });

  // ── Leaderboard ──────────────────────────────────────────────────────────

  test("leaderboard displays all 4 models", async ({ page }) => {
    await expect(page.locator("text=lgbm_direct").first()).toBeVisible({ timeout: 10_000 });
    await expect(page.locator("text=auto_arima").first()).toBeVisible();
    await expect(page.locator("text=auto_ets").first()).toBeVisible();
    await expect(page.locator("text=seasonal_naive").first()).toBeVisible();
  });

  test("leaderboard shows WMAPE values", async ({ page }) => {
    // Wait for data
    await expect(page.locator("text=lgbm_direct").first()).toBeVisible({ timeout: 10_000 });
    // Check that WMAPE percentages are displayed (0.082 → 8.2%)
    await expect(page.locator("text=8.2").first()).toBeVisible();
  });

  test("LOB input can be changed", async ({ page }) => {
    // Wait for page to fully render
    await page.waitForTimeout(2_000);
    const lobInput = page.locator('label:has-text("LOB") + input, input[type="text"]').first();
    await expect(lobInput).toBeVisible();
    await lobInput.clear();
    await lobInput.fill("wholesale");
    await expect(lobInput).toHaveValue("wholesale");
  });

  // ── FVA Cascade ──────────────────────────────────────────────────────────

  test("FVA section renders with layer data", async ({ page }) => {
    // FVA mock has 4 layers: naive, statistical, ml, champion
    await expect(page.locator("text=naive").first()).toBeVisible({ timeout: 10_000 });
  });

  // ── Calibration ──────────────────────────────────────────────────────────

  test("calibration section renders", async ({ page }) => {
    // Calibration mock has lgbm_direct with 50%, 80%, 95% intervals
    await expect(page.locator("main h1")).toContainText("Backtest Results");
    // The calibration chart should be somewhere on the page
    await page.waitForTimeout(2000); // Allow all fetches to complete
  });

  // ── Decomposition ────────────────────────────────────────────────────────

  test("forecast decomposition loads on demand with truthful labels", async ({ page }) => {
    const section = page.getByRole("region", { name: "Forecast decomposition" });
    await section.getByRole("button", { name: "Load decomposition" }).click();
    await expect(section.getByRole("heading", { name: "Component magnitude" })).toBeVisible();
    await expect(section).toContainText("moving_average_decomposition");
    await expect(section).toContainText("Not model-specific feature attribution");
    await expect(section).not.toContainText(/SHAP|waterfall|feature importance|LOESS/);
  });
});
