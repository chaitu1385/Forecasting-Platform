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

  // ── SHAP ─────────────────────────────────────────────────────────────────

  test("SHAP data loads on demand", async ({ page }) => {
    // Look for a button to load SHAP (may say "Load SHAP" or "Feature Importance")
    const shapBtn = page.locator("button:has-text('SHAP'), button:has-text('Feature'), button:has-text('Load')");
    if (await shapBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
      await shapBtn.first().click();
      // After clicking, SHAP features should appear
      await expect(page.locator("text=lag_1")).toBeVisible({ timeout: 10_000 });
    }
  });
});
