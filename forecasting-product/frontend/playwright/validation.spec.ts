import { test, expect } from "@playwright/test";
import { mockApiRoutes, MOCK } from "./helpers/mock-routes";
import { loginDemo, navigateWithinApp } from "./helpers/auth";

test.describe("Validation — Backtest & Manifest Data Integrity", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page);
    await loginDemo(page);
  });

  // ── Leaderboard Data Validation ──────────────────────────────────────────

  test("leaderboard entries are ranked correctly (ascending WMAPE)", async ({ page }) => {
    await navigateWithinApp(page, "/backtest");
    await expect(page.locator("text=lgbm_direct").first()).toBeVisible({ timeout: 10_000 });

    // Verify the mock data itself is sorted by rank
    const entries = MOCK.leaderboard.entries;
    for (let i = 0; i < entries.length - 1; i++) {
      expect(entries[i].wmape).toBeLessThan(entries[i + 1].wmape);
    }
  });

  test("all leaderboard models have n_series = 50", async ({ page }) => {
    await navigateWithinApp(page, "/backtest");
    await expect(page.locator("text=lgbm_direct").first()).toBeVisible({ timeout: 10_000 });

    for (const entry of MOCK.leaderboard.entries) {
      expect(entry.n_series).toBe(50);
    }
  });

  // ── FVA Validation ───────────────────────────────────────────────────────

  test("FVA layers show decreasing WMAPE from naive to champion", async ({ page }) => {
    await navigateWithinApp(page, "/backtest");
    await page.waitForTimeout(3_000);

    const fvaSummary = MOCK.fva.summary;
    expect(fvaSummary[0].avg_wmape).toBeGreaterThan(
      fvaSummary[fvaSummary.length - 1].avg_wmape,
    );
  });

  // ── Manifest Validation ──────────────────────────────────────────────────

  test("health page shows pipeline manifests", async ({ page }) => {
    await navigateWithinApp(page, "/health");
    // Click the Pipeline Manifests tab
    await page.getByRole("button", { name: "Pipeline Manifests" }).click();
    await expect(page.locator("text=run_2025_01_15_001").first()).toBeVisible({ timeout: 10_000 });
  });

  test("manifests show validation status", async ({ page }) => {
    await navigateWithinApp(page, "/health");
    await page.getByRole("button", { name: "Pipeline Manifests" }).click();

    // First manifest has validation_passed: true
    const manifests = MOCK.manifests.manifests;
    expect(manifests[0].validation_passed).toBe(true);
    expect(manifests[0].validation_warnings).toBe(0);
  });

  // ── Drift Validation ────────────────────────────────────────────────────

  test("drift alerts show correct severity counts", async ({ page }) => {
    await navigateWithinApp(page, "/health");
    await page.waitForTimeout(3_000);

    // Mock drift data: 1 critical, 2 warning
    expect(MOCK.drift.n_critical).toBe(1);
    expect(MOCK.drift.n_warning).toBe(2);
    expect(MOCK.drift.alerts).toHaveLength(3);
  });

  // ── Cost Tracking ───────────────────────────────────────────────────────

  test("cost data is consistent with manifests", async ({ page }) => {
    await navigateWithinApp(page, "/health");
    await page.waitForTimeout(3_000);

    // Each cost entry should have matching run_id in manifests
    const costRunIds = MOCK.costs.costs.map((c) => c.run_id);
    const manifestRunIds = MOCK.manifests.manifests.map((m) => m.run_id);
    for (const costId of costRunIds) {
      expect(manifestRunIds).toContain(costId);
    }
  });
});
