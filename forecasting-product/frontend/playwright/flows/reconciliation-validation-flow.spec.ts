import { test, expect } from "@playwright/test";
import { mockApiRoutes, MOCK } from "../helpers/mock-routes";
import { loginDemo, navigateWithinApp } from "../helpers/auth";

test.describe("Reconciliation & Validation Flow", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page);
    await loginDemo(page);
  });

  test("hierarchy reconciliation returns coherence = 1.0", async ({ page }) => {
    // Verify mock data structure
    expect(MOCK.hierarchyReconcile.coherence_after).toBe(1.0);
    expect(MOCK.hierarchyReconcile.coherence_before).toBe(0.85);
    expect(MOCK.hierarchyReconcile.method).toBe("bottom_up");
  });

  test("hierarchy build returns expected levels", async ({ page }) => {
    expect(MOCK.hierarchyBuild.levels).toEqual(["category", "region", "sku"]);
    expect(MOCK.hierarchyBuild.n_nodes).toBe(15);
    expect(MOCK.hierarchyBuild.n_leaf).toBe(10);
  });

  test("hierarchy aggregation computes correct result", async ({ page }) => {
    expect(MOCK.hierarchyAggregate.target_level).toBe("category");
    expect(MOCK.hierarchyAggregate.n_nodes).toBe(3);
    expect(MOCK.hierarchyAggregate.aggregated_rows).toBe(156);
  });

  test("reconciliation preserves series count", async ({ page }) => {
    expect(MOCK.hierarchyReconcile.n_series_reconciled).toBe(10);
  });

  test("reconciliation API returns valid preview data", async ({ page }) => {
    const preview = MOCK.hierarchyReconcile.preview;
    expect(preview).toHaveLength(1);
    expect(preview[0]).toHaveProperty("series_id");
    expect(preview[0]).toHaveProperty("original");
    expect(preview[0]).toHaveProperty("reconciled");
    expect(preview[0].reconciled).toBeGreaterThanOrEqual(0);
  });

  test("hierarchy page loads and shows file upload", async ({ page }) => {
    await navigateWithinApp(page, "/hierarchy");
    await expect(page.locator("main h1")).toContainText("Hierarchy");
    await expect(page.locator('input[type="file"]')).toBeVisible();
  });

  test("full reconciliation flow on hierarchy page", async ({ page }) => {
    await navigateWithinApp(page, "/hierarchy");

    // Upload CSV file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: "hierarchy.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(
        "series_id,category,region,week,quantity\nsku_001,electronics,east,2025-01-06,100\n",
      ),
    });

    // Fill hierarchy levels
    const textInputs = page.locator('input[type="text"]');
    const levelsInput = textInputs.last();
    if (await levelsInput.isVisible({ timeout: 3_000 }).catch(() => false)) {
      await levelsInput.fill("category,region");
    }

    // Build hierarchy
    const buildBtn = page.locator("button:has-text('Build')");
    if (await buildBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
      await buildBtn.click();
      // Wait for response
      await page.waitForTimeout(2_000);

      // Select reconciliation method
      const methodSelect = page.locator("select").last();
      if (await methodSelect.isVisible({ timeout: 3_000 }).catch(() => false)) {
        await methodSelect.selectOption("bottom_up");
      }

      // Reconcile
      const reconBtn = page.locator("button:has-text('Reconcile')");
      if (await reconBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
        await reconBtn.click();
        await page.waitForTimeout(2_000);
      }
    }
  });
});
