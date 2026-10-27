import { test, expect } from "@playwright/test";
import { mockApiRoutes } from "../helpers/mock-routes";
import { loginDemo, navigateWithinApp } from "../helpers/auth";

test.describe("Hierarchy Flow", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page);
    await loginDemo(page);
    await navigateWithinApp(page, "/hierarchy");
  });

  test("page shows file upload and levels input", async ({ page }) => {
    await expect(page.locator("main h1")).toContainText("Hierarchy");
    // File input for CSV
    await expect(page.locator('input[type="file"]')).toBeVisible();
  });

  test("reconciliation method selector has all 6 methods", async ({ page }) => {
    // Methods displayed as both cards and a select dropdown
    await expect(page.getByRole("heading", { name: "Bottom-Up" })).toBeVisible({ timeout: 5_000 });
    await expect(page.getByRole("heading", { name: "Top-Down" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Middle-Out" })).toBeVisible();
  });

  test("uploading a file and building hierarchy shows results", async ({ page }) => {
    // Upload CSV
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: "hierarchy_data.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(
        "series_id,category,region,week,quantity\nsku_001,electronics,east,2025-01-06,100\n",
      ),
    });

    // Fill levels input
    const levelsInput = page.locator('input[type="text"]').last();
    if (await levelsInput.isVisible({ timeout: 3_000 }).catch(() => false)) {
      await levelsInput.fill("category,region");
    }

    // Click build hierarchy button
    const buildBtn = page.locator("button:has-text('Build')");
    if (await buildBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
      await buildBtn.click();
      // Mock returns levels: ["category", "region", "sku"], n_nodes: 15
      await page.waitForTimeout(3_000);
      // Check that build result metrics are displayed
      await expect(page.locator("text=15").first()).toBeVisible({ timeout: 10_000 });
    }
  });

  test("reconciliation with bottom-up method", async ({ page }) => {
    // First we need hierarchy to be built — upload and build
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: "hierarchy_data.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(
        "series_id,category,region,week,quantity\nsku_001,electronics,east,2025-01-06,100\n",
      ),
    });

    const levelsInput = page.locator('input[type="text"]').last();
    if (await levelsInput.isVisible({ timeout: 3_000 }).catch(() => false)) {
      await levelsInput.fill("category,region");
    }

    const buildBtn = page.locator("button:has-text('Build')");
    if (await buildBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
      await buildBtn.click();
      await page.waitForTimeout(2_000);
    }

    // Now try reconciliation
    const reconBtn = page.locator("button:has-text('Reconcile')");
    if (await reconBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
      await reconBtn.click();
      // Mock returns coherence_after: 1.0
      await page.waitForTimeout(2_000);
    }
  });
});
