import { test, expect, type Page } from "@playwright/test";
import { mockApiRoutes } from "../helpers/mock-routes";
import { loginDemo, navigateWithinApp } from "../helpers/auth";

async function uploadHierarchy(page: Page) {
  await page.getByLabel("Upload CSV or Parquet").setInputFiles({
    name: "hierarchy_data.csv", mimeType: "text/csv",
    buffer: Buffer.from("sku,category,region,week,quantity,forecast\nsku_001,electronics,east,2025-01-06,100,100\n"),
  });
  await page.getByLabel("Hierarchy Levels (comma-separated)").fill("category,region,sku");
}

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

  test("reconciliation selector exposes supported methods and discloses middle-out", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Bottom-Up" })).toBeVisible({ timeout: 5_000 });
    await expect(page.getByRole("heading", { name: "Top-Down" })).toBeVisible();
    await expect(page.getByLabel("Method", { exact: true }).locator("option")).toHaveCount(5);
    await expect(page.getByRole("option", { name: /middle-out/i })).toHaveCount(0);
    await expect(page.getByText(/Middle-out is unavailable/)).toBeVisible();
    await expect(page.getByRole("heading", { name: "WLS (structural)", exact: true })).toBeVisible();
    await expect(page.getByRole("heading", { name: "MinT (structural fallback)", exact: true })).toBeVisible();
  });

  test("uploading a file and building hierarchy shows results", async ({ page }) => {
    await uploadHierarchy(page);
    await page.getByRole("button", { name: "Build Hierarchy", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Level Stats", exact: true })).toBeVisible();
    await expect(page.getByText("3 x 3", { exact: true })).toBeVisible();
  });

  test("reconciliation with bottom-up method", async ({ page }) => {
    await uploadHierarchy(page);
    await page.getByRole("button", { name: "Reconcile", exact: true }).click();
    const result = page.getByTestId("reconciliation-result");
    await expect(result.getByRole("cell", { name: "sku_001", exact: true })).toBeVisible();
    await expect(result.getByRole("cell", { name: "100", exact: true })).toBeVisible();
    await expect(page.getByTestId("reconciliation-effective-method")).toHaveText("Bottom-Up");
  });
});
