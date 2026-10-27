import { test, expect } from "@playwright/test";
import { mockApiRoutes } from "../helpers/mock-routes";
import { loginDemo, navigateWithinApp } from "../helpers/auth";
import path from "path";

test.describe("Data Onboarding Flow", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page);
    await loginDemo(page);
  });

  test("page shows upload section with LOB input", async ({ page }) => {
    await navigateWithinApp(page, "/data-onboarding");
    await expect(page.locator("main h1")).toContainText("Data Onboarding");
    await expect(page.locator("text=Upload Data")).toBeVisible();
    // LOB name text input
    await expect(page.locator('input[type="text"]').first()).toBeVisible();
  });

  test("LOB name field defaults to retail", async ({ page }) => {
    await navigateWithinApp(page, "/data-onboarding");
    const lobInput = page.locator('input[type="text"]').first();
    await expect(lobInput).toHaveValue("retail");
  });

  test("AI interpretation checkbox is present", async ({ page }) => {
    await navigateWithinApp(page, "/data-onboarding");
    await expect(page.locator("text=Enable AI Interpretation")).toBeVisible();
    const checkbox = page.locator('input[type="checkbox"]');
    await expect(checkbox).not.toBeChecked();
  });

  test("file upload triggers analysis and shows results", async ({ page }) => {
    await navigateWithinApp(page, "/data-onboarding");

    // Trigger file upload via the FileUpload component's hidden input
    const fileInput = page.locator('input[type="file"]');
    // Create a synthetic CSV file in-memory
    await fileInput.setInputFiles({
      name: "test_data.csv",
      mimeType: "text/csv",
      buffer: Buffer.from("series_id,week,quantity\nsku_001,2024-01-01,100\nsku_002,2024-01-01,200\n"),
    });

    // Wait for the mock analysis response to render
    // The analysis mock returns n_series=50, n_rows=5200
    await expect(page.locator("text=Schema Detection")).toBeVisible({ timeout: 15_000 });
  });

  test("changing LOB name is reflected in the input", async ({ page }) => {
    await navigateWithinApp(page, "/data-onboarding");
    const lobInput = page.locator('input[type="text"]').first();
    await lobInput.clear();
    await lobInput.fill("wholesale");
    await expect(lobInput).toHaveValue("wholesale");
  });
});
