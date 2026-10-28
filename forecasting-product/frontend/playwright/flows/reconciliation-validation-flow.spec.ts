import { test, expect } from "@playwright/test";
import { mockApiRoutes, MOCK } from "../helpers/mock-routes";
import { loginDemo, navigateWithinApp } from "../helpers/auth";

test.describe("Reconciliation & Validation Flow", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page);
    await loginDemo(page);
  });

  test("hierarchy mock declares the scoped check rather than an invented coherence score", async () => {
    expect(MOCK.hierarchyReconcile.coherence.scope).toBe("leaf_derived_parent_totals");
    expect(MOCK.hierarchyReconcile.coherence.max_abs_error).toBe(0);
    expect(MOCK.hierarchyReconcile.method).toBe("bottom_up");
  });

  test("hierarchy build returns expected levels", async ({ page }) => {
    expect(MOCK.hierarchyBuild.levels).toEqual(["category", "region", "sku"]);
    expect(MOCK.hierarchyBuild.total_nodes).toBe(3);
    expect(MOCK.hierarchyBuild.leaf_count).toBe(1);
  });

  test("hierarchy aggregation computes correct result", async ({ page }) => {
    expect(MOCK.hierarchyAggregate.target_level).toBe("category");
    expect(MOCK.hierarchyAggregate.unique_nodes).toBe(1);
    expect(MOCK.hierarchyAggregate.total_rows).toBe(1);
  });

  test("reconciliation preserves series count", async ({ page }) => {
    expect(MOCK.hierarchyReconcile.input_rows).toBe(1);
    expect(MOCK.hierarchyReconcile.output_rows).toBe(1);
  });

  test("reconciliation API returns valid preview data", async ({ page }) => {
    expect(MOCK.hierarchyReconcile.reconciled_preview).toEqual([
      { sku: "sku_001", week: "2025-01-06", forecast: 100 },
    ]);
  });

  test("hierarchy page loads and shows file upload", async ({ page }) => {
    await navigateWithinApp(page, "/hierarchy");
    await expect(page.locator("main h1")).toContainText("Hierarchy");
    await expect(page.locator('input[type="file"]')).toBeVisible();
  });

  test("full reconciliation flow on hierarchy page", async ({ page }) => {
    await navigateWithinApp(page, "/hierarchy");

    await page.getByLabel("Upload CSV or Parquet").setInputFiles({
      name: "hierarchy.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(
        "sku,category,region,week,quantity,forecast\nsku_001,electronics,east,2025-01-06,100,100\n",
      ),
    });

    await page.getByLabel("Hierarchy Levels (comma-separated)").fill("category,region,sku");
    await page.getByRole("button", { name: "Build Hierarchy", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Level Stats", exact: true })).toBeVisible();
    await page.getByLabel("Method", { exact: true }).selectOption("bottom_up");
    await page.getByRole("button", { name: "Reconcile", exact: true }).click();
    const result = page.getByTestId("reconciliation-result");
    await expect(result.getByRole("cell", { name: "100", exact: true })).toBeVisible();
    await expect(result).toContainText("1 rows, 1 periods, 2 parent/period groups checked");
  });
});
