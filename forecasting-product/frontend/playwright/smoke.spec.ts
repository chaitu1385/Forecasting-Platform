import { test, expect } from "@playwright/test";
import { mockApiRoutes } from "./helpers/mock-routes";
import { loginDemo, navigateWithinApp } from "./helpers/auth";

test.describe("Smoke Tests — All Pages Load", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page);
    await loginDemo(page);
  });

  test("home page loads with persona cards", async ({ page }) => {
    await navigateWithinApp(page, "/");
    await expect(page.locator("main h1")).toContainText("Sales Forecasting Product");
    await expect(page.locator("text=Data Scientist")).toBeVisible();
    await expect(page.locator("text=Demand Planner")).toBeVisible();
    await expect(page.locator("text=S&OP Leader")).toBeVisible();
    await expect(page.locator("text=Platform Engineer")).toBeVisible();
  });

  test("login page loads", async ({ page }) => {
    await page.goto("/login");
    await expect(page.locator("h2")).toContainText("Forecasting Product");
    await expect(page.locator("#username")).toBeVisible();
    await expect(page.locator("#role")).toBeVisible();
    await expect(page.locator('button[type="submit"]')).toBeVisible();
  });

  test("data onboarding page loads", async ({ page }) => {
    await navigateWithinApp(page, "/data-onboarding");
    await expect(page.locator("main h1")).toContainText("Data Onboarding");
    await expect(page.locator("text=Upload Data")).toBeVisible();
  });

  test("series explorer page loads", async ({ page }) => {
    await navigateWithinApp(page, "/series-explorer");
    await expect(page.locator("main h1")).toContainText("Series Explorer");
  });

  test("backtest page loads with leaderboard", async ({ page }) => {
    await navigateWithinApp(page, "/backtest");
    await expect(page.locator("main h1")).toContainText("Backtest Results");
    // Wait for leaderboard data to render
    await expect(page.locator("text=lgbm_direct").first()).toBeVisible({ timeout: 10_000 });
  });

  test("forecast viewer page loads", async ({ page }) => {
    await navigateWithinApp(page, "/forecast");
    await expect(page.locator("main h1")).toContainText("Forecast");
  });

  test("hierarchy page loads", async ({ page }) => {
    await navigateWithinApp(page, "/hierarchy");
    await expect(page.locator("main h1")).toContainText("Hierarchy");
  });

  test("SKU transitions page loads", async ({ page }) => {
    await navigateWithinApp(page, "/sku-transitions");
    await expect(page.locator("main h1")).toContainText("SKU");
  });

  test("platform health page loads", async ({ page }) => {
    await navigateWithinApp(page, "/health");
    await expect(page.locator("main h1")).toContainText("Platform Health");
  });

  test("S&OP meeting page loads", async ({ page }) => {
    await navigateWithinApp(page, "/sop");
    await expect(page.locator("main h1")).toContainText("S&OP");
  });
});
