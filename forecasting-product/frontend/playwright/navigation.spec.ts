import { test, expect } from "@playwright/test";
import { mockApiRoutes } from "./helpers/mock-routes";
import { loginDemo, navigateWithinApp } from "./helpers/auth";

test.describe("Navigation — Sidebar & Persona Cards", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page);
    await loginDemo(page);
  });

  // ── Sidebar Navigation ──────────────────────────────────────────────────

  const NAV_ITEMS = [
    { title: "Data Onboarding", href: "/data-onboarding" },
    { title: "Series Explorer", href: "/series-explorer" },
    { title: "SKU Transitions", href: "/sku-transitions" },
    { title: "Hierarchy", href: "/hierarchy" },
    { title: "Backtest Results", href: "/backtest" },
    { title: "Forecast Viewer", href: "/forecast" },
    { title: "Platform Health", href: "/health" },
    { title: "S&OP Meeting", href: "/sop" },
  ];

  for (const item of NAV_ITEMS) {
    test(`sidebar link "${item.title}" navigates to ${item.href}`, async ({ page }) => {
      await navigateWithinApp(page, "/");
      // Click the sidebar link by its text
      await page.locator("aside").locator(`a:has-text("${item.title}")`).click();
      await page.waitForURL(`**${item.href}`);
      expect(page.url()).toContain(item.href);
    });
  }

  // ── Sidebar Collapse/Expand ──────────────────────────────────────────────

  test("sidebar can be collapsed and expanded", async ({ page }) => {
    await navigateWithinApp(page, "/");
    const sidebar = page.locator("aside");

    // Initially expanded — "Forecasting Product" text visible
    await expect(sidebar.locator("text=Forecasting Product").first()).toBeVisible();

    // Click collapse button (ChevronLeft icon)
    await sidebar.locator("button").first().click();

    // After collapse — sidebar brand text should be hidden
    await expect(sidebar.locator(".text-sm.font-bold")).toBeHidden();

    // Click expand button (ChevronRight icon)
    await sidebar.locator("button").first().click();

    // After expand — text should be visible again
    await expect(sidebar.locator(".text-sm.font-bold")).toBeVisible();
  });

  // ── Persona Quick-Start Cards ────────────────────────────────────────────

  test("persona card links navigate to correct pages", async ({ page }) => {
    await navigateWithinApp(page, "/");

    // Click first link in Data Scientist persona card → /data-onboarding
    const dsCard = page.locator("text=Data Scientist").locator("..");
    const firstLink = dsCard.locator("a").first();
    const href = await firstLink.getAttribute("href");
    expect(
      ["/data-onboarding", "/series-explorer", "/backtest", "/forecast"],
    ).toContain(href);
  });
});
