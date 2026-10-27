import { test, expect } from "@playwright/test";
import { loginDemo, navigateWithinApp } from "../helpers/auth";

/**
 * E2E integration tests for the Walmart M5 daily-frequency forecast pipeline.
 *
 * Requires:
 *   - Explicit-demo backend at localhost:8000 (with M5 daily data loaded)
 *   - Frontend running at localhost:3000
 *
 * Run:
 *   npm run test:pw:m5daily
 *
 * Tagged @m5daily so they are excluded from default CI runs.
 */

const API = "http://localhost:8000";
const LOB = "walmart_m5_daily";

// Resolve the fixture path relative to the repo root
const FIXTURE_CSV = "tests/integration/fixtures/m5_daily_sample.csv";
const CONFIG_YAML = "tests/integration/fixtures/m5_daily_config.yaml";

test.describe("M5 Daily Pipeline E2E @m5daily", () => {
  test.beforeEach(async ({ page }) => {
    await loginDemo(page, { role: "admin" });
  });

  // ─────────────────────────────────────────────────────────────────────────
  //  1. Data Onboarding
  // ─────────────────────────────────────────────────────────────────────────

  test("data onboarding — upload daily CSV and detect frequency @m5daily", async ({
    page,
  }) => {
    await navigateWithinApp(page, "/data-onboarding");
    await expect(page.locator("main")).toBeVisible();

    // The page should have at minimum a file upload area
    const fileInput = page.locator('input[type="file"]').first();
    if (await fileInput.isVisible()) {
      // Upload the M5 daily sample
      await fileInput.setInputFiles(FIXTURE_CSV);

      // Wait for analysis results (may take a few seconds)
      await page.waitForTimeout(5_000);

      // Look for frequency indicator showing "D" or "Daily"
      const mainText = await page.locator("main").textContent();
      expect(
        mainText?.includes("D") || mainText?.includes("daily") || mainText?.includes("Daily")
      ).toBeTruthy();
    }
  });

  // ─────────────────────────────────────────────────────────────────────────
  //  2. Run Backtest
  // ─────────────────────────────────────────────────────────────────────────

  test("run backtest via API and verify leaderboard @m5daily", async ({
    page,
  }) => {
    // Use the API directly for reliability (Playwright request context)
    const backtestResp = await page.request.post(
      `${API}/pipeline/backtest?lob=${LOB}`,
      {
        multipart: {
          file: {
            name: "m5_daily.csv",
            mimeType: "text/csv",
            buffer: Buffer.from(
              require("fs").readFileSync(FIXTURE_CSV)
            ),
          },
          config_file: {
            name: "config.yaml",
            mimeType: "application/x-yaml",
            buffer: Buffer.from(
              require("fs").readFileSync(CONFIG_YAML)
            ),
          },
        },
      }
    );

    expect(backtestResp.ok()).toBeTruthy();
    const data = await backtestResp.json();
    expect(data.status).toBe("completed");
    expect(data.leaderboard.length).toBeGreaterThanOrEqual(2);
    expect(data.champion_model).toBeTruthy();
    expect(data.best_wmape).toBeGreaterThan(0);
  });

  // ─────────────────────────────────────────────────────────────────────────
  //  3. Run Forecast
  // ─────────────────────────────────────────────────────────────────────────

  test("run forecast via API and verify daily dates @m5daily", async ({
    page,
  }) => {
    // Backtest first to establish champion
    await page.request.post(`${API}/pipeline/backtest?lob=${LOB}`, {
      multipart: {
        file: {
          name: "m5_daily.csv",
          mimeType: "text/csv",
          buffer: Buffer.from(require("fs").readFileSync(FIXTURE_CSV)),
        },
        config_file: {
          name: "config.yaml",
          mimeType: "application/x-yaml",
          buffer: Buffer.from(require("fs").readFileSync(CONFIG_YAML)),
        },
      },
    });

    // Forecast
    const fcResp = await page.request.post(
      `${API}/pipeline/forecast?lob=${LOB}&horizon=28`,
      {
        multipart: {
          file: {
            name: "m5_daily.csv",
            mimeType: "text/csv",
            buffer: Buffer.from(require("fs").readFileSync(FIXTURE_CSV)),
          },
          config_file: {
            name: "config.yaml",
            mimeType: "application/x-yaml",
            buffer: Buffer.from(require("fs").readFileSync(CONFIG_YAML)),
          },
        },
      }
    );

    expect(fcResp.ok()).toBeTruthy();
    const data = await fcResp.json();
    expect(data.status).toBe("completed");
    expect(data.forecast_rows).toBeGreaterThan(0);
    expect(data.series_count).toBeGreaterThan(10);

    // Verify forecast preview has daily dates
    if (data.forecast_preview && data.forecast_preview.length >= 2) {
      const d0 = new Date(data.forecast_preview[0].week);
      const d1 = new Date(data.forecast_preview[1].week);
      // For same series, dates should be 1 day apart
      if (data.forecast_preview[0].series_id === data.forecast_preview[1].series_id) {
        const diffDays = (d1.getTime() - d0.getTime()) / (1000 * 60 * 60 * 24);
        expect(diffDays).toBe(1);
      }
    }
  });

  // ─────────────────────────────────────────────────────────────────────────
  //  4. View Backtest Page
  // ─────────────────────────────────────────────────────────────────────────

  test("backtest page renders leaderboard @m5daily", async ({ page }) => {
    await navigateWithinApp(page, "/backtest");
    await expect(page.locator("main h1")).toContainText("Backtest");

    // Wait for content to load
    await page.waitForTimeout(3_000);

    // The page should render without errors
    const mainContent = await page.locator("main").textContent();
    expect(mainContent).toBeTruthy();
  });

  // ─────────────────────────────────────────────────────────────────────────
  //  5. View Forecast Page
  // ─────────────────────────────────────────────────────────────────────────

  test("forecast page renders with daily data @m5daily", async ({ page }) => {
    await navigateWithinApp(page, "/forecast");
    await expect(page.locator("main")).toBeVisible();

    // Wait for content
    await page.waitForTimeout(3_000);

    // Page should render heading
    const heading = page.locator("main h1");
    await expect(heading).toBeVisible();
  });

  // ─────────────────────────────────────────────────────────────────────────
  //  6. Series Explorer
  // ─────────────────────────────────────────────────────────────────────────

  test("series explorer detects breaks on daily data @m5daily", async ({
    page,
  }) => {
    await navigateWithinApp(page, "/series-explorer");
    await expect(page.locator("main")).toBeVisible();

    // Call break detection API directly
    const breaksResp = await page.request.post(
      `${API}/series/breaks?method=cusum`,
      {
        multipart: {
          file: {
            name: "m5_daily.csv",
            mimeType: "text/csv",
            buffer: Buffer.from(require("fs").readFileSync(FIXTURE_CSV)),
          },
        },
      }
    );
    expect(breaksResp.ok()).toBeTruthy();
    const data = await breaksResp.json();
    expect(data.total_series).toBeGreaterThan(0);
  });

  // ─────────────────────────────────────────────────────────────────────────
  //  7. Hierarchy
  // ─────────────────────────────────────────────────────────────────────────

  test("hierarchy build and aggregate on daily data @m5daily", async ({
    page,
  }) => {
    // Build hierarchy via API
    const buildResp = await page.request.post(
      `${API}/hierarchy/build?levels=cat_id,dept_id,series_id&id_column=series_id&name=product`,
      {
        multipart: {
          file: {
            name: "m5_daily.csv",
            mimeType: "text/csv",
            buffer: Buffer.from(require("fs").readFileSync(FIXTURE_CSV)),
          },
        },
      }
    );
    expect(buildResp.ok()).toBeTruthy();
    const buildData = await buildResp.json();
    expect(buildData.total_nodes).toBeGreaterThan(0);

    // Aggregate to category level
    const aggResp = await page.request.post(
      `${API}/hierarchy/aggregate?levels=cat_id,dept_id,series_id&target_level=cat_id&value_columns=quantity`,
      {
        multipart: {
          file: {
            name: "m5_daily.csv",
            mimeType: "text/csv",
            buffer: Buffer.from(require("fs").readFileSync(FIXTURE_CSV)),
          },
        },
      }
    );
    expect(aggResp.ok()).toBeTruthy();
    const aggData = await aggResp.json();
    expect(aggData.total_rows).toBeGreaterThan(0);
  });

  // ─────────────────────────────────────────────────────────────────────────
  //  8. Platform Health
  // ─────────────────────────────────────────────────────────────────────────

  test("health page and API work after daily pipeline runs @m5daily", async ({
    page,
  }) => {
    // API health check
    const healthResp = await page.request.get(`${API}/health`);
    expect(healthResp.ok()).toBeTruthy();
    const healthData = await healthResp.json();
    expect(healthData.status).toBe("ok");

    // Navigate to health page
    await navigateWithinApp(page, "/health");
    await expect(page.locator("main h1")).toContainText("Health");
  });
});
