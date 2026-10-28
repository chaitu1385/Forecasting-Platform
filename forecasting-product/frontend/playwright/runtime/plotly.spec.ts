import { expect, test } from "@playwright/test";
import type { PlotlyHTMLElement } from "plotly.js";
import type { ForecastResponse, HierarchyBuildResponse } from "../../src/lib/types";
import { loginDemo, navigateWithinApp } from "../helpers/auth";
import { mockApiRoutes } from "../helpers/mock-routes";

const forecast: ForecastResponse = {
  lob: "retail",
  series_count: 2,
  forecast_origin: "2026-01-01",
  points: ["runtime-a", "runtime-b"].flatMap((series_id, series) =>
    [5, 12, 19, 26].map((day, period) => ({
      series_id,
      week: `2026-01-${String(day).padStart(2, "0")}`,
      forecast: 100 + series * 100 + period * 10,
      forecast_p10: 90 + series * 100 + period * 10,
      forecast_p50: 100 + series * 100 + period * 10,
      forecast_p90: 110 + series * 100 + period * 10,
      model: "synthetic",
    })),
  ),
};

const hierarchy: HierarchyBuildResponse = {
  name: "synthetic",
  levels: ["region", "sku"],
  level_stats: [{ level: "region", node_count: 2 }, { level: "sku", node_count: 2 }],
  total_nodes: 5,
  leaf_count: 2,
  s_matrix_shape: [5, 2],
  s_matrix_sample: [],
  tree_nodes: [
    { key: "All", level: "total", parent: "", is_leaf: false },
    { key: "West", level: "region", parent: "All", is_leaf: false },
    { key: "East", level: "region", parent: "All", is_leaf: false },
    { key: "sku-a", level: "sku", parent: "West", is_leaf: true },
    { key: "sku-b", level: "sku", parent: "East", is_leaf: true },
  ],
};

let pageErrors: string[];
let externalRequests: string[];

test.beforeEach(async ({ page, context, baseURL }) => {
  if (!baseURL) throw new Error("The runtime browser suite requires an isolated frontend URL");
  pageErrors = [];
  externalRequests = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  const allowedOrigins = new Set([new URL(baseURL).origin, "http://localhost:8000"]);
  await context.route("**/*", (route) => {
    const origin = new URL(route.request().url()).origin;
    if (allowedOrigins.has(origin)) return route.continue();
    externalRequests.push(origin);
    return route.abort();
  });
  await mockApiRoutes(page);
  await page.route((url) =>
    url.origin === "http://localhost:8000" && url.pathname === "/forecast/retail", (route) =>
    route.fulfill({ json: forecast }),
  );
  await page.route((url) =>
    url.origin === "http://localhost:8000" && url.pathname === "/hierarchy/build", (route) =>
    route.fulfill({ json: hierarchy }),
  );
  await loginDemo(page, { username: "runtime-review" });
});

test.afterEach(() => {
  expect(pageErrors).toEqual([]);
  expect(externalRequests).toEqual([]);
});

test("real fan chart renders quantiles, updates series and toggles its legend", async ({ page }) => {
  await navigateWithinApp(page, "/forecast");
  const plot = page.locator(".js-plotly-plot");
  await expect(plot.locator(".scatterlayer .trace")).toHaveCount(3);
  await expect(plot.locator(".scatterlayer .js-fill").first()).toBeVisible();
  expect(await plot.evaluate((element) =>
    (element as PlotlyHTMLElement).data.map((trace) => ({
      name: trace.name,
      y: "y" in trace ? trace.y : undefined,
    })),
  )).toEqual([
    { name: "P10\u2013P90", y: [110, 120, 130, 140, 120, 110, 100, 90] },
    { name: "P50 (median)", y: [100, 110, 120, 130] },
    { name: "Forecast", y: [100, 110, 120, 130] },
  ]);
  await page.locator("select")
    .filter({ has: page.locator('option[value="runtime-b"]') })
    .first().selectOption("runtime-b");
  await expect.poll(() => plot.evaluate((element) => {
    const trace = (element as PlotlyHTMLElement).data.at(-1);
    return trace && "y" in trace ? trace.y : undefined;
  })).toEqual([200, 210, 220, 230]);
  await plot.locator(".legend .traces")
    .filter({ hasText: /^Forecast$/ }).locator(".legendtoggle").click();
  await expect.poll(() => plot.evaluate((element) =>
    (element as PlotlyHTMLElement).data.at(-1)?.visible,
  )).toBe("legendonly");
});

test("real hierarchy sunburst renders all nodes and drills into a region", async ({ page }) => {
  await navigateWithinApp(page, "/hierarchy");
  await page.locator('input[type="file"]').setInputFiles({
    name: "hierarchy.csv",
    mimeType: "text/csv",
    buffer: Buffer.from(
      "series_id,region,week,quantity\nsku-a,West,2026-01-05,100\nsku-b,East,2026-01-05,200\n",
    ),
  });
  await page.getByPlaceholder("e.g. category,subcategory,brand").fill("region,sku");
  await page.getByRole("button", { name: "Build Hierarchy", exact: true }).click();
  const plot = page.locator(".js-plotly-plot");
  await expect(plot.locator(".sunburstlayer .slice")).toHaveCount(5);
  expect(await plot.evaluate((element) => {
    const trace = (element as PlotlyHTMLElement).data[0];
    return {
      labels: "labels" in trace ? trace.labels : undefined,
      values: "values" in trace ? trace.values : undefined,
    };
  })).toEqual({ labels: ["All", "West", "East", "sku-a", "sku-b"], values: [2, 1, 1, 1, 1] });
  await plot.locator(".sunburstlayer .slice")
    .filter({ hasText: /^West$/ }).locator("path.surface").click();
  await expect.poll(() => plot.evaluate((element) => {
    const trace = (element as PlotlyHTMLElement).data[0];
    return "level" in trace ? trace.level : undefined;
  })).toBe("West");
});
