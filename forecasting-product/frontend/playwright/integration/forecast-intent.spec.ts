import { expect, test } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { loginDemo, navigateWithinApp } from "../helpers/auth";

function snapshot(root: string): Record<string, string> {
  const files: Record<string, string> = {};
  function walk(directory: string) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const filename = path.join(directory, entry.name);
      if (entry.isDirectory()) walk(filename);
      else if (entry.isFile()) {
        files[path.relative(root, filename)] = createHash("sha256").update(readFileSync(filename)).digest("hex");
      }
    }
  }
  walk(root);
  return files;
}

test("preserves a Croston five-period plan through the real UI/API @integration @forecast-intent", async ({ page }, testInfo) => {
  const apiUrl = process.env.FORECAST_INTENT_API_URL;
  const dataDir = process.env.FORECAST_INTENT_DATA_DIR;
  test.skip(!apiUrl || !dataDir, "Requires the isolated forecast_intent_server.py fixture and explicit output root");
  if (!apiUrl || !dataDir) throw new Error("Missing forecast intent fixture configuration");
  const frontendUrl = new URL(testInfo.project.use.baseURL ?? "");
  expect(["localhost", "127.0.0.1"]).toContain(frontendUrl.hostname);
  expect(new URL(apiUrl).hostname).toBe("127.0.0.1");
  expect(new URL(apiUrl).protocol).toBe("http:");
  const marker = JSON.parse(readFileSync(path.join(dataDir, ".forecast-intent-fixture.json"), "utf-8"));
  expect(marker.api_url).toBe(apiUrl);
  const schema = await page.request.get(`${apiUrl}/openapi.json`);
  expect((await schema.json()).info.title).toBe("Forecast intent test fixture");

  // No mocks: allow only the dedicated UI and API origins.
  await page.route("**/*", (route) => {
    const origin = new URL(route.request().url()).origin;
    return [frontendUrl.origin, apiUrl].includes(origin) ? route.continue() : route.abort();
  });
  await page.setViewportSize({ width: 1440, height: 1800 });
  await loginDemo(page, { username: "intent-demo", role: "admin" });
  await navigateWithinApp(page, "/data-onboarding");
  await page.getByLabel("LOB Name").fill("intent_browser");
  const rows = Array.from({ length: 60 }, (_, i) =>
    `sku,${new Date(Date.UTC(2024, 0, 1 + i * 7)).toISOString().slice(0, 10)},${10 + i * 3}`
  );
  const csv = `series_id,week,quantity\n${rows.join("\n")}\n`;
  const analysis = page.waitForResponse(r => r.url().startsWith(`${apiUrl}/analyze?`) && r.request().method() === "POST");
  await page.getByRole("button", { name: /Time Series Data/ }).locator("input").setInputFiles({
    name: "intent-synthetic.csv", mimeType: "text/csv", buffer: Buffer.from(csv),
  });
  expect((await analysis).status()).toBe(200);
  await expect(page.getByRole("heading", { name: "Pipeline Execution", exact: true })).toBeVisible();

  const yaml = [
    "forecast:", "  frequency: W", "  horizon_periods: 5", "  forecasters: [naive_seasonal, croston]",
    "backtest:", "  n_folds: 1", "  val_periods: 4",
    "parallelism:", "  n_workers: 1",
    "data_quality:", "  fill_gaps: false", "  min_series_length_periods: 1",
    "post_validation:", "  enabled: false", "",
  ].join("\n");
  const configInput = page.getByRole("button", { name: /Pipeline Configuration/ }).locator("input");
  await configInput.setInputFiles({ name: "five-period-plan.yaml", mimeType: "text/yaml", buffer: Buffer.from(yaml) });
  const backtestResponse = page.waitForResponse(r => r.url().startsWith(`${apiUrl}/pipeline/backtest?`));
  await page.getByRole("button", { name: "Run Backtest", exact: true }).click();
  const backtest = await backtestResponse;
  expect(backtest.status()).toBe(200);
  const winner = await backtest.json();
  expect(winner.champion_model).toBe("croston");
  expect(winner.execution.horizon_periods).toBe(5);

  const forecastResponse = page.waitForResponse(r => r.url().startsWith(`${apiUrl}/pipeline/forecast?`));
  await page.getByRole("button", { name: "Run Forecast", exact: true }).click();
  const forecast = await forecastResponse;
  expect(forecast.status()).toBe(200);
  const requestUrl = new URL(forecast.request().url());
  expect(requestUrl.searchParams.get("model_id")).toBe("croston");
  expect(requestUrl.searchParams.get("horizon")).toBe("5");
  expect(forecast.request().postData()).toContain(yaml);
  const result = await forecast.json();
  expect(result.model_id).toBe("croston");
  expect(result.horizon).toBe(5);
  expect(result.forecast_rows).toBe(5);
  expect(result.forecast_preview).toHaveLength(5);
  expect(result.forecast_preview.map((row: { model: string }) => row.model)).toEqual(Array(5).fill("croston"));
  expect(result.execution.config).toEqual(winner.execution.config);
  await expect(page.getByText("Effective model", { exact: true }).locator("..")).toContainText("croston");
  await expect(page.getByText("Horizon (periods)", { exact: true }).locator("..")).toContainText("5");
  const pipelinePanel = page.locator("section").filter({ has: page.getByRole("heading", { name: "Pipeline Execution", exact: true }) });
  await pipelinePanel.screenshot({ path: testInfo.outputPath("forecast-intent.png") });
  await testInfo.attach("real-execution", {
    body: JSON.stringify({ request: forecast.request().url(), backtest: winner, forecast: result }, null, 2),
    contentType: "application/json",
  });

  const before = snapshot(dataDir);
  expect(Object.keys(before).filter(filename =>
    filename.includes("intent_browser") && path.basename(filename) === "manifest.json")).toHaveLength(2);
  expect(result.run_id).not.toBe(winner.run_id);
  await configInput.setInputFiles({ name: "invalid-plan.yaml", mimeType: "text/yaml", buffer: Buffer.from("forecast: [") });
  await expect(page.getByRole("heading", { name: "Backtest Result", exact: true })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Forecast Result", exact: true })).toHaveCount(0);
  for (const [endpoint, button] of [["forecast", "Run Forecast"], ["backtest", "Run Backtest"]]) {
    const failure = page.waitForResponse(r => r.url().startsWith(`${apiUrl}/pipeline/${endpoint}?`));
    await page.getByRole("button", { name: button, exact: true }).click();
    expect((await failure).status()).toBe(400);
    await expect(page.getByText(/Invalid config YAML syntax/)).toBeVisible();
    expect(snapshot(dataDir)).toEqual(before);
  }
});
