import { expect, test, type Page, type Request, type TestInfo } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import type {
  AnalysisResponse, BIExportResponse, ForecastResponse, LeaderboardResponse,
  PipelineRunResponse, RunConfigResponse,
} from "../../src/lib/types";
import { submitPrivately } from "../fixtures/authentication";
import { navigateWithinApp } from "../helpers/auth";

type Row = Record<string, unknown>;
interface Frame { row_count: number; columns: string[]; rows: Row[] }
interface Marker { api_url: string; frontend_origin: string; mode: string }

test.use({ timezoneId: "America/Los_Angeles" });

const digest = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
const day = (start: string, offset: number) =>
  new Date(Date.parse(`${start}T00:00:00Z`) + offset * 86_400_000).toISOString().slice(0, 10);

function demand(series: number, index: number, daily = false, offset = 0): number {
  if (daily && [160, 190].includes(index)) return 0;
  return (series === 0 ? 100 + 2 * index + 5 * (index % 4) : 40 + index + 2 * (index % 4)) + offset;
}

function history(daily = false, offset = 0, omit?: "time" | "target"): string {
  const rows = [omit === "time" ? "series_id,quantity"
    : omit === "target" ? "series_id,week" : `series_id,${daily ? "date" : "week"},quantity`];
  for (let series = 0; series < 2; series++) {
    for (let index = 0; index < (daily ? 208 : 104); index++) {
      const id = `${daily ? "DAILY" : "CORE"}_${series === 0 ? "A" : "B"}`;
      const date = day(daily ? "2025-01-01" : "2024-01-01", index * (daily ? 1 : 7));
      const value = demand(series, index, daily, offset);
      rows.push(omit === "time" ? `${id},${value}` : omit === "target" ? `${id},${date}` : `${id},${date},${value}`);
    }
  }
  return rows.join("\n") + "\n";
}

function productHierarchy(): string {
  const rows = ["product_id,category,week,quantity"];
  for (let category = 0; category < 4; category++) {
    for (let product = 0; product < 2; product++) {
      for (let index = 0; index < 104; index++) {
        rows.push(`PRODUCT_${category}_${product},CATEGORY_${category},${day("2024-01-01", index * 7)},${(100 + 20 * category + 10 * product + 2 * index + 5 * (index % 4)).toFixed(1)}`);
      }
    }
  }
  return rows.join("\n") + "\n";
}

const plan = (model: string, horizon: number) => `forecast:
  frequency: W
  forecasters: [${model}]
  horizon_periods: ${horizon}
  quantiles: [0.1, 0.5, 0.9]
backtest:
  n_folds: 2
  val_periods: 8
  gap_periods: 0
  champion_granularity: lob
data_quality:
  fill_gaps: false
  min_series_length_periods: 52
  validation:
    enabled: true
  cleansing:
    enabled: false
post_validation:
  enabled: true
  halt_on_blocker: true
parallelism:
  n_workers: 1
  n_jobs_statsforecast: 1
  num_threads_mlforecast: 1
metrics: [wmape, normalized_bias, mae, rmse]
`;

function number(row: Row, column: string): number {
  const value = row[column];
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`Expected finite numeric ${column}`);
  return value;
}

function text(row: Row, column: string): string {
  const value = row[column];
  if (typeof value !== "string") throw new Error(`Expected string ${column}`);
  return value;
}

function project(rows: Row[], columns: string[]): Row[] {
  return rows.map(row => Object.fromEntries(columns.map(column => [column, row[column]])));
}

function ordered(rows: Row[], id = "series_id", time = "week"): Row[] {
  return [...rows].sort((a, b) => text(a, id).localeCompare(text(b, id)) || text(a, time).localeCompare(text(b, time)));
}

function hashes(directory: string): Record<string, string> {
  const result: Record<string, string> = {};
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      for (const [member, hash] of Object.entries(hashes(file))) result[path.join(entry.name, member)] = hash;
    } else if (entry.isFile()) result[entry.name] = digest(readFileSync(file));
    else throw new Error("Unexpected nonregular fixture member");
  }
  return result;
}

function unchanged(before: Record<string, string>, directory: string) {
  const after = hashes(directory);
  for (const [member, hash] of Object.entries(before)) expect(after[member], member).toBe(hash);
}

function parquet(file: string): Frame {
  const result: Frame = JSON.parse(execFileSync(process.env.AUTH_E2E_PYTHON || "python", ["-c", [
    "import json,sys,polars as pl",
    "frame=pl.read_parquet(sys.argv[1])",
    "print(json.dumps({'row_count':frame.height,'columns':frame.columns,'rows':frame.to_dicts()},default=str,allow_nan=False))",
  ].join(";"), file], { encoding: "utf8", windowsHide: true, maxBuffer: 8 * 1024 * 1024 }));
  expect(result.rows).toHaveLength(result.row_count);
  return result;
}

function fixture(page: Page, testInfo: TestInfo) {
  expect(path.basename(testInfo.config.configFile ?? "")).toBe("playwright.auth.config.ts");
  expect(process.env.AUTH_E2E_FRONTEND_MODE).toBe("production");
  for (const option of ["trace", "screenshot", "video"] as const) expect(testInfo.project.use[option]).toBe("off");
  expect(Boolean(process.env.DEBUG || process.env.PWDEBUG)).toBe(false);
  const root = process.env.AUTH_E2E_ROOT;
  if (!root || !path.isAbsolute(root)) throw new Error("A private absolute authentication fixture root is required");
  const marker: Marker = JSON.parse(readFileSync(path.join(root, "data", ".authentication-fixture.json"), "utf8"));
  expect(marker.frontend_origin).toBe(testInfo.project.use.baseURL);
  expect(marker.mode).toBe("production");
  for (const url of [marker.frontend_origin, marker.api_url]) expect(new URL(url).hostname).toBe("127.0.0.1");
  const credentials = {
    admin: readFileSync(path.join(root, "credentials", "admin.jwt"), "utf8").trim(),
    viewer: readFileSync(path.join(root, "credentials", "viewer.jwt"), "utf8").trim(),
  };
  const privateValues = Object.values(credentials);
  let exposed = false;
  const safe = (value: string) => {
    exposed ||= privateValues.some(secret => value.includes(secret));
    return privateValues.reduce((value, secret) => value.replaceAll(secret, "[REDACTED]"), value);
  };
  const diagnostics = {
    page_errors: [] as string[],
    console: [] as { type: string; message: string }[],
    http_errors: [] as { path: string; method: string; http_status: number }[],
    request_failures: [] as { path: string; method: string; error: string }[],
  };
  page.on("console", event => {
    const message = safe(event.text());
    if (["error", "warning"].includes(event.type())) diagnostics.console.push({ type: event.type(), message });
  });
  page.on("pageerror", error => diagnostics.page_errors.push(safe(error.message)));
  page.on("response", response => {
    if (response.status() >= 400) diagnostics.http_errors.push({
      path: new URL(response.url()).pathname, method: response.request().method(), http_status: response.status(),
    });
  });
  page.on("requestfailed", request => diagnostics.request_failures.push({
    path: new URL(request.url()).pathname, method: request.method(), error: safe(request.failure()?.errorText || "Unknown transport failure"),
  }));
  const runs = path.join(root, "data", "runs");
  const before = hashes(runs);
  const runDirectory = (lob: string, id: string) => {
    expect(id).toMatch(/^[0-9a-f]{32}$/);
    return path.join(runs, lob, id);
  };
  return {
    root, marker, runs, before, runDirectory,
    frame: (lob: string, id: string, member: string) => parquet(path.join(runDirectory(lob, id), `${member}.parquet`)),
    configuration: (lob: string, id: string): unknown => JSON.parse(readFileSync(path.join(runDirectory(lob, id), "config.json"), "utf8")),
    async login(role: "admin" | "viewer" = "admin") {
      await page.goto("/login");
      await expect(page.getByLabel("Access token", { exact: true })).toHaveAttribute("type", "password");
      await expect(page.locator("#role")).toHaveCount(0);
      await submitPrivately(page, credentials[role]);
      await page.waitForURL(`${marker.frontend_origin}/`);
      await expect(page.locator("header").getByText(`browser-${role}`, { exact: true })).toBeVisible();
    },
    async guardOrigins() {
      await page.route("**/*", route => {
        const url = new URL(safe(route.request().url()));
        return [marker.frontend_origin, marker.api_url].includes(url.origin) && !exposed
          ? route.continue() : route.abort();
      });
    },
    async save(evidence: Row) {
      const stored = await page.evaluate(() => JSON.stringify([Object.entries(localStorage), Object.entries(sessionStorage), document.cookie]));
      expect(privateValues.some(value => stored.includes(value))).toBe(false);
      expect(exposed).toBe(false);
      expect(diagnostics.page_errors).toEqual([]);
      expect(diagnostics.request_failures.filter(item => item.error !== "net::ERR_ABORTED" || item.method !== "GET")).toEqual([]);
      unchanged(before, runs);
      writeFileSync(testInfo.outputPath("core-product-evidence.json"), JSON.stringify({
        source_sha: process.env.GITHUB_SHA ?? process.env.CORE_SOURCE_SHA ?? null,
        build_id: readFileSync(path.join(process.cwd(), ".next", "BUILD_ID"), "utf8").trim(),
        outcome: "passed", mode: "production", captures: "off", timezone: "America/Los_Angeles",
        browser: page.context().browser()?.version(), ...evidence, diagnostics,
        original_run_hashes: before, original_runs_unchanged: true, private_values_exposed: false,
      }, null, 2));
    },
  };
}

function response(page: Page, endpoint: string, method = "POST") {
  return page.waitForResponse(item =>
    new URL(item.url()).pathname === endpoint && item.request().method() === method);
}

async function upload(page: Page, label: string, name: string, contents: string) {
  await page.getByRole("button", { name: new RegExp(label) }).locator('input[type="file"]').setInputFiles({
    name, mimeType: name.endsWith(".yaml") ? "text/yaml" : "text/csv", buffer: Buffer.from(contents),
  });
}

async function onboarding(page: Page, lob: string, contents = history(), name = "weekly.csv"): Promise<AnalysisResponse> {
  await navigateWithinApp(page, "/data-onboarding");
  await page.getByLabel("LOB Name", { exact: true }).fill(lob);
  const pending = response(page, "/analyze");
  await upload(page, "Time Series Data", name, contents);
  const received = await pending;
  expect(received.status()).toBe(200);
  const analysis: AnalysisResponse = await received.json();
  await expect(page.getByRole("heading", { name: "Schema Detection", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Pipeline Execution", exact: true })).toBeVisible();
  return analysis;
}

async function downloadYaml(page: Page): Promise<string> {
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download YAML", exact: true }).click();
  const file = await pending;
  const downloaded = await file.path();
  if (!downloaded) throw new Error("The generated YAML download is unavailable");
  return readFileSync(downloaded, "utf8");
}

async function pipeline(page: Page, kind: "backtest" | "forecast") {
  const pending = response(page, `/pipeline/${kind}`);
  await page.getByRole("button", { name: kind === "backtest" ? "Run Backtest" : "Run Forecast", exact: true }).click();
  const received = await pending;
  const body: PipelineRunResponse = await received.json();
  expect(received.status(), JSON.stringify(body)).toBe(200);
  expect(body.status).toBe("completed");
  expect(body.run_type).toBe(kind);
  expect(body.run_id).toMatch(/^[0-9a-f]{32}$/);
  await expect(page.getByRole("heading", { name: kind === "backtest" ? "Backtest Result" : "Forecast Result", exact: true })).toBeVisible();
  return { http_status: received.status(), query: Object.fromEntries(new URL(received.url()).searchParams), body };
}

function intervals(rows: Row[], medianIsPoint = false) {
  for (const row of rows) {
    expect(number(row, "forecast_p10")).toBeLessThanOrEqual(number(row, "forecast_p50"));
    expect(number(row, "forecast_p50")).toBeLessThanOrEqual(number(row, "forecast_p90"));
    if (medianIsPoint) expect(number(row, "forecast_p50")).toBe(number(row, "forecast"));
  }
}

function naiveForecast(rows: Row[], horizon: number, daily = false, offset = 0) {
  const time = daily ? "date" : "week";
  const cycles = daily
    ? [[507, 514, 521, 508, 515, 522, 529], [243, 246, 249, 244, 247, 250, 253]]
    : [[204, 211, 218, 225, 212, 219, 226, 233], [92, 95, 98, 101, 96, 99, 102, 105]];
  const expected = cycles.flatMap((cycle, series) => Array.from({ length: horizon }, (_, index) => ({
    series_id: `${daily ? "DAILY" : "CORE"}_${series === 0 ? "A" : "B"}`,
    [time]: day(daily ? "2025-07-28" : "2025-12-29", index * (daily ? 1 : 7)),
    forecast: cycle[index % cycle.length] + offset,
  })));
  expect(ordered(project(rows, ["series_id", time, "forecast"]), "series_id", time)).toEqual(expected);
  expect(rows.every(row => row.model === "naive_seasonal")).toBe(true);
  intervals(rows, true);
}

function backtestOracle(rows: Row[], daily = false) {
  const naive = rows.filter(row => row.model_id === "naive_seasonal");
  const base = daily ? "2025-01-01" : "2024-01-01";
  const cadence = daily ? 1 : 7;
  const season = daily ? 7 : 52;
  const validation = daily ? 51 : 8;
  const expected: Row[] = [];
  // Independent integer arithmetic, with explicit origins rather than production CV/fold helpers.
  for (const origin of daily ? [105, 156] : [87, 95]) {
    for (let series = 0; series < 2; series++) {
      for (let step = 1; step <= validation; step++) expected.push({
        origin: day(base, origin * cadence),
        series_id: `${daily ? "DAILY" : "CORE"}_${series === 0 ? "A" : "B"}`,
        target_week: day(base, (origin + step) * cadence),
        forecast_step: step,
        actual: demand(series, origin + step, daily),
        forecast: demand(series, origin - season + 1 + ((step - 1) % season), daily),
      });
    }
  }
  const actual = project(naive, ["series_id", "target_week", "forecast_step", "actual", "forecast"])
    .map(row => ({ ...row, origin: day(text(row, "target_week"), -number(row, "forecast_step") * cadence) }));
  expect(ordered(actual, "series_id", "target_week")).toEqual(ordered(expected, "series_id", "target_week"));
  const absolute = naive.reduce((sum, row) => sum + Math.abs(number(row, "forecast") - number(row, "actual")), 0);
  const volume = naive.reduce((sum, row) => sum + Math.abs(number(row, "actual")), 0);
  const signed = naive.reduce((sum, row) => sum + number(row, "forecast") - number(row, "actual"), 0);
  const squared = naive.reduce((sum, row) => sum + (number(row, "forecast") - number(row, "actual")) ** 2, 0);
  expect([naive.length, absolute, volume, signed, squared]).toEqual(
    daily ? [204, 10024, 61910, -7560, 972712] : [32, 2496, 6992, -2496, 216320],
  );
  if (daily) {
    const zeros = naive.filter(row => row.actual === 0);
    expect(zeros).toHaveLength(4);
    expect(zeros.every(row => row.wmape === null && row.normalized_bias === null)).toBe(true);
  }
  return { observations: actual, absolute_error: absolute, actual_volume: volume, signed_error: signed, squared_error: squared };
}

async function forecastViewer(page: Page, lob: string, id: string): Promise<ForecastResponse> {
  await navigateWithinApp(page, "/forecast");
  const input = page.locator('main input[type="text"]').first();
  const pending = response(page, `/forecast/${lob}`, "GET");
  if (await input.inputValue() !== lob) await input.fill(lob);
  else await page.getByRole("button", { name: "Load Forecasts", exact: true }).click();
  const received = await pending;
  expect(received.status()).toBe(200);
  const body: ForecastResponse = await received.json();
  expect(body.run_id).toBe(id);
  await expect(page.getByTestId("forecast-run-id")).toHaveText(id);
  return body;
}

async function chartsAndTable(page: Page, series: string, first: string, value: string, last?: string, horizon = 8) {
  await page.locator("select").filter({ has: page.locator(`option[value="${series}"]`) }).selectOption(series);
  await expect(page.getByRole("heading", { name: `Forecast: ${series}`, exact: true })).toBeVisible();
  await expect(page.locator(".recharts-line-curve").first()).toBeVisible();
  await expect(page.locator(".js-plotly-plot")).toBeVisible();
  const summary = page.getByText("Forecast data", { exact: true });
  const data = summary.locator("..");
  if (await data.getAttribute("open") === null) await summary.click();
  await expect(data.getByRole("cell", { name: first, exact: true })).toBeVisible();
  await expect(data.getByRole("cell", { name: value, exact: true }).first()).toBeVisible();
  if (last) {
    for (let index = 1; index < Math.ceil(horizon / 10); index++) await data.getByRole("button").last().click();
    await expect(data.getByRole("cell", { name: last, exact: true })).toBeVisible();
  }
}

async function storedConfiguration(page: Page, lob: string, id: string): Promise<RunConfigResponse> {
  const pending = response(page, `/pipeline/runs/${lob}/${id}/config`, "GET");
  await page.getByRole("button", { name: "View stored configuration", exact: true }).click();
  const result: RunConfigResponse = await (await pending).json();
  expect(result.run_id).toBe(id);
  return result;
}

async function nextPaint(page: Page) {
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function pauseRequest(page: Page, apiUrl: string, endpoint: string, trigger: () => Promise<void>) {
  let original: Request | undefined;
  let observed!: () => void;
  let release!: () => void;
  const started = new Promise<void>(resolve => { observed = resolve; });
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route(`${apiUrl}${endpoint}*`, async route => {
    original = route.request();
    observed();
    await gate;
    await route.continue();
  }, { times: 1 });
  const pending = page.waitForResponse(item => item.request() === original);
  await trigger();
  await started;
  return {
    release,
    async finish() {
      release();
      const received = await pending;
      await received.finished();
      await nextPaint(page);
      return received;
    },
  };
}

test("core weekly planner journey verifies arithmetic, learned intent, exports, exact history and roles", async ({ page }, testInfo) => {
  const f = fixture(page, testInfo);
  const lob = "core_weekly";
  expect(digest(history())).toBe("f62b39cac0a4d40abc01e15cc093dda6a42cfe829e0c73afd6b6bf5cfbb776cf");
  await f.guardOrigins();
  await f.login();
  const analysis = await onboarding(page, lob);
  expect(analysis).toMatchObject({ n_rows: 208, n_series: 2, frequency: "W", time_column: "week", target_column: "quantity" });
  const downloaded = await downloadYaml(page);
  expect(downloaded).toBe(analysis.recommended_config_yaml);
  await upload(page, "Pipeline Configuration", "naive-eight.yaml", plan("naive_seasonal", 8));
  const backtest = await pipeline(page, "backtest");
  expect(backtest.body.champion_model).toBe("naive_seasonal");
  expect(backtest.body.best_wmape).toBeCloseTo(156 / 437, 13);
  await expect(page.getByText("35.7%", { exact: true })).toBeVisible();
  const arithmetic = backtestOracle(f.frame(lob, backtest.body.run_id, "metrics").rows);
  const first = await pipeline(page, "forecast");
  expect(first.query).toMatchObject({ model_id: "naive_seasonal", horizon: "8" });
  expect(first.body).toMatchObject({ series_count: 2, forecast_rows: 16 });
  expect(first.body.run_id).not.toBe(backtest.body.run_id);
  const original = f.frame(lob, first.body.run_id, "forecast");
  naiveForecast(original.rows, 8);
  const input = f.frame(lob, first.body.run_id, "input");
  const prepared = f.frame(lob, first.body.run_id, "prepared");
  expect(input.row_count).toBe(208);
  expect(ordered(project(prepared.rows, ["series_id", "week", "quantity"])))
    .toEqual(ordered(project(input.rows, ["series_id", "week", "quantity"])));
  const originalHashes = hashes(f.runDirectory(lob, first.body.run_id));
  const viewed = await forecastViewer(page, lob, first.body.run_id);
  expect(viewed.points).toHaveLength(16);
  expect(viewed.forecast_origin).toBe("2025-12-22");
  await chartsAndTable(page, "CORE_A", "2025-12-29", "204.00");
  const config = await storedConfiguration(page, lob, first.body.run_id);
  expect(config).toMatchObject({ config_hash: viewed.config_hash, configuration: { settings: { forecast: { horizon_weeks: 8 } } } });

  const leaderboardRead = response(page, `/metrics/leaderboard/${lob}`, "GET");
  await navigateWithinApp(page, "/backtest");
  const leaderboard: LeaderboardResponse = await (await leaderboardRead).json();
  expect(leaderboard.run_id).toBe(backtest.body.run_id);
  await expect(page.getByRole("table", { name: "Stored model leaderboard" })).toContainText("35.7%");
  await expect(page.locator(".recharts-bar-rectangle").first()).toBeVisible();
  await navigateWithinApp(page, "/sop");
  await expect(page.getByText("16.2%", { exact: true })).toHaveCount(0);
  await expect(page.getByText(/Summary metrics are not connected/)).toBeVisible();
  const exports: Row[] = [];
  for (const [label, report] of [
    ["Export Leaderboard", "leaderboard"], ["Export Bias Report", "bias-report"], ["Export Forecast vs Actual", "forecast-actual"],
  ]) {
    const pending = response(page, `/governance/export/${report}`);
    await page.getByRole("button", { name: label, exact: true }).click();
    const received = await pending;
    expect(received.status()).toBe(200);
    const body: BIExportResponse = await received.json();
    const directory = path.resolve(f.root, "data", "bi_exports");
    const file = path.resolve(directory, body.export_path);
    expect(file.startsWith(directory + path.sep)).toBe(true);
    const contents = parquet(file);
    expect(body.status).toBe("exported");
    if (report === "leaderboard") {
      expect(body.run_id).toBe(backtest.body.run_id);
      expect(contents.rows.map(({ source_run_id, actuals_run_id, ...row }) => {
        expect(source_run_id).toBe(backtest.body.run_id);
        expect(actuals_run_id).toBeNull();
        return row;
      })).toEqual(backtest.body.leaderboard);
    } else if (report === "bias-report") expect(contents.row_count).toBe(2);
    else {
      expect(body.run_id).toBe(first.body.run_id);
      expect(contents.row_count).toBe(0);
    }
    exports.push({ outcome: "passed", http_status: received.status(), response: body, file_sha256: digest(readFileSync(file)), contents });
  }

  await onboarding(page, lob);
  await upload(page, "Pipeline Configuration", "naive-four.yaml", plan("naive_seasonal", 4));
  const shorter = await pipeline(page, "forecast");
  expect(shorter.query.model_id).toBeUndefined();
  expect(shorter.query.horizon).toBeUndefined();
  naiveForecast(f.frame(lob, shorter.body.run_id, "forecast").rows, 4);
  await upload(page, "Pipeline Data File", "offset.csv", history(false, 50));
  await expect(page.getByRole("heading", { name: "Forecast Result", exact: true })).toHaveCount(0);
  const changed = await pipeline(page, "forecast");
  naiveForecast(f.frame(lob, changed.body.run_id, "forecast").rows, 4, false, 50);
  await upload(page, "Pipeline Data File", "weekly-again.csv", history());
  await upload(page, "Pipeline Configuration", "ets-four.yaml", plan("auto_ets", 4));
  const learnedBacktest = await pipeline(page, "backtest");
  expect(learnedBacktest.body.champion_model).toBe("auto_ets");
  const learned = await pipeline(page, "forecast");
  const learnedFrame = f.frame(lob, learned.body.run_id, "forecast");
  expect(learned.body.execution).toMatchObject({ model_id: "auto_ets", horizon_periods: 4 });
  expect(learnedFrame.row_count).toBe(8);
  expect(learnedFrame.rows.every(row => row.model === "auto_ets" && Number.isFinite(row.forecast))).toBe(true);
  intervals(learnedFrame.rows);
  await forecastViewer(page, lob, learned.body.run_id);
  await page.reload();
  await page.waitForURL(`${f.marker.frontend_origin}/login`);
  await f.login();
  await forecastViewer(page, lob, learned.body.run_id);
  const exact = page.waitForResponse(item => {
    const url = new URL(item.url());
    return url.pathname === `/forecast/${lob}` && url.searchParams.get("run_id") === first.body.run_id;
  });
  await page.getByLabel("Forecast run", { exact: true }).selectOption(first.body.run_id);
  const older: ForecastResponse = await (await exact).json();
  expect(older).toEqual(viewed);
  await expect(page.getByTestId("forecast-run-id")).toHaveText(first.body.run_id);
  await chartsAndTable(page, "CORE_A", "2025-12-29", "204.00");
  expect(await storedConfiguration(page, lob, first.body.run_id)).toEqual(config);
  expect(hashes(f.runDirectory(lob, first.body.run_id))).toEqual(originalHashes);
  await page.getByTitle("Logout", { exact: true }).click();
  await page.waitForURL(`${f.marker.frontend_origin}/login`);
  await f.login("viewer");
  await navigateWithinApp(page, "/data-onboarding");
  await page.getByLabel("LOB Name", { exact: true }).fill(lob);
  const denied = response(page, "/analyze");
  await upload(page, "Time Series Data", "viewer.csv", history());
  const denial = await denied;
  expect(denial.status()).toBe(403);
  await expect(page.getByText(await denial.text(), { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Pipeline Execution", exact: true })).toHaveCount(0);
  await f.save({
    journey: "weekly custom plan", analysis, downloaded_yaml_sha256: digest(downloaded),
    backtest, arithmetic, first_forecast: first, full_forecast: original, viewed, stored_configuration: config,
    exports, export_boundary: "Server-side reports, not a forecast download; future dates have no same-run actual overlap.",
    shorter, changed, learned_backtest: learnedBacktest, learned_forecast: learned,
    learned_full_forecast: learnedFrame, exact_older_run: older, original_member_hashes: originalHashes,
    viewer_denial: { outcome: "passed", http_status: 403 },
  });
});

test("unchanged generated plans and downloaded YAML execute the supported weekly and non-fixed leaf workflow", async ({ page }, testInfo) => {
  const f = fixture(page, testInfo);
  const lob = "core_generated";
  await f.guardOrigins();
  await f.login();
  const analysis = await onboarding(page, lob);
  const firstBacktest = await pipeline(page, "backtest");
  expect(firstBacktest.body.execution.config.cleansing_enabled).toBe(false);
  expect(firstBacktest.body.champion_model).toBeTruthy();
  const firstForecast = await pipeline(page, "forecast");
  expect(firstForecast.body.execution.model_id).toBe(firstBacktest.body.champion_model);
  expect(firstForecast.body).toMatchObject({ forecast_rows: 50, series_count: 2 });
  const yaml = await downloadYaml(page);
  expect(yaml).toBe(analysis.recommended_config_yaml);
  await upload(page, "Pipeline Configuration", "generated-roundtrip.yaml", yaml);
  const secondBacktest = await pipeline(page, "backtest");
  expect(secondBacktest.body.champion_model).toBeTruthy();
  const secondForecast = await pipeline(page, "forecast");
  expect(secondForecast.body.execution.model_id).toBe(secondBacktest.body.champion_model);
  expect(secondForecast.body.execution.horizon_periods).toBe(25);
  expect(f.frame(lob, secondForecast.body.run_id, "forecast").row_count).toBe(50);
  for (const id of [firstBacktest.body.run_id, secondBacktest.body.run_id, secondForecast.body.run_id]) {
    expect(f.configuration(lob, id)).toMatchObject({
      settings: { data_quality: { cleansing: { enabled: false }, structural_breaks: { enabled: false } }, reconciliation: { method: "bottom_up" } },
    });
  }
  const hierarchyLob = "core_generated_hierarchy";
  const hierarchy = await onboarding(page, hierarchyLob, productHierarchy(), "product-hierarchy.csv");
  expect(hierarchy).toMatchObject({ n_rows: 832, n_series: 8 });
  const hierarchyYaml = await downloadYaml(page);
  expect(hierarchyYaml).toBe(hierarchy.recommended_config_yaml);
  const direct = await pipeline(page, "forecast");
  await upload(page, "Pipeline Configuration", "hierarchy-roundtrip.yaml", hierarchyYaml);
  const roundtrip = await pipeline(page, "forecast");
  const full = f.frame(hierarchyLob, roundtrip.body.run_id, "forecast");
  expect([direct.body.forecast_rows, roundtrip.body.forecast_rows, full.row_count]).toEqual([200, 200, 200]);
  expect(new Set(full.rows.map(row => row.product_id)).size).toBe(8);
  expect(new Set(full.rows.map(row => `${row.product_id}:${row.week}`)).size).toBe(200);
  intervals(full.rows, true);
  expect(f.configuration(hierarchyLob, roundtrip.body.run_id)).toMatchObject({
    settings: { reconciliation: { method: "bottom_up" }, hierarchies: [{ fixed: false, levels: ["category", "product_id"] }] },
  });
  await f.save({
    journey: "unchanged generated plans", analysis, yaml_sha256: digest(yaml),
    default_backtest: firstBacktest, default_forecast: firstForecast,
    roundtrip_backtest: secondBacktest, roundtrip_forecast: secondForecast,
    hierarchy_analysis: hierarchy, hierarchy_default_forecast: direct,
    hierarchy_roundtrip_forecast: roundtrip, full_hierarchy_forecast: full,
  });
});

test("daily generated plan preserves date columns, zero demand, cohort scoring and full forecast calendar", async ({ page }, testInfo) => {
  const f = fixture(page, testInfo);
  const lob = "core_daily";
  expect(digest(history(true))).toBe("157e315091dce8762054d8651ea323afe0a0d5e2a5eebcc4bf49d04b8e332fe1");
  await f.guardOrigins();
  await f.login();
  const analysis = await onboarding(page, lob, history(true), "daily.csv");
  expect(analysis).toMatchObject({ n_rows: 416, n_series: 2, frequency: "D", time_column: "date", target_column: "quantity" });
  const yaml = await downloadYaml(page);
  expect(yaml).toBe(analysis.recommended_config_yaml);
  const naive = await pipeline(page, "forecast");
  expect(naive.query.model_id).toBeUndefined();
  expect(naive.body.execution).toMatchObject({
    model_id: "naive_seasonal", horizon_periods: 51,
    config: { frequency: "D", season_length: 7, time_column: "date", backtest_n_folds: 2, backtest_val_periods: 51 },
  });
  const full = f.frame(lob, naive.body.run_id, "forecast");
  naiveForecast(full.rows, 51, true);
  expect(full.rows.reduce((sum, row) => sum + number(row, "forecast"), 0)).toBe(38946);
  const viewed = await forecastViewer(page, lob, naive.body.run_id);
  expect(viewed.points).toHaveLength(102);
  expect(viewed.forecast_origin).toBe("2025-07-27");
  await chartsAndTable(page, "DAILY_A", "2025-07-28", "507.00", "2025-09-16", 51);
  const config = await storedConfiguration(page, lob, naive.body.run_id);
  expect(config).toMatchObject({
    config_hash: viewed.config_hash,
    configuration: { settings: { forecast: { frequency: "D", time_column: "date", horizon_weeks: 51 } } },
  });
  await onboarding(page, lob, history(true), "daily-roundtrip.csv");
  await upload(page, "Pipeline Configuration", "daily-generated.yaml", yaml);
  const backtest = await pipeline(page, "backtest");
  expect(backtest.body.execution.config).toMatchObject({
    frequency: "D", season_length: 7, time_column: "date", backtest_n_folds: 2, backtest_val_periods: 51, champion_granularity: "series",
  });
  const arithmetic = backtestOracle(f.frame(lob, backtest.body.run_id, "metrics").rows, true);
  expect(backtest.body.best_wmape).toBeNull();
  for (const [id, score] of [["DAILY_A", 6698 / 41991], ["DAILY_B", 3326 / 19919]] as const) {
    const row = backtest.body.leaderboard?.find(row => row.model_id === "naive_seasonal" && row.group_key === id);
    expect(row).toMatchObject({ n_observations: 102, competition_scope: "series_id" });
    if (!row) throw new Error("Missing daily cohort leaderboard row");
    expect(number(row, "wmape")).toBeCloseTo(score, 13);
  }
  expect(backtest.body.champion_model).toBeTruthy();
  const champion = await pipeline(page, "forecast");
  expect(champion.body.execution.model_id).toBe(backtest.body.champion_model);
  expect(champion.body.forecast_rows).toBe(102);
  const boardRead = response(page, `/metrics/leaderboard/${lob}`, "GET");
  await navigateWithinApp(page, "/backtest");
  expect((await (await boardRead).json()).run_id).toBe(backtest.body.run_id);
  await expect(page.getByTestId("scoring-provenance")).toContainText("Scope: series_id");
  await expect(page.getByText("Per competition", { exact: true })).toBeVisible();
  const board = page.getByRole("table", { name: "Stored model leaderboard" });
  await expect(board.locator('[data-model="naive_seasonal"][data-group="DAILY_A"]')).toContainText("16.0%");
  await expect(board.locator('[data-model="naive_seasonal"][data-group="DAILY_B"]')).toContainText("16.7%");
  await f.save({
    journey: "daily generated plan", input_sha256: digest(history(true)), analysis,
    generated_yaml_sha256: digest(yaml), naive_forecast: naive, full_naive_forecast: full,
    viewed, stored_configuration: config, backtest, arithmetic, champion_forecast: champion,
    pooled_ratio_for_audit_only: "5012/30955", selection_scope: "series_id", zero_actual_rows_preserved: 4,
  });
});

test("real late Analyze and pipeline responses cannot replace current intent or turn bad input into success", async ({ page }, testInfo) => {
  const f = fixture(page, testInfo);
  const lob = "core_ownership";
  await f.guardOrigins();
  await f.login();
  await onboarding(page, lob);
  const counts = { backtest: 0, forecast: 0 };
  page.on("request", request => {
    if (request.method() !== "POST") return;
    const name = new URL(request.url()).pathname;
    if (name === "/pipeline/backtest") counts.backtest++;
    if (name === "/pipeline/forecast") counts.forecast++;
  });
  const analysisRaces: Row[] = [];
  for (const [label, old, status] of [["success", history(), 200], ["error", history(false, 0, "target"), 400]] as const) {
    const delayed = await pauseRequest(page, f.marker.api_url, "/analyze",
      () => upload(page, "Time Series Data", `older-${label}.csv`, old));
    try {
      const currentRead = response(page, "/analyze");
      await upload(page, "Time Series Data", `current-daily-${label}.csv`, history(true));
      const currentResponse = await currentRead;
      expect(currentResponse.status()).toBe(200);
      const current: AnalysisResponse = await currentResponse.json();
      expect(current).toMatchObject({ frequency: "D", time_column: "date", n_rows: 416 });
      const older = await delayed.finish();
      expect(older.status()).toBe(status);
      await expect(page.getByText("Time Column", { exact: true }).locator("..")).toContainText("date");
      await expect(page.getByText(/Could not detect a numeric target column|Unable to reach the API server/)).toHaveCount(0);
      expect(await downloadYaml(page)).toBe(current.recommended_config_yaml);
      analysisRaces.push({ outcome: "passed", older_http_status: status, current_frequency: "D", yaml_sha256: digest(current.recommended_config_yaml) });
    } finally { delayed.release(); }
  }
  await onboarding(page, lob, history(), "current-weekly.csv");
  await upload(page, "Pipeline Configuration", "naive-eight.yaml", plan("naive_seasonal", 8));
  const delayedBacktest = await pauseRequest(page, f.marker.api_url, "/pipeline/backtest",
    () => page.getByRole("button", { name: "Run Backtest", exact: true }).click());
  let oldBacktest: PipelineRunResponse;
  try {
    await upload(page, "Pipeline Configuration", "ets-four.yaml", plan("auto_ets", 4));
    await upload(page, "Pipeline Data File", "offset.csv", history(false, 50));
    const received = await delayedBacktest.finish();
    expect(received.status()).toBe(200);
    oldBacktest = await received.json();
    expect(oldBacktest.champion_model).toBe("naive_seasonal");
    await expect(page.getByRole("heading", { name: "Backtest Result", exact: true })).toHaveCount(0);
  } finally { delayedBacktest.release(); }
  const currentLearned = await pipeline(page, "forecast");
  expect(currentLearned.query.model_id).toBeUndefined();
  expect(currentLearned.query.horizon).toBeUndefined();
  expect(currentLearned.body.execution).toMatchObject({ model_id: "auto_ets", horizon_periods: 4, model_source: "config" });
  await upload(page, "Pipeline Configuration", "naive-eight-again.yaml", plan("naive_seasonal", 8));
  await upload(page, "Pipeline Data File", "weekly-again.csv", history());
  const delayedForecast = await pauseRequest(page, f.marker.api_url, "/pipeline/forecast",
    () => page.getByRole("button", { name: "Run Forecast", exact: true }).click());
  let oldForecast: PipelineRunResponse;
  try {
    await upload(page, "Pipeline Configuration", "naive-four.yaml", plan("naive_seasonal", 4));
    await upload(page, "Pipeline Data File", "offset-again.csv", history(false, 50));
    const received = await delayedForecast.finish();
    expect(received.status()).toBe(200);
    oldForecast = await received.json();
    expect(oldForecast.forecast_rows).toBe(16);
    await expect(page.getByRole("heading", { name: "Forecast Result", exact: true })).toHaveCount(0);
  } finally { delayedForecast.release(); }
  const current = await pipeline(page, "forecast");
  naiveForecast(f.frame(lob, current.body.run_id, "forecast").rows, 4, false, 50);
  const beforeFailures = hashes(f.runs);
  await upload(page, "Pipeline Data File", "missing-target.csv", history(false, 0, "target"));
  const missingRun = response(page, "/pipeline/forecast");
  await page.getByRole("button", { name: "Run Forecast", exact: true }).click();
  const failedRun = await missingRun;
  expect(failedRun.status()).toBe(500);
  const failedBody = await failedRun.json();
  expect(failedBody.detail).toMatchObject({ code: "forecast_failed", outcome: "not_committed" });
  await expect(page.getByText(/forecast_failed/)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Forecast Result", exact: true })).toHaveCount(0);
  expect(hashes(f.runs)).toEqual(beforeFailures);
  await upload(page, "Pipeline Data File", "valid-weekly.csv", history());
  await upload(page, "Pipeline Configuration", "invalid.yaml", "forecast:\n  horizon_periods: 0\n");
  const invalidConfig = response(page, "/pipeline/forecast");
  await page.getByRole("button", { name: "Run Forecast", exact: true }).click();
  expect((await invalidConfig).status()).toBe(400);
  await expect(page.getByText(/Invalid config/)).toBeVisible();
  const admissions: Row[] = [];
  for (const [name, csv, message] of [
    ["malformed.csv", 'series_id,week,quantity\n"CORE_A,2024-01-01,10\n', /Failed to read uploaded file/],
    ["missing-demand.csv", history(false, 0, "target"), /numeric target column/],
    ["missing-time.csv", history(false, 0, "time"), /time\/date column/],
  ] as const) {
    const pending = response(page, "/analyze");
    await upload(page, "Time Series Data", name, csv);
    const received = await pending;
    expect(received.status()).toBe(400);
    await expect(page.getByText(message).first()).toBeVisible();
    await expect(page.getByRole("heading", { name: "Pipeline Execution", exact: true })).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "Schema Detection", exact: true })).toHaveCount(0);
    admissions.push({ outcome: "passed", input: name, http_status: 400, response: await received.json() });
  }
  expect(hashes(f.runs)).toEqual(beforeFailures);
  expect(counts).toEqual({ backtest: 1, forecast: 5 });
  await f.save({
    journey: "real response ownership and input rejection", analysis_races: analysisRaces,
    old_backtest: oldBacktest, current_learned_forecast: currentLearned,
    old_forecast: oldForecast, current_forecast: current,
    failed_pipeline: { outcome: "passed", http_status: 500, response: failedBody },
    invalid_config: { outcome: "passed", http_status: 400 }, admissions, post_counts: counts,
    transport: "Real requests were delayed then continued; no response was mocked or fulfilled.",
    failures_preserve_all_committed_members: true,
  });
});
