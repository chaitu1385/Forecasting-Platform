import { readdirSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

vi.mock("@playwright/test", () => ({
  defineConfig: (config: unknown) => config,
  devices: { "Desktop Chrome": {} },
}));

async function configuration(suite?: string) {
  vi.resetModules();
  vi.stubEnv("AUTH_E2E_SUITE", suite ?? "");
  return (await import("../playwright.auth.config")).default;
}

function pattern(value: unknown): RegExp {
  if (!(value instanceof RegExp)) throw new Error("Expected an explicit suite selector");
  return value;
}

beforeEach(() => {
  vi.stubEnv("AUTH_E2E_ROOT", path.join(os.tmpdir(), "auth-config-selection-only"));
  vi.stubEnv("AUTH_E2E_FRONTEND_MODE", "production");
  vi.stubEnv("DEBUG", "");
  vi.stubEnv("PWDEBUG", "");
});

afterEach(() => vi.unstubAllEnvs());

it("defaults to the existing contracts without silently running core cases in the same rate bucket", async () => {
  const config = await configuration();
  expect(config.testMatch).toBeUndefined();
  expect(pattern(config.testIgnore).test(path.join("auth", "core-product.spec.ts"))).toBe(true);
});

it("selects only the core file for its fresh fixture", async () => {
  const config = await configuration("core");
  expect(config.testIgnore).toBeUndefined();
  const match = pattern(config.testMatch);
  expect(match.test(path.join("auth", "core-product.spec.ts"))).toBe(true);
  expect(match.test(path.join("auth", "production.spec.ts"))).toBe(false);
});

it("partitions every private spec exactly once while retaining authentication and browser guards", async () => {
  const contracts = await configuration("contracts");
  const core = await configuration("core");
  const files = readdirSync(path.join(process.cwd(), "playwright", "auth"))
    .filter(file => file.endsWith(".spec.ts")).sort();
  const contractFiles = files.filter(file => !pattern(contracts.testIgnore).test(file));
  const coreFiles = files.filter(file => pattern(core.testMatch).test(file));
  expect(coreFiles).toEqual(["core-product.spec.ts"]);
  expect([...contractFiles, ...coreFiles].sort()).toEqual(files);
  expect(contractFiles.filter(file => coreFiles.includes(file))).toEqual([]);
  for (const config of [contracts, core]) {
    expect(config.retries).toBe(0);
    expect(config.workers).toBe(1);
    expect(config.timeout).toBe(180_000);
    expect(config.globalTimeout).toBe(360_000);
    expect(config.use).toMatchObject({ trace: "off", screenshot: "off", video: "off" });
    expect(config.webServer).toEqual(expect.arrayContaining([
      expect.objectContaining({ reuseExistingServer: false }),
    ]));
  }
});

it("rejects unknown selection rather than passing an accidentally empty suite", async () => {
  await expect(configuration("all")).rejects.toThrow("AUTH_E2E_SUITE must be contracts or core");
});
