import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.PLAYWRIGHT_RUNTIME_PORT || 3285);
if (!Number.isInteger(port) || port <= 1024 || port >= 65536) {
  throw new Error("PLAYWRIGHT_RUNTIME_PORT must be a valid unprivileged port");
}
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./playwright/runtime",
  fullyParallel: false,
  forbidOnly: true,
  retries: 0,
  workers: 1,
  reporter: "line",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL,
    viewport: { width: 1440, height: 1100 },
    trace: "off",
    screenshot: "only-on-failure",
    launchOptions: { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "node .next/standalone/server.js",
    url: `${baseURL}/login`,
    reuseExistingServer: false,
    timeout: 60_000,
    env: { HOSTNAME: "127.0.0.1", PORT: String(port), NEXT_TELEMETRY_DISABLED: "1" },
  },
});
