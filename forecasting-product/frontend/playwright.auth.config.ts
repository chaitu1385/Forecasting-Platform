import { defineConfig, devices } from "@playwright/test";
import path from "node:path";

const root = process.env.AUTH_E2E_ROOT;
if (!root || !path.isAbsolute(root)) throw new Error("AUTH_E2E_ROOT must be a fresh absolute temporary directory outside the repository");
if (process.env.DEBUG || process.env.PWDEBUG) throw new Error("Credential checks cannot run with protocol/debug logging enabled");
const apiPort = Number(process.env.AUTH_E2E_API_PORT || 8197);
const frontendPort = Number(process.env.AUTH_E2E_FRONTEND_PORT || 3197);
if (![apiPort, frontendPort].every(port => Number.isInteger(port) && port > 1024 && port < 65536)) {
  throw new Error("Authentication fixture ports must be valid unprivileged ports");
}
const apiUrl = `http://127.0.0.1:${apiPort}`;
const frontendUrl = `http://127.0.0.1:${frontendPort}`;
const frontendMode = process.env.AUTH_E2E_FRONTEND_MODE || "development";
if (!["development", "production"].includes(frontendMode)) {
  throw new Error("AUTH_E2E_FRONTEND_MODE must be development or production");
}
function quote(value: string) {
  if (process.platform === "win32") {
    if (value.includes('"')) throw new Error("Unsupported fixture path");
    return `"${value}"`;
  }
  return `'${value.replace(/'/g, "'\\''")}'`;
}

export default defineConfig({
  testDir: "./playwright/auth",
  fullyParallel: false,
  forbidOnly: true,
  retries: 0,
  workers: 1,
  reporter: "line",
  timeout: 180_000,
  globalTimeout: 360_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: frontendUrl,
    trace: "off",
    screenshot: "off",
    video: "off",
    launchOptions: { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: [
        quote(process.env.AUTH_E2E_PYTHON || "python"),
        "playwright/fixtures/authentication_server.py",
        "--port", String(apiPort),
        "--data-dir", quote(path.join(root, "data")),
        "--credentials-dir", quote(path.join(root, "credentials")),
        "--frontend-origin", frontendUrl,
      ].join(" "),
      url: `${apiUrl}/health`,
      reuseExistingServer: false,
      timeout: 90_000,
      env: {
        POLARS_MAX_THREADS: "2", OMP_NUM_THREADS: "1", OPENBLAS_NUM_THREADS: "1",
        MKL_NUM_THREADS: "1", NUMBA_NUM_THREADS: "1", JWT_SECRET_KEY: "",
      },
    },
    {
      command: frontendMode === "production"
        ? `node -e "require('node:fs').cpSync('.next/static', '.next/standalone/.next/static', { recursive: true })" && node .next/standalone/server.js`
        : `node node_modules/next/dist/bin/next dev --webpack --hostname 127.0.0.1 --port ${frontendPort}`,
      url: `${frontendUrl}/login`,
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        NEXT_PUBLIC_API_URL: apiUrl, NEXT_TELEMETRY_DISABLED: "1", JWT_SECRET_KEY: "",
        HOSTNAME: "127.0.0.1", PORT: String(frontendPort),
      },
    },
  ],
});
