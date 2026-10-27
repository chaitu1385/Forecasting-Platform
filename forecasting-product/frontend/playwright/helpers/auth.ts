import { expect, type Page } from "@playwright/test";
import type { Role } from "../../src/lib/types";

export async function loginViaUI(page: Page, username: string, role: Role) {
  await page.goto("/login");
  await expect(page.getByText("Explicit demo mode:", { exact: false })).toBeVisible();
  await page.getByLabel("Demo username", { exact: true }).fill(username);
  await page.getByLabel("Role", { exact: true }).selectOption(role);
  await page.getByRole("button", { name: "Sign In to Demo", exact: true }).click();
  await page.waitForURL("/", { timeout: 30_000 });
  await expect(page.locator("header").getByText(username, { exact: true })).toBeVisible();
}

export async function loginDemo(page: Page, opts: { username?: string; role?: Role } = {}) {
  await loginViaUI(page, opts.username ?? "test_admin", opts.role ?? "admin");
}

export async function navigateWithinApp(page: Page, targetPath: string) {
  if (new URL(page.url()).pathname === targetPath) return;
  await page.locator(`aside a[href="${targetPath}"]`).click();
  await page.waitForURL(targetPath);
}

export async function clearAuth(page: Page) {
  await page.getByTitle("Logout", { exact: true }).click();
  await page.waitForURL("/login");
}
