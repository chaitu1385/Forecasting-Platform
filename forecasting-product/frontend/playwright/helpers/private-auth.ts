import type { Page } from "@playwright/test";

// Evaluate rather than fill: Playwright action diagnostics must never contain a token argument.
export async function submitPrivately(page: Page, token: string) {
  try {
    await page.evaluate(value => {
      const input = document.querySelector("#access-token");
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      if (!(input instanceof HTMLInputElement) || input.type !== "password" || !setter) {
        throw new Error("Private credential input is unavailable");
      }
      setter.call(input, value);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }, token);
    await page.getByRole("button", { name: "Sign In", exact: true }).click();
  } catch {
    await page.evaluate(() => document.querySelector("#access-token")?.remove());
    throw new Error("Private credential submission failed");
  }
}
