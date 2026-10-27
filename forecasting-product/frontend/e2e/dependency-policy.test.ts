import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, test } from "vitest";

interface LockedPackage {
  version?: string;
  dev?: boolean;
  resolved?: string;
  integrity?: string;
  link?: boolean;
  dependencies?: Record<string, string>;
}

const lock: { packages: Record<string, LockedPackage> } = JSON.parse(
  readFileSync(resolve(process.cwd(), "package-lock.json"), "utf8"),
);

const developmentExceptions = [
  {
    path: "node_modules/eslint-plugin-react/node_modules/resolve",
    version: "2.0.0-next.6",
    parent: "node_modules/eslint-plugin-react",
    dependency: "resolve",
    range: "^2.0.0-next.5",
  },
  {
    path: "node_modules/gensync",
    version: "1.0.0-beta.2",
    parent: "node_modules/@babel/core",
    dependency: "gensync",
    range: "^1.0.0-beta.2",
  },
];

describe("dependency release policy", () => {
  test("permits only reviewed upstream-required development release labels", () => {
    const prereleases = Object.entries(lock.packages)
      .filter(([, entry]) => entry.version && /^\d+\.\d+\.\d+-/.test(entry.version))
      .map(([path, entry]) => ({ path, version: entry.version }))
      .sort((a, b) => a.path.localeCompare(b.path));
    expect(prereleases).toEqual(
      developmentExceptions.map(({ path, version }) => ({ path, version }))
        .sort((a, b) => a.path.localeCompare(b.path)),
    );
    for (const exception of developmentExceptions) {
      expect(lock.packages[exception.path].dev).toBe(true);
      expect(lock.packages[exception.parent].dev).toBe(true);
      expect(lock.packages[exception.parent].dependencies?.[exception.dependency])
        .toBe(exception.range);
    }
  });

  test("retains artifact integrity without local or mirror-specific URLs", () => {
    for (const [path, entry] of Object.entries(lock.packages)) {
      if (!path) continue;
      expect(entry.resolved, path).toBeUndefined();
      expect(entry.link, path).not.toBe(true);
      expect(entry.integrity, path).toMatch(/^sha512-[A-Za-z0-9+/]+={0,2}$/);
    }
  });
});
