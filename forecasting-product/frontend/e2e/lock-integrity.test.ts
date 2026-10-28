import { createHash } from "node:crypto";
import { describe, expect, test } from "vitest";
import { assertAuthoritative, assertGraphUnchanged, assertOriginal } from "../scripts/lock-integrity.cjs";

const integrity = (text: string, algorithm: "sha1" | "sha512") =>
  `${algorithm}-${createHash(algorithm).update(text).digest("base64")}`;

describe("lock integrity negative controls", () => {
  test("rejects a corrupted original tarball rather than signing its replacement", () => {
    expect(() => assertOriginal(Buffer.from("corrupted"), integrity("original", "sha1")))
      .toThrow("Original tarball integrity mismatch");
  });

  test("rejects a digest that differs from authoritative registry data", () => {
    expect(() => assertAuthoritative(integrity("mirror", "sha512"), integrity("official", "sha512")))
      .toThrow("authoritative upstream SHA-512 mismatch");
  });

  test.each(["version", "dependencies", "os", "cpu", "optional"] as const)(
    "rejects an accidental %s graph change",
    (field) => {
      const before = {
        lockfileVersion: 3,
        packages: {
          "node_modules/example": {
            version: "1.0.0", dependencies: { child: "^2.0.0" },
            os: ["linux"], cpu: ["x64"], optional: false,
            integrity: integrity("original", "sha1"),
          },
        },
      };
      const after = structuredClone(before);
      const entry = after.packages["node_modules/example"];
      if (field === "version") entry.version = "2.0.0";
      if (field === "dependencies") entry.dependencies.child = "^3.0.0";
      if (field === "os") entry.os = ["win32"];
      if (field === "cpu") entry.cpu = ["arm64"];
      if (field === "optional") entry.optional = true;
      expect(() => assertGraphUnchanged(before, after)).toThrow("graph changed");
    },
  );

  test("allows only integrity strengthening while preserving the complete graph", () => {
    const before = { packages: { example: { version: "1.0.0", integrity: integrity("same", "sha1") } } };
    const after = structuredClone(before);
    after.packages.example.integrity = integrity("same", "sha512");
    expect(() => assertGraphUnchanged(before, after)).not.toThrow();
    expect(() => assertOriginal(Buffer.from("same"), before.packages.example.integrity)).not.toThrow();
    expect(() => assertAuthoritative(after.packages.example.integrity, integrity("same", "sha512"))).not.toThrow();
  });
});
