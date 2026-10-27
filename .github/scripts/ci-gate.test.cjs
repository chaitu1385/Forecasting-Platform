"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const { assertSuccessfulJobs, required } = require("./ci-gate.cjs");

const successful = () => Object.fromEntries(required.map((name) => [name, { result: "success" }]));

test("the complete successful job set passes", () => {
  assert.doesNotThrow(() => assertSuccessfulJobs(successful()));
});

for (const name of required) {
  for (const result of ["failure", "cancelled", "skipped", undefined]) {
    test(`${name}=${result} cannot pass the required gate`, () => {
      const needs = successful();
      needs[name] = { result };
      assert.throws(() => assertSuccessfulJobs(needs), /not successful/);
    });
  }
}

test("missing, empty, malformed, or unexpected job sets fail closed", () => {
  for (const value of [null, [], {}, { ...successful(), unexpected: { result: "success" } }]) {
    assert.throws(() => assertSuccessfulJobs(value), /complete required job set/);
  }
  const needs = successful();
  delete needs.test;
  assert.throws(() => assertSuccessfulJobs(needs), /complete required job set/);
});
