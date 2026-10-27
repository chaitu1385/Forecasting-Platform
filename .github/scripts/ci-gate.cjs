"use strict";

const required = [
  "lint", "validation", "forecast-intent", "process-safety",
  "test", "build", "frontend", "automation",
];

function assertSuccessfulJobs(needs) {
  if (!needs || typeof needs !== "object" || Array.isArray(needs)
      || Object.keys(needs).length !== required.length) {
    throw new Error("CI gate did not receive the complete required job set");
  }
  const failed = required.filter((name) => needs[name]?.result !== "success");
  if (failed.length) {
    throw new Error(`Required checks not successful: ${failed.map((name) => `${name}=${needs[name]?.result ?? "missing"}`).join(", ")}`);
  }
}

if (require.main === module) {
  assertSuccessfulJobs(JSON.parse(process.env.CI_NEEDS));
  console.log("All required CI jobs succeeded, including the full backend matrix and Docker smoke.");
}

module.exports = { assertSuccessfulJobs, required };
