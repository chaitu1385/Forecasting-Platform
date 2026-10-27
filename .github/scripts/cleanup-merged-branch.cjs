"use strict";

const { execFileSync } = require("node:child_process");

function runGit(args) {
  return execFileSync("git", args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
}

function deleteWithLease(candidate, git = runGit) {
  const ref = `refs/heads/${candidate.branch}`;
  if (!/^[0-9a-f]{40}$/i.test(candidate.sha)) {
    throw new Error("A verified full commit SHA is required for conditional deletion");
  }
  git(["check-ref-format", ref]);
  // An explicit expected SHA protects new work pushed after the API checks.
  return git(["push", "--porcelain", `--force-with-lease=${ref}:${candidate.sha}`, "origin", `:${ref}`]);
}

async function inspectCandidate({ github, context, pullNumber }) {
  const params = { owner: context.repo.owner, repo: context.repo.repo };
  const { data: repository } = await github.rest.repos.get(params);
  const { data: pr } = await github.rest.pulls.get({ ...params, pull_number: pullNumber });
  const retained = (reason) => ({ eligible: false, reason });
  if (!repository.id || !repository.default_branch || !context.ref?.startsWith("refs/heads/")) {
    throw new Error("Cannot establish repository default/current branch; refusing cleanup");
  }
  if (pr.merged !== true || pr.state !== "closed") return retained("PR is not merged");
  if (pr.head?.repo?.id !== repository.id || pr.base?.repo?.id !== repository.id) {
    return retained("PR is not entirely in this repository");
  }
  if (pr.base.ref !== repository.default_branch) return retained("PR did not merge into the default branch");
  const branch = pr.head.ref;
  const sha = pr.head.sha;
  if (typeof branch !== "string" || !branch || !/^[0-9a-f]{40}$/i.test(sha ?? "")) {
    throw new Error("PR head identity is incomplete; refusing cleanup");
  }
  const keep = new Set([repository.default_branch, "master", "main", context.ref.slice("refs/heads/".length)]);
  if (keep.has(branch)) return retained("Default, conventional main, or current branch");
  let live;
  try {
    ({ data: live } = await github.rest.repos.getBranch({ ...params, branch }));
  } catch (error) {
    if (error.status === 404) return retained("Branch no longer exists");
    throw error;
  }
  if (live.protected !== false) return retained("Branch is protected or protection status is unknown");
  if (live.commit?.sha !== sha) return retained("Branch head differs from the merged PR head");
  const open = await github.paginate(github.rest.pulls.list, { ...params, state: "open", per_page: 100 });
  if (open.some((item) =>
    (item.head?.repo?.id === repository.id && item.head.ref === branch)
    || (item.base?.repo?.id === repository.id && item.base.ref === branch))) {
    return retained("Branch is used by an open PR");
  }
  return { eligible: true, branch, sha };
}

async function cleanupMergedBranch({ github, context, core, pullNumber, dryRun = true, git = runGit }) {
  if (!/^[1-9]\d*$/.test(String(pullNumber)) || !Number.isSafeInteger(Number(pullNumber))) {
    throw new Error("Specify one positive PR number");
  }
  if (![true, false, "true", "false"].includes(dryRun)) {
    throw new Error("dry_run must be true or false");
  }
  const args = { github, context, pullNumber: Number(pullNumber) };
  const candidate = await inspectCandidate(args);
  if (!candidate.eligible) {
    core.info(`Retained: ${candidate.reason}`);
    return { status: "retained", ...candidate };
  }
  git(["check-ref-format", `refs/heads/${candidate.branch}`]);
  if (dryRun === true || dryRun === "true") {
    core.info(`Dry run: eligible merged PR branch ${candidate.branch} at ${candidate.sha}; nothing deleted`);
    return { status: "dry-run", ...candidate };
  }
  const current = await inspectCandidate(args);
  if (!current.eligible || current.branch !== candidate.branch || current.sha !== candidate.sha) {
    throw new Error("Cleanup eligibility changed during verification; nothing deleted");
  }
  deleteWithLease(current, git);
  core.info(`Deleted merged PR branch ${current.branch} at verified head ${current.sha}`);
  return { status: "deleted", ...current };
}

module.exports = { cleanupMergedBranch, deleteWithLease, inspectCandidate };
