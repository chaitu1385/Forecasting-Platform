"use strict";

const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { mkdtempSync, readFileSync, rmSync, writeFileSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join } = require("node:path");
const { test } = require("node:test");
const { cleanupMergedBranch, deleteWithLease } = require("./cleanup-merged-branch.cjs");

const sha = "a".repeat(40);

test("manual cleanup pins reviewed Node24 actions", () => {
  const workflow = readFileSync(join(__dirname, "..", "workflows", "cleanup-branches.yml"), "utf8");
  const actions = [...workflow.matchAll(/^\s*(?:-\s*)?uses:\s+([^\s#]+)/gm)]
    .map((match) => match[1]);
  assert.deepEqual(actions, [
    "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
    "actions/github-script@3a2844b7e9c422d3c10d287c895573f7108da1b3",
  ]);
});

function fixture() {
  const repository = { id: 1, default_branch: "trunk" };
  const pr = {
    merged: true, state: "closed",
    head: { ref: "feature/merged", sha, repo: { id: 1 } },
    base: { ref: "trunk", repo: { id: 1 } },
  };
  const branch = { protected: false, commit: { sha } };
  const open = [];
  const calls = [];
  const options = {
    github: {
      rest: {
        repos: { get: async () => ({ data: repository }), getBranch: async () => ({ data: branch }) },
        pulls: { get: async () => ({ data: pr }), list: () => {} },
      },
      paginate: async () => open,
    },
    context: { repo: { owner: "example", repo: "forecast" }, ref: "refs/heads/trunk" },
    core: { info: () => {} }, pullNumber: 42,
    git: (args) => { calls.push(args); return ""; },
  };
  return { repository, pr, branch, open, calls, options };
}

test("dry run is the default and cannot mutate refs", async () => {
  const f = fixture();
  const result = await cleanupMergedBranch(f.options);
  assert.equal(result.status, "dry-run");
  assert.deepEqual(f.calls, [["check-ref-format", "refs/heads/feature/merged"]]);
});

for (const [name, alter] of [
  ["unmerged PR", (f) => { f.pr.merged = false; }],
  ["open PR", (f) => { f.pr.state = "open"; }],
  ["fork head", (f) => { f.pr.head.repo.id = 2; }],
  ["deleted fork", (f) => { f.pr.head.repo = null; }],
  ["foreign base", (f) => { f.pr.base.repo.id = 2; }],
  ["non-default target", (f) => { f.pr.base.ref = "release"; }],
  ["actual default branch", (f) => { f.pr.head.ref = "trunk"; }],
  ["master branch", (f) => { f.pr.head.ref = "master"; }],
  ["main branch", (f) => { f.pr.head.ref = "main"; }],
  ["current branch", (f) => { f.options.context.ref = "refs/heads/feature/merged"; }],
  ["protected branch", (f) => { f.branch.protected = true; }],
  ["unknown protection", (f) => { delete f.branch.protected; }],
  ["advanced branch", (f) => { f.branch.commit.sha = "b".repeat(40); }],
  ["open PR head", (f) => { f.open.push({ head: f.pr.head }); }],
  ["open PR base", (f) => { f.open.push({ base: { ref: f.pr.head.ref, repo: { id: 1 } } }); }],
]) {
  test(`retains ${name}`, async () => {
    const f = fixture();
    alter(f);
    const result = await cleanupMergedBranch({ ...f.options, dryRun: false });
    assert.equal(result.status, "retained");
    assert.equal(f.calls.length, 0);
  });
}

test("deletion requires fresh checks and one exact explicit lease", async () => {
  const f = fixture();
  let reads = 0;
  f.options.github.rest.pulls.get = async () => { reads++; return { data: f.pr }; };
  const result = await cleanupMergedBranch({ ...f.options, dryRun: false });
  assert.equal(result.status, "deleted");
  assert.equal(reads, 2);
  assert.deepEqual(f.calls.at(-1), [
    "push", "--porcelain", `--force-with-lease=refs/heads/feature/merged:${sha}`,
    "origin", ":refs/heads/feature/merged",
  ]);
});

test("an eligibility change on recheck fails without deletion", async () => {
  const f = fixture();
  let reads = 0;
  f.options.github.rest.repos.getBranch = async () => ({
    data: ++reads === 1 ? f.branch : { ...f.branch, protected: true },
  });
  await assert.rejects(cleanupMergedBranch({ ...f.options, dryRun: false }), /eligibility changed/);
  assert.equal(f.calls.some((args) => args[0] === "push"), false);
});

test("API failures and delete failures are not swallowed", async () => {
  const f = fixture();
  f.options.github.rest.repos.getBranch = async () => { throw new Error("permission denied"); };
  await assert.rejects(cleanupMergedBranch(f.options), /permission denied/);
  const g = fixture();
  g.options.git = (args) => { if (args[0] === "push") throw new Error("stale lease"); };
  await assert.rejects(cleanupMergedBranch({ ...g.options, dryRun: false }), /stale lease/);
});

test("invalid inputs cannot enable destructive operation", async () => {
  const f = fixture();
  await assert.rejects(cleanupMergedBranch({ ...f.options, pullNumber: "42; command" }), /positive PR/);
  await assert.rejects(cleanupMergedBranch({ ...f.options, dryRun: "anything" }), /dry_run/);
  delete f.repository.default_branch;
  await assert.rejects(cleanupMergedBranch(f.options), /default\/current/);
  assert.equal(f.calls.length, 0);
});

function localRepository(t) {
  const root = mkdtempSync(join(tmpdir(), "forecast-cleanup-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const remote = join(root, "remote.git");
  const work = join(root, "work");
  const run = (args, cwd = root) => execFileSync("git", args, {
    cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"],
  }).trim();
  run(["init", "--bare", "-q", remote]);
  run(["init", "-q", work]);
  const git = (args) => run(args, work);
  git(["config", "user.name", "Cleanup test"]);
  git(["config", "user.email", "cleanup@example.invalid"]);
  git(["config", "commit.gpgsign", "false"]);
  writeFileSync(join(work, "fixture.txt"), "initial\n");
  git(["add", "fixture.txt"]);
  git(["commit", "-qm", "initial"]);
  git(["remote", "add", "origin", remote]);
  git(["push", "-q", "origin", "HEAD:refs/heads/merged"]);
  const initial = git(["rev-parse", "HEAD"]);
  const advance = () => {
    writeFileSync(join(work, "fixture.txt"), "new work\n");
    git(["add", "fixture.txt"]);
    git(["commit", "-qm", "new work"]);
    git(["push", "-q", "origin", "HEAD:refs/heads/merged"]);
    return git(["rev-parse", "HEAD"]);
  };
  return { git, initial, advance, remoteGit: (args) => run(["--git-dir", remote, ...args]) };
}

test("real git deletes only the exact expected head in a disposable bare repository", (t) => {
  const f = localRepository(t);
  deleteWithLease({ branch: "merged", sha: f.initial }, f.git);
  assert.throws(() => f.remoteGit(["rev-parse", "--verify", "refs/heads/merged"]));
});

for (const replacement of [false, true]) {
  test(`real lease rejects ${replacement ? "same-named replacement branch" : "post-check head advance"}`, (t) => {
    const f = localRepository(t);
    if (replacement) f.remoteGit(["update-ref", "-d", "refs/heads/merged"]);
    const advanced = f.advance();
    assert.throws(() => deleteWithLease({ branch: "merged", sha: f.initial }, f.git));
    assert.equal(f.remoteGit(["rev-parse", "refs/heads/merged"]), advanced);
  });
}
