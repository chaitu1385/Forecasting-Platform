const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { createHash } = require("node:crypto");
const fs = require("node:fs");
const { createRequire } = require("node:module");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..");
const LOCK = path.join(ROOT, "package-lock.json");
const RECORD = path.join(ROOT, "npm-integrity-normalization.json");
const PUBLIC_REGISTRY = "https://registry.npmjs.org/";
const MAX_TARBALL_BYTES = 64 * 1024 * 1024;
let tools;

function npmTools() {
  if (tools) return tools;
  const entry = process.env.npm_execpath;
  if (!entry) throw new Error("Run this command through the pinned npm script");
  const requireNpm = createRequire(entry);
  assert.equal(requireNpm("../package.json").version, "11.19.1", "Use npm 11.19.1");
  tools = { pacote: requireNpm("pacote"), ssri: requireNpm("ssri"), entry };
  return tools;
}

function graph(lock) {
  const copy = structuredClone(lock);
  for (const entry of Object.values(copy.packages)) delete entry.integrity;
  return copy;
}

function assertGraphUnchanged(before, after) {
  assert.deepEqual(graph(after), graph(before), "Package/version/dependency/platform graph changed");
}

function fingerprint(lock) {
  return createHash("sha256").update(JSON.stringify(graph(lock))).digest("hex");
}

function assertOriginal(bytes, integrity) {
  if (!npmTools().ssri.checkData(bytes, integrity)) {
    throw new Error("Original tarball integrity mismatch");
  }
}

function assertAuthoritative(expected, actual, context = "artifact") {
  const ssri = npmTools().ssri;
  const expectedHashes = ssri.parse(expected).sha512;
  const actualHashes = ssri.parse(actual).sha512;
  if (!expectedHashes || expectedHashes.length !== 1 || !actualHashes
      || !actualHashes.some((hash) => hash.digest === expectedHashes[0].digest)) {
    throw new Error(`${context}: authoritative upstream SHA-512 mismatch`);
  }
}

function npmConfig(key) {
  return execFileSync(process.execPath, [npmTools().entry, "config", "get", key], {
    encoding: "utf8", stdio: ["ignore", "pipe", "pipe"],
  }).trim();
}

function secureRegistry(value) {
  const url = new URL(value);
  if (url.protocol !== "https:" || url.username || url.password) {
    throw new Error("A credential-free HTTPS registry URL is required");
  }
  return url.href;
}

function privateCache() {
  const cache = process.env.npm_config_cache || process.env.NPM_CONFIG_CACHE;
  if (!cache || !path.isAbsolute(cache)) {
    throw new Error("Set npm_config_cache to an absolute task-owned cache directory");
  }
  return cache;
}

async function tarball(spec, integrity, options) {
  return npmTools().pacote.tarball.stream(spec, async (stream) => {
    const chunks = [];
    let size = 0;
    for await (const chunk of stream) {
      size += chunk.length;
      if (size > MAX_TARBALL_BYTES) throw new Error("Tarball exceeds the bounded normalization limit");
      chunks.push(chunk);
    }
    const bytes = Buffer.concat(chunks);
    assertOriginal(bytes, integrity);
    return bytes;
  }, { ...options, integrity, ignoreScripts: true, fetchTimeout: 60_000, fetchRetries: 1 });
}

async function bounded(entries, operation) {
  if (entries.length > 256) throw new Error("Review normalization scope before processing more than 256 artifacts");
  const pending = [...entries];
  let failed = false;
  await Promise.all(Array.from({ length: 4 }, async () => {
    while (pending.length && !failed) {
      try {
        await operation(pending.shift());
      } catch (error) {
        failed = true;
        throw error;
      }
    }
  }));
}

function checkRecord(lock, record) {
  assert.equal(record.tool, "npm@11.19.1");
  assert.equal(record.graph_sha256, fingerprint(lock), "Normalization record does not match the complete graph");
  for (const entry of record.entries) {
    for (const location of entry.paths) {
      assert.equal(lock.packages[location].version, entry.version);
      assert.equal(lock.packages[location].integrity, entry.normalized_integrity);
    }
  }
}

async function refresh() {
  const before = JSON.parse(fs.readFileSync(LOCK, "utf8"));
  const after = structuredClone(before);
  const registry = secureRegistry(npmConfig("registry"));
  const options = { registry, cache: privateCache(), preferOffline: true };
  const groups = new Map();
  for (const [location, entry] of Object.entries(before.packages)) {
    if (!location || !entry.integrity?.startsWith("sha1-")) continue;
    const name = entry.name || location.slice(location.lastIndexOf("node_modules/") + 13);
    const spec = `${name}@${entry.version}`;
    if (!groups.has(spec)) groups.set(spec, { name, version: entry.version, original_integrity: entry.integrity, paths: [] });
    const group = groups.get(spec);
    assert.equal(group.original_integrity, entry.integrity, `Conflicting original integrity: ${spec}`);
    group.paths.push(location);
  }
  const entries = [...groups.values()];
  if (!entries.length && !fs.existsSync(RECORD)) {
    throw new Error("No original legacy entries remain and the normalization provenance is missing");
  }
  await bounded(entries, async (entry) => {
    const bytes = await tarball(`${entry.name}@${entry.version}`, entry.original_integrity, options);
    entry.normalized_integrity = npmTools().ssri.fromData(bytes, { algorithms: ["sha512"] }).toString();
    for (const location of entry.paths) after.packages[location].integrity = entry.normalized_integrity;
  });
  if (fs.existsSync(RECORD)) {
    const previous = JSON.parse(fs.readFileSync(RECORD, "utf8"));
    for (const entry of previous.entries) {
      if (groups.has(`${entry.name}@${entry.version}`)) continue;
      const locations = entry.paths.filter((location) => after.packages[location]?.version === entry.version);
      for (const location of locations) {
        assert.equal(after.packages[location].integrity, entry.normalized_integrity,
          "A previously normalized artifact changed without a package version change");
      }
      if (locations.length) entries.push({ ...entry, paths: locations });
    }
  }
  assertGraphUnchanged(before, after);
  const record = {
    tool: "npm@11.19.1", graph_sha256: fingerprint(after),
    entries: entries.sort((a, b) => `${a.name}@${a.version}`.localeCompare(`${b.name}@${b.version}`)),
  };
  checkRecord(after, record);
  const temporary = `${LOCK}.${process.pid}.tmp`;
  const recordTemporary = `${RECORD}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(after, null, 2) + "\n", { flag: "wx" });
  try {
    fs.writeFileSync(recordTemporary, JSON.stringify(record, null, 2) + "\n", { flag: "wx" });
    fs.renameSync(recordTemporary, RECORD);
    fs.renameSync(temporary, LOCK);
  } finally {
    if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
    if (fs.existsSync(recordTemporary)) fs.unlinkSync(recordTemporary);
  }
  console.log(`Normalized ${groups.size} exact artifacts; complete graph unchanged. Upstream acceptance is still required.`);
}

async function verifyUpstream() {
  const lock = JSON.parse(fs.readFileSync(LOCK, "utf8"));
  const record = JSON.parse(fs.readFileSync(RECORD, "utf8"));
  checkRecord(lock, record);
  // Only public-registry options are passed; never forward private registry credentials.
  const options = { registry: PUBLIC_REGISTRY, cache: privateCache(), preferOnline: true,
    fetchTimeout: 60_000, fetchRetries: 1 };
  await bounded(record.entries, async (entry) => {
    const spec = `${entry.name}@${entry.version}`;
    const metadata = await npmTools().pacote.manifest(spec, options);
    assert.equal(metadata.name, entry.name);
    assert.equal(metadata.version, entry.version);
    const published = metadata.dist?.integrity || metadata._integrity;
    if (published && npmTools().ssri.parse(published).sha512) {
      assertAuthoritative(entry.normalized_integrity, published, spec);
    } else {
      const url = new URL(metadata.dist?.tarball || metadata._resolved);
      if (url.protocol !== "https:" || url.hostname !== "registry.npmjs.org" || url.username || url.password) {
        throw new Error(`Not an official HTTPS registry tarball: ${spec}`);
      }
      await tarball(url.href, entry.normalized_integrity, options);
    }
  });
  console.log(`Verified ${record.entries.length} normalized digests against authoritative public-registry data.`);
}

module.exports = { assertOriginal, assertAuthoritative, assertGraphUnchanged, checkRecord };

if (require.main === module) {
  const mode = process.argv[2];
  if (!["--refresh", "--check-upstream"].includes(mode) || process.argv.length !== 3) {
    throw new Error("Use --refresh or --check-upstream explicitly");
  }
  (mode === "--refresh" ? refresh() : verifyUpstream()).catch((error) => {
    console.error(String(error.message).replace(/https?:\/\/\S+/g, "[registry URL]"));
    process.exitCode = 1;
  });
}
