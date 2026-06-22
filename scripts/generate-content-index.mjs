#!/usr/bin/env node
/**
 * Build-time content index generator.
 *
 * Walks src/content/, validates every meta.json, and emits
 * src/generated/content-index.json — the search/relationship index the app
 * imports statically. Runs automatically via the `prebuild` script; run
 * manually with `npm run generate:content`.
 *
 * Validation here is intentionally dependency-free (plain Node) so the script
 * works before `npm install`. The runtime loader (src/lib/content/loader.ts)
 * re-validates with zod, so invalid content fails the build in either path.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CONTENT_DIR = path.join(ROOT, "src", "content");
const OUT_DIR = path.join(ROOT, "src", "generated");
const OUT_FILE = path.join(OUT_DIR, "content-index.json");

const TYPE_DIRS = {
  architecture: "architectures",
  paper: "papers",
  math: "math",
  "system-design": "system-design",
  problem: "problems",
  interview: "interview",
  roadmap: "roadmaps",
  implementation: "implementations",
};

const DIFFICULTIES = ["beginner", "intermediate", "advanced"];
const SLUG_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const RELATIONSHIP_KEYS = [
  "architectures",
  "papers",
  "math",
  "problems",
  "systemDesign",
  "interview",
  "roadmaps",
  "implementations",
];

/** Per-type required extra fields (beyond the base meta). */
const TYPE_REQUIRED = {
  architecture: ["year"],
  paper: ["authors", "year"],
  math: ["category"],
  "system-design": [],
  problem: [],
  interview: ["questionType"],
  roadmap: [],
  implementation: ["paperSlug"],
};

const errors = [];
const entries = [];

function fail(file, message) {
  errors.push(`${path.relative(ROOT, file)}: ${message}`);
}

function validateMeta(meta, type, slug, file) {
  if (meta.type !== type)
    fail(file, `type is "${meta.type}", expected "${type}" for this directory`);
  if (meta.slug !== slug)
    fail(file, `slug "${meta.slug}" does not match directory name "${slug}"`);
  if (!SLUG_RE.test(meta.slug ?? ""))
    fail(file, `slug must be kebab-case, got "${meta.slug}"`);
  if (typeof meta.title !== "string" || meta.title.length === 0)
    fail(file, "title is required");
  if (typeof meta.description !== "string" || meta.description.length === 0)
    fail(file, "description is required");
  if (!DIFFICULTIES.includes(meta.difficulty))
    fail(file, `difficulty must be one of ${DIFFICULTIES.join("|")}`);
  if (meta.tags !== undefined && !Array.isArray(meta.tags))
    fail(file, "tags must be an array");

  for (const key of TYPE_REQUIRED[type]) {
    if (meta[key] === undefined) fail(file, `missing required field "${key}"`);
  }

  const rels = meta.relationships ?? {};
  for (const key of Object.keys(rels)) {
    if (!RELATIONSHIP_KEYS.includes(key))
      fail(file, `unknown relationship key "${key}"`);
    else if (!Array.isArray(rels[key]))
      fail(file, `relationships.${key} must be an array of slugs`);
  }
}

// --- Pass 1: collect and validate every item -------------------------------

for (const [type, dirName] of Object.entries(TYPE_DIRS)) {
  const typeDir = path.join(CONTENT_DIR, dirName);
  if (!fs.existsSync(typeDir)) continue;

  for (const entry of fs.readdirSync(typeDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const metaFile = path.join(typeDir, entry.name, "meta.json");
    if (!fs.existsSync(metaFile)) {
      fail(metaFile, "missing meta.json");
      continue;
    }

    let meta;
    try {
      meta = JSON.parse(fs.readFileSync(metaFile, "utf-8"));
    } catch (e) {
      fail(metaFile, `invalid JSON: ${e.message}`);
      continue;
    }

    validateMeta(meta, type, entry.name, metaFile);

    entries.push({
      slug: meta.slug,
      type,
      title: meta.title,
      description: meta.description,
      tags: meta.tags ?? [],
      difficulty: meta.difficulty,
      relationships: meta.relationships ?? {},
    });
  }
}

// --- Pass 2: verify relationship targets exist ------------------------------

const KEY_TO_TYPE = {
  architectures: "architecture",
  papers: "paper",
  math: "math",
  problems: "problem",
  systemDesign: "system-design",
  interview: "interview",
  roadmaps: "roadmap",
  implementations: "implementation",
};

const known = new Set(entries.map((e) => `${e.type}:${e.slug}`));
for (const entry of entries) {
  for (const [key, slugs] of Object.entries(entry.relationships)) {
    const targetType = KEY_TO_TYPE[key];
    for (const slug of slugs ?? []) {
      if (!known.has(`${targetType}:${slug}`)) {
        fail(
          path.join(CONTENT_DIR, TYPE_DIRS[entry.type], entry.slug, "meta.json"),
          `relationship target not found: ${key} → "${slug}" (no ${targetType} with that slug)`
        );
      }
    }
  }
}

// --- Emit -------------------------------------------------------------------

if (errors.length > 0) {
  console.error(`\nContent validation failed (${errors.length} error(s)):\n`);
  for (const err of errors) console.error(`  ✗ ${err}`);
  console.error("");
  process.exit(1);
}

entries.sort((a, b) =>
  a.type === b.type ? a.slug.localeCompare(b.slug) : a.type.localeCompare(b.type)
);

fs.mkdirSync(OUT_DIR, { recursive: true });
fs.writeFileSync(
  OUT_FILE,
  JSON.stringify(
    { generatedAt: new Date().toISOString(), count: entries.length, entries },
    null,
    2
  )
);

console.log(
  `✓ content-index.json generated: ${entries.length} items across ${
    new Set(entries.map((e) => e.type)).size
  } types`
);
