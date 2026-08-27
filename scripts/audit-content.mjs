#!/usr/bin/env node

import { spawnSync } from "node:child_process";

const node = process.execPath;
const checks = [
  ["Generate and validate content index", node, ["scripts/generate-content-index.mjs"]],
  ["Architecture coverage and MDX", node, ["scripts/audit-architecture-content.mjs"]],
  ["Curriculum coverage and MDX", node, ["scripts/audit-curriculum-content.mjs"]],
  ["System-design coverage and MDX", node, ["scripts/audit-system-design-content.mjs"]],
  ["Paper coverage, metadata, and MDX", node, ["scripts/audit-paper-content.mjs"]],
  ["Physical-path orphan classification", node, ["scripts/audit-content-orphans.mjs"]],
  ["Registry-aware internal links", node, ["scripts/audit-internal-content-links.mjs"]],
  ["TypeScript", node, ["node_modules/typescript/bin/tsc", "--noEmit", "--incremental", "false"]],
];

const failures = [];
for (const [label, command, args] of checks) {
  console.log(`\n=== ${label} ===`);
  const result = spawnSync(command, args, {
    cwd: process.cwd(),
    encoding: "utf8",
    stdio: "inherit",
    shell: false,
  });
  if (result.error || result.status !== 0) {
    failures.push(`${label} (exit ${result.status ?? "unavailable"})`);
  }
}

if (failures.length > 0) {
  console.error(`\nContent audit failed:\n- ${failures.join("\n- ")}`);
  process.exitCode = 1;
} else {
  console.log("\nAll content, metadata, route, and TypeScript audits passed.");
}
