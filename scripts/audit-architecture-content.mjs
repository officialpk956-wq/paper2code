#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { compile } from "@mdx-js/mdx";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const root = process.cwd();
const source = fs.readFileSync(path.join(root, "src/data/content/architectures.ts"), "utf8");
const literal = source.match(/export const ARCHITECTURES[^=]*=\s*(\[[\s\S]*?\]);/)?.[1];
if (!literal) throw new Error("Could not parse ARCHITECTURES");
const architectures = Function(`"use strict"; return (${literal});`)();
const errors = [];
let compiled = 0;

if (new Set(architectures.map((item) => item.slug)).size !== architectures.length) {
  errors.push("Architecture registry contains duplicate slugs");
}

for (const architecture of architectures) {
  const directory = path.join(root, "src/content/architectures", architecture.slug);
  const articleFile = path.join(directory, "content.mdx");
  const metaFile = path.join(directory, "meta.json");
  if (!fs.existsSync(articleFile)) {
    errors.push(`${architecture.slug}: missing content.mdx`);
    continue;
  }
  if (!fs.existsSync(metaFile)) {
    errors.push(`${architecture.slug}: missing meta.json`);
    continue;
  }

  try {
    const metadata = JSON.parse(fs.readFileSync(metaFile, "utf8"));
    if (metadata.type !== "architecture") {
      errors.push(`${architecture.slug}: metadata type must be architecture`);
    }
    if (metadata.slug !== architecture.slug) {
      errors.push(`${architecture.slug}: metadata slug mismatch`);
    }
  } catch (error) {
    errors.push(`${architecture.slug}: invalid metadata JSON: ${error.message}`);
  }

  try {
    await compile(fs.readFileSync(articleFile, "utf8"), {
      remarkPlugins: [remarkGfm, remarkMath],
      rehypePlugins: [rehypeKatex],
    });
    compiled += 1;
  } catch (error) {
    errors.push(`${architecture.slug}: MDX compilation failed: ${error.message}`);
  }
}

console.log(`Canonical architectures: ${architectures.length}`);
console.log(`MDX articles compiled: ${compiled}`);

if (errors.length > 0) {
  console.error(`Architecture audit failed (${errors.length} error(s)):\n`);
  for (const error of errors) console.error(`  - ${error}`);
  process.exitCode = 1;
} else {
  console.log("Architecture content audit passed.");
}
