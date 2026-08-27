#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { compile } from "@mdx-js/mdx";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const root = process.cwd();
const source = fs.readFileSync(path.join(root, "src/data/content/papers.ts"), "utf8");
const literal = source.match(/export const PAPERS[^=]*=\s*(\[[\s\S]*?\]);/)?.[1];
if (!literal) throw new Error("Could not parse PAPERS");
const papers = Function(`"use strict"; return (${literal});`)();

const args = Object.fromEntries(
  process.argv.slice(2).map((argument) => {
    const [key, value = "true"] = argument.replace(/^--/, "").split("=");
    return [key, value];
  }),
);
const from = Number(args.from ?? 1);
const to = Number(args.to ?? Number.MAX_SAFE_INTEGER);
const targets = papers.filter((paper) => paper.rank >= from && paper.rank <= to);
const paperSlugs = new Set(papers.map((paper) => paper.slug));
const errors = [];
let compiled = 0;

if (paperSlugs.size !== papers.length) errors.push("Paper registry contains duplicate slugs");
if (new Set(papers.map((paper) => paper.rank)).size !== papers.length) {
  errors.push("Paper registry contains duplicate ranks");
}

for (const paper of targets) {
  const directory = path.join(root, "src/content/papers", paper.slug);
  const articleFile = path.join(directory, "content.mdx");
  const metaFile = path.join(directory, "meta.json");
  if (!fs.existsSync(articleFile)) {
    errors.push(`${paper.slug}: missing content.mdx`);
    continue;
  }
  if (!fs.existsSync(metaFile)) {
    errors.push(`${paper.slug}: missing meta.json`);
    continue;
  }

  const article = fs.readFileSync(articleFile, "utf8");
  const wordCount = article.match(/\b[\p{L}\p{N}_-]+\b/gu)?.length ?? 0;
  const sectionCount = article.match(/^## /gm)?.length ?? 0;
  if (wordCount < 650) errors.push(`${paper.slug}: only ${wordCount} words (minimum 650)`);
  if (sectionCount < 15) errors.push(`${paper.slug}: only ${sectionCount} H2 sections (minimum 15)`);

  try {
    await compile(article, {
      remarkPlugins: [remarkGfm, remarkMath],
      rehypePlugins: [rehypeKatex],
    });
    compiled += 1;
  } catch (error) {
    errors.push(`${paper.slug}: MDX compilation failed: ${error.message}`);
  }

  try {
    const meta = JSON.parse(fs.readFileSync(metaFile, "utf8"));
    if (meta.type !== "paper") errors.push(`${paper.slug}: metadata type must be paper`);
    if (meta.slug !== paper.slug) errors.push(`${paper.slug}: metadata slug mismatch`);
    if (!meta.title) errors.push(`${paper.slug}: metadata title is empty`);
    for (const related of meta.relationships?.papers ?? []) {
      if (!paperSlugs.has(related)) errors.push(`${paper.slug}: invalid paper relationship ${related}`);
    }
  } catch (error) {
    errors.push(`${paper.slug}: invalid metadata JSON: ${error.message}`);
  }
}

console.log(`Paper registry: ${papers.length}`);
console.log(`Audited rank range: ${from}-${to}`);
console.log(`Target papers: ${targets.length}`);
console.log(`MDX articles compiled: ${compiled}`);

if (errors.length > 0) {
  console.error(`Paper audit failed (${errors.length} error(s)):\n`);
  for (const error of errors) console.error(`  - ${error}`);
  process.exitCode = 1;
} else {
  console.log("Paper content audit passed.");
}
