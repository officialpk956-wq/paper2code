#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const source = fs.readFileSync(path.join(root, "src/data/content/papers.ts"), "utf8");
const literal = source.match(/export const PAPERS[^=]*=\s*(\[[\s\S]*?\]);/)?.[1];
if (!literal) throw new Error("Could not parse PAPERS");
const papers = Function(`"use strict"; return (${literal});`)();

let changed = 0;
for (const paper of papers) {
  const file = path.join(root, "src/content/papers", paper.slug, "meta.json");
  if (!fs.existsSync(file)) throw new Error(`${paper.slug}: missing meta.json`);
  const metadata = JSON.parse(fs.readFileSync(file, "utf8"));
  const difficulty = String(metadata.difficulty ?? paper.difficulty ?? "intermediate").toLowerCase();
  const normalizedDifficulty = difficulty === "expert"
    ? "advanced"
    : ["beginner", "intermediate", "advanced"].includes(difficulty)
      ? difficulty
      : "intermediate";
  const normalized = {
    ...metadata,
    type: "paper",
    slug: paper.slug,
    title: paper.title,
    difficulty: normalizedDifficulty,
    year: metadata.year ?? paper.year,
  };
  if (!normalized.year) throw new Error(`${paper.slug}: year is missing from registry and metadata`);
  const output = `${JSON.stringify(normalized, null, 2)}\n`;
  if (output !== fs.readFileSync(file, "utf8")) {
    fs.writeFileSync(file, output);
    changed += 1;
  }
}

console.log(`Paper metadata normalized: ${changed} file(s) changed`);
