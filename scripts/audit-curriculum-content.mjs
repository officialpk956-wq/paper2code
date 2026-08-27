#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { compile } from "@mdx-js/mdx";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = fs.readFileSync(path.join(ROOT, "src/data/content/curriculum.ts"), "utf8");
const literal = source.match(/export const CURRICULUM[^=]*=\s*(\[[\s\S]*\]);/)?.[1];
if (!literal) throw new Error("Could not parse CURRICULUM metadata");
const curriculum = Function(`"use strict"; return (${literal});`)();

const errors = [];
let compiled = 0;

for (const domain of curriculum) {
  for (const topic of domain.topics) {
    const relative = path.join(
      "src/content/curriculum",
      domain.slug,
      topic.slug,
      "content.mdx",
    );
    const file = path.join(ROOT, relative);
    if (!fs.existsSync(file)) {
      errors.push(`${relative}: missing canonical lesson`);
      continue;
    }

    const body = fs.readFileSync(file, "utf8");
    const headings = [...body.matchAll(/^## (\d+)\. /gm)].map((match) => Number(match[1]));
    if (headings.join(",") !== "1,2,3,4,5,6,7,8,9,10,11") {
      errors.push(`${relative}: expected numbered sections 1 through 11`);
    }

    const questions = body.match(/^\*\*Q[1-5]:/gm)?.length ?? 0;
    if (questions !== 5) errors.push(`${relative}: expected 5 Q&A pairs, got ${questions}`);

    const words = body.trim().split(/\s+/).length;
    if (words < 700) errors.push(`${relative}: lesson is too short (${words} words)`);

    for (const byte of Buffer.from(body)) {
      if (byte < 32 && ![9, 10, 13].includes(byte)) {
        errors.push(`${relative}: contains a control character`);
        break;
      }
    }

    try {
      await compile(body, {
        remarkPlugins: [remarkMath, remarkGfm],
        rehypePlugins: [rehypeKatex],
      });
      compiled += 1;
    } catch (error) {
      errors.push(`${relative}: MDX compilation failed: ${error.message}`);
    }
  }
}

console.log(`Curriculum topics: ${curriculum.flatMap((domain) => domain.topics).length}`);
console.log(`MDX lessons compiled: ${compiled}`);
if (errors.length) {
  console.error(`Curriculum audit failed (${errors.length} error(s)):\n`);
  for (const error of errors) console.error(`  - ${error}`);
  process.exit(1);
}
console.log("Curriculum audit passed.");
