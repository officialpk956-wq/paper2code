#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { compile } from "@mdx-js/mdx";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = fs.readFileSync(path.join(ROOT, "src/data/content/systemDesign.ts"), "utf8");
const literal = source.match(/export const SD_SYSTEMS[^=]*=\s*(\[[\s\S]*\]);/)?.[1];
if (!literal) throw new Error("Could not parse SD_SYSTEMS");
const systems = Function(`"use strict"; return (${literal});`)();

const expectedHeadings = [
  "Beginner: Foundations",
  "Intermediate: Design Deep Dive",
  "Advanced: Production at Scale",
  "Research: Frontiers",
  "Hands-On Projects",
];
const errors = [];
let compiled = 0;

function words(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function mdxText(text) {
  return text.replaceAll("<", "&lt;");
}

for (const system of systems) {
  const directory = path.join(ROOT, "src/content/system-design", system.slug);
  const contentFile = path.join(directory, "content.mdx");
  const metaFile = path.join(directory, "meta.json");
  const relative = path.relative(ROOT, contentFile);

  if (!fs.existsSync(contentFile)) {
    errors.push(`${relative}: missing content.mdx`);
    continue;
  }
  if (!fs.existsSync(metaFile)) {
    errors.push(`${path.relative(ROOT, metaFile)}: missing meta.json`);
    continue;
  }

  const body = fs.readFileSync(contentFile, "utf8");
  const headings = [...body.matchAll(/^## (.+)$/gm)].map((match) => match[1]);
  const positions = expectedHeadings.map((heading) => headings.indexOf(heading));
  if (positions.some((position) => position < 0) || positions.some((position, index) => index && position <= positions[index - 1])) {
    errors.push(`${relative}: required H2 sections are missing or out of order`);
  }
  const internalLinks = body.match(/\[[^\]]+\]\(\/[a-z0-9/?#&=._-]+\)/gi)?.length ?? 0;
  if (internalLinks < 4) errors.push(`${relative}: expected at least 4 internal cross-links, got ${internalLinks}`);

  for (let index = 0; index < system.modules.length; index += 1) {
    const module = system.modules[index];
    const start = body.indexOf(`## ${expectedHeadings[index]}`);
    const end = body.indexOf(`## ${expectedHeadings[index + 1]}`, start + 1);
    const section = body.slice(start, end < 0 ? body.length : end);
    const proseEnd = section.indexOf("### Diagrams");
    const prose = section.slice(0, proseEnd);

    if (words(prose) < 300) {
      errors.push(`${relative}: ${module.level} teaching prose has ${words(prose)} words; expected at least 300`);
    }
    if (!section.includes("**Prerequisites:**")) {
      errors.push(`${relative}: ${module.level} prerequisite callout missing`);
    }

    for (const objective of module.learningObjectives) {
      if (!section.includes(mdxText(objective))) errors.push(`${relative}: ${module.level} objective missing: ${objective}`);
    }
    for (const prerequisite of module.prerequisites) {
      if (!section.includes(prerequisite)) errors.push(`${relative}: ${module.level} prerequisite missing: ${prerequisite}`);
    }
    for (const diagram of module.diagramsNeeded) {
      if (!section.includes(mdxText(diagram))) errors.push(`${relative}: ${module.level} diagram missing: ${diagram}`);
    }
    const diagramCount = section.match(/^#### Diagram \d+:/gm)?.length ?? 0;
    if (diagramCount !== module.diagramsNeeded.length) {
      errors.push(`${relative}: ${module.level} has ${diagramCount} diagrams; expected ${module.diagramsNeeded.length}`);
    }

    for (const study of module.caseStudies) {
      const marker = `#### ${mdxText(study)}`;
      const studyStart = section.indexOf(marker);
      if (studyStart < 0) {
        errors.push(`${relative}: ${module.level} case study missing: ${study}`);
        continue;
      }
      const nextHeading = section.indexOf("\n#### ", studyStart + marker.length);
      const caseBody = section.slice(studyStart + marker.length, nextHeading < 0 ? section.length : nextHeading);
      if (words(caseBody) < 150) {
        errors.push(`${relative}: case study '${study}' has ${words(caseBody)} words; expected at least 150`);
      }
    }

    for (const question of module.interviewQuestions) {
      if (!section.includes(mdxText(question))) errors.push(`${relative}: ${module.level} interview question missing: ${question}`);
    }

    if (module.handsOnProjects.length === 0 && !body.includes(`**${module.level} project (proposed):`)) {
      errors.push(`${relative}: ${module.level} proposed projects missing`);
    }
    for (const project of module.handsOnProjects) {
      if (!body.includes(project)) errors.push(`${relative}: listed hands-on project missing: ${project}`);
    }
  }

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

  let meta;
  try {
    meta = JSON.parse(fs.readFileSync(metaFile, "utf8"));
  } catch (error) {
    errors.push(`${path.relative(ROOT, metaFile)}: invalid JSON: ${error.message}`);
    continue;
  }
  if (meta.type !== "system-design") errors.push(`${path.relative(ROOT, metaFile)}: wrong type`);
  if (meta.slug !== system.slug) errors.push(`${path.relative(ROOT, metaFile)}: slug mismatch`);
  if (meta.title !== system.name) errors.push(`${path.relative(ROOT, metaFile)}: title mismatch`);
  if (meta.difficulty !== "advanced") errors.push(`${path.relative(ROOT, metaFile)}: difficulty must be advanced`);
  if (!Array.isArray(meta.tags) || !meta.tags.includes("system-design")) {
    errors.push(`${path.relative(ROOT, metaFile)}: system-design tag missing`);
  }
}

console.log(`Canonical systems: ${systems.length}`);
console.log(`MDX articles compiled: ${compiled}`);
if (errors.length) {
  console.error(`System-design audit failed (${errors.length} error(s)):\n`);
  for (const error of errors) console.error(`  - ${error}`);
  process.exit(1);
}
console.log("System-design audit passed.");
