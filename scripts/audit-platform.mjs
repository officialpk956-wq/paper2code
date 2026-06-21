import fs from 'fs';
import path from 'path';

// Load content index
const contentIndex = JSON.parse(fs.readFileSync('./src/generated/content-index.json', 'utf8'));
const entries = contentIndex.entries;

const getEntry = (slug, type) => entries.find(e => e.slug === slug && (!type || e.type === type));

const reports = {
  missingContent: [],
  brokenRelationships: [],
  orphanNodes: [],
  searchCoverage: [],
  learningPathDeadEnds: [],
};

// 1. Content Audit (Major Papers)
// Let's identify "major" papers based on if they are present in the evolution timeline or just check all papers.
// We will check if they have: architecture links, implementation, tensor trace, math, problems, evolution.

// To check evolution, we parse src/data/paper-timeline.ts
const timelineDataStr = fs.readFileSync('./src/data/paper-timeline.ts', 'utf8');

// To check tensor traces, we read src/data/tensor-traces.ts
const tensorTracesStr = fs.readFileSync('./src/data/tensor-traces.ts', 'utf8');
const tensorTracesMatches = [...tensorTracesStr.matchAll(/paperSlug:\s*"([^"]+)"/g)].map(m => m[1]);

const papers = entries.filter(e => e.type === 'paper');

for (const paper of papers) {
  const rels = paper.relationships || {};
  const hasArch = rels.architectures && rels.architectures.length > 0;
  
  // Implementation
  const hasImpl = entries.some(e => e.type === 'implementation' && (e.paperSlug === paper.slug || (e.relationships?.papers?.includes(paper.slug))));
  
  // Tensor Trace
  const hasTensorTrace = tensorTracesMatches.includes(paper.slug);
  
  // Math
  const hasMath = rels.math && rels.math.length > 0;
  
  // Problems
  const hasProblems = rels.problems && rels.problems.length > 0;
  
  // Evolution
  const hasEvolution = timelineDataStr.includes(`"${paper.slug}"`) || timelineDataStr.includes(`'${paper.slug}'`);

  const missing = [];
  if (!hasArch) missing.push('Architecture');
  if (!hasImpl) missing.push('Implementation Journey');
  if (!hasTensorTrace) missing.push('Tensor Trace');
  if (!hasMath) missing.push('Math Concepts');
  if (!hasProblems) missing.push('Problems');
  if (!hasEvolution) missing.push('Evolution Path');

  if (missing.length > 0) {
    reports.missingContent.push({ paper: paper.slug, missing });
  }
}

// 2. Relationship Audit
const allSlugs = new Set(entries.map(e => e.slug));
const incomingEdges = new Map();
entries.forEach(e => incomingEdges.set(e.slug, 0));

for (const entry of entries) {
  const rels = entry.relationships || {};
  for (const [key, targets] of Object.entries(rels)) {
    if (!Array.isArray(targets)) continue;
    for (const target of targets) {
      if (!allSlugs.has(target)) {
        reports.brokenRelationships.push(`Entry "${entry.slug}" (${entry.type}) references non-existent target "${target}" in "${key}"`);
      } else {
        incomingEdges.set(target, (incomingEdges.get(target) || 0) + 1);
      }
    }
  }
}

// Check for orphans (nodes with 0 incoming and 0 outgoing edges, excluding roadmaps which are roots)
for (const entry of entries) {
  if (entry.type === 'roadmap') continue;
  const rels = entry.relationships || {};
  const outgoingCount = Object.values(rels).flat().length;
  const incomingCount = incomingEdges.get(entry.slug) || 0;
  
  if (outgoingCount === 0 && incomingCount === 0) {
    reports.orphanNodes.push(`Orphan node: "${entry.slug}" (${entry.type}) has no incoming or outgoing relationships.`);
  }
}

// 3. Search Audit
// Check if tensor traces are in the search index
// Search index in our app relies on content-index.json.
// Tensor traces are defined in a separate TS file.
reports.searchCoverage.push("Tensor Traces are defined in `src/data/tensor-traces.ts` and are NOT currently included in `src/generated/content-index.json` or `searchContent` in `src/lib/content/relationships.ts`. They will not appear in global search.");

// 4. Learning Path Audit (Roadmaps)
const roadmaps = entries.filter(e => e.type === 'roadmap');
for (const rm of roadmaps) {
  // Roadmaps link to topics, papers, etc.
  const rels = rm.relationships || {};
  const targets = Object.values(rels).flat();
  if (targets.length === 0) {
    reports.learningPathDeadEnds.push(`Roadmap "${rm.slug}" has no content linked.`);
  }
  // Check depth-1 progression
  for (const targetSlug of targets) {
    const target = getEntry(targetSlug);
    if (!target) continue;
    const targetRels = target.relationships || {};
    const targetOutgoing = Object.values(targetRels).flat();
    if (targetOutgoing.length === 0) {
      reports.learningPathDeadEnds.push(`Roadmap "${rm.slug}" -> "${targetSlug}" leads to a dead end (no further relationships).`);
    }
  }
}

console.log(JSON.stringify(reports, null, 2));
