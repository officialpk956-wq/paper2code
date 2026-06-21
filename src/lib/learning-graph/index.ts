/**
 * Learning Graph Engine
 * Builds a directed prerequisite graph from EVOLUTION_NODES, ROADMAPS,
 * and PROBLEMS data. Provides path-finding and unlock queries.
 */

import { EVOLUTION_NODES } from '@/data/evolution';
import { ROADMAPS } from '@/data/roadmaps';
import { PROBLEMS } from '@/data/problems';
import { isCompleted } from '@/lib/progress';
import type { ProgressContentType } from '@/lib/progress';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GraphNode {
  id: string; // "{type}:{slug}"
  type: ProgressContentType;
  slug: string;
  title: string;
  prerequisites: string[]; // other node IDs
  successors: string[]; // other node IDs
  estimatedMinutes: number;
}

export interface LearningPath {
  nodes: GraphNode[]; // topological order, prereqs first
  completedIds: string[]; // already done
  remainingMinutes: number;
}

// ---------------------------------------------------------------------------
// Static time estimates per type (minutes)
// ---------------------------------------------------------------------------

const TIME_ESTIMATES: Record<ProgressContentType, number> = {
  paper: 120,
  architecture: 60,
  problem: 30,
  implementation: 90,
  roadmap: 0,
  'roadmap-node': 60,
  'system-design': 45,
  'tensor-trace': 20,
  math: 40,
  topic: 30,
};

export const CONTENT_TOTALS: Record<ProgressContentType, number> = {
  paper: 24,
  architecture: 42,
  problem: 156,
  implementation: 18,
  roadmap: 6,
  'roadmap-node': 120,
  'system-design': 15,
  'tensor-trace': 8,
  math: 12,
  topic: 12,
};

// ---------------------------------------------------------------------------
// Graph construction (lazy, built once)
// ---------------------------------------------------------------------------

function makeId(type: ProgressContentType, slug: string): string {
  return `${type}:${slug}`;
}

function buildGraph(): Map<string, GraphNode> {
  const graph = new Map<string, GraphNode>();

  function upsert(node: GraphNode): void {
    const existing = graph.get(node.id);
    if (existing) {
      // Merge in any new prerequisites / successors
      for (const p of node.prerequisites) {
        if (!existing.prerequisites.includes(p)) existing.prerequisites.push(p);
      }
    } else {
      graph.set(node.id, node);
    }
  }

  // ── EVOLUTION NODES (papers & architectures) ──────────────────────────────
  for (const evo of EVOLUTION_NODES) {
    const type: ProgressContentType = 'paper';
    const id = makeId(type, evo.slug);
    upsert({
      id,
      type,
      slug: evo.slug,
      title: evo.title,
      prerequisites: evo.parents.map((p) => makeId('paper', p)),
      successors: evo.children.map((c) => makeId('paper', c)),
      estimatedMinutes: TIME_ESTIMATES.paper,
    });
  }

  // ── ROADMAP NODES ─────────────────────────────────────────────────────────
  for (const roadmap of Object.values(ROADMAPS)) {
    for (const phase of roadmap.phases) {
      for (const node of phase.nodes) {
        const id = makeId('roadmap-node', node.id);
        upsert({
          id,
          type: 'roadmap-node',
          slug: node.id,
          title: node.title,
          prerequisites: node.prerequisites.map((p) =>
            makeId('roadmap-node', p),
          ),
          successors: [],
          estimatedMinutes: TIME_ESTIMATES['roadmap-node'],
        });
      }
    }
  }

  // ── PROBLEMS (soft dependency: related papers / architectures) ────────────
  for (const problem of PROBLEMS) {
    const id = makeId('problem', problem.slug);
    const prereqs: string[] = [
      ...problem.relatedPapers.map((s) => makeId('paper', s)),
      ...problem.relatedArchitectures.map((s) => makeId('architecture', s)),
    ];
    upsert({
      id,
      type: 'problem',
      slug: problem.slug,
      title: problem.title,
      prerequisites: prereqs,
      successors: [],
      estimatedMinutes: problem.estimatedTime ?? TIME_ESTIMATES.problem,
    });
  }

  // ── Compute reverse edges (successors) for all nodes ─────────────────────
  for (const node of graph.values()) {
    for (const prereqId of node.prerequisites) {
      const prereq = graph.get(prereqId);
      if (prereq && !prereq.successors.includes(node.id)) {
        prereq.successors.push(node.id);
      }
    }
  }

  return graph;
}

let _graph: Map<string, GraphNode> | null = null;

function getGraph(): Map<string, GraphNode> {
  if (_graph === null) _graph = buildGraph();
  return _graph;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function getGraphNode(
  type: ProgressContentType,
  slug: string,
): GraphNode | null {
  return getGraph().get(makeId(type, slug)) ?? null;
}

export function getPrerequisites(
  type: ProgressContentType,
  slug: string,
): GraphNode[] {
  const node = getGraph().get(makeId(type, slug));
  if (!node) return [];
  return node.prerequisites.flatMap((id) => {
    const n = getGraph().get(id);
    return n ? [n] : [];
  });
}

export function getNextTopics(
  type: ProgressContentType,
  slug: string,
): GraphNode[] {
  const node = getGraph().get(makeId(type, slug));
  if (!node) return [];
  return node.successors.flatMap((id) => {
    const n = getGraph().get(id);
    return n ? [n] : [];
  });
}

export function getMissingPrerequisites(
  type: ProgressContentType,
  slug: string,
): GraphNode[] {
  return getPrerequisites(type, slug).filter(
    (n) => !isCompleted(n.type, n.slug),
  );
}

export function isUnlocked(
  type: ProgressContentType,
  slug: string,
): boolean {
  const node = getGraph().get(makeId(type, slug));
  if (!node || node.prerequisites.length === 0) return true;
  return node.prerequisites.every((prereqId) => {
    const prereq = getGraph().get(prereqId);
    return !prereq || isCompleted(prereq.type, prereq.slug);
  });
}

/** Returns all nodes whose prerequisites are fully satisfied and that are not yet completed. */
export function getUnlockedContent(): GraphNode[] {
  return Array.from(getGraph().values()).filter(
    (n) => !isCompleted(n.type, n.slug) && isUnlocked(n.type, n.slug),
  );
}

/**
 * BFS backward from target to collect all prerequisite nodes in topological
 * order, then returns the full path with completion info.
 */
export function getLearningPath(
  type: ProgressContentType,
  slug: string,
): LearningPath {
  const graph = getGraph();
  const targetId = makeId(type, slug);
  const visited = new Set<string>();
  const ordered: GraphNode[] = [];

  function collect(id: string): void {
    if (visited.has(id)) return;
    visited.add(id);
    const node = graph.get(id);
    if (!node) return;
    for (const prereqId of node.prerequisites) collect(prereqId);
    ordered.push(node);
  }

  collect(targetId);

  const completedIds = ordered
    .filter((n) => isCompleted(n.type, n.slug))
    .map((n) => n.id);

  const remainingMinutes = ordered
    .filter((n) => !isCompleted(n.type, n.slug))
    .reduce((sum, n) => sum + n.estimatedMinutes, 0);

  return { nodes: ordered, completedIds, remainingMinutes };
}

export function getGraphStats(): {
  nodeCount: number;
  edgeCount: number;
  completedCount: number;
  unlockedCount: number;
} {
  const graph = getGraph();
  const nodes = Array.from(graph.values());
  return {
    nodeCount: nodes.length,
    edgeCount: nodes.reduce((s, n) => s + n.prerequisites.length, 0),
    completedCount: nodes.filter((n) => isCompleted(n.type, n.slug)).length,
    unlockedCount: getUnlockedContent().length,
  };
}
