/**
 * Roadmap Intelligence
 * Augments static roadmap data with live progress to produce adaptive state:
 * auto-skip mastered nodes, surface revisions, estimate completion dates.
 */

import { ROADMAPS, type Roadmap, type RoadmapNode, type RoadmapId } from '@/data/roadmaps';
import { isCompleted, getCompletedByType } from '@/lib/progress';
import { getMastery } from '@/lib/persistence';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type AdaptiveNodeState =
  | 'locked'
  | 'available'
  | 'in_progress'
  | 'completed'
  | 'mastered'
  | 'needs_revision';

export interface AdaptiveNode extends RoadmapNode {
  adaptiveState: AdaptiveNodeState;
  canSkip: boolean;
  estimatedRemainingMinutes: number;
}

export interface AdaptivePhase {
  id: string;
  name: string;
  description: string;
  nodes: AdaptiveNode[];
  completedCount: number;
  totalCount: number;
  progressPct: number;
}

export interface AdaptiveRoadmap {
  id: RoadmapId;
  title: string;
  description: string;
  icon: string;
  color: string;
  phases: AdaptivePhase[];
  totalProgress: number;
  estimatedCompletionDays: number;
  nextNode: AdaptiveNode | null;
  suggestedRevisions: AdaptiveNode[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseEstimatedMinutes(timeStr: string): number {
  const match = timeStr.match(/(\d+)/);
  if (!match) return 60;
  const num = parseInt(match[1], 10);
  // "20 hours" → 1200 minutes; "30 minutes" → 30 minutes
  if (timeStr.toLowerCase().includes('hour')) return num * 60;
  return num;
}

function isPrereqSatisfied(prereqId: string, completedNodeIds: Set<string>): boolean {
  return completedNodeIds.has(prereqId) || isCompleted('roadmap-node', prereqId);
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

function adaptNode(
  node: RoadmapNode,
  completedNodeIds: Set<string>,
): AdaptiveNode {
  const completedByEngine = isCompleted('roadmap-node', node.id);
  const staticCompleted = node.state === 'completed';
  const actuallyCompleted = completedByEngine || staticCompleted;

  const mastery = getMastery(node.id);
  const allPrereqsSatisfied =
    node.prerequisites.length === 0 ||
    node.prerequisites.every((p) => isPrereqSatisfied(p, completedNodeIds));

  let adaptiveState: AdaptiveNodeState;
  if (mastery >= 3 && actuallyCompleted) {
    adaptiveState = 'mastered';
  } else if (actuallyCompleted) {
    adaptiveState = mastery < 2 ? 'needs_revision' : 'completed';
  } else if (node.state === 'in_progress') {
    adaptiveState = 'in_progress';
  } else if (!allPrereqsSatisfied) {
    adaptiveState = 'locked';
  } else {
    adaptiveState = 'available';
  }

  const canSkip = mastery >= 4 || (actuallyCompleted && mastery >= 3);
  const minutes = parseEstimatedMinutes(node.estimatedTime);

  return {
    ...node,
    adaptiveState,
    canSkip,
    estimatedRemainingMinutes: actuallyCompleted ? 0 : minutes,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function getAdaptiveRoadmap(id: RoadmapId): AdaptiveRoadmap | null {
  const roadmap: Roadmap = ROADMAPS[id];
  if (!roadmap) return null;

  // Build a set of all completed node IDs (both static + engine-tracked)
  const completedNodeIds = new Set<string>(getCompletedByType('roadmap-node'));
  // Also include nodes whose static state is completed
  for (const phase of roadmap.phases) {
    for (const node of phase.nodes) {
      if (node.state === 'completed') completedNodeIds.add(node.id);
    }
  }

  const adaptivePhases: AdaptivePhase[] = roadmap.phases.map((phase) => {
    const adaptedNodes = phase.nodes.map((node) =>
      adaptNode(node, completedNodeIds),
    );
    const completedCount = adaptedNodes.filter(
      (n) =>
        n.adaptiveState === 'completed' ||
        n.adaptiveState === 'mastered' ||
        n.adaptiveState === 'needs_revision',
    ).length;
    return {
      id: phase.id,
      name: phase.name,
      description: phase.description,
      nodes: adaptedNodes,
      completedCount,
      totalCount: adaptedNodes.length,
      progressPct:
        adaptedNodes.length > 0
          ? Math.round((completedCount / adaptedNodes.length) * 100)
          : 0,
    };
  });

  // Totals
  const allNodes = adaptivePhases.flatMap((p) => p.nodes);
  const totalCompleted = allNodes.filter(
    (n) =>
      n.adaptiveState === 'completed' ||
      n.adaptiveState === 'mastered' ||
      n.adaptiveState === 'needs_revision',
  ).length;
  const totalProgress =
    allNodes.length > 0
      ? Math.round((totalCompleted / allNodes.length) * 100)
      : 0;

  // Estimated completion
  const remainingMinutes = allNodes.reduce(
    (s, n) => s + n.estimatedRemainingMinutes,
    0,
  );
  const dailyStudyMinutes = 60; // assume 1h/day
  const estimatedCompletionDays = Math.ceil(
    remainingMinutes / dailyStudyMinutes,
  );

  // Next node to work on
  const nextNode =
    allNodes.find(
      (n) => n.adaptiveState === 'in_progress' || n.adaptiveState === 'available',
    ) ?? null;

  // Nodes that need revision (completed but low mastery)
  const suggestedRevisions = allNodes.filter(
    (n) => n.adaptiveState === 'needs_revision',
  );

  return {
    id,
    title: roadmap.title,
    description: roadmap.description,
    icon: roadmap.icon,
    color: roadmap.color,
    phases: adaptivePhases,
    totalProgress,
    estimatedCompletionDays,
    nextNode,
    suggestedRevisions,
  };
}

export function getAllAdaptiveRoadmaps(): AdaptiveRoadmap[] {
  return (Object.keys(ROADMAPS) as RoadmapId[])
    .map(getAdaptiveRoadmap)
    .filter((r): r is AdaptiveRoadmap => r !== null);
}

export function getMissingPrerequisites(
  roadmapId: RoadmapId,
  nodeId: string,
): RoadmapNode[] {
  const roadmap = ROADMAPS[roadmapId];
  if (!roadmap) return [];
  const allNodes = roadmap.phases.flatMap((p) => p.nodes);
  const nodeMap = new Map(allNodes.map((n) => [n.id, n]));
  const target = nodeMap.get(nodeId);
  if (!target) return [];
  return target.prerequisites
    .map((id) => nodeMap.get(id))
    .filter((n): n is RoadmapNode => n !== undefined)
    .filter((n) => !isCompleted('roadmap-node', n.id) && n.state !== 'completed');
}
