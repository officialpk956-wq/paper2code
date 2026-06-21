/**
 * Problem Difficulty Calibration
 * Computes difficulty scores and recommends problem sets by tier.
 */

import { PROBLEMS, type Problem } from '@/data/problems';
import { getCompletedByType, getCompletionRate } from '@/lib/progress';
import { getMastery } from '@/lib/persistence';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type DifficultyTier = 'easy' | 'medium' | 'hard' | 'expert';

export interface CalibratedProblem {
  problem: Problem;
  tier: DifficultyTier;
  readinessScore: number; // 0–100 — how ready is this user
  requiredConcepts: string[];
  relatedPaperCount: number;
  estimatedMinutes: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DIFFICULTY_MAP: Record<Problem['difficulty'], DifficultyTier> = {
  beginner: 'easy',
  intermediate: 'medium',
  advanced: 'hard',
};

function computeTier(problem: Problem): DifficultyTier {
  const base = DIFFICULTY_MAP[problem.difficulty];
  // Upgrade tier based on relatedArchitectures + relatedPapers count
  const complexity =
    problem.relatedArchitectures.length + problem.relatedPapers.length;
  if (base === 'hard' && complexity >= 3) return 'expert';
  return base;
}

function computeReadiness(
  problem: Problem,
  completedPapers: Set<string>,
  completedArch: Set<string>,
): number {
  if (problem.relatedPapers.length === 0 && problem.relatedArchitectures.length === 0) {
    return 100; // no prerequisites → always ready
  }
  const totalPrereqs =
    problem.relatedPapers.length + problem.relatedArchitectures.length;
  const metPrereqs =
    problem.relatedPapers.filter((p) => completedPapers.has(p)).length +
    problem.relatedArchitectures.filter((a) => completedArch.has(a)).length;

  // Also factor in mastery
  const masteryBonus = [
    ...problem.relatedPapers,
    ...problem.relatedArchitectures,
  ].reduce((s, sl) => s + getMastery(sl), 0);

  const prereqScore = totalPrereqs > 0 ? (metPrereqs / totalPrereqs) * 70 : 70;
  const masteryScore = Math.min(masteryBonus * 5, 30);
  return Math.round(prereqScore + masteryScore);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function calibrateAll(): CalibratedProblem[] {
  const completedPapers = new Set(getCompletedByType('paper'));
  const completedArch = new Set(getCompletedByType('architecture'));

  return PROBLEMS.map((problem) => ({
    problem,
    tier: computeTier(problem),
    readinessScore: computeReadiness(problem, completedPapers, completedArch),
    requiredConcepts: [
      ...problem.relatedPapers,
      ...problem.relatedArchitectures,
    ],
    relatedPaperCount: problem.relatedPapers.length,
    estimatedMinutes: problem.estimatedTime ?? 30,
  }));
}

export function recommendProblemSet(tier?: DifficultyTier): CalibratedProblem[] {
  const completedProblems = new Set(getCompletedByType('problem'));
  const calibrated = calibrateAll().filter(
    (c) => !completedProblems.has(c.problem.slug),
  );

  const filtered = tier ? calibrated.filter((c) => c.tier === tier) : calibrated;

  return filtered
    .sort((a, b) => b.readinessScore - a.readinessScore)
    .slice(0, 10);
}

export function getProblemsByTier(): Record<DifficultyTier, CalibratedProblem[]> {
  const all = calibrateAll();
  return {
    easy: all.filter((c) => c.tier === 'easy'),
    medium: all.filter((c) => c.tier === 'medium'),
    hard: all.filter((c) => c.tier === 'hard'),
    expert: all.filter((c) => c.tier === 'expert'),
  };
}

export function getNextChallenge(): CalibratedProblem | null {
  const completedProblems = new Set(getCompletedByType('problem'));
  const completionPct = getCompletionRate('problem');

  // Calibrate tier to current progress level
  let targetTier: DifficultyTier = 'easy';
  if (completionPct >= 60) targetTier = 'expert';
  else if (completionPct >= 35) targetTier = 'hard';
  else if (completionPct >= 15) targetTier = 'medium';

  const calibrated = calibrateAll().filter(
    (c) => !completedProblems.has(c.problem.slug) && c.tier === targetTier,
  );

  return calibrated.sort((a, b) => b.readinessScore - a.readinessScore)[0] ?? null;
}
