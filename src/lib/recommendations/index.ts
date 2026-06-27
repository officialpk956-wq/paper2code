/**
 * Real Recommendation Engine
 * Uses completion history, mastery state, prerequisites, and recent activity
 * to generate ranked, categorized content recommendations.
 */

import { PROBLEMS } from '@/data/problems';
import { PAPER_TIMELINE } from '@/data/paper-timeline';
import { ARCHITECTURE_CATALOG } from '@/data/architecture-catalog';
import { ROADMAPS, type RoadmapId } from '@/data/roadmaps';
import { LEARNING_TRACKS } from '@/data/learning-tracks';
import {
  getCompletedByType,
  getInProgress,
  getCompletedCount,
} from '@/lib/progress';
import { isUnlocked, getMissingPrerequisites } from '@/lib/learning-graph';
import { getMastery } from '@/lib/persistence';
import { getRecentActivity } from '@/lib/study-sessions';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type RecommendationCategory =
  | 'continue_learning'
  | 'ready_now'
  | 'skill_gap'
  | 'advanced_challenge'
  | 'suggested_revision';

export interface Recommendation {
  type: 'problem' | 'paper' | 'architecture' | 'roadmap' | 'track' | 'topic';
  slug: string;
  title: string;
  description: string;
  href: string;
  category: RecommendationCategory;
  reason: string;
  difficulty?: string;
  estimatedMinutes?: number;
  score: number;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function recentSlugs(limit = 20): Set<string> {
  return new Set(getRecentActivity(limit).map((a) => a.slug));
}

// ---------------------------------------------------------------------------
// Individual recommendation generators
// ---------------------------------------------------------------------------

export function recommendProblems(limit = 5): Recommendation[] {
  const completedSet = new Set(getCompletedByType('problem'));
  const inProgressSlugs = new Set(
    getInProgress()
      .filter((p) => p.type === 'problem')
      .map((p) => p.slug),
  );
  const completedPapers = new Set(getCompletedByType('paper'));
  const completedArch = new Set(getCompletedByType('architecture'));
  const recent = recentSlugs();

  const recs: Recommendation[] = [];

  for (const problem of PROBLEMS) {
    if (completedSet.has(problem.slug)) continue;

    let score = 0;
    let category: RecommendationCategory = 'ready_now';
    let reason = 'Matches your current level';

    // In-progress — highest priority
    if (inProgressSlugs.has(problem.slug)) {
      score += 15;
      category = 'continue_learning';
      reason = 'Continue where you left off';
    }

    // Related content overlap
    const paperHits = problem.relatedPapers.filter((p) =>
      completedPapers.has(p),
    ).length;
    const archHits = problem.relatedArchitectures.filter((a) =>
      completedArch.has(a),
    ).length;
    score += paperHits * 3 + archHits * 2;

    if (paperHits + archHits > 0 && category !== 'continue_learning') {
      reason = `Builds on ${paperHits + archHits} topic${paperHits + archHits > 1 ? 's' : ''} you've studied`;
    }

    // Mastery-based advanced challenge
    const masteryTotal = [
      ...problem.relatedPapers,
      ...problem.relatedArchitectures,
    ].reduce((s, sl) => s + getMastery(sl), 0);
    if (masteryTotal >= 6 && category === 'ready_now') {
      category = 'advanced_challenge';
      reason = 'Your mastery qualifies you for this challenge';
      score += 5;
    }

    // Unlocked check
    if (!isUnlocked('problem', problem.slug)) {
      category = 'skill_gap';
      const missing = getMissingPrerequisites('problem', problem.slug);
      reason =
        missing.length > 0
          ? `Complete "${missing[0].title}" first`
          : 'Complete prerequisites first';
      score = Math.max(score - 5, 0);
    }

    // Recent activity boost
    if (recent.has(problem.slug)) score += 2;

    // Penalise advanced problems with no related content studied
    if (
      problem.difficulty === 'advanced' &&
      paperHits === 0 &&
      archHits === 0 &&
      category === 'ready_now'
    ) {
      category = 'skill_gap';
      reason = 'Study related papers and architectures first';
      score -= 3;
    }

    recs.push({
      type: 'problem',
      slug: problem.slug,
      title: problem.title,
      description: problem.description,
      href: `/problems/${problem.slug}`,
      category,
      reason,
      difficulty: problem.difficulty,
      estimatedMinutes: problem.estimatedTime,
      score,
    });
  }

  return recs.sort((a, b) => b.score - a.score).slice(0, limit);
}

export function recommendPapers(limit = 5): Recommendation[] {
  const completedPapers = new Set(getCompletedByType('paper'));
  const completedArch = new Set(getCompletedByType('architecture'));
  const recent = recentSlugs();
  const recs: Recommendation[] = [];

  for (const paper of PAPER_TIMELINE) {
    if (paper.status !== 'complete') continue;
    if (completedPapers.has(paper.slug)) continue;

    let score = 0;
    const category: RecommendationCategory = 'ready_now';
    let reason = 'Key paper in the field';

    // Architecture connection
    if (paper.architectureSlug && completedArch.has(paper.architectureSlug)) {
      score += 4;
      reason = `Introduces the ${paper.architectureSlug} architecture you're exploring`;
    }

    // Foundation papers score higher (chronological = foundational)
    const eraBoost: Record<string, number> = {
      'Deep Learning Era': 4,
      'Attention Era': 3,
      'Scaling Era': 2,
      'Generative Era': 2,
      'Reasoning Era': 1,
    };
    score += eraBoost[paper.era] ?? 1;

    if (recent.has(paper.slug)) score += 2;

    recs.push({
      type: 'paper',
      slug: paper.slug,
      title: paper.shortTitle,
      description: paper.description,
      href: `/papers/${paper.slug}`,
      category,
      reason,
      difficulty: 'intermediate',
      estimatedMinutes: 90,
      score,
    });
  }

  return recs.sort((a, b) => b.score - a.score).slice(0, limit);
}

export function recommendArchitectures(limit = 4): Recommendation[] {
  const completedArch = new Set(getCompletedByType('architecture'));
  const completedPapers = new Set(getCompletedByType('paper'));
  const recent = recentSlugs();
  const recs: Recommendation[] = [];

  for (const arch of ARCHITECTURE_CATALOG) {
    if (arch.status !== 'complete') continue;
    if (completedArch.has(arch.slug)) continue;

    let score = 0;
    let category: RecommendationCategory = 'ready_now';
    let reason = 'Important architecture to understand';

    const paperMatches = (arch.papers ?? []).filter((p) =>
      completedPapers.has(p.slug),
    ).length;
    if (paperMatches > 0) {
      score += paperMatches * 3;
      reason = `You've read ${paperMatches} related paper${paperMatches > 1 ? 's' : ''}`;
    }

    // Mastery on this arch's related papers
    const masterySum = (arch.papers ?? []).reduce(
      (s, p) => s + getMastery(p.slug),
      0,
    );
    if (masterySum >= 4) {
      category = 'advanced_challenge';
      reason = 'Your paper mastery supports deep architecture study';
      score += 3;
    }

    // Foundational architectures
    if (['transformer', 'resnet', 'alexnet', 'bert', 'gpt'].includes(arch.slug)) {
      score += 5;
    }

    if (recent.has(arch.slug)) score += 2;

    recs.push({
      type: 'architecture',
      slug: arch.slug,
      title: arch.title,
      description: arch.description,
      href: `/architectures/${arch.slug}`,
      category,
      reason,
      difficulty: 'intermediate',
      score,
    });
  }

  return recs.sort((a, b) => b.score - a.score).slice(0, limit);
}

export function recommendRoadmap(): Recommendation | null {
  const completedProblems = getCompletedCount('problem');

  let roadmapId: RoadmapId = 'ai-engineer';
  let reason = 'Best starting point for your level';

  if (completedProblems >= 30) {
    roadmapId = 'llm-engineer';
    reason = 'Your problem-solving depth is ready for LLM engineering';
  } else if (completedProblems >= 15) {
    roadmapId = 'ml-engineer';
    reason = 'Ready to advance to full ML engineering scope';
  }

  const roadmap = ROADMAPS[roadmapId];
  if (!roadmap) return null;

  return {
    type: 'roadmap',
    slug: roadmapId,
    title: roadmap.title,
    description: roadmap.description,
    href: `/roadmaps/${roadmapId}`,
    category: 'ready_now',
    reason,
    score: 10,
  };
}

export function recommendTrack(): Recommendation | null {
  const completedProblems = new Set(getCompletedByType('problem'));

  for (const track of LEARNING_TRACKS) {
    // Pick first track where not all problems are done
    const notDone = track.problems.some(
      (step) => !completedProblems.has(step.problemId),
    );
    if (notDone) {
      const doneCount = track.problems.filter((step) =>
        completedProblems.has(step.problemId),
      ).length;
      const pct = Math.round((doneCount / track.problems.length) * 100);
      return {
        type: 'track',
        slug: track.slug,
        title: track.title,
        description: track.subtitle,
        href: `/learning-tracks/${track.slug}`,
        category: doneCount > 0 ? 'continue_learning' : 'ready_now',
        reason:
          doneCount > 0
            ? `${pct}% complete — keep going`
            : 'Structured path through transformer fundamentals',
        estimatedMinutes: track.estimatedHours * 60,
        score: doneCount > 0 ? 12 : 8,
      };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Aggregated recommendation feed
// ---------------------------------------------------------------------------

export function getRecommendations(limit = 8): Recommendation[] {
  const problems = recommendProblems(3);
  const papers = recommendPapers(2);
  const archs = recommendArchitectures(2);
  const roadmap = recommendRoadmap();
  const track = recommendTrack();

  return [
    ...problems,
    ...papers,
    ...archs,
    ...(roadmap ? [roadmap] : []),
    ...(track ? [track] : []),
  ]
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

export const CATEGORY_LABELS: Record<RecommendationCategory, string> = {
  continue_learning: 'Continue',
  ready_now: 'Ready Now',
  skill_gap: 'Skill Gap',
  advanced_challenge: 'Challenge',
  suggested_revision: 'Revisit',
};

export const CATEGORY_COLORS: Record<RecommendationCategory, string> = {
  continue_learning: 'var(--accent-primary)',
  ready_now: 'var(--color-easy)',
  skill_gap: 'var(--color-medium)',
  advanced_challenge: 'var(--color-hard)',
  suggested_revision: 'var(--accent-cyan)',
};
