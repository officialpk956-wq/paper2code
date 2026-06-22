import { ASSESSMENT_QUESTIONS, GOAL_OPTIONS } from '@/data/assessment';
import type { SkillArea } from '@/data/assessment';
import { ROADMAPS } from '@/data/roadmaps';
import type { RoadmapId } from '@/data/roadmaps';
import { save, load, KEYS } from '@/lib/persistence';

export type Level = 'foundation-builder' | 'architecture-reader' | 'system-builder' | 'research-practitioner';

export interface Profile {
  level: Level;
  goal: string;
  roadmapId: RoadmapId;
  startPhaseId: string;
  hoursPerWeek: string;
  skillProfile: Record<SkillArea, number>;
  completedAt: number;
}

/**
 * Pure function to score an assessment deterministically based on user answers.
 * Returns the computed per-area fractions (0..1) and the overall Level.
 *
 * @param answers Map of question ID to selected option ID.
 */
export function scoreAssessment(answers: Record<string, string>): { skillProfile: Record<SkillArea, number>; level: Level } {
  const areaTotals: Record<SkillArea, number> = {
    foundations: 0,
    architectures: 0,
    systems: 0,
    implementation: 0
  };
  const areaEarned: Record<SkillArea, number> = {
    foundations: 0,
    architectures: 0,
    systems: 0,
    implementation: 0
  };

  let totalWeight = 0;
  let earnedWeight = 0;

  for (const q of ASSESSMENT_QUESTIONS) {
    areaTotals[q.area] += q.weight;
    totalWeight += q.weight;

    if (answers[q.id] === q.correctOptionId) {
      areaEarned[q.area] += q.weight;
      earnedWeight += q.weight;
    }
  }

  const skillProfile: Record<SkillArea, number> = {
    foundations: areaTotals.foundations > 0 ? areaEarned.foundations / areaTotals.foundations : 0,
    architectures: areaTotals.architectures > 0 ? areaEarned.architectures / areaTotals.architectures : 0,
    systems: areaTotals.systems > 0 ? areaEarned.systems / areaTotals.systems : 0,
    implementation: areaTotals.implementation > 0 ? areaEarned.implementation / areaTotals.implementation : 0
  };

  const fraction = totalWeight > 0 ? earnedWeight / totalWeight : 0;

  let level: Level = 'foundation-builder';
  if (fraction >= 0.8) {
    level = 'research-practitioner';
  } else if (fraction >= 0.5) {
    level = 'system-builder';
  } else if (fraction >= 0.25) {
    level = 'architecture-reader';
  }

  return { skillProfile, level };
}

/**
 * Pure function to map a Level and Goal to a specific Roadmap phase.
 */
export function mapLevelToEntry(level: Level, goalText: string): { roadmapId: RoadmapId; startPhaseId: string } {
  const roadmapId = GOAL_OPTIONS[goalText] || 'ai-engineer';
  const roadmap = ROADMAPS[roadmapId];

  const levelMapping: Record<Level, number> = {
    'foundation-builder': 0,
    'architecture-reader': 1,
    'system-builder': 2,
    'research-practitioner': 3
  };

  const levelIdx = levelMapping[level];
  
  if (!roadmap || !roadmap.phases.length) {
    return { roadmapId, startPhaseId: 'phase-1' };
  }

  const phaseIdx = Math.min(levelIdx, roadmap.phases.length - 1);
  return { roadmapId, startPhaseId: roadmap.phases[phaseIdx].id };
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

export function saveProfile(profile: Profile): void {
  save(KEYS.userProfile, profile);
}

export function loadProfile(): Profile | null {
  return load<Profile>(KEYS.userProfile);
}

export function hasProfile(): boolean {
  return loadProfile() !== null;
}
