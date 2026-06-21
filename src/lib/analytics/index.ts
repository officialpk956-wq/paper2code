/**
 * Analytics Engine
 * Computes all dashboard metrics from real tracked data.
 */

import {
  getCompletedCount,
  getCompletionRate,
  getKnowledgeCoverage,
  getAllCompleted,
  getInProgress,
  type ProgressContentType,
} from '@/lib/progress';
import {
  getTotalStudyMs,
  getWeekStudyMs,
  getMonthStudyMs,
  getTodayStudyMs,
  getStudyStreak,
  getDailyStudy,
  formatDuration,
} from '@/lib/study-sessions';
import { getAllMasteryLevels } from '@/lib/persistence';
import { getGraphStats } from '@/lib/learning-graph';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PlatformMetrics {
  // Time
  totalStudyHours: number;
  todayStudyFormatted: string;
  weekStudyFormatted: string;
  monthStudyFormatted: string;

  // Streaks
  studyStreakDays: number;

  // Completion
  overallCompletionRate: number;
  problemCompletionRate: number;
  paperCompletionRate: number;

  // Counts
  completedProblems: number;
  completedPapers: number;
  completedArchitectures: number;
  inProgressCount: number;

  // Mastery
  masteredCount: number; // mastery >= 3
  masteryAverage: number; // 0–4

  // Knowledge
  knowledgeCoverage: Partial<Record<ProgressContentType, number>>;
  graphStats: { nodeCount: number; edgeCount: number; completedCount: number; unlockedCount: number };

  // Velocity (weekly completion rate)
  weeklyVelocity: number; // items completed this week
}

export interface HeatmapDay {
  date: string;
  totalMs: number;
  level: 0 | 1 | 2 | 3 | 4; // intensity bucket
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function getPlatformMetrics(): PlatformMetrics {
  const masteryLevels = Object.values(getAllMasteryLevels()) as number[];
  const masteredCount = masteryLevels.filter((v) => v >= 3).length;
  const masteryAverage =
    masteryLevels.length > 0
      ? masteryLevels.reduce((s, v) => s + v, 0) / masteryLevels.length
      : 0;

  const totalMs = getTotalStudyMs();
  const totalStudyHours = Math.round((totalMs / 3_600_000) * 10) / 10;

  // Weekly velocity: count items completed in the last 7 days
  const sevenDaysAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
  const weeklyVelocity = getAllCompleted().filter(
    (p) => (p.completedAt ?? 0) > sevenDaysAgo,
  ).length;

  return {
    totalStudyHours,
    todayStudyFormatted: formatDuration(getTodayStudyMs()),
    weekStudyFormatted: formatDuration(getWeekStudyMs()),
    monthStudyFormatted: formatDuration(getMonthStudyMs()),
    studyStreakDays: getStudyStreak(),
    overallCompletionRate: getCompletionRate(),
    problemCompletionRate: getCompletionRate('problem'),
    paperCompletionRate: getCompletionRate('paper'),
    completedProblems: getCompletedCount('problem'),
    completedPapers: getCompletedCount('paper'),
    completedArchitectures: getCompletedCount('architecture'),
    inProgressCount: getInProgress().length,
    masteredCount,
    masteryAverage: Math.round(masteryAverage * 10) / 10,
    knowledgeCoverage: getKnowledgeCoverage(),
    graphStats: getGraphStats(),
    weeklyVelocity,
  };
}

export function getHeatmapData(days = 30): HeatmapDay[] {
  const daily = getDailyStudy(days);

  // Find max for bucketing
  const maxMs = Math.max(...daily.map((d) => d.totalMs), 1);

  return daily.map((d) => {
    let level: 0 | 1 | 2 | 3 | 4 = 0;
    if (d.totalMs > 0) {
      const ratio = d.totalMs / maxMs;
      if (ratio >= 0.75) level = 4;
      else if (ratio >= 0.5) level = 3;
      else if (ratio >= 0.25) level = 2;
      else level = 1;
    }
    return { date: d.date, totalMs: d.totalMs, level };
  });
}

export function getLearningVelocityTrend(
  weeks = 4,
): Array<{ week: string; count: number }> {
  const completed = getAllCompleted();
  const now = Date.now();
  const trend: Array<{ week: string; count: number }> = [];

  for (let i = weeks - 1; i >= 0; i--) {
    const weekStart = now - (i + 1) * 7 * 24 * 60 * 60 * 1000;
    const weekEnd = now - i * 7 * 24 * 60 * 60 * 1000;
    const count = completed.filter(
      (p) =>
        (p.completedAt ?? 0) >= weekStart && (p.completedAt ?? 0) < weekEnd,
    ).length;
    const weekLabel = `W-${i === 0 ? 'current' : String(i)}`;
    trend.push({ week: weekLabel, count });
  }

  return trend;
}
