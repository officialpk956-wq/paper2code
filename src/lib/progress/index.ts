/**
 * Unified Progress Engine
 * Tracks completion state for every content type across the platform.
 * Extends (and stays backward-compatible with) the existing persistence layer.
 */

import { save, load } from '@/lib/persistence';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ProgressContentType =
  | 'paper'
  | 'architecture'
  | 'problem'
  | 'implementation'
  | 'roadmap'
  | 'roadmap-node'
  | 'system-design'
  | 'tensor-trace'
  | 'math'
  | 'topic';

export type ProgressState = 'not_started' | 'started' | 'in_progress' | 'completed';

export interface ContentProgress {
  type: ProgressContentType;
  slug: string;
  state: ProgressState;
  startedAt?: number;
  completedAt?: number;
  timeSpentMs: number;
}

// ---------------------------------------------------------------------------
// Known total counts per type (for completion-rate calculation)
// ---------------------------------------------------------------------------

export const CONTENT_TOTALS: Record<ProgressContentType, number> = {
  paper: 24,
  architecture: 24,
  problem: 110,
  implementation: 12,
  roadmap: 6,
  'roadmap-node': 120,
  'system-design': 8,
  'tensor-trace': 6,
  math: 16,
  topic: 12,
};

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

const PROGRESS_KEY = 'p2c:content-progress';

function progressKey(type: ProgressContentType, slug: string): string {
  return `${type}:${slug}`;
}

function loadAll(): Record<string, ContentProgress> {
  return load<Record<string, ContentProgress>>(PROGRESS_KEY) ?? {};
}

function saveAll(all: Record<string, ContentProgress>): void {
  save(PROGRESS_KEY, all);
}

// ---------------------------------------------------------------------------
// Core mutation API
// ---------------------------------------------------------------------------

export function markStarted(type: ProgressContentType, slug: string): void {
  const all = loadAll();
  const k = progressKey(type, slug);
  const existing = all[k];
  if (existing?.state === 'completed') return;
  all[k] = {
    type,
    slug,
    state: 'started',
    startedAt: existing?.startedAt ?? Date.now(),
    timeSpentMs: existing?.timeSpentMs ?? 0,
  };
  saveAll(all);
}

export function markInProgress(type: ProgressContentType, slug: string): void {
  const all = loadAll();
  const k = progressKey(type, slug);
  const existing = all[k];
  if (existing?.state === 'completed') return;
  all[k] = {
    type,
    slug,
    state: 'in_progress',
    startedAt: existing?.startedAt ?? Date.now(),
    timeSpentMs: existing?.timeSpentMs ?? 0,
  };
  saveAll(all);
}

export function markCompleted(type: ProgressContentType, slug: string): void {
  const all = loadAll();
  const k = progressKey(type, slug);
  const existing = all[k];
  all[k] = {
    type,
    slug,
    state: 'completed',
    startedAt: existing?.startedAt ?? Date.now(),
    completedAt: Date.now(),
    timeSpentMs: existing?.timeSpentMs ?? 0,
  };
  saveAll(all);

  // Backward-compat: mirror problem completions to old key
  if (type === 'problem') {
    const old = load<string[]>('p2c:completed-problems') ?? [];
    if (!old.includes(slug)) {
      save('p2c:completed-problems', [...old, slug]);
    }
  }
}

export function setCompleted(type: ProgressContentType, slug: string, completed: boolean): void {
  if (completed) {
    markCompleted(type, slug);
  } else {
    const all = loadAll();
    const k = progressKey(type, slug);
    const existing = all[k];
    if (!existing) return;
    all[k] = {
      ...existing,
      state: 'not_started',
      completedAt: undefined,
    };
    saveAll(all);

    if (type === 'problem') {
      const old = load<string[]>('p2c:completed-problems') ?? [];
      save('p2c:completed-problems', old.filter(s => s !== slug));
    }
  }
}

export function addTimeSpent(
  type: ProgressContentType,
  slug: string,
  ms: number,
): void {
  const all = loadAll();
  const k = progressKey(type, slug);
  const existing = all[k] ?? { type, slug, state: 'not_started' as ProgressState, timeSpentMs: 0 };
  all[k] = { ...existing, timeSpentMs: (existing.timeSpentMs ?? 0) + ms };
  saveAll(all);
}

// ---------------------------------------------------------------------------
// Query API
// ---------------------------------------------------------------------------

export function getProgress(
  type: ProgressContentType,
  slug: string,
): ContentProgress {
  const all = loadAll();
  return (
    all[progressKey(type, slug)] ?? {
      type,
      slug,
      state: 'not_started',
      timeSpentMs: 0,
    }
  );
}

export function isCompleted(type: ProgressContentType, slug: string): boolean {
  return getProgress(type, slug).state === 'completed';
}

export function isStarted(type: ProgressContentType, slug: string): boolean {
  const state = getProgress(type, slug).state;
  return state === 'started' || state === 'in_progress';
}

export function getCompletedByType(type: ProgressContentType): string[] {
  return Object.values(loadAll())
    .filter((p) => p.type === type && p.state === 'completed')
    .map((p) => p.slug);
}

export function getAllCompleted(): ContentProgress[] {
  return Object.values(loadAll()).filter((p) => p.state === 'completed');
}

export function getInProgress(): ContentProgress[] {
  return Object.values(loadAll()).filter(
    (p) => p.state === 'in_progress' || p.state === 'started',
  );
}

export function getCompletedCount(type?: ProgressContentType): number {
  return Object.values(loadAll()).filter(
    (p) =>
      p.state === 'completed' && (type === undefined || p.type === type),
  ).length;
}

export function getCompletionRate(type?: ProgressContentType): number {
  const completed = getCompletedCount(type);
  const total =
    type !== undefined
      ? (CONTENT_TOTALS[type] ?? 1)
      : Object.values(CONTENT_TOTALS).reduce((a, b) => a + b, 0);
  return total > 0 ? Math.round((completed / total) * 100) : 0;
}

export function getTotalTimeSpentMs(): number {
  return Object.values(loadAll()).reduce(
    (sum, p) => sum + (p.timeSpentMs ?? 0),
    0,
  );
}

export function getStreakDays(): number {
  const all = Object.values(loadAll())
    .filter(p => p.state === 'completed' && p.completedAt)
    .map(p => p.completedAt as number)
    .sort((a, b) => b - a); // descending

  if (all.length === 0) return 0;

  const getDayString = (ts: number) => {
    const d = new Date(ts);
    return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`;
  };

  const daysSet = new Set(all.map(getDayString));

  let streak = 0;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  const todayStr = getDayString(today.getTime());
  const yesterdayStr = getDayString(yesterday.getTime());

  if (!daysSet.has(todayStr) && !daysSet.has(yesterdayStr)) {
    return 0; // The streak is broken if neither today nor yesterday has activity
  }

  // Count backwards starting from today
  let currDate = new Date(today);
  while (daysSet.has(getDayString(currDate.getTime()))) {
    streak++;
    currDate.setDate(currDate.getDate() - 1);
  }

  // If today is empty but yesterday had activity, we count from yesterday
  if (streak === 0 && daysSet.has(yesterdayStr)) {
    currDate = new Date(yesterday);
    while (daysSet.has(getDayString(currDate.getTime()))) {
      streak++;
      currDate.setDate(currDate.getDate() - 1);
    }
  }

  return streak;
}

/** Counts unique types of content the user has touched (breadth metric). */
export function getKnowledgeCoverage(): Record<ProgressContentType, number> {
  const all = loadAll();
  const coverage: Partial<Record<ProgressContentType, number>> = {};
  for (const p of Object.values(all)) {
    if (p.state !== 'not_started') {
      coverage[p.type] = (coverage[p.type] ?? 0) + 1;
    }
  }
  return coverage as Record<ProgressContentType, number>;
}
