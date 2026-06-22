/**
 * Persistence layer — localStorage with hydration-safe reads and graceful
 * fallback when storage is unavailable (SSR, private browsing, quota errors).
 *
 * Usage:
 *   import { save, load, remove } from '@/lib/persistence';
 *   save('my-key', { foo: 'bar' });       // void
 *   load<{ foo: string }>('my-key');       // { foo: 'bar' } | null
 *   remove('my-key');                     // void
 */

// ---------------------------------------------------------------------------
// Core primitives
// ---------------------------------------------------------------------------

function isClient(): boolean {
  return typeof window !== 'undefined';
}

/** Persist any JSON-serializable value under `key`. */
export function save<T>(key: string, value: T): void {
  if (!isClient()) return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Quota exceeded or storage unavailable — fail silently
  }
}

/** Load a value previously saved under `key`. Returns null if absent/invalid. */
export function load<T>(key: string): T | null {
  if (!isClient()) return null;
  try {
    const raw = window.localStorage.getItem(key);
    if (raw === null) return null;
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

/** Remove a key from storage. */
export function remove(key: string): void {
  if (!isClient()) return;
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Fail silently
  }
}

// ---------------------------------------------------------------------------
// Namespaced storage keys
// ---------------------------------------------------------------------------

export const KEYS = {
  // Problem completion set
  completedProblems: 'p2c:completed-problems',
  // Per-problem notes: Record<slug, string>
  problemNotes: 'p2c:problem-notes',
  // Mastery levels: Record<slug, 0-4>
  masteryLevels: 'p2c:mastery-levels',
  // Bookmarked papers: string[]
  bookmarkedPapers: 'p2c:bookmarked-papers',
  // Saved research journeys: string[]
  savedJourneys: 'p2c:saved-journeys',
  // Version tags: Record<itemId, string[]>
  versionTags: 'p2c:version-tags',
  // Dashboard last-viewed problem
  lastProblemSlug: 'p2c:last-problem-slug',
  // User display name
  displayName: 'p2c:display-name',
  // Diagnostic skill profile
  userProfile: 'p2c:profile',
} as const;

// ---------------------------------------------------------------------------
// Typed domain helpers — built on top of save/load/remove
// ---------------------------------------------------------------------------

export type MasteryLevel = 0 | 1 | 2 | 3 | 4;

// --- Completed Problems ---

export function markProblemComplete(slug: string): void {
  const set = load<string[]>(KEYS.completedProblems) ?? [];
  if (!set.includes(slug)) {
    save(KEYS.completedProblems, [...set, slug]);
  }
}

export function markProblemIncomplete(slug: string): void {
  const set = load<string[]>(KEYS.completedProblems) ?? [];
  save(KEYS.completedProblems, set.filter((s) => s !== slug));
}

export function isProblemComplete(slug: string): boolean {
  const set = load<string[]>(KEYS.completedProblems) ?? [];
  return set.includes(slug);
}

export function getCompletedProblems(): string[] {
  return load<string[]>(KEYS.completedProblems) ?? [];
}

// --- Mastery Levels ---

export function setMastery(slug: string, level: MasteryLevel): void {
  const map = load<Record<string, MasteryLevel>>(KEYS.masteryLevels) ?? {};
  save(KEYS.masteryLevels, { ...map, [slug]: level });
}

export function getMastery(slug: string): MasteryLevel {
  const map = load<Record<string, MasteryLevel>>(KEYS.masteryLevels) ?? {};
  return map[slug] ?? 0;
}

export function getAllMasteryLevels(): Record<string, MasteryLevel> {
  return load<Record<string, MasteryLevel>>(KEYS.masteryLevels) ?? {};
}

// --- Problem Notes ---

export function saveProblemNote(slug: string, note: string): void {
  const map = load<Record<string, string>>(KEYS.problemNotes) ?? {};
  save(KEYS.problemNotes, { ...map, [slug]: note });
}

export function getProblemNote(slug: string): string {
  const map = load<Record<string, string>>(KEYS.problemNotes) ?? {};
  return map[slug] ?? '';
}

// --- Bookmarked Papers ---

export function bookmarkPaper(slug: string): void {
  const list = load<string[]>(KEYS.bookmarkedPapers) ?? [];
  if (!list.includes(slug)) {
    save(KEYS.bookmarkedPapers, [...list, slug]);
  }
}

export function unbookmarkPaper(slug: string): void {
  const list = load<string[]>(KEYS.bookmarkedPapers) ?? [];
  save(KEYS.bookmarkedPapers, list.filter((s) => s !== slug));
}

export function isPaperBookmarked(slug: string): boolean {
  const list = load<string[]>(KEYS.bookmarkedPapers) ?? [];
  return list.includes(slug);
}

export function getBookmarkedPapers(): string[] {
  return load<string[]>(KEYS.bookmarkedPapers) ?? [];
}

// --- Display Name ---

export function getDisplayName(): string {
  return load<string>(KEYS.displayName) ?? 'Researcher';
}

export function setDisplayName(name: string): void {
  save(KEYS.displayName, name.trim() || 'Researcher');
}

// --- Progress summary (for dashboard) ---

export interface ProgressSummary {
  completedCount: number;
  masteredCount: number;
  bookmarkCount: number;
  displayName: string;
}

export function getProgressSummary(): ProgressSummary {
  const completed = getCompletedProblems();
  const mastery = getAllMasteryLevels();
  const mastered = Object.values(mastery).filter((v) => v >= 3).length;
  const bookmarks = getBookmarkedPapers();
  return {
    completedCount: completed.length,
    masteredCount: mastered,
    bookmarkCount: bookmarks.length,
    displayName: getDisplayName(),
  };
}
