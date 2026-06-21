/**
 * Study Session Tracking
 * Records session boundaries, active time, and content visited.
 * All data stored in localStorage; SSR-safe via isClient guard.
 */

import { save, load } from '@/lib/persistence';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface StudySession {
  id: string;
  startedAt: number;
  endedAt?: number;
  activeMs: number;
  contentVisited: string[]; // "type:slug" entries
}

export interface DailyStudy {
  date: string; // "YYYY-MM-DD"
  totalMs: number;
  sessionCount: number;
  contentVisited: string[];
}

// ---------------------------------------------------------------------------
// Storage keys
// ---------------------------------------------------------------------------

const SESSIONS_KEY = 'p2c:study-sessions';
const CURRENT_KEY = 'p2c:current-session';
const DAILY_KEY = 'p2c:daily-study';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isClient(): boolean {
  return typeof window !== 'undefined';
}

function dateKey(ts = Date.now()): string {
  const d = new Date(ts);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

export function startSession(): StudySession | null {
  if (!isClient()) return null;
  const session: StudySession = {
    id: String(Date.now()),
    startedAt: Date.now(),
    activeMs: 0,
    contentVisited: [],
  };
  save(CURRENT_KEY, session);
  return session;
}

export function getCurrentSession(): StudySession | null {
  return load<StudySession>(CURRENT_KEY);
}

export function recordContentVisit(type: string, slug: string): void {
  const session = getCurrentSession();
  if (!session) return;
  const entry = `${type}:${slug}`;
  if (!session.contentVisited.includes(entry)) {
    session.contentVisited = [...session.contentVisited, entry];
    save(CURRENT_KEY, session);
  }
}

export function updateSessionTime(activeMs: number): void {
  const session = getCurrentSession();
  if (!session) return;
  save(CURRENT_KEY, { ...session, activeMs });
}

export function endSession(): void {
  const session = getCurrentSession();
  if (!session) return;

  const now = Date.now();
  const ended: StudySession = {
    ...session,
    endedAt: now,
    activeMs: now - session.startedAt,
  };

  // Append to history (keep 90 days)
  const cutoff = now - 90 * 24 * 60 * 60 * 1000;
  const history = (load<StudySession[]>(SESSIONS_KEY) ?? []).filter(
    (s) => s.startedAt > cutoff,
  );
  save(SESSIONS_KEY, [...history, ended]);

  // Update daily aggregate
  const daily = load<Record<string, DailyStudy>>(DAILY_KEY) ?? {};
  const today = dateKey();
  const existing: DailyStudy = daily[today] ?? {
    date: today,
    totalMs: 0,
    sessionCount: 0,
    contentVisited: [],
  };
  const mergedContent = Array.from(
    new Set([...existing.contentVisited, ...ended.contentVisited]),
  );
  daily[today] = {
    ...existing,
    totalMs: existing.totalMs + ended.activeMs,
    sessionCount: existing.sessionCount + 1,
    contentVisited: mergedContent,
  };
  save(DAILY_KEY, daily);

  save(CURRENT_KEY, null);
}

// ---------------------------------------------------------------------------
// Query API
// ---------------------------------------------------------------------------

export function getDailyStudy(days = 30): DailyStudy[] {
  const daily = load<Record<string, DailyStudy>>(DAILY_KEY) ?? {};
  const result: DailyStudy[] = [];
  for (let i = days - 1; i >= 0; i--) {
    const k = dateKey(Date.now() - i * 24 * 60 * 60 * 1000);
    result.push(
      daily[k] ?? { date: k, totalMs: 0, sessionCount: 0, contentVisited: [] },
    );
  }
  return result;
}

export function getTodayStudyMs(): number {
  const daily = load<Record<string, DailyStudy>>(DAILY_KEY) ?? {};
  return daily[dateKey()]?.totalMs ?? 0;
}

export function getWeekStudyMs(): number {
  return getDailyStudy(7).reduce((s, d) => s + d.totalMs, 0);
}

export function getMonthStudyMs(): number {
  return getDailyStudy(30).reduce((s, d) => s + d.totalMs, 0);
}

export function getTotalStudyMs(): number {
  const sessions = load<StudySession[]>(SESSIONS_KEY) ?? [];
  return sessions.reduce((s, sess) => s + sess.activeMs, 0);
}

export function getStudyStreak(): number {
  const daily = getDailyStudy(90);
  let streak = 0;
  // Work backwards from yesterday (today may still be accumulating)
  for (let i = daily.length - 2; i >= 0; i--) {
    if (daily[i].totalMs > 0) {
      streak++;
    } else {
      break;
    }
  }
  // Also count today if there's activity
  if (daily[daily.length - 1].totalMs > 0) streak++;
  return streak;
}

export function getRecentActivity(
  limit = 10,
): Array<{ type: string; slug: string; visitedAt: number }> {
  const sessions = load<StudySession[]>(SESSIONS_KEY) ?? [];
  const seen = new Set<string>();
  const results: Array<{ type: string; slug: string; visitedAt: number }> = [];

  for (const session of [...sessions].reverse()) {
    for (const entry of [...session.contentVisited].reverse()) {
      if (seen.has(entry)) continue;
      seen.add(entry);
      const colonIdx = entry.indexOf(':');
      if (colonIdx === -1) continue;
      results.push({
        type: entry.slice(0, colonIdx),
        slug: entry.slice(colonIdx + 1),
        visitedAt: session.endedAt ?? session.startedAt,
      });
      if (results.length >= limit) return results;
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Formatting helper
// ---------------------------------------------------------------------------

export function formatDuration(ms: number): string {
  if (ms <= 0) return '0m';
  const totalMinutes = Math.floor(ms / 60_000);
  if (totalMinutes < 1) return '< 1m';
  if (totalMinutes < 60) return `${totalMinutes}m`;
  const hours = Math.floor(totalMinutes / 60);
  const mins = totalMinutes % 60;
  return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
}
