/**
 * Dojo submission tracking — localStorage-backed.
 * Mirrors writes to the shared progress engine for dashboard integration.
 */

import { markCompleted, isCompleted } from '@/lib/progress';

const STORAGE_KEY = 'dojo:submissions';

export type SubmissionStatus =
  | 'accepted'
  | 'wrong_answer'
  | 'error'
  | 'timeout';

export interface TestResult {
  index: number;
  passed: boolean;
  input?: Record<string, unknown>;
  expected: unknown;
  actual: unknown;
  runtime_ms: number | null;
  error?: string;
}

export interface Submission {
  id: string;
  slug: string;
  code: string;
  status: SubmissionStatus;
  passedTests: number;
  totalTests: number;
  runtimeMs: number | null;
  submittedAt: string; // ISO string
  results: TestResult[];
}

export interface DojoProblemProgress {
  slug: string;
  solved: boolean;
  attempted: boolean;
  lastSubmittedAt: string | null;
  bestRuntimeMs: number | null;
  submissionCount: number;
}

// ---------------------------------------------------------------------------
// Storage helpers
// ---------------------------------------------------------------------------

function loadAll(): Record<string, Submission[]> {
  if (typeof window === 'undefined') return {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveAll(data: Record<string, Submission[]>): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch {}
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function saveSubmission(submission: Submission): void {
  const all = loadAll();
  const existing = all[submission.slug] ?? [];
  all[submission.slug] = [submission, ...existing].slice(0, 50); // keep last 50
  saveAll(all);

  // Mirror accepted submissions to the global progress engine
  if (submission.status === 'accepted') {
    markCompleted('problem', submission.slug);
  }
}

export function getSubmissions(slug: string): Submission[] {
  return loadAll()[slug] ?? [];
}

export function getAllSubmissions(): Submission[] {
  const all = loadAll();
  return Object.values(all)
    .flat()
    .sort(
      (a, b) =>
        new Date(b.submittedAt).getTime() - new Date(a.submittedAt).getTime(),
    );
}

export function getProblemProgress(slug: string): DojoProblemProgress {
  const subs = getSubmissions(slug);
  const solved = subs.some((s) => s.status === 'accepted') || isCompleted('problem', slug);
  const best = subs
    .filter((s) => s.status === 'accepted' && s.runtimeMs !== null)
    .map((s) => s.runtimeMs as number)
    .sort((a, b) => a - b)[0] ?? null;

  return {
    slug,
    solved,
    attempted: subs.length > 0,
    lastSubmittedAt: subs[0]?.submittedAt ?? null,
    bestRuntimeMs: best,
    submissionCount: subs.length,
  };
}

export function getDojoStats(): {
  solved: number;
  attempted: number;
  totalSubmissions: number;
} {
  const all = loadAll();
  const slugs = Object.keys(all);
  let solved = 0;
  let attempted = 0;
  let totalSubmissions = 0;

  for (const slug of slugs) {
    const subs = all[slug];
    totalSubmissions += subs.length;
    if (subs.length > 0) attempted++;
    if (subs.some((s) => s.status === 'accepted')) solved++;
  }

  return { solved, attempted, totalSubmissions };
}

export function formatRuntime(ms: number | null): string {
  if (ms === null) return '—';
  if (ms < 1) return '<1 ms';
  return `${Math.round(ms)} ms`;
}

export function generateSubmissionId(): string {
  return `sub_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}
