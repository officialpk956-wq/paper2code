'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import type { Problem, ProblemCategory, Difficulty } from '@/data/problems';
import { getProblemProgress } from '@/lib/dojo';

const CATEGORY_LABELS: Record<ProblemCategory, string> = {
  'linear-algebra': 'Linear Algebra',
  'deep-learning': 'Deep Learning',
  cnn: 'CNN',
  transformer: 'Transformer',
  'llm-engineering': 'LLM Engineering',
};

const DIFFICULTY_STYLES: Record<Difficulty, string> = {
  beginner: 'badge-easy',
  intermediate: 'badge-medium',
  advanced: 'badge-hard',
};

const DIFFICULTY_LABELS: Record<Difficulty, string> = {
  beginner: 'Easy',
  intermediate: 'Medium',
  advanced: 'Hard',
};

interface DojoProblemListProps {
  problems: Problem[];
}

export function DojoProblemList({ problems }: DojoProblemListProps) {
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState<ProblemCategory | 'all'>('all');
  const [difficulty, setDifficulty] = useState<Difficulty | 'all'>('all');

  const categories = useMemo<ProblemCategory[]>(
    () => [...new Set(problems.map((p) => p.category))],
    [problems],
  );

  const filtered = useMemo(() => {
    let result = problems;
    if (category !== 'all') result = result.filter((p) => p.category === category);
    if (difficulty !== 'all') result = result.filter((p) => p.difficulty === difficulty);
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (p) =>
          p.title.toLowerCase().includes(q) ||
          p.tags.some((t) => t.toLowerCase().includes(q)) ||
          p.category.includes(q),
      );
    }
    return result;
  }, [problems, category, difficulty, search]);

  const stats = useMemo(() => {
    const total = problems.length;
    const solved = problems.filter((p) => getProblemProgress(p.slug).solved).length;
    return { total, solved };
  }, [problems]);

  return (
    <div className="max-w-5xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <h1 className="text-2xl font-bold text-[--color-text-primary]">
            DS Coding Dojo
          </h1>
          <span
            className="text-xs px-2 py-0.5 rounded font-medium"
            style={{
              background: 'rgba(99,102,241,0.15)',
              color: 'var(--accent-primary)',
              border: '1px solid rgba(99,102,241,0.3)',
            }}
          >
            LeetCode for Data Science
          </span>
        </div>
        <p className="text-sm text-[--color-text-secondary]">
          Practice ML/DS problems with Python — real test cases, instant feedback.
        </p>
      </div>

      {/* Stats Bar */}
      <div className="flex items-center gap-6">
        <StatBadge label="Solved" value={`${stats.solved}/${stats.total}`} color="#10b981" />
        <StatBadge label="Easy" value={String(problems.filter((p) => p.difficulty === 'beginner').length)} color="#10b981" />
        <StatBadge label="Medium" value={String(problems.filter((p) => p.difficulty === 'intermediate').length)} color="#f59e0b" />
        <StatBadge label="Hard" value={String(problems.filter((p) => p.difficulty === 'advanced').length)} color="#ef4444" />
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Search */}
        <input
          type="text"
          placeholder="Search problems…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="input-field text-sm w-52"
        />

        {/* Category */}
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value as ProblemCategory | 'all')}
          className="input-field text-sm pr-8"
        >
          <option value="all">All Categories</option>
          {categories.map((c) => (
            <option key={c} value={c}>
              {CATEGORY_LABELS[c]}
            </option>
          ))}
        </select>

        {/* Difficulty */}
        <div className="flex gap-1">
          {(['all', 'beginner', 'intermediate', 'advanced'] as const).map((d) => (
            <button
              key={d}
              onClick={() => setDifficulty(d)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                difficulty === d
                  ? 'text-[--color-text-primary]'
                  : 'text-[--color-text-muted] hover:text-[--color-text-secondary]'
              }`}
              style={
                difficulty === d
                  ? { background: 'var(--bg-surface)', border: '1px solid var(--color-border)' }
                  : { border: '1px solid transparent' }
              }
            >
              {d === 'all' ? 'All' : DIFFICULTY_LABELS[d]}
            </button>
          ))}
        </div>

        <span className="text-xs text-[--color-text-muted] ml-auto">
          {filtered.length} problem{filtered.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Problem Table */}
      {filtered.length === 0 ? (
        <div className="text-center py-16 text-[--color-text-muted] text-sm">
          No problems match your filters.
        </div>
      ) : (
        <div
          className="rounded-xl overflow-hidden"
          style={{ border: '1px solid var(--color-border)' }}
        >
          {/* Table Header */}
          <div
            className="grid grid-cols-[2.5rem_1fr_9rem_9rem_6rem] px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-[--color-text-muted]"
            style={{ background: 'var(--bg-surface)', borderBottom: '1px solid var(--color-border)' }}
          >
            <span>#</span>
            <span>Title</span>
            <span>Category</span>
            <span>Difficulty</span>
            <span className="text-right">Time</span>
          </div>

          {/* Rows */}
          {filtered.map((problem, i) => (
            <ProblemRow key={problem.slug} problem={problem} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}

function ProblemRow({ problem, index }: { problem: Problem; index: number }) {
  const progress = getProblemProgress(problem.slug);

  return (
    <Link
      href={`/dojo/${problem.slug}`}
      className="grid grid-cols-[2.5rem_1fr_9rem_9rem_6rem] px-4 py-3.5 items-center transition-colors hover:bg-[--bg-hover] group"
      style={{
        borderBottom: '1px solid var(--color-divider)',
        background: index % 2 === 0 ? 'var(--bg-body)' : 'transparent',
      }}
    >
      {/* Number / Status */}
      <div className="text-xs font-mono">
        {progress.solved ? (
          <span style={{ color: '#10b981' }}>✓</span>
        ) : progress.attempted ? (
          <span style={{ color: '#f59e0b' }}>○</span>
        ) : (
          <span className="text-[--color-text-muted]">{index + 1}</span>
        )}
      </div>

      {/* Title + Tags */}
      <div className="min-w-0 pr-4">
        <span className="text-sm font-medium text-[--color-text-primary] group-hover:text-[--accent-primary] transition-colors">
          {problem.title}
        </span>
        <div className="flex gap-1 mt-1 flex-wrap">
          {problem.tags.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="text-xs px-1.5 rounded"
              style={{
                background: 'var(--bg-surface)',
                color: 'var(--color-text-muted)',
              }}
            >
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* Category */}
      <span className="text-xs text-[--color-text-muted]">
        {CATEGORY_LABELS[problem.category] ?? problem.category}
      </span>

      {/* Difficulty */}
      <span className={`badge ${DIFFICULTY_STYLES[problem.difficulty]} w-fit`}>
        {DIFFICULTY_LABELS[problem.difficulty]}
      </span>

      {/* Time */}
      <span className="text-xs text-[--color-text-muted] text-right">
        ~{problem.estimatedTime}m
      </span>
    </Link>
  );
}

function StatBadge({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-lg font-bold" style={{ color }}>
        {value}
      </span>
      <span className="text-xs text-[--color-text-muted]">{label}</span>
    </div>
  );
}
