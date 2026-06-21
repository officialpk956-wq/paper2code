'use client';

import Link from 'next/link';
import { PROBLEMS } from '@/data/problems';

const TOPIC_TO_PROBLEM_SLUGS: Record<string, string[]> = {
  'attention':           ['scaled-dot-product-attention', 'positional-encoding', 'multi-head-attention', 'layer-normalization'],
  'multi-head-attention': ['scaled-dot-product-attention', 'multi-head-attention', 'masked-attention', 'layer-normalization'],
};

const DIFFICULTY_STYLES: Record<string, { text: string; bg: string; label: string }> = {
  beginner:     { text: '#10b981', bg: 'rgba(16,185,129,0.12)', label: 'Easy' },
  intermediate: { text: '#f59e0b', bg: 'rgba(245,158,11,0.12)', label: 'Medium' },
  advanced:     { text: '#ef4444', bg: 'rgba(239,68,68,0.12)', label: 'Hard' },
};

interface RecommendedProblemsProps {
  topicSlug: string;
}

export function RecommendedProblems({ topicSlug }: RecommendedProblemsProps) {
  const slugs = TOPIC_TO_PROBLEM_SLUGS[topicSlug];
  if (!slugs) return null;

  const problems = slugs
    .map(slug => PROBLEMS.find(p => p.slug === slug))
    .filter((p): p is NonNullable<typeof p> => p !== undefined);

  if (problems.length === 0) return null;

  return (
    <section
      id="recommended-problems"
      className="mb-8 mt-2"
      aria-labelledby="recommended-problems-heading"
    >
      <div className="flex items-center gap-3 mb-4">
        <div
          className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0"
          style={{ background: 'rgba(99,102,241,0.12)' }}
          aria-hidden="true"
        >
          <span style={{ fontSize: 15 }}>⚡</span>
        </div>
        <div>
          <h3
            id="recommended-problems-heading"
            className="text-[13px] font-bold"
            style={{ color: 'var(--color-text-primary)' }}
          >
            Recommended Problems
          </h3>
          <p className="text-[11px]" style={{ color: 'var(--color-text-tertiary)' }}>
            Practice what you just learned
          </p>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        {problems.map(problem => {
          const style = DIFFICULTY_STYLES[problem.difficulty] ?? DIFFICULTY_STYLES.intermediate;
          return (
            <div
              key={problem.slug}
              className="rounded-xl border p-3 flex items-center gap-3"
              style={{
                background: 'var(--bg-surface)',
                borderColor: 'var(--color-border)',
              }}
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <span
                    className="text-[10px] font-semibold rounded px-1.5 py-0.5"
                    style={{ color: style.text, background: style.bg }}
                  >
                    {style.label}
                  </span>
                  <span className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
                    ~{problem.estimatedTime} min
                  </span>
                </div>
                <p className="text-xs font-semibold truncate" style={{ color: 'var(--color-text-primary)' }}>
                  {problem.title}
                </p>
              </div>
              <Link
                href={`/dojo/${problem.slug}`}
                className="flex-shrink-0 px-3 py-1.5 rounded-lg text-xs font-semibold transition-opacity hover:opacity-80"
                style={{ background: 'var(--accent-primary)', color: '#fff' }}
              >
                Solve →
              </Link>
            </div>
          );
        })}
      </div>
    </section>
  );
}
