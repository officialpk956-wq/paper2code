'use client';

import type { Problem } from '@/data/problems';

interface TheoryPanelProps {
  problem: Problem;
  activeTab: 'problem' | 'theory' | 'hints' | 'history';
  onTabChange: (tab: 'problem' | 'theory' | 'hints' | 'history') => void;
}

const DIFFICULTY_STYLES: Record<string, string> = {
  beginner: 'badge-easy',
  intermediate: 'badge-medium',
  advanced: 'badge-hard',
};

const DIFFICULTY_LABELS: Record<string, string> = {
  beginner: 'Easy',
  intermediate: 'Medium',
  advanced: 'Hard',
};

const CATEGORY_LABELS: Record<string, string> = {
  'linear-algebra': 'Linear Algebra',
  'deep-learning': 'Deep Learning',
  cnn: 'CNN',
  transformer: 'Transformer',
  'llm-engineering': 'LLM Engineering',
};

const TABS = [
  { key: 'problem', label: 'Problem' },
  { key: 'theory', label: 'Theory' },
  { key: 'hints', label: 'Hints' },
  { key: 'history', label: 'History' },
] as const;

export function TheoryPanel({
  problem,
  activeTab,
  onTabChange,
}: TheoryPanelProps) {
  return (
    <div className="flex flex-col h-full">
      {/* Tab Bar */}
      <div className="flex border-b border-[--color-border] flex-shrink-0">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => onTabChange(tab.key)}
            className={`px-4 py-3 text-sm font-medium transition-colors border-b-2 -mb-px ${
              activeTab === tab.key
                ? 'border-[--accent-primary] text-[--accent-primary]'
                : 'border-transparent text-[--color-text-secondary] hover:text-[--color-text-primary]'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-5 space-y-5">
        {activeTab === 'problem' && (
          <ProblemTab problem={problem} />
        )}
        {activeTab === 'theory' && (
          <TheoryTab problem={problem} />
        )}
        {activeTab === 'hints' && (
          <HintsTab problem={problem} />
        )}
        {activeTab === 'history' && (
          <div className="text-[--color-text-secondary] text-sm">
            Submission history appears here after you submit.
          </div>
        )}
      </div>
    </div>
  );
}

function ProblemTab({ problem }: { problem: Problem }) {
  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 flex-wrap mb-2">
          <span className={`badge ${DIFFICULTY_STYLES[problem.difficulty]}`}>
            {DIFFICULTY_LABELS[problem.difficulty]}
          </span>
          <span className="badge" style={{ background: 'var(--bg-surface)', color: 'var(--color-text-secondary)', border: '1px solid var(--color-border)' }}>
            {CATEGORY_LABELS[problem.category] ?? problem.category}
          </span>
          <span className="text-xs text-[--color-text-muted]">
            ~{problem.estimatedTime} min
          </span>
        </div>
        <h1 className="text-lg font-semibold text-[--color-text-primary]">
          {problem.title}
        </h1>
      </div>

      {/* Description */}
      <div>
        <p className="text-sm text-[--color-text-secondary] leading-relaxed whitespace-pre-line">
          {problem.description}
        </p>
      </div>

      {/* Learning Points */}
      {problem.learningPoints.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-widest text-[--color-text-muted] mb-2">
            Learning Objectives
          </h3>
          <ul className="space-y-1.5">
            {problem.learningPoints.map((point, i) => (
              <li key={i} className="flex gap-2 text-sm text-[--color-text-secondary]">
                <span className="text-[--accent-primary] mt-0.5 flex-shrink-0">▸</span>
                <span>{point}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Tags */}
      {problem.tags.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-widest text-[--color-text-muted] mb-2">
            Topics
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {problem.tags.map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 rounded text-xs"
                style={{
                  background: 'var(--bg-surface)',
                  color: 'var(--color-text-secondary)',
                  border: '1px solid var(--color-border)',
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Visible Test Cases */}
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-widest text-[--color-text-muted] mb-2">
          Examples
        </h3>
        <div className="space-y-3">
          {problem.testCases
            .filter((tc) => tc.visible)
            .map((tc, i) => (
              <div
                key={i}
                className="rounded-lg p-3 text-xs font-mono space-y-2"
                style={{ background: 'var(--bg-surface)', border: '1px solid var(--color-border)' }}
              >
                <div>
                  <span className="text-[--color-text-muted]">Input: </span>
                  <span className="text-[--color-text-secondary]">
                    {JSON.stringify(tc.input)}
                  </span>
                </div>
                <div>
                  <span className="text-[--color-text-muted]">Output: </span>
                  <span className="text-[--accent-cyan]">
                    {JSON.stringify(tc.output)}
                  </span>
                </div>
              </div>
            ))}
        </div>
        <p className="mt-2 text-xs text-[--color-text-muted]">
          + {problem.testCases.filter((tc) => !tc.visible).length} hidden test{problem.testCases.filter((tc) => !tc.visible).length !== 1 ? 's' : ''}
        </p>
      </div>
    </div>
  );
}

function TheoryTab({ problem }: { problem: Problem }) {
  return (
    <div className="space-y-5">
      <div>
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
          Step-by-Step Explanation
        </h3>
        <p className="text-sm text-[--color-text-secondary] leading-relaxed whitespace-pre-line">
          {problem.explanation.stepByStep}
        </p>
      </div>
      <div>
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
          Complexity Analysis
        </h3>
        <p className="text-sm text-[--color-text-secondary] leading-relaxed whitespace-pre-line">
          {problem.explanation.complexityAnalysis}
        </p>
      </div>
      <div>
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
          Common Mistakes
        </h3>
        <p className="text-sm text-[--color-text-secondary] leading-relaxed whitespace-pre-line">
          {problem.explanation.commonMistakes}
        </p>
      </div>
      {problem.relatedMath.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-widest text-[--color-text-muted] mb-2">
            Related Math
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {problem.relatedMath.map((m) => (
              <span
                key={m}
                className="px-2 py-0.5 rounded text-xs"
                style={{ background: 'rgba(139,92,246,0.12)', color: '#a78bfa', border: '1px solid rgba(139,92,246,0.25)' }}
              >
                {m}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function HintsTab({ problem }: { problem: Problem }) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-[--color-text-secondary]">
        Progressive hints — reveal one at a time to avoid spoilers.
      </p>

      <HintCard label="Hint 1" content={problem.hints.hint1} />
      <HintCard label="Hint 2" content={problem.hints.hint2} />
      <HintCard label="Hint 3" content={problem.hints.hint3} />

      <div className="pt-2 border-t border-[--color-border]">
        <HintCard label="Solution Outline" content={problem.hints.solutionOutline} accent />
      </div>

      <div>
        <h3 className="text-xs font-semibold uppercase tracking-widest text-[--color-text-muted] mb-2">
          Interview Discussion
        </h3>
        <p className="text-sm text-[--color-text-secondary] leading-relaxed whitespace-pre-line">
          {problem.explanation.interviewDiscussion}
        </p>
      </div>
    </div>
  );
}

function HintCard({
  label,
  content,
  accent = false,
}: {
  label: string;
  content: string;
  accent?: boolean;
}) {
  return (
    <details className="group rounded-lg overflow-hidden" style={{ border: `1px solid var(--color-border)` }}>
      <summary
        className="flex items-center justify-between px-4 py-2.5 cursor-pointer select-none text-sm font-medium"
        style={{ background: 'var(--bg-surface)' }}
      >
        <span style={{ color: accent ? 'var(--accent-primary)' : 'var(--color-text-primary)' }}>
          {label}
        </span>
        <span className="text-[--color-text-muted] group-open:rotate-180 transition-transform">▼</span>
      </summary>
      <div className="px-4 py-3" style={{ background: 'var(--bg-body)' }}>
        <p className="text-sm text-[--color-text-secondary] leading-relaxed whitespace-pre-line">
          {content}
        </p>
      </div>
    </details>
  );
}
