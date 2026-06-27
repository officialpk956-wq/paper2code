'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';
import type { Problem } from '@/data/problems';
import { TheoryPanel } from './TheoryPanel';
import { DojoEditor } from './DojoEditor';
import { TestResultPanel } from './TestResultPanel';
import { SubmissionHistory } from './SubmissionHistory';
import type { TestResult, Submission } from '@/lib/dojo';
import {
  saveSubmission,
  getProblemProgress,
  generateSubmissionId,
} from '@/lib/dojo';
import type { RunState } from './DojoEditor';

// ---------------------------------------------------------------------------
// Related Research panel (Phase 16F)
// ---------------------------------------------------------------------------

const PAPER_LABELS: Record<string, string> = {
  'attention-is-all-you-need': 'Attention Is All You Need',
  'deep-residual-learning':    'Deep Residual Learning',
  'gpt-3':                     'Language Models are Few-Shot Learners',
  'gpt':                       'Improving Language Understanding by GPT',
  'bert':                      'BERT: Pre-training of Transformers',
};

const ARCH_LABELS: Record<string, string> = {
  'transformer': 'Transformer Architecture',
  'resnet':      'ResNet Architecture',
  'bert':        'BERT Architecture',
  'gpt':         'GPT Architecture',
  'vit':         'Vision Transformer (ViT)',
  'llama':       'LLaMA Architecture',
};

function RelatedResearchPanel({ problem }: { problem: Problem }) {
  const papers = problem.relatedPapers.filter(s => PAPER_LABELS[s]);
  const archs  = problem.relatedArchitectures.filter(s => ARCH_LABELS[s]);
  if (papers.length === 0 && archs.length === 0) return null;

  return (
    <div
      className="border-t px-4 py-3 flex-shrink-0"
      style={{ borderColor: 'var(--color-border)', background: 'var(--bg-surface)' }}
    >
      <p
        className="text-[11px] font-bold uppercase tracking-wider mb-2"
        style={{ color: 'var(--color-text-tertiary)' }}
      >
        Related Research
      </p>
      <div className="flex flex-wrap gap-2">
        {papers.map(slug => (
          <Link
            key={`paper-${slug}`}
            href={`/papers/${slug}`}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-opacity hover:opacity-80"
            style={{ background: 'rgba(99,102,241,0.12)', color: 'var(--accent-primary)' }}
          >
            📄 {PAPER_LABELS[slug]}
          </Link>
        ))}
        {archs.map(slug => (
          <Link
            key={`arch-${slug}`}
            href={`/architectures/${slug}`}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-opacity hover:opacity-80"
            style={{ background: 'rgba(6,182,212,0.12)', color: 'var(--accent-cyan)' }}
          >
            🏗 {ARCH_LABELS[slug]}
          </Link>
        ))}
      </div>
    </div>
  );
}

type LeftTab = 'problem' | 'theory' | 'hints' | 'history';

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

function extractFunctionName(template: string): string {
  const match = template.match(/def\s+([a-zA-Z_]\w*)\s*\(/);
  return match ? match[1] : 'solution';
}

interface DojoProblemPageProps {
  problem: Problem;
}

export function DojoProblemPage({ problem }: DojoProblemPageProps) {
  const [leftTab, setLeftTab] = useState<LeftTab>('problem');
  const [runState, setRunState] = useState<RunState>('idle');
  const [submitState, setSubmitState] = useState<RunState>('idle');
  const [runResults, setRunResults] = useState<TestResult[]>([]);
  const [submitStatus, setSubmitStatus] = useState<Submission['status'] | null>(null);
  const [submitResults, setSubmitResults] = useState<TestResult[]>([]);
  const [submitPassedTests, setSubmitPassedTests] = useState<number>(0);
  const [submitTotalTests, setSubmitTotalTests] = useState<number>(0);
  const [submitRuntimeMs, setSubmitRuntimeMs] = useState<number | null>(null);
  const [runError, setRunError] = useState<string | undefined>();
  const [submitError, setSubmitError] = useState<string | undefined>();
  const [activeResultTab, setActiveResultTab] = useState<'run' | 'submit'>('run');
  const [submissionKey, setSubmissionKey] = useState(0);

  const progress = getProblemProgress(problem.slug);
  const functionName = extractFunctionName(problem.pythonTemplate);

  const handleRun = useCallback(
    async (code: string) => {
      setRunState('running');
      setRunError(undefined);
      setActiveResultTab('run');

      try {
        const res = await fetch('/api/dojo/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            code,
            testCases: problem.testCases,
            functionName,
            visibleOnly: true,
          }),
        });

        const data = await res.json();

        if (!res.ok) {
          setRunError(data.error ?? 'Execution failed');
          setRunState('error');
          return;
        }

        setRunResults(data.results ?? []);
        setRunState('done');
      } catch (err) {
        setRunError(err instanceof Error ? err.message : 'Network error');
        setRunState('error');
      }
    },
    [problem.testCases, functionName],
  );

  const handleSubmit = useCallback(
    async (code: string) => {
      setSubmitState('running');
      setSubmitError(undefined);
      setActiveResultTab('submit');

      try {
        const res = await fetch('/api/dojo/submit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            code,
            testCases: problem.testCases,
            functionName,
          }),
        });

        const data = await res.json();

        if (!res.ok) {
          setSubmitError(data.error ?? 'Submission failed');
          setSubmitState('error');
          return;
        }

        const status: Submission['status'] = data.status ?? 'error';
        const results: TestResult[] = data.results ?? [];
        const passedTests: number = data.passedTests ?? 0;
        const totalTests: number = data.totalTests ?? problem.testCases.length;
        const runtimeMs: number | null = data.runtimeMs ?? null;

        setSubmitResults(results);
        setSubmitStatus(status);
        setSubmitPassedTests(passedTests);
        setSubmitTotalTests(totalTests);
        setSubmitRuntimeMs(runtimeMs);
        setSubmitState('done');

        // Persist to localStorage
        const submission: Submission = {
          id: generateSubmissionId(),
          slug: problem.slug,
          code,
          status,
          passedTests,
          totalTests,
          runtimeMs,
          submittedAt: new Date().toISOString(),
          results,
        };
        saveSubmission(submission);
        setSubmissionKey((k) => k + 1);

        // Switch to history tab to show the new submission
        if (status === 'accepted') {
          setLeftTab('history');
        }
      } catch (err) {
        setSubmitError(err instanceof Error ? err.message : 'Network error');
        setSubmitState('error');
      }
    },
    [problem.testCases, problem.slug, functionName],
  );

  const isSolved = progress.solved || submitStatus === 'accepted';

  return (
    <div className="flex flex-col h-screen overflow-hidden" style={{ background: 'var(--bg-body)' }}>
      {/* Top Bar */}
      <header
        className="flex items-center justify-between px-4 h-12 border-b flex-shrink-0 gap-4"
        style={{ background: 'var(--bg-surface)', borderColor: 'var(--color-border)' }}
      >
        <div className="flex items-center gap-2 min-w-0">
          <Link
            href="/dojo"
            aria-label="Back to Problems"
            className="text-xs text-[--color-text-muted] hover:text-[--accent-primary] transition-colors flex-shrink-0"
          >
            ← Problems
          </Link>
          <span className="text-[--color-text-muted]">/</span>
          <span className="text-sm font-medium text-[--color-text-primary] truncate">
            {problem.title}
          </span>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          {isSolved && (
            <span
              className="text-xs px-2 py-0.5 rounded"
              style={{
                background: 'rgba(16,185,129,0.15)',
                color: '#10b981',
                border: '1px solid rgba(16,185,129,0.3)',
              }}
            >
              ✓ Solved
            </span>
          )}
          <span className={`badge ${DIFFICULTY_STYLES[problem.difficulty]}`}>
            {DIFFICULTY_LABELS[problem.difficulty]}
          </span>
          <span className="text-xs text-[--color-text-muted]">
            ~{problem.estimatedTime} min
          </span>
        </div>
      </header>

      {/* Body: Left Panel + Right Panel */}
      <div className="flex flex-1 min-h-0">
        {/* Left Panel (40%) — problem/theory/hints */}
        <div
          className="w-[42%] min-w-[320px] max-w-[520px] flex flex-col border-r flex-shrink-0"
          style={{ borderColor: 'var(--color-border)' }}
        >
          {leftTab !== 'history' ? (
            <TheoryPanel
              problem={problem}
              activeTab={leftTab}
              onTabChange={(tab) => setLeftTab(tab as LeftTab)}
            />
          ) : (
            <div className="flex flex-col h-full">
              {/* Tab Bar (mirrors TheoryPanel's tab bar) */}
              <div role="tablist" aria-label="Problem sections" className="flex border-b border-[--color-border] flex-shrink-0">
                {(['problem', 'theory', 'hints', 'history'] as const).map((t) => (
                  <button
                    key={t}
                    role="tab"
                    aria-selected={leftTab === t}
                    onClick={() => setLeftTab(t)}
                    className={`px-4 py-3 text-sm font-medium transition-colors border-b-2 -mb-px capitalize ${
                      leftTab === t
                        ? 'border-[--accent-primary] text-[--accent-primary]'
                        : 'border-transparent text-[--color-text-secondary] hover:text-[--color-text-primary]'
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
              <div className="flex-1 overflow-y-auto">
                {isSolved && <RelatedResearchPanel problem={problem} />}
                <SubmissionHistory key={submissionKey} slug={problem.slug} />
              </div>
            </div>
          )}
        </div>

        {/* Right Panel (60%) — editor on top, results on bottom */}
        <div className="flex flex-1 flex-col min-w-0 min-h-0">
          {/* Editor — 60% height */}
          <div className="flex-[3] min-h-0">
            <DojoEditor
              initialCode={problem.pythonTemplate}
              onRun={handleRun}
              onSubmit={handleSubmit}
              runState={runState}
              submitState={submitState}
            />
          </div>

          {/* Results Panel — 40% height */}
          <div
            className="flex-[2] min-h-0 border-t flex flex-col"
            style={{ borderColor: 'rgba(255,255,255,0.08)' }}
          >
            {/* Result Tab Bar */}
            <div
              role="tablist"
              aria-label="Result panels"
              className="flex items-center gap-1 px-4 py-2 border-b flex-shrink-0"
              style={{
                background: 'var(--bg-surface)',
                borderColor: 'var(--color-border)',
              }}
            >
              <button
                role="tab"
                aria-selected={activeResultTab === 'run'}
                onClick={() => setActiveResultTab('run')}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  activeResultTab === 'run'
                    ? 'text-[--accent-primary]'
                    : 'text-[--color-text-muted] hover:text-[--color-text-secondary]'
                }`}
                style={
                  activeResultTab === 'run'
                    ? { background: 'rgba(var(--accent-primary-rgb, 99,102,241),0.12)' }
                    : {}
                }
              >
                Test Output
              </button>
              <button
                role="tab"
                aria-selected={activeResultTab === 'submit'}
                onClick={() => setActiveResultTab('submit')}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  activeResultTab === 'submit'
                    ? 'text-[--accent-primary]'
                    : 'text-[--color-text-muted] hover:text-[--color-text-secondary]'
                }`}
                style={
                  activeResultTab === 'submit'
                    ? { background: 'rgba(var(--accent-primary-rgb, 99,102,241),0.12)' }
                    : {}
                }
              >
                Submission
                {submitStatus === 'accepted' && (
                  <span className="ml-1.5 text-[#10b981]">✓</span>
                )}
              </button>
            </div>

            {/* Result Content */}
            <div className="flex-1 min-h-0 overflow-hidden">
              {activeResultTab === 'run' ? (
                <TestResultPanel
                  results={runResults}
                  error={runError}
                  isRunning={runState === 'running'}
                  mode="run"
                  testCases={problem.testCases}
                />
              ) : (
                <TestResultPanel
                  results={submitResults}
                  error={submitError}
                  isRunning={submitState === 'running'}
                  mode="submit"
                  status={submitStatus}
                  passedTests={submitPassedTests}
                  totalTests={submitTotalTests}
                  runtimeMs={submitRuntimeMs}
                  testCases={problem.testCases}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
