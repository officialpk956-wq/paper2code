'use client';

import type { TestResult } from '@/lib/dojo';
import type { TestCase } from '@/data/problems';

interface TestResultPanelProps {
  results: TestResult[];
  error?: string;
  isRunning?: boolean;
  mode: 'run' | 'submit';
  status?: 'accepted' | 'wrong_answer' | 'error' | 'timeout' | null;
  passedTests?: number;
  totalTests?: number;
  runtimeMs?: number | null;
  testCases?: TestCase[];
}

const STATUS_CONFIG = {
  accepted:     { label: 'Accepted',           color: 'var(--color-success)', bg: 'rgba(var(--color-success-rgb), 0.12)', border: 'rgba(var(--color-success-rgb), 0.30)' },
  wrong_answer: { label: 'Wrong Answer',        color: 'var(--color-warning)', bg: 'rgba(var(--color-warning-rgb), 0.12)', border: 'rgba(var(--color-warning-rgb), 0.30)' },
  error:        { label: 'Runtime Error',       color: 'var(--color-error)',   bg: 'rgba(var(--color-error-rgb),   0.12)', border: 'rgba(var(--color-error-rgb),   0.30)' },
  timeout:      { label: 'Time Limit Exceeded', color: 'var(--color-error)',   bg: 'rgba(var(--color-error-rgb),   0.12)', border: 'rgba(var(--color-error-rgb),   0.30)' },
};

export function TestResultPanel({
  results,
  error,
  isRunning = false,
  mode,
  status,
  passedTests,
  totalTests,
  runtimeMs,
  testCases,
}: TestResultPanelProps) {
  if (isRunning) {
    return (
      <div className="flex items-center gap-3 h-full px-5 py-4">
        <span
          className="inline-block w-4 h-4 border-2 border-t-transparent rounded-full animate-spin"
          style={{ borderColor: 'var(--accent-primary)', borderTopColor: 'transparent' }}
        />
        <span className="text-sm text-[--color-text-secondary]">
          {mode === 'run' ? 'Running against sample cases…' : 'Running all test cases…'}
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4">
        <div
          className="rounded-lg p-4"
          style={{
            background: 'rgba(var(--color-error-rgb), 0.08)',
            border: '1px solid rgba(var(--color-error-rgb), 0.25)',
          }}
        >
          <p className="text-xs font-semibold mb-2" style={{ color: 'var(--color-error)' }}>Execution Error</p>
          <pre className="text-xs text-[--color-text-secondary] whitespace-pre-wrap font-mono">
            {error}
          </pre>
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    if (testCases && testCases.length > 0) {
      const visibleCases = testCases.filter((tc) => tc.visible);
      return (
        <div className="p-4 space-y-3 overflow-y-auto h-full">
          <div className="text-sm font-medium text-[--color-text-secondary] mb-2">
            Sample Test Cases
          </div>
          <div className="space-y-2">
            {visibleCases.map((tc, i) => (
              <div
                key={i}
                className="rounded-lg overflow-hidden border border-[--color-border] bg-[--bg-surface]"
              >
                <div className="px-3 py-2 border-b border-[--color-border] bg-[rgba(255,255,255,0.02)]">
                  <span className="text-xs font-medium text-[--color-text-primary]">Case {i + 1}</span>
                </div>
                <div className="px-3 py-3 space-y-2">
                  <InfoRow label="Input" value={JSON.stringify(tc.input)} />
                  <InfoRow label="Expected" value={JSON.stringify(tc.output)} color="var(--accent-cyan)" />
                </div>
              </div>
            ))}
          </div>
        </div>
      );
    }

    return (
      <div className="flex items-center justify-center h-full text-sm text-[--color-text-muted]">
        Run your code to see test results
      </div>
    );
  }

  const cfg = status ? STATUS_CONFIG[status] : null;

  return (
    <div className="p-4 space-y-3 overflow-y-auto h-full">
      {/* Verdict Banner (submit mode) */}
      {cfg && (
        <div
          className="rounded-lg px-4 py-3 flex items-center justify-between"
          style={{ background: cfg.bg, border: `1px solid ${cfg.border}` }}
        >
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold" style={{ color: cfg.color }}>
              {cfg.label}
            </span>
            {passedTests !== undefined && totalTests !== undefined && (
              <span className="text-xs" style={{ color: cfg.color, opacity: 0.8 }}>
                {passedTests}/{totalTests} tests passed
              </span>
            )}
          </div>
          {runtimeMs !== null && runtimeMs !== undefined && (
            <span className="text-xs text-[--color-text-muted]">
              {runtimeMs < 1 ? '<1 ms' : `${Math.round(runtimeMs)} ms`}
            </span>
          )}
        </div>
      )}

      {/* Individual Test Cases */}
      <div className="space-y-2">
        {results.map((result, i) => (
          <TestCaseRow key={i} result={result} index={i} />
        ))}
      </div>
    </div>
  );
}

function TestCaseRow({ result, index }: { result: TestResult; index: number }) {
  const passed = result.passed;
  const hasError = !!result.error;

  return (
    <details className="group rounded-lg overflow-hidden" open={!passed}>
      <summary
        className="flex items-center gap-3 px-3 py-2.5 cursor-pointer select-none"
        style={{
          background: passed
            ? 'rgba(var(--color-success-rgb), 0.08)'
            : hasError
              ? 'rgba(var(--color-error-rgb), 0.08)'
              : 'rgba(var(--color-warning-rgb), 0.08)',
          border: `1px solid ${
            passed
              ? 'rgba(var(--color-success-rgb), 0.20)'
              : hasError
                ? 'rgba(var(--color-error-rgb), 0.20)'
                : 'rgba(var(--color-warning-rgb), 0.20)'
          }`,
        }}
      >
        <span
          className="text-sm"
          style={{ color: passed ? 'var(--color-success)' : hasError ? 'var(--color-error)' : 'var(--color-warning)' }}
        >
          {passed ? '✓' : '✗'}
        </span>
        <span className="text-xs font-medium text-[--color-text-primary]">
          Case {index + 1}
        </span>
        {result.runtime_ms !== null && (
          <span className="text-xs text-[--color-text-muted] ml-auto">
            {result.runtime_ms < 1 ? '<1 ms' : `${Math.round(result.runtime_ms)} ms`}
          </span>
        )}
        <span className="text-[--color-text-muted] group-open:rotate-180 transition-transform text-xs ml-1">
          ▼
        </span>
      </summary>
      <div
        className="px-3 py-3 space-y-2"
        style={{ background: 'var(--bg-body)', borderTop: '1px solid var(--color-border)' }}
      >
        {result.error ? (
          <div>
            <p className="text-xs text-[--color-text-muted] mb-1">Error</p>
            <pre className="text-xs font-mono whitespace-pre-wrap" style={{ color: 'var(--color-error)' }}>
              {result.error}
            </pre>
          </div>
        ) : (
          <>
            {result.input && (
              <InfoRow label="Input" value={JSON.stringify(result.input)} />
            )}
            <InfoRow label="Expected" value={JSON.stringify(result.expected)} color="var(--accent-cyan)" />
            <InfoRow
              label="Output"
              value={result.actual !== null ? JSON.stringify(result.actual) : 'None'}
              color={passed ? 'var(--color-success)' : 'var(--color-warning)'}
            />
          </>
        )}
      </div>
    </details>
  );
}

function InfoRow({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex gap-2 text-xs font-mono">
      <span className="text-[--color-text-muted] flex-shrink-0 w-16">{label}:</span>
      <span style={{ color: color ?? 'var(--color-text-secondary)' }}>{value}</span>
    </div>
  );
}
