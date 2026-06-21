'use client';

import { useEffect, useState } from 'react';
import type { Submission } from '@/lib/dojo';
import { getSubmissions, formatRuntime } from '@/lib/dojo';

interface SubmissionHistoryProps {
  slug: string;
}

const STATUS_STYLES: Record<string, { label: string; color: string }> = {
  accepted: { label: 'Accepted', color: '#10b981' },
  wrong_answer: { label: 'Wrong Answer', color: '#f59e0b' },
  error: { label: 'Runtime Error', color: '#ef4444' },
  timeout: { label: 'TLE', color: '#ef4444' },
};

export function SubmissionHistory({ slug }: SubmissionHistoryProps) {
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  const [selected, setSelected] = useState<Submission | null>(null);

  useEffect(() => {
    setSubmissions(getSubmissions(slug));
  }, [slug]);

  if (submissions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 py-12">
        <span className="text-3xl opacity-30">📋</span>
        <p className="text-sm text-[--color-text-muted]">No submissions yet</p>
        <p className="text-xs text-[--color-text-muted]">Submit your solution to see history</p>
      </div>
    );
  }

  if (selected) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex items-center gap-2 p-4 border-b border-[--color-border] flex-shrink-0">
          <button
            onClick={() => setSelected(null)}
            className="text-xs text-[--accent-primary] hover:underline"
          >
            ← Back
          </button>
          <span className="text-xs text-[--color-text-muted]">·</span>
          <span
            className="text-xs font-medium"
            style={{ color: STATUS_STYLES[selected.status]?.color ?? 'inherit' }}
          >
            {STATUS_STYLES[selected.status]?.label ?? selected.status}
          </span>
          <span className="text-xs text-[--color-text-muted] ml-auto">
            {new Date(selected.submittedAt).toLocaleString()}
          </span>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          <pre
            className="text-xs font-mono leading-relaxed p-4 rounded-lg overflow-x-auto"
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--color-border)',
              color: 'var(--color-text-secondary)',
            }}
          >
            {selected.code}
          </pre>
          <div className="mt-4 flex gap-4 text-xs text-[--color-text-muted]">
            <span>{selected.passedTests}/{selected.totalTests} tests</span>
            <span>{formatRuntime(selected.runtimeMs)}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2 p-4">
      {submissions.map((sub) => {
        const style = STATUS_STYLES[sub.status];
        return (
          <button
            key={sub.id}
            onClick={() => setSelected(sub)}
            className="w-full text-left rounded-lg px-4 py-3 transition-colors hover:opacity-90"
            style={{ background: 'var(--bg-surface)', border: '1px solid var(--color-border)' }}
          >
            <div className="flex items-center justify-between">
              <span
                className="text-sm font-medium"
                style={{ color: style?.color ?? 'var(--color-text-primary)' }}
              >
                {style?.label ?? sub.status}
              </span>
              <span className="text-xs text-[--color-text-muted]">
                {new Date(sub.submittedAt).toLocaleTimeString()}
              </span>
            </div>
            <div className="flex items-center gap-3 mt-1 text-xs text-[--color-text-muted]">
              <span>{sub.passedTests}/{sub.totalTests} tests</span>
              <span>{formatRuntime(sub.runtimeMs)}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
