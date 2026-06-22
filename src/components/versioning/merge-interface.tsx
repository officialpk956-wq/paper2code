'use client';

import { GitMerge, AlertCircle, CheckCircle, X } from 'lucide-react';
import { useState } from 'react';

interface ConflictBlock {
  id: string;
  type: 'conflict' | 'clean';
  lineStart: number;
  lineEnd: number;
  ourContent?: string;
  theirContent?: string;
  status: 'unresolved' | 'resolved_ours' | 'resolved_theirs' | 'resolved_manual';
}

const conflicts: ConflictBlock[] = [
  {
    id: 'conflict-1',
    type: 'conflict',
    lineStart: 15,
    lineEnd: 22,
    ourContent: '        self.scale = 1.0 / math.sqrt(self.d_k)',
    theirContent: '        self.scale = 1.0 / np.sqrt(self.d_k)',
    status: 'unresolved',
  },
  {
    id: 'conflict-2',
    type: 'conflict',
    lineStart: 45,
    lineEnd: 52,
    ourContent: '    def attention(self, q, k, v):\n        scores = torch.matmul(q, k.T)',
    theirContent: '    def attention(self, q, k, v):\n        scores = q @ k.T  # optimized',
    status: 'resolved_ours',
  },
];

interface MergeInterfaceProps {
  sourceBranch?: string;
  targetBranch?: string;
}

export function MergeInterface({
  sourceBranch = 'feature/attention-optimization',
  targetBranch = 'main',
}: MergeInterfaceProps) {
  const [conflictResolutions, setConflictResolutions] = useState<Record<string, string>>(
    Object.fromEntries(conflicts.map((c) => [c.id, c.status]))
  );

  const resolveConflict = (id: string, resolution: 'ours' | 'theirs' | 'manual') => {
    setConflictResolutions((prev) => ({
      ...prev,
      [id]: `resolved_${resolution}`,
    }));
  };

  const unresolvedCount = conflicts.filter(
    (c) => conflictResolutions[c.id] === 'unresolved'
  ).length;

  const resolvedCount = conflicts.length - unresolvedCount;

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border border-[--color-border] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-body]">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <GitMerge size={18} className="text-[--accent-primary]" />
              <h3 className="text-sm font-semibold text-[--color-text-primary]">
                Merge Review
              </h3>
            </div>
            <div className="flex items-center gap-2 text-xs text-[--color-text-secondary]">
              <span className="px-2 py-1 rounded bg-[--accent-primary]/20 text-[--accent-primary]">
                {sourceBranch}
              </span>
              <span>→</span>
              <span className="px-2 py-1 rounded bg-[--bg-surface]">
                {targetBranch}
              </span>
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="bg-[--bg-surface] rounded h-2 overflow-hidden">
          <div
            className="bg-green-500 h-full transition-all"
            style={{ width: `${(resolvedCount / conflicts.length) * 100}%` }}
          />
        </div>
        <div className="mt-2 text-xs text-[--color-text-secondary]">
          {resolvedCount} of {conflicts.length} conflicts resolved
        </div>
      </div>

      {/* Conflicts List */}
      <div className="flex-1 overflow-y-auto">
        <div className="divide-y divide-[--color-border]">
          {conflicts.map((conflict) => {
            const resolution = conflictResolutions[conflict.id];
            const isResolved = !resolution.includes('unresolved');

            return (
              <div
                key={conflict.id}
                className={`p-4 border-l-4 ${
                  isResolved
                    ? 'border-l-green-500 bg-green-500/5'
                    : 'border-l-yellow-500 bg-yellow-500/5'
                }`}
              >
                {/* Conflict Header */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    {isResolved ? (
                      <CheckCircle size={16} className="text-green-400" />
                    ) : (
                      <AlertCircle size={16} className="text-yellow-400" />
                    )}
                    <span className="text-sm font-semibold text-[--color-text-primary]">
                      Conflict at lines {conflict.lineStart}-{conflict.lineEnd}
                    </span>
                  </div>
                  <span className="text-xs px-2 py-1 rounded bg-[--bg-surface] text-[--color-text-tertiary]">
                    {isResolved ? 'Resolved' : 'Unresolved'}
                  </span>
                </div>

                {/* Conflict Content */}
                <div className="grid grid-cols-2 gap-3 mb-3 text-xs font-mono">
                  {/* Our Version */}
                  <div className="rounded-lg bg-blue-500/10 border border-blue-500/30 p-3">
                    <div className="text-blue-400 font-semibold mb-2 text-xs">
                      {targetBranch} (ours)
                    </div>
                    <div className="text-[--color-text-secondary] break-words whitespace-pre-wrap">
                      {conflict.ourContent}
                    </div>
                  </div>

                  {/* Their Version */}
                  <div className="rounded-lg bg-purple-500/10 border border-purple-500/30 p-3">
                    <div className="text-purple-400 font-semibold mb-2 text-xs">
                      {sourceBranch} (theirs)
                    </div>
                    <div className="text-[--color-text-secondary] break-words whitespace-pre-wrap">
                      {conflict.theirContent}
                    </div>
                  </div>
                </div>

                {/* Resolution Buttons */}
                {!isResolved ? (
                  <div className="flex gap-2">
                    <button
                      onClick={() => resolveConflict(conflict.id, 'ours')}
                      className="flex-1 px-3 py-2 rounded text-xs font-medium bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 transition-colors"
                    >
                      Use Ours
                    </button>
                    <button
                      onClick={() => resolveConflict(conflict.id, 'theirs')}
                      className="flex-1 px-3 py-2 rounded text-xs font-medium bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 transition-colors"
                    >
                      Use Theirs
                    </button>
                    <button
                      onClick={() => resolveConflict(conflict.id, 'manual')}
                      className="flex-1 px-3 py-2 rounded text-xs font-medium bg-[--accent-primary]/20 hover:bg-[--accent-primary]/30 text-[--accent-primary] transition-colors"
                    >
                      Edit Manually
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 px-3 py-2 rounded text-xs bg-green-500/10 text-green-400 border border-green-500/30">
                    <CheckCircle size={14} />
                    <span>Resolved: {resolution.split('_').pop()}</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-[--color-border] bg-[--bg-body] space-y-3">
        {unresolvedCount > 0 && (
          <div className="flex items-center gap-2 px-3 py-2 rounded bg-yellow-500/10 border border-yellow-500/30 text-xs text-yellow-400">
            <AlertCircle size={14} />
            <span>{unresolvedCount} conflict{unresolvedCount !== 1 ? 's' : ''} remaining</span>
          </div>
        )}

        <div className="flex gap-2">
          <button
            disabled={unresolvedCount > 0}
            className={`flex-1 px-3 py-2 rounded text-xs font-medium transition-colors ${
              unresolvedCount > 0
                ? 'bg-[--color-border] text-[--color-text-tertiary] cursor-not-allowed'
                : 'bg-[--accent-primary] hover:opacity-90 text-white'
            }`}
          >
            Complete Merge
          </button>
          <button className="px-3 py-2 rounded text-xs font-medium bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
            <X size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
