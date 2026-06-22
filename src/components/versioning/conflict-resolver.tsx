'use client';

import { AlertTriangle, CheckCircle, Edit2, RefreshCw } from 'lucide-react';
import { useState } from 'react';

interface Conflict {
  id: string;
  file: string;
  line: number;
  severity: 'high' | 'medium' | 'low';
  type: string;
}

const conflicts: Conflict[] = [
  {
    id: 'conflict-1',
    file: 'attention.py',
    line: 42,
    severity: 'high',
    type: 'Import statement mismatch',
  },
  {
    id: 'conflict-2',
    file: 'model.py',
    line: 105,
    severity: 'high',
    type: 'Function signature changed',
  },
  {
    id: 'conflict-3',
    file: 'utils.py',
    line: 23,
    severity: 'medium',
    type: 'Variable initialization',
  },
  {
    id: 'conflict-4',
    file: 'config.py',
    line: 8,
    severity: 'low',
    type: 'Configuration parameter',
  },
];

interface ConflictResolverProps {
  title?: string;
  description?: string;
}

export function ConflictResolver({
  title = 'Merge Conflicts Detected',
  description = 'Resolve conflicts before completing the merge',
}: ConflictResolverProps) {
  const [resolvedConflicts, setResolvedConflicts] = useState<Set<string>>(new Set());

  const toggleResolved = (id: string) => {
    setResolvedConflicts((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const resolutionRate = (resolvedConflicts.size / conflicts.length) * 100;
  const unresolvedCount = conflicts.length - resolvedConflicts.size;

  const highSeverityCount = conflicts.filter((c) => c.severity === 'high').length;
  const mediumSeverityCount = conflicts.filter((c) => c.severity === 'medium').length;
  const lowSeverityCount = conflicts.filter((c) => c.severity === 'low').length;

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border border-[--color-border] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-body]">
        <div className="flex items-center gap-2 mb-2">
          <AlertTriangle size={18} className="text-yellow-400" />
          <h3 className="text-sm font-semibold text-[--color-text-primary]">
            {title}
          </h3>
        </div>
        <p className="text-xs text-[--color-text-secondary] ml-6">
          {description}
        </p>

        {/* Stats Grid */}
        <div className="grid grid-cols-4 gap-3 mt-4">
          <div className="px-3 py-2 rounded-lg bg-[--bg-panel] text-center">
            <div className="text-lg font-bold text-yellow-400">{unresolvedCount}</div>
            <div className="text-xs text-[--color-text-tertiary]">Unresolved</div>
          </div>
          <div className="px-3 py-2 rounded-lg bg-[--bg-panel] text-center">
            <div className="text-lg font-bold text-red-400">{highSeverityCount}</div>
            <div className="text-xs text-[--color-text-tertiary]">High</div>
          </div>
          <div className="px-3 py-2 rounded-lg bg-[--bg-panel] text-center">
            <div className="text-lg font-bold text-yellow-400">{mediumSeverityCount}</div>
            <div className="text-xs text-[--color-text-tertiary]">Medium</div>
          </div>
          <div className="px-3 py-2 rounded-lg bg-[--bg-panel] text-center">
            <div className="text-lg font-bold text-blue-400">{lowSeverityCount}</div>
            <div className="text-xs text-[--color-text-tertiary]">Low</div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-4">
          <div className="flex justify-between mb-1">
            <span className="text-xs font-medium text-[--color-text-secondary]">
              Resolution Progress
            </span>
            <span className="text-xs font-bold text-[--accent-primary]">
              {Math.round(resolutionRate)}%
            </span>
          </div>
          <div className="bg-[--bg-surface] rounded h-2 overflow-hidden">
            <div
              className="bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] h-full transition-all"
              style={{ width: `${resolutionRate}%` }}
            />
          </div>
        </div>
      </div>

      {/* Conflicts List */}
      <div className="flex-1 overflow-y-auto">
        <div className="divide-y divide-[--color-border]">
          {conflicts.map((conflict) => {
            const isResolved = resolvedConflicts.has(conflict.id);
            const severityColor = {
              high: 'bg-red-500/10 border-l-4 border-l-red-500',
              medium: 'bg-yellow-500/10 border-l-4 border-l-yellow-500',
              low: 'bg-blue-500/10 border-l-4 border-l-blue-500',
            };

            const severityBadgeColor = {
              high: 'bg-red-500/20 text-red-400',
              medium: 'bg-yellow-500/20 text-yellow-400',
              low: 'bg-blue-500/20 text-blue-400',
            };

            return (
              <div
                key={conflict.id}
                className={`p-4 transition-all cursor-pointer hover:bg-[--bg-surface] ${severityColor[conflict.severity]} ${
                  isResolved ? 'opacity-60' : ''
                }`}
                onClick={() => toggleResolved(conflict.id)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-3 flex-1">
                    <div className="mt-1">
                      {isResolved ? (
                        <CheckCircle size={18} className="text-green-400" />
                      ) : (
                        <AlertTriangle size={18} className="text-yellow-400" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-semibold text-[--color-text-primary]">
                        {conflict.file}
                        <span className="text-xs text-[--color-text-tertiary] ml-2">
                          line {conflict.line}
                        </span>
                      </div>
                      <div className="text-xs text-[--color-text-secondary] mt-0.5">
                        {conflict.type}
                      </div>
                    </div>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs font-medium flex-shrink-0 ${severityBadgeColor[conflict.severity]}`}>
                    {conflict.severity.charAt(0).toUpperCase() + conflict.severity.slice(1)}
                  </span>
                </div>

                {/* Expanded Actions */}
                {!isResolved && (
                  <div className="flex gap-2 ml-9 mt-2 pt-2 border-t border-[--color-border]">
                    <button className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 rounded text-xs font-medium bg-[--bg-panel] hover:bg-[--bg-surface] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
                      <Edit2 size={12} />
                      Review
                    </button>
                    <button className="px-2 py-1.5 rounded text-xs font-medium bg-[--accent-primary]/20 hover:bg-[--accent-primary]/30 text-[--accent-primary] transition-colors">
                      <RefreshCw size={12} />
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-[--color-border] bg-[--bg-body] space-y-3">
        {unresolvedCount > 0 ? (
          <div className="flex items-center gap-2 px-3 py-2 rounded text-xs bg-yellow-500/10 text-yellow-400 border border-yellow-500/30">
            <AlertTriangle size={14} />
            <span>Resolve all conflicts before merging</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-2 rounded text-xs bg-green-500/10 text-green-400 border border-green-500/30">
            <CheckCircle size={14} />
            <span>All conflicts resolved</span>
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
          <button className="px-3 py-2 rounded text-xs font-medium bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary] transition-colors">
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
