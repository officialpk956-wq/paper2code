'use client';

import { Clock, RotateCcw, Download, Trash2 } from 'lucide-react';
import { useState } from 'react';

interface Version {
  id: string;
  timestamp: string;
  author: string;
  description: string;
  changes: string;
  icon: string;
}

const versions: Version[] = [
  {
    id: 'current',
    timestamp: 'Now',
    author: 'You',
    description: 'Added attention head visualization',
    changes: '5 layers added, 2 modified',
    icon: '✏️',
  },
  {
    id: 'v1-2',
    timestamp: '2 hours ago',
    author: 'You',
    description: 'Updated transformer block structure',
    changes: '3 layers modified',
    icon: '⚙️',
  },
  {
    id: 'v1-1',
    timestamp: '5 hours ago',
    author: 'Sarah Chen',
    description: 'Added pooling layer',
    changes: '1 layer added, 1 modified',
    icon: '➕',
  },
  {
    id: 'v1-0',
    timestamp: '1 day ago',
    author: 'You',
    description: 'Initial workspace creation',
    changes: 'Created with 7 layers',
    icon: '🆕',
  },
  {
    id: 'v0-5',
    timestamp: '2 days ago',
    author: 'Alex Kumar',
    description: 'Feedback: Consider attention optimization',
    changes: 'Comment added',
    icon: '💬',
  },
];

interface VersionHistoryProps {
  workspaceName?: string;
}

export function VersionHistory({ workspaceName = 'Transformer Architecture' }: VersionHistoryProps) {
  const [selectedVersion, setSelectedVersion] = useState<string>('current');

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-r border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-1">
          Version History
        </h3>
        <p className="text-xs text-[--color-text-secondary]">
          {workspaceName}
        </p>
      </div>

      {/* Versions Timeline */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-3">
          {versions.map((version, idx) => (
            <div
              key={version.id}
              className={`group relative cursor-pointer transition-all rounded-lg border-2 p-3 ${
                selectedVersion === version.id
                  ? 'bg-[--accent-primary]/10 border-[--accent-primary]'
                  : 'bg-[--bg-body] border-[--color-border] hover:border-[--accent-primary]/50'
              }`}
              onClick={() => setSelectedVersion(version.id)}
            >
              {/* Timeline Connector */}
              {idx < versions.length - 1 && (
                <div className="absolute left-8 top-14 w-0.5 h-4 bg-[--color-border]" />
              )}

              {/* Icon Circle */}
              <div className="absolute -left-4 top-3 w-8 h-8 rounded-full bg-[--bg-panel] border-2 border-[--color-border] flex items-center justify-center text-sm">
                {version.icon}
              </div>

              {/* Content */}
              <div className="ml-6">
                {/* Header */}
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <div className="text-xs font-semibold text-[--color-text-primary]">
                      {version.description}
                    </div>
                    <div className="text-xs text-[--color-text-tertiary] mt-1 flex items-center gap-1">
                      <Clock size={12} />
                      {version.timestamp}
                    </div>
                  </div>
                  {version.id !== 'current' && (
                    <span className="text-xs text-[--color-text-tertiary]">
                      {version.author}
                    </span>
                  )}
                </div>

                {/* Changes */}
                <div className="text-xs text-[--color-text-secondary] bg-[--bg-surface] rounded px-2 py-1 mb-2 inline-block">
                  {version.changes}
                </div>

                {/* Actions */}
                {version.id !== 'current' && (
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="p-1 rounded hover:bg-[--bg-body] text-[--color-text-tertiary] hover:text-[--accent-primary]" title="Restore to this version">
                      <RotateCcw size={12} />
                    </button>
                    <button className="p-1 rounded hover:bg-[--bg-body] text-[--color-text-tertiary]" title="Download version">
                      <Download size={12} />
                    </button>
                    <button className="p-1 rounded hover:bg-red-500/20 text-red-400" title="Delete version">
                      <Trash2 size={12} />
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer Stats */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body] space-y-2 text-xs text-[--color-text-secondary]">
        <div className="flex justify-between">
          <span>Total Versions:</span>
          <span className="text-[--accent-primary]">{versions.length}</span>
        </div>
        <div className="flex justify-between">
          <span>Auto-save Enabled:</span>
          <span className="text-green-400">✓ Every 5 min</span>
        </div>
        <button className="w-full mt-2 px-3 py-2 rounded text-xs font-medium bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          Clear History
        </button>
      </div>
    </div>
  );
}
