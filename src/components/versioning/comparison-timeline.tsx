'use client';

import { GitCompare, Plus, Minus, Edit2 } from 'lucide-react';
import { useState } from 'react';

interface TimelineEvent {
  id: string;
  version: string;
  timestamp: string;
  author: string;
  avatar: string;
  title: string;
  changes: {
    added: number;
    modified: number;
    deleted: number;
  };
  description: string;
}

const timeline: TimelineEvent[] = [
  {
    id: '1',
    version: 'v1.2',
    timestamp: 'Now',
    author: 'You',
    avatar: '👤',
    title: 'Added attention optimization',
    changes: { added: 3, modified: 5, deleted: 1 },
    description: 'Optimized attention mechanism with bias=False',
  },
  {
    id: '2',
    version: 'v1.1',
    timestamp: '2 hours ago',
    author: 'You',
    avatar: '👤',
    title: 'Updated transformer block structure',
    changes: { added: 2, modified: 8, deleted: 0 },
    description: 'Refactored multi-head attention implementation',
  },
  {
    id: '3',
    version: 'v1.0',
    timestamp: '1 day ago',
    author: 'Sarah Chen',
    avatar: '👩',
    title: 'Added pooling layer',
    changes: { added: 5, modified: 3, deleted: 2 },
    description: 'Implemented global average pooling',
  },
  {
    id: '4',
    version: 'v0.9',
    timestamp: '3 days ago',
    author: 'Alex Kumar',
    avatar: '👨',
    title: 'Initial transformer architecture',
    changes: { added: 12, modified: 0, deleted: 0 },
    description: 'Created base transformer with 6 layers',
  },
];

interface ComparisonTimelineProps {
  workspaceName?: string;
  selectedVersions?: string[];
}

export function ComparisonTimeline({
  workspaceName = 'Transformer Architecture',
  selectedVersions: initialSelectedVersions = ['1', '3'],
}: ComparisonTimelineProps) {
  const [selectedVersions, setSelectedVersions] = useState<Set<string>>(
    new Set(initialSelectedVersions)
  );

  const toggleVersion = (id: string) => {
    setSelectedVersions((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        if (next.size >= 2) {
          next.delete(Array.from(next)[0]);
        }
        next.add(id);
      }
      return next;
    });
  };

  const isSelected = (id: string) => selectedVersions.has(id);
  const selectedArray = Array.from(selectedVersions).map((id) =>
    timeline.find((t) => t.id === id)
  );

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border border-[--color-border] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-body]">
        <div className="flex items-center gap-2 mb-2">
          <GitCompare size={18} className="text-[--accent-primary]" />
          <h3 className="text-sm font-semibold text-[--color-text-primary]">
            Version Comparison
          </h3>
        </div>
        <p className="text-xs text-[--color-text-secondary]">
          {workspaceName} • Select 2 versions to compare
        </p>
      </div>

      {/* Timeline */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          {/* Timeline Visualization */}
          <div className="relative">
            {timeline.map((event, idx) => (
              <div
                key={event.id}
                className="mb-6 relative cursor-pointer"
                onClick={() => toggleVersion(event.id)}
              >
                {/* Timeline Connector */}
                {idx < timeline.length - 1 && (
                  <div
                    className={`absolute left-5 top-12 w-0.5 h-8 transition-colors ${
                      isSelected(event.id)
                        ? 'bg-[--accent-primary]'
                        : 'bg-[--color-border]'
                    }`}
                  />
                )}

                {/* Timeline Dot */}
                <div
                  className={`absolute -left-2 top-3 w-5 h-5 rounded-full border-2 transition-all ${
                    isSelected(event.id)
                      ? 'bg-[--accent-primary] border-[--accent-primary] scale-125'
                      : 'bg-[--bg-panel] border-[--color-border] hover:scale-110'
                  }`}
                />

                {/* Event Card */}
                <div
                  className={`ml-8 p-3 rounded-lg border transition-all ${
                    isSelected(event.id)
                      ? 'bg-[--accent-primary]/10 border-[--accent-primary]'
                      : 'bg-[--bg-body] border-[--color-border] hover:border-[--accent-primary]/50'
                  }`}
                >
                  {/* Header */}
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold text-[--color-text-primary]">
                          {event.avatar} {event.author}
                        </span>
                        <span className="text-xs px-2 py-0.5 rounded bg-[--bg-surface] text-[--color-text-tertiary]">
                          {event.version}
                        </span>
                      </div>
                      <div className="text-xs text-[--color-text-tertiary] mt-1">
                        {event.timestamp}
                      </div>
                    </div>
                    {isSelected(event.id) && (
                      <span className="text-xs px-2 py-1 rounded bg-[--accent-primary] text-white font-medium flex-shrink-0">
                        Selected
                      </span>
                    )}
                  </div>

                  {/* Title & Description */}
                  <div className="mb-2">
                    <div className="text-xs font-semibold text-[--color-text-primary]">
                      {event.title}
                    </div>
                    <div className="text-xs text-[--color-text-secondary] mt-0.5">
                      {event.description}
                    </div>
                  </div>

                  {/* Change Stats */}
                  <div className="flex gap-2">
                    <div className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-green-500/10 text-green-400">
                      <Plus size={12} />
                      {event.changes.added}
                    </div>
                    <div className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-blue-500/10 text-blue-400">
                      <Edit2 size={12} />
                      {event.changes.modified}
                    </div>
                    <div className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-red-500/10 text-red-400">
                      <Minus size={12} />
                      {event.changes.deleted}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Comparison Panel */}
      {selectedArray.length === 2 && (
        <div className="border-t border-[--color-border] bg-[--bg-body] p-4 space-y-3">
          <div className="text-xs font-semibold text-[--color-text-primary]">
            Comparing Versions
          </div>

          <div className="grid grid-cols-2 gap-3">
            {selectedArray.map((event) =>
              event ? (
                <div
                  key={event.id}
                  className="p-3 rounded-lg bg-[--bg-panel] border border-[--color-border]"
                >
                  <div className="text-xs font-semibold text-[--color-text-primary]">
                    {event.version}
                  </div>
                  <div className="text-xs text-[--color-text-tertiary] mt-0.5">
                    {event.title}
                  </div>
                  <div className="flex gap-1 mt-2">
                    <span className="text-xs px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">
                      +{event.changes.added}
                    </span>
                    <span className="text-xs px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400">
                      ±{event.changes.modified}
                    </span>
                    <span className="text-xs px-1.5 py-0.5 rounded bg-red-500/20 text-red-400">
                      -{event.changes.deleted}
                    </span>
                  </div>
                </div>
              ) : null
            )}
          </div>

          <button className="w-full px-3 py-2 rounded text-xs font-medium bg-[--accent-primary] hover:opacity-90 text-white transition-opacity">
            View Detailed Diff
          </button>
        </div>
      )}

      {/* Footer */}
      <div className="border-t border-[--color-border] bg-[--bg-panel] px-4 py-3">
        <div className="text-xs text-[--color-text-tertiary]">
          {selectedArray.length} of 2 versions selected
        </div>
      </div>
    </div>
  );
}
