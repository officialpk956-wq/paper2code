'use client';

import { ChevronDown, Copy } from 'lucide-react';
import { useState } from 'react';

interface DiffLine {
  type: 'add' | 'remove' | 'context';
  lineNumber: number;
  content: string;
}

const sampleDiff: DiffLine[] = [
  { type: 'context', lineNumber: 1, content: 'class MultiHeadAttention(nn.Module):' },
  { type: 'context', lineNumber: 2, content: '    def __init__(self, d_model, n_heads):' },
  { type: 'context', lineNumber: 3, content: '        super().__init__()' },
  { type: 'context', lineNumber: 4, content: '        self.d_model = d_model' },
  { type: 'remove', lineNumber: 5, content: '        self.n_heads = n_heads  # old implementation' },
  { type: 'add', lineNumber: 5, content: '        self.n_heads = n_heads  # optimized v2' },
  { type: 'context', lineNumber: 6, content: '        self.d_k = d_model // n_heads' },
  { type: 'remove', lineNumber: 7, content: '        self.W_q = nn.Linear(d_model, d_model)' },
  { type: 'add', lineNumber: 7, content: '        self.W_q = nn.Linear(d_model, d_model, bias=False)' },
  { type: 'context', lineNumber: 8, content: '        self.W_k = nn.Linear(d_model, d_model)' },
  { type: 'context', lineNumber: 9, content: '        self.W_v = nn.Linear(d_model, d_model)' },
  { type: 'add', lineNumber: 10, content: '        self.scale = 1 / sqrt(self.d_k)  # new scaling' },
  { type: 'context', lineNumber: 11, content: '' },
  { type: 'context', lineNumber: 12, content: '    def forward(self, x):' },
];

interface VersionDiffViewerProps {
  fromVersion?: string;
  toVersion?: string;
  viewMode?: 'split' | 'unified';
}

export function VersionDiffViewer({
  fromVersion = 'v1.0 (2 days ago)',
  toVersion = 'v1.2 (current)',
  viewMode: initialViewMode = 'split',
}: VersionDiffViewerProps) {
  const [viewMode, setViewMode] = useState<'split' | 'unified'>(initialViewMode);
  const [showStats, setShowStats] = useState(true);

  const addLines = sampleDiff.filter((l) => l.type === 'add').length;
  const removeLines = sampleDiff.filter((l) => l.type === 'remove').length;

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border border-[--color-border] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-body]">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-sm font-semibold text-[--color-text-primary] mb-1">
              Version Comparison
            </h3>
            <div className="flex items-center gap-2 text-xs text-[--color-text-secondary]">
              <span className="px-2 py-1 rounded bg-[--bg-surface] text-[--color-text-tertiary]">
                {fromVersion}
              </span>
              <span>→</span>
              <span className="px-2 py-1 rounded bg-[--accent-primary]/20 text-[--accent-primary]">
                {toVersion}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowStats(!showStats)}
              className="p-1.5 rounded hover:bg-[--bg-surface] text-[--color-text-tertiary] hover:text-[--color-text-primary] transition-colors"
            >
              <ChevronDown size={16} className={`transition-transform ${showStats ? 'rotate-180' : ''}`} />
            </button>
          </div>
        </div>

        {/* Stats Bar */}
        {showStats && (
          <div className="flex gap-4 text-xs">
            <div className="flex items-center gap-2 px-3 py-2 rounded bg-green-500/10 text-green-400">
              <div className="w-2 h-2 rounded-full bg-green-400" />
              <span>+{addLines} additions</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-2 rounded bg-red-500/10 text-red-400">
              <div className="w-2 h-2 rounded-full bg-red-400" />
              <span>-{removeLines} deletions</span>
            </div>
          </div>
        )}
      </div>

      {/* View Mode Toggle */}
      <div className="px-6 py-3 border-b border-[--color-border] bg-[--bg-panel] flex gap-2">
        <button
          onClick={() => setViewMode('split')}
          className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
            viewMode === 'split'
              ? 'bg-[--accent-primary] text-white'
              : 'bg-[--bg-body] text-[--color-text-secondary] hover:text-[--color-text-primary]'
          }`}
        >
          Split View
        </button>
        <button
          onClick={() => setViewMode('unified')}
          className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
            viewMode === 'unified'
              ? 'bg-[--accent-primary] text-white'
              : 'bg-[--bg-body] text-[--color-text-secondary] hover:text-[--color-text-primary]'
          }`}
        >
          Unified View
        </button>
      </div>

      {/* Diff Content */}
      <div className="flex-1 overflow-auto font-mono text-xs">
        {viewMode === 'split' ? (
          <div className="flex">
            {/* Left Side - Original */}
            <div className="flex-1 border-r border-[--color-border]">
              <div className="bg-[--bg-body] px-4 py-2 border-b border-[--color-border] text-xs text-[--color-text-secondary] sticky top-0 z-10">
                {fromVersion}
              </div>
              <div className="divide-y divide-[--color-border]">
                {sampleDiff.map((line, idx) => (
                  <div
                    key={idx}
                    className={`flex items-start hover:bg-[--bg-surface] transition-colors ${
                      line.type === 'remove' ? 'bg-red-500/5' : 'bg-transparent'
                    }`}
                  >
                    <div className="w-12 px-2 py-1 bg-[--bg-body] border-r border-[--color-border] text-[--color-text-tertiary] text-right flex-shrink-0">
                      {line.type === 'remove' ? line.lineNumber : ''}
                    </div>
                    <div className="flex-1 px-4 py-1 text-[--color-text-secondary]">
                      {line.type === 'remove' && (
                        <span className="text-red-400">{line.content}</span>
                      )}
                      {(line.type === 'context' || line.type === 'add') && (
                        <span className="text-[--color-text-secondary]">{line.content}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right Side - Modified */}
            <div className="flex-1">
              <div className="bg-[--bg-body] px-4 py-2 border-b border-[--color-border] text-xs text-[--color-text-secondary] sticky top-0 z-10">
                {toVersion}
              </div>
              <div className="divide-y divide-[--color-border]">
                {sampleDiff.map((line, idx) => (
                  <div
                    key={idx}
                    className={`flex items-start hover:bg-[--bg-surface] transition-colors ${
                      line.type === 'add' ? 'bg-green-500/5' : 'bg-transparent'
                    }`}
                  >
                    <div className="w-12 px-2 py-1 bg-[--bg-body] border-r border-[--color-border] text-[--color-text-tertiary] text-right flex-shrink-0">
                      {line.type === 'add' ? line.lineNumber : ''}
                    </div>
                    <div className="flex-1 px-4 py-1 text-[--color-text-secondary]">
                      {line.type === 'add' && (
                        <span className="text-green-400">{line.content}</span>
                      )}
                      {(line.type === 'context' || line.type === 'remove') && (
                        <span className="text-[--color-text-secondary]">{line.content}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          /* Unified View */
          <div className="bg-[--bg-body]">
            <div className="divide-y divide-[--color-border]">
              {sampleDiff.map((line, idx) => (
                <div
                  key={idx}
                  className={`flex items-start hover:bg-[--bg-surface] transition-colors ${
                    line.type === 'add'
                      ? 'bg-green-500/5'
                      : line.type === 'remove'
                        ? 'bg-red-500/5'
                        : 'bg-transparent'
                  }`}
                >
                  <div className="w-8 px-2 py-1 bg-[--bg-panel] border-r border-[--color-border] text-[--color-text-tertiary] text-right flex-shrink-0">
                    {line.type === 'add' ? '+' : line.type === 'remove' ? '-' : ' '}
                  </div>
                  <div className="w-12 px-2 py-1 bg-[--bg-panel] border-r border-[--color-border] text-[--color-text-tertiary] text-right flex-shrink-0">
                    {line.lineNumber}
                  </div>
                  <div className="flex-1 px-4 py-1">
                    {line.type === 'add' && <span className="text-green-400">{line.content}</span>}
                    {line.type === 'remove' && <span className="text-red-400">{line.content}</span>}
                    {line.type === 'context' && <span className="text-[--color-text-secondary]">{line.content}</span>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-6 py-3 border-t border-[--color-border] bg-[--bg-body] flex justify-between items-center">
        <div className="text-xs text-[--color-text-tertiary]">
          Showing {sampleDiff.length} lines
        </div>
        <button className="flex items-center gap-1 px-3 py-1.5 rounded text-xs font-medium bg-[--accent-primary]/10 hover:bg-[--accent-primary]/20 text-[--accent-primary] transition-colors">
          <Copy size={12} />
          Copy Diff
        </button>
      </div>
    </div>
  );
}
