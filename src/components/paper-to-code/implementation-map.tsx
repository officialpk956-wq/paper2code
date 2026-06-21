'use client';

import { ArrowRight, CheckCircle, AlertCircle } from 'lucide-react';
import { useState } from 'react';

interface Mapping {
  id: string;
  paperSection: string;
  codeLines: string;
  status: 'complete' | 'partial' | 'todo';
  description: string;
}

const mappings: Mapping[] = [
  {
    id: '1',
    paperSection: 'Self-attention mechanism',
    codeLines: '31-33 (Linear projections)',
    status: 'complete',
    description: 'Query, Key, Value projections implemented',
  },
  {
    id: '2',
    paperSection: 'Attention scores computation',
    codeLines: '41-44 (Scoring & softmax)',
    status: 'complete',
    description: 'QK^T / √d_k computation with softmax',
  },
  {
    id: '3',
    paperSection: 'Multi-head parallel processing',
    codeLines: '16-18, 36-39 (Reshape & transpose)',
    status: 'complete',
    description: 'Splits embeddings into h heads and processes in parallel',
  },
  {
    id: '4',
    paperSection: 'Head concatenation',
    codeLines: '51-54 (Concat & linear)',
    status: 'complete',
    description: 'Concatenates outputs and applies final linear layer',
  },
  {
    id: '5',
    paperSection: 'Gradient computation',
    codeLines: 'TBD',
    status: 'todo',
    description: 'Backward pass and gradient flow analysis',
  },
];

const statusConfig = {
  complete: {
    icon: CheckCircle,
    color: 'text-green-400',
    bg: 'bg-green-500/10',
    border: 'border-green-500/30',
  },
  partial: {
    icon: AlertCircle,
    color: 'text-yellow-400',
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/30',
  },
  todo: {
    icon: AlertCircle,
    color: 'text-slate-400',
    bg: 'bg-slate-500/10',
    border: 'border-slate-500/30',
  },
};

interface ImplementationMapProps {
  selectedExcerpt?: string;
}

export function ImplementationMap({ selectedExcerpt }: ImplementationMapProps) {
  const [expandedMapping, setExpandedMapping] = useState<string | null>(null);

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-l border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-3">
          Paper-to-Code Map
        </h3>
        <div className="space-y-1 text-xs text-[--color-text-tertiary]">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-400" />
            <span>Complete</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-yellow-400" />
            <span>Partial</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-slate-400" />
            <span>To Do</span>
          </div>
        </div>
      </div>

      {/* Mappings List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {mappings.map((mapping) => {
          const config = statusConfig[mapping.status];
          const Icon = config.icon;
          const isSelected = selectedExcerpt === mapping.id;
          const isExpanded = expandedMapping === mapping.id;

          return (
            <div
              key={mapping.id}
              className={`rounded-lg border transition-all cursor-pointer ${
                isSelected
                  ? `${config.bg} ${config.border} border-2`
                  : `bg-[--bg-body] border-[--color-border] hover:${config.bg}`
              }`}
            >
              {/* Header */}
              <button
                onClick={() =>
                  setExpandedMapping(isExpanded ? null : mapping.id)
                }
                className="w-full flex items-start gap-3 p-3"
              >
                <Icon size={16} className={`flex-shrink-0 mt-1 ${config.color}`} />
                <div className="flex-1 text-left min-w-0">
                  <div className="text-xs font-semibold text-[--color-text-primary] mb-1">
                    {mapping.paperSection}
                  </div>
                  <div className="text-xs text-[--color-text-tertiary]">
                    {mapping.codeLines}
                  </div>
                </div>
                <ArrowRight
                  size={14}
                  className={`flex-shrink-0 text-[--color-text-tertiary] transition-transform ${
                    isExpanded ? 'rotate-90' : ''
                  }`}
                />
              </button>

              {/* Expanded Content */}
              {isExpanded && (
                <div className="px-3 pb-3 pt-0 border-t border-[--color-border]">
                  <p className="text-xs text-[--color-text-secondary] leading-relaxed">
                    {mapping.description}
                  </p>
                  <div className="mt-3 pt-3 border-t border-[--color-border]">
                    <div className="flex gap-2">
                      <button className="flex-1 px-2 py-1.5 text-xs rounded bg-[--bg-surface] hover:bg-[--bg-body] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
                        View Paper
                      </button>
                      <button className="flex-1 px-2 py-1.5 text-xs rounded bg-[--bg-surface] hover:bg-[--bg-body] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
                        View Code
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Progress Stats */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body]">
        <div className="space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-[--color-text-secondary]">Implementation Progress</span>
            <span className="font-semibold text-[--accent-primary]">80%</span>
          </div>
          <div className="h-1.5 bg-[--bg-surface] rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan]"
              style={{ width: '80%' }}
            />
          </div>
          <div className="text-xs text-[--color-text-tertiary] pt-1">
            4 of 5 concepts implemented
          </div>
        </div>
      </div>
    </div>
  );
}
