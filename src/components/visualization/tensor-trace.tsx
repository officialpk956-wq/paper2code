'use client';

import { ChevronRight, Eye, Copy } from 'lucide-react';
import { useState } from 'react';

interface TensorOperation {
  id: string;
  name: string;
  operation: string;
  inputShape: string;
  outputShape: string;
  params?: number;
  complexity?: string;
}

const operations: TensorOperation[] = [
  {
    id: '1',
    name: 'Input Embedding',
    operation: 'embedding(input_ids)',
    inputShape: '[B, L]',
    outputShape: '[B, L, D]',
    params: 3072000,
  },
  {
    id: '2',
    name: 'Query Projection',
    operation: 'linear(x)',
    inputShape: '[B, L, D]',
    outputShape: '[B, L, D]',
    params: 1024000,
  },
  {
    id: '3',
    name: 'Reshape for Heads',
    operation: 'reshape(x)',
    inputShape: '[B, L, D]',
    outputShape: '[B, H, L, D/H]',
  },
  {
    id: '4',
    name: 'Attention Scores',
    operation: 'matmul(Q, K.T)',
    inputShape: '[B, H, L, D/H]',
    outputShape: '[B, H, L, L]',
    complexity: 'O(n²)',
  },
  {
    id: '5',
    name: 'Softmax',
    operation: 'softmax(scores)',
    inputShape: '[B, H, L, L]',
    outputShape: '[B, H, L, L]',
  },
  {
    id: '6',
    name: 'Apply Attention',
    operation: 'matmul(attn, V)',
    inputShape: '[B, H, L, L]',
    outputShape: '[B, H, L, D/H]',
    complexity: 'O(n²)',
  },
  {
    id: '7',
    name: 'Concatenate Heads',
    operation: 'reshape(x)',
    inputShape: '[B, H, L, D/H]',
    outputShape: '[B, L, D]',
  },
  {
    id: '8',
    name: 'Output Projection',
    operation: 'linear(x)',
    inputShape: '[B, L, D]',
    outputShape: '[B, L, D]',
    params: 1024000,
  },
];

export function TensorTrace() {
  const [expandedOp, setExpandedOp] = useState<string | null>('1');
  const [hoveredOp, setHoveredOp] = useState<string | null>(null);

  const formatParams = (params?: number) => {
    if (!params) return null;
    if (params >= 1000000) return `${(params / 1000000).toFixed(1)}M`;
    if (params >= 1000) return `${(params / 1000).toFixed(0)}K`;
    return params.toString();
  };

  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h2 className="text-lg font-bold text-[--color-text-primary]">
          Tensor Trace
        </h2>
        <p className="text-xs text-[--color-text-secondary] mt-1">
          Multi-Head Attention Forward Pass • Interactive Tensor Flow
        </p>
      </div>

      {/* Operations Flow */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-3xl space-y-2">
          {operations.map((op, idx) => {
            const isExpanded = expandedOp === op.id;
            const isHovered = hoveredOp === op.id;

            return (
              <div key={op.id}>
                {/* Operation Card */}
                <div
                  onMouseEnter={() => setHoveredOp(op.id)}
                  onMouseLeave={() => setHoveredOp(null)}
                  onClick={() => setExpandedOp(isExpanded ? null : op.id)}
                  className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                    isExpanded
                      ? 'bg-[--accent-primary]/10 border-[--accent-primary]'
                      : isHovered
                      ? 'bg-[--bg-panel] border-[--accent-cyan]'
                      : 'bg-[--bg-panel] border-[--color-border]'
                  }`}
                >
                  {/* Header */}
                  <div className="flex items-start gap-3">
                    {/* Step Number */}
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-[--accent-primary] to-[--accent-cyan] flex items-center justify-center">
                      <span className="text-xs font-bold text-white">{idx + 1}</span>
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-semibold text-[--color-text-primary]">
                        {op.name}
                      </div>
                      <div className="text-xs text-[--color-text-tertiary] font-mono mt-1">
                        {op.operation}
                      </div>
                    </div>

                    {/* Chevron */}
                    <ChevronRight
                      size={16}
                      className={`flex-shrink-0 text-[--color-text-tertiary] transition-transform ${
                        isExpanded ? 'rotate-90' : ''
                      }`}
                    />
                  </div>

                  {/* Shapes Row */}
                  <div className="mt-3 flex items-center gap-3 text-xs">
                    <div className="px-2 py-1 rounded bg-[--bg-body] text-[--color-text-secondary] font-mono">
                      {op.inputShape}
                    </div>
                    <div className="text-[--color-text-tertiary]">→</div>
                    <div className="px-2 py-1 rounded bg-[--bg-body] text-[--accent-cyan] font-mono">
                      {op.outputShape}
                    </div>
                  </div>

                  {/* Expanded Details */}
                  {isExpanded && (
                    <div className="mt-4 pt-4 border-t border-[--color-border] space-y-3">
                      <div className="grid grid-cols-2 gap-3">
                        {op.params && (
                          <div className="bg-[--bg-body] rounded p-2">
                            <div className="text-xs text-[--color-text-tertiary] mb-1">
                              Parameters
                            </div>
                            <div className="text-sm font-mono font-semibold text-[--accent-primary]">
                              {formatParams(op.params)}
                            </div>
                          </div>
                        )}
                        {op.complexity && (
                          <div className="bg-[--bg-body] rounded p-2">
                            <div className="text-xs text-[--color-text-tertiary] mb-1">
                              Time Complexity
                            </div>
                            <div className="text-sm font-mono font-semibold text-[--accent-cyan]">
                              {op.complexity}
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Memory Calculation */}
                      <div className="bg-[--accent-primary]/10 rounded p-3 border border-[--accent-primary]/20">
                        <div className="text-xs text-[--color-text-secondary] mb-2">
                          Memory Usage Estimate
                        </div>
                        <div className="text-sm font-mono text-[--color-text-primary]">
                          Input: {op.inputShape} = 98.3 KB
                          <br />
                          Output: {op.outputShape} = 98.3 KB
                          <br />
                          <span className="text-[--accent-cyan]">Total: 196.6 KB</span>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex gap-2">
                        <button className="flex-1 flex items-center justify-center gap-1.5 px-2 py-2 rounded text-xs font-medium bg-[--bg-body] text-[--color-text-secondary] hover:text-[--color-text-primary] hover:bg-[--bg-surface] transition-colors">
                          <Eye size={12} />
                          Visualize
                        </button>
                        <button className="flex-1 flex items-center justify-center gap-1.5 px-2 py-2 rounded text-xs font-medium bg-[--bg-body] text-[--color-text-secondary] hover:text-[--color-text-primary] hover:bg-[--bg-surface] transition-colors">
                          <Copy size={12} />
                          Code
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Connector */}
                {idx < operations.length - 1 && (
                  <div className="flex justify-center py-1">
                    <div className="w-0.5 h-4 bg-gradient-to-b from-[--color-border] to-transparent" />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Stats Footer */}
      <div className="px-6 py-4 border-t border-[--color-border] bg-[--bg-panel] grid grid-cols-3 gap-4">
        <div>
          <div className="text-xs text-[--color-text-tertiary] mb-1">Total Params</div>
          <div className="text-sm font-semibold text-[--accent-primary]">5.12M</div>
        </div>
        <div>
          <div className="text-xs text-[--color-text-tertiary] mb-1">Peak Memory</div>
          <div className="text-sm font-semibold text-[--accent-cyan]">2.4 GB</div>
        </div>
        <div>
          <div className="text-xs text-[--color-text-tertiary] mb-1">FLOPs</div>
          <div className="text-sm font-semibold text-[--accent-primary]">1.2 × 10⁹</div>
        </div>
      </div>
    </div>
  );
}
