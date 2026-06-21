'use client';

import { RotateCcw, Download } from 'lucide-react';
import { useState } from 'react';

interface AttentionVizProps {
  layerId?: string;
}

export function AttentionViz({ layerId = 'transformer-1' }: AttentionVizProps) {
  const [selectedHead, setSelectedHead] = useState(0);

  const heads = 8;
  const seqLen = 12; // for visualization

  // Generate attention weights (deterministic but looks random)
  const getAttentionWeight = (i: number, j: number, head: number): number => {
    const hash = (i * seqLen + j + head * seqLen * seqLen) % 100;
    return 0.3 + (hash / 100) * 0.7;
  };

  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h2 className="text-lg font-bold text-[--color-text-primary]">
          Attention Head Visualization
        </h2>
        <p className="text-xs text-[--color-text-secondary] mt-1">
          {layerId} • 8 attention heads • Sequence length 12
        </p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="space-y-6">
          {/* Head Selector */}
          <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
            <div className="text-xs font-semibold text-[--color-text-primary] mb-3">
              Select Attention Head
            </div>
            <div className="grid grid-cols-4 gap-2">
              {Array.from({ length: heads }, (_, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedHead(i)}
                  className={`px-3 py-2 rounded text-xs font-medium transition-colors ${
                    selectedHead === i
                      ? 'bg-[--accent-primary] text-white'
                      : 'bg-[--bg-body] text-[--color-text-secondary] hover:bg-[--bg-surface]'
                  }`}
                >
                  Head {i + 1}
                </button>
              ))}
            </div>
          </div>

          {/* Attention Matrix */}
          <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
            <div className="text-xs font-semibold text-[--color-text-primary] mb-3">
              Attention Weights Matrix
            </div>
            <div className="overflow-x-auto">
              <div className="inline-block">
                {/* Row labels and matrix */}
                <div className="flex">
                  {/* Column labels */}
                  <div className="flex flex-col">
                    <div className="w-8" />
                    {Array.from({ length: seqLen }, (_, i) => (
                      <div key={`row-${i}`} className="w-8 h-8 flex items-center justify-center text-xs text-[--color-text-tertiary]">
                        {i}
                      </div>
                    ))}
                  </div>

                  {/* Matrix */}
                  <div className="flex flex-col">
                    {/* Header row */}
                    <div className="flex">
                      {Array.from({ length: seqLen }, (_, i) => (
                        <div key={`col-${i}`} className="w-8 h-8 flex items-center justify-center text-xs text-[--color-text-tertiary]">
                          {i}
                        </div>
                      ))}
                    </div>

                    {/* Data rows */}
                    {Array.from({ length: seqLen }, (_, i) => (
                      <div key={`row-data-${i}`} className="flex">
                        {Array.from({ length: seqLen }, (_, j) => {
                          const weight = getAttentionWeight(i, j, selectedHead);
                          return (
                            <div
                              key={`cell-${i}-${j}`}
                              className="w-8 h-8 border border-[--color-border]"
                              style={{
                                backgroundColor: `rgba(124, 58, 237, ${weight * 0.8})`,
                                cursor: 'pointer',
                              }}
                              title={`Attention[${i}][${j}] = ${weight.toFixed(3)}`}
                            />
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Legend */}
            <div className="mt-4 flex items-center gap-4">
              <div className="text-xs text-[--color-text-secondary]">
                Intensity: Low
              </div>
              <div className="flex gap-1">
                {Array.from({ length: 5 }, (_, i) => (
                  <div
                    key={`legend-${i}`}
                    className="w-6 h-6 border border-[--color-border]"
                    style={{ backgroundColor: `rgba(124, 58, 237, ${(i / 4) * 0.8})` }}
                  />
                ))}
              </div>
              <div className="text-xs text-[--color-text-secondary]">
                High
              </div>
            </div>
          </div>

          {/* Head Analysis */}
          <div className="bg-[--accent-cyan]/10 rounded-lg p-4 border border-[--accent-cyan]/20">
            <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
              Head {selectedHead + 1} Analysis
            </div>
            <div className="space-y-2 text-xs text-[--color-text-secondary]">
              <div className="flex justify-between">
                <span>Average Attention Span:</span>
                <span className="text-[--accent-cyan]">4.2 tokens</span>
              </div>
              <div className="flex justify-between">
                <span>Max Attention Weight:</span>
                <span className="text-[--accent-cyan]">0.89</span>
              </div>
              <div className="flex justify-between">
                <span>Entropy (Diversity):</span>
                <span className="text-[--accent-cyan]">2.1 bits</span>
              </div>
              <div className="flex justify-between">
                <span>Attention Pattern:</span>
                <span className="text-[--accent-cyan]">Mixed (Local + Long-range)</span>
              </div>
            </div>
          </div>

          {/* Head Interpretation */}
          <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
            <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
              Likely Role
            </div>
            <p className="text-xs text-[--color-text-secondary] leading-relaxed">
              This head shows moderate attention to nearby tokens with occasional long-range jumps.
              It likely captures both local syntactic structure and longer-term semantic dependencies.
            </p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-3 border-t border-[--color-border] bg-[--bg-panel] flex gap-2">
        <button className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-medium bg-[--bg-body] hover:bg-[--bg-surface] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          <RotateCcw size={12} />
          Reset
        </button>
        <button className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-medium bg-[--accent-primary] hover:opacity-90 text-white transition-opacity">
          <Download size={12} />
          Export
        </button>
      </div>
    </div>
  );
}
