'use client';

import { Copy, Download } from 'lucide-react';
import { useState } from 'react';

interface Excerpt {
  id: string;
  text: string;
  lineStart: number;
  lineEnd: number;
  highlighted?: boolean;
}

const excerpts: Excerpt[] = [
  {
    id: '1',
    text: 'The key innovation of the Transformer architecture is the use of self-attention mechanisms. Unlike recurrent neural networks, which process sequences sequentially, transformers process all tokens in parallel.',
    lineStart: 1,
    lineEnd: 3,
    highlighted: true,
  },
  {
    id: '2',
    text: 'Multi-head attention allows the model to attend to different parts of the input simultaneously. We compute attention using three linear projections: Query (Q), Key (K), and Value (V).',
    lineStart: 5,
    lineEnd: 7,
  },
  {
    id: '3',
    text: 'The attention mechanism computes a weighted sum of the values, where the weights are determined by the similarity between the query and keys. This is computed as: Attention(Q, K, V) = softmax(QK^T / √d_k) V',
    lineStart: 9,
    lineEnd: 11,
  },
  {
    id: '4',
    text: 'For multi-head attention with h heads, we split the embeddings into h parts and apply attention in parallel. The outputs from all heads are concatenated and passed through a final linear layer.',
    lineStart: 13,
    lineEnd: 15,
  },
];

interface PaperExcerptProps {
  onSelectExcerpt?: (id: string) => void;
  selectedExcerpt?: string;
}

export function PaperExcerpt({ onSelectExcerpt, selectedExcerpt }: PaperExcerptProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h2 className="text-lg font-bold text-[--color-text-primary]">
          Attention Is All You Need
        </h2>
        <p className="text-xs text-[--color-text-secondary] mt-1">
          Vaswani et al., 2017 • Abstract & Introduction
        </p>
      </div>

      {/* Excerpts */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="space-y-4 max-w-2xl">
          {excerpts.map((excerpt) => (
            <div
              key={excerpt.id}
              onClick={() => onSelectExcerpt?.(excerpt.id)}
              className={`group p-4 rounded-lg border-2 transition-all cursor-pointer ${
                selectedExcerpt === excerpt.id
                  ? 'bg-[--accent-primary]/10 border-[--accent-primary]'
                  : excerpt.highlighted
                  ? 'bg-[--accent-cyan]/10 border-[--accent-cyan] hover:border-[--accent-primary]'
                  : 'bg-[--bg-panel] border-[--color-border] hover:border-[--accent-primary]'
              }`}
            >
              {/* Line numbers */}
              <div className="flex gap-4">
                <div className="flex-shrink-0 text-xs text-[--color-text-tertiary] font-mono">
                  <div>{excerpt.lineStart}</div>
                  {excerpt.lineEnd > excerpt.lineStart && (
                    <>
                      {Array.from(
                        { length: excerpt.lineEnd - excerpt.lineStart },
                        (_, i) => (
                          <div key={i + 1}>{excerpt.lineStart + i + 1}</div>
                        )
                      )}
                    </>
                  )}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-[--color-text-primary] leading-relaxed">
                    {excerpt.text}
                  </p>

                  {/* Actions */}
                  <div className="flex gap-2 mt-3 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleCopy(excerpt.text, excerpt.id);
                      }}
                      className={`flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors ${
                        copiedId === excerpt.id
                          ? 'bg-[--accent-primary] text-white'
                          : 'bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary]'
                      }`}
                    >
                      <Copy size={12} />
                      {copiedId === excerpt.id ? 'Copied!' : 'Copy'}
                    </button>
                    <button className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
                      <span>📌</span>
                      Bookmark
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-[--color-border] bg-[--bg-panel] flex items-center justify-between">
        <span className="text-xs text-[--color-text-tertiary]">
          {excerpts.length} excerpts
        </span>
        <button className="flex items-center gap-2 px-3 py-2 text-xs rounded bg-[--accent-primary] hover:opacity-90 text-white transition-opacity">
          <Download size={14} />
          Export
        </button>
      </div>
    </div>
  );
}
