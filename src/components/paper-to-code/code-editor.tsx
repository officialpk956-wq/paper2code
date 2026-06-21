'use client';

import { Copy, Download, Play } from 'lucide-react';

const codeSnippet = `import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
  """Multi-head self-attention mechanism.

  Allows the model to attend to different parts of
  the input sequence in parallel.
  """

  def __init__(self, d_model: int, n_heads: int):
    super().__init__()
    assert d_model % n_heads == 0

    self.n_heads = n_heads
    self.d_k = d_model // n_heads

    # Linear projections
    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)
    self.W_o = nn.Linear(d_model, d_model)

  def forward(self, Q, K, V, mask=None):
    """Compute multi-head attention.

    Args:
      Q: Query [batch_size, seq_len, d_model]
      K: Key [batch_size, seq_len, d_model]
      V: Value [batch_size, seq_len, d_model]
      mask: Optional attention mask

    Returns:
      output: [batch_size, seq_len, d_model]
    """
    batch_size = Q.shape[0]

    # Linear transformations
    Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k)
    K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k)
    V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k)

    # Transpose for attention computation
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

    # Apply mask if provided
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax
    attention = torch.softmax(scores, dim=-1)

    # Apply attention to values
    context = torch.matmul(attention, V)

    # Concatenate heads
    context = context.transpose(1, 2).contiguous()
    context = context.view(batch_size, -1, self.d_model)

    # Final linear transformation
    output = self.W_o(context)

    return output`;

const lines = codeSnippet.split('\n');

interface CodeEditorProps {
  selectedExcerpt?: string;
}

export function CodeEditor({ selectedExcerpt }: CodeEditorProps) {
  const highlightedLines = selectedExcerpt === '1'
    ? [0, 1, 2, 3, 4, 5]
    : selectedExcerpt === '3'
    ? [40, 41, 42, 43, 44]
    : [];

  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h2 className="text-lg font-bold text-[--color-text-primary]">
          Implementation
        </h2>
        <p className="text-xs text-[--color-text-secondary] mt-1">
          multi_head_attention.py • PyTorch
        </p>
      </div>

      {/* Toolbar */}
      <div className="px-6 py-3 border-b border-[--color-border] bg-[--bg-surface] flex gap-2">
        <button className="flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium text-white bg-[--accent-primary] hover:opacity-90 transition-opacity">
          <Play size={14} />
          Run
        </button>
        <button className="flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          <Copy size={14} />
          Copy
        </button>
        <button className="flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          <Download size={14} />
          Export
        </button>
      </div>

      {/* Editor */}
      <div className="flex-1 overflow-auto bg-[--bg-body]">
        <div className="p-6">
          <pre className="font-mono text-sm leading-relaxed">
            {lines.map((line, idx) => (
              <div
                key={idx}
                className={`transition-colors ${
                  highlightedLines.includes(idx)
                    ? 'bg-[--accent-primary]/20 text-[--accent-primary]'
                    : 'text-[--color-text-primary]'
                }`}
              >
                <span className="inline-block w-12 text-right pr-4 text-[--color-text-tertiary] select-none">
                  {String(idx + 1).padStart(2, ' ')}
                </span>
                <span className="text-[--color-text-primary]">{line || ' '}</span>
              </div>
            ))}
          </pre>
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-3 border-t border-[--color-border] bg-[--bg-panel] text-xs text-[--color-text-tertiary]">
        {lines.length} lines • Python • Syntax highlighting enabled
      </div>
    </div>
  );
}
