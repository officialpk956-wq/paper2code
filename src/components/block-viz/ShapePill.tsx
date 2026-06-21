'use client';

import React from 'react';

interface ShapePillProps {
  shape: number[] | null | undefined;
  label?: string;
  compact?: boolean;
}

function tensorRankColor(rank: number): { bg: string; text: string; border: string } {
  switch (rank) {
    case 2: return { bg: 'rgba(6,182,212,0.12)', text: '#22d3ee', border: 'rgba(6,182,212,0.35)' };
    case 3: return { bg: 'rgba(139,92,246,0.12)', text: '#a78bfa', border: 'rgba(139,92,246,0.35)' };
    case 4: return { bg: 'rgba(249,115,22,0.12)', text: '#fb923c', border: 'rgba(249,115,22,0.35)' };
    default: return { bg: 'rgba(239,68,68,0.12)', text: '#f87171', border: 'rgba(239,68,68,0.35)' };
  }
}

export const ShapePill = React.memo(function ShapePill({ shape, label, compact = false }: ShapePillProps) {
  if (!shape || shape.length === 0) {
    return (
      <span style={{
        display: 'inline-block',
        padding: '1px 6px',
        borderRadius: '4px',
        fontSize: '11px',
        fontFamily: 'var(--font-mono, monospace)',
        background: 'rgba(255,255,255,0.06)',
        color: 'var(--text-muted, #6b7280)',
        border: '1px solid rgba(255,255,255,0.1)',
      }}>
        —
      </span>
    );
  }

  const rank = shape.length;
  const { bg, text, border } = tensorRankColor(rank);

  // Format: skip batch dim (index 0) if it's 1, show as "64×56×56" for 4D tensors
  const dims = shape.slice(1); // skip batch
  const formatted = compact
    ? dims.join('×')
    : `(${shape.join(', ')})`;

  return (
    <span
      title={`Tensor shape: (${shape.join(', ')}) — rank ${rank}D`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '3px',
        padding: compact ? '1px 5px' : '2px 7px',
        borderRadius: '5px',
        fontSize: compact ? '10px' : '11px',
        fontFamily: 'var(--font-mono, monospace)',
        fontWeight: 500,
        background: bg,
        color: text,
        border: `1px solid ${border}`,
        whiteSpace: 'nowrap',
      }}
    >
      {label && <span style={{ opacity: 0.7, marginRight: 2 }}>{label}:</span>}
      {formatted}
    </span>
  );
});
