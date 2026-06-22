'use client';

import React from 'react';

interface FLOPsBadgeProps {
  flops: number;
  params?: number;
  compact?: boolean;
}

function flopsColor(mflops: number): { bg: string; text: string; border: string } {
  if (mflops >= 1000) return { bg: 'rgba(239,68,68,0.12)', text: '#f87171', border: 'rgba(239,68,68,0.35)' };
  if (mflops >= 100) return { bg: 'rgba(245,158,11,0.12)', text: '#fbbf24', border: 'rgba(245,158,11,0.35)' };
  return { bg: 'rgba(16,185,129,0.12)', text: '#34d399', border: 'rgba(16,185,129,0.35)' };
}

function formatFlops(mflops: number): string {
  if (mflops >= 1000) return `${(mflops / 1000).toFixed(1)}G`;
  if (mflops >= 1) return `${mflops.toFixed(0)}M`;
  if (mflops > 0) return `${(mflops * 1000).toFixed(0)}K`;
  return '—';
}

function formatParams(paramsM: number): string {
  if (paramsM >= 1000) return `${(paramsM / 1000).toFixed(1)}B`;
  if (paramsM >= 1) return `${paramsM.toFixed(1)}M`;
  if (paramsM > 0) return `${(paramsM * 1000).toFixed(0)}K`;
  return '—';
}

export const FLOPsBadge = React.memo(function FLOPsBadge({ flops, params, compact = false }: FLOPsBadgeProps) {
  if (flops === 0 && (!params || params === 0)) return null;

  const { bg, text, border } = flopsColor(flops);
  const label = flops > 0 ? `${formatFlops(flops)}FLOPs` : '';
  const paramsLabel = params && params > 0 ? `${formatParams(params)}` : '';

  return (
    <span
      title={`FLOPs: ${flops.toFixed(2)}M${params ? ` · Params: ${params.toFixed(4)}M` : ''}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '4px',
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
      {label}
      {paramsLabel && (
        <span style={{ opacity: 0.65, color: 'var(--text-muted, #9ca3af)' }}>
          {compact ? '' : '·'} {paramsLabel}
        </span>
      )}
    </span>
  );
});
