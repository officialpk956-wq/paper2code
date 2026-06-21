'use client';

import React, { useState } from 'react';
import { ChevronRight, ChevronDown } from 'lucide-react';
import { ShapePill } from './ShapePill';
import { FLOPsBadge } from './FLOPsBadge';

export interface LayerNode {
  id: string;
  name: string;
  type: string;
  input_shape: number[] | null;
  output_shape: number[] | null;
  flops_mflops: number;
  params_M: number;
  formula?: string;
  complexity?: string;
  severity?: string;
}

export interface BlockNode {
  id: string;
  name: string;
  type: string;
  input_shape: number[] | null;
  output_shape: number[] | null;
  flops_mflops: number;
  params_M: number;
  layers: LayerNode[];
}

interface BlockBoxProps {
  block: BlockNode;
  isSelected: boolean;
  isActive?: boolean;
  onSelect: (block: BlockNode) => void;
  defaultExpanded?: boolean;
}

const SEVERITY_COLORS: Record<string, string> = {
  low:      'rgba(var(--color-success-rgb), 0.6)',
  medium:   'rgba(var(--color-warning-rgb), 0.6)',
  high:     'rgba(var(--color-error-rgb), 0.6)',
  critical: 'var(--color-error)',
};

export const BlockBox = React.memo(function BlockBox({
  block,
  isSelected,
  isActive = false,
  onSelect,
  defaultExpanded = false,
}: BlockBoxProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  const hasLayers = block.layers.length > 0;
  const borderColor = isActive
    ? 'var(--accent-primary)'
    : isSelected
    ? 'rgba(var(--accent-secondary-rgb), 0.6)'
    : 'rgba(255,255,255,0.08)';

  return (
    <div
      style={{
        border: `1px solid ${borderColor}`,
        borderRadius: '8px',
        overflow: 'hidden',
        transition: 'border-color 0.15s',
        background: isActive
          ? 'rgba(var(--accent-primary-rgb), 0.08)'
          : isSelected
          ? 'rgba(var(--accent-secondary-rgb), 0.06)'
          : 'rgba(255,255,255,0.02)',
      }}
    >
      {/* Block header */}
      <div
        role="button"
        tabIndex={0}
        aria-expanded={hasLayers ? expanded : undefined}
        onClick={() => { onSelect(block); if (hasLayers) setExpanded((e) => !e); }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onSelect(block);
            if (hasLayers) setExpanded((ex) => !ex);
          }
        }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '8px 10px',
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        {/* Expand chevron */}
        <span style={{ color: 'var(--text-muted, #6b7280)', flexShrink: 0, width: 16 }}>
          {hasLayers
            ? expanded
              ? <ChevronDown size={14} />
              : <ChevronRight size={14} />
            : <span style={{ display: 'inline-block', width: 14 }} />}
        </span>

        {/* Active indicator */}
        {isActive && (
          <span style={{
            display: 'inline-block',
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: 'var(--accent-primary)',
            flexShrink: 0,
            boxShadow: '0 0 6px var(--accent-primary)',
          }} />
        )}

        {/* Name + type */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: '12px',
            fontWeight: 600,
            color: isSelected ? 'var(--color-text-primary)' : 'var(--color-text-primary)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}>
            {block.name}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--text-muted, #6b7280)', marginTop: 1 }}>
            {block.type}
          </div>
        </div>

        {/* Shape flow */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
          <ShapePill shape={block.input_shape} compact />
          <span style={{ color: 'var(--text-muted, #6b7280)', fontSize: 10 }}>→</span>
          <ShapePill shape={block.output_shape} compact />
        </div>

        {/* FLOPs */}
        <FLOPsBadge flops={block.flops_mflops} params={block.params_M} compact />
      </div>

      {/* Layers (Level 3) */}
      {expanded && hasLayers && (
        <div style={{
          borderTop: '1px solid rgba(255,255,255,0.06)',
          padding: '4px 0',
        }}>
          {block.layers.map((layer) => (
            <LayerRow key={layer.id} layer={layer} />
          ))}
        </div>
      )}
    </div>
  );
});


function LayerRow({ layer }: { layer: LayerNode }) {
  const severityDot = layer.severity && layer.severity !== 'low'
    ? SEVERITY_COLORS[layer.severity] ?? null
    : null;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
      padding: '4px 10px 4px 32px',
    }}>
      {/* Severity dot */}
      {severityDot ? (
        <span style={{
          flexShrink: 0,
          width: 5,
          height: 5,
          borderRadius: '50%',
          background: severityDot,
        }} />
      ) : (
        <span style={{ flexShrink: 0, width: 5 }} />
      )}

      {/* Layer name */}
      <div style={{
        flex: 1,
        minWidth: 0,
        fontSize: '11px',
        color: 'var(--text-secondary, #94a3b8)',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}>
        {layer.name}
        <span style={{ fontSize: 9, marginLeft: 4, color: 'var(--text-muted, #6b7280)' }}>
          {layer.type}
        </span>
      </div>

      {/* Shapes */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 3, flexShrink: 0 }}>
        <ShapePill shape={layer.input_shape} compact />
        <span style={{ color: 'var(--text-muted, #6b7280)', fontSize: 9 }}>→</span>
        <ShapePill shape={layer.output_shape} compact />
      </div>

      {/* FLOPs */}
      <FLOPsBadge flops={layer.flops_mflops} compact />
    </div>
  );
}
