'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { ChevronRight, ChevronDown } from 'lucide-react';
import { BlockBox, type BlockNode } from './BlockBox';
import { ShapePill } from './ShapePill';
import { FLOPsBadge } from './FLOPsBadge';

export interface StageNode {
  id: string;
  name: string;
  type: string;
  description: string;
  input_shape: number[] | null;
  output_shape: number[] | null;
  flops_mflops: number;
  params_M: number;
  blocks: BlockNode[];
}

export interface HierarchyData {
  paper_id: string;
  name: string;
  description: string;
  input_shape: number[];
  total_flops_mflops: number;
  total_params_M: number;
  stages: StageNode[];
}

interface BlockGraphProps {
  data: HierarchyData;
  selectedBlock: BlockNode | null;
  activeBlockId?: string | null;
  onSelectBlock: (block: BlockNode) => void;
}

const STAGE_TYPE_COLORS: Record<string, string> = {
  stem: 'rgba(16,185,129,0.15)',
  backbone: 'rgba(59,130,246,0.12)',
  encoder: 'rgba(59,130,246,0.12)',
  decoder: 'rgba(139,92,246,0.12)',
  bottleneck: 'rgba(245,158,11,0.12)',
  head: 'rgba(236,72,153,0.12)',
  default: 'rgba(255,255,255,0.04)',
};
const STAGE_TYPE_BORDER: Record<string, string> = {
  stem: 'rgba(16,185,129,0.25)',
  backbone: 'rgba(59,130,246,0.2)',
  encoder: 'rgba(59,130,246,0.2)',
  decoder: 'rgba(139,92,246,0.2)',
  bottleneck: 'rgba(245,158,11,0.2)',
  head: 'rgba(236,72,153,0.2)',
  default: 'rgba(255,255,255,0.08)',
};

export const BlockGraph = React.memo(function BlockGraph({
  data,
  selectedBlock,
  activeBlockId,
  onSelectBlock,
}: BlockGraphProps) {
  const [expandedStages, setExpandedStages] = useState<Set<string>>(
    () => new Set(data.stages.map((s) => s.id)),
  );
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to active block
  useEffect(() => {
    if (!activeBlockId || !containerRef.current) return;
    const el = containerRef.current.querySelector(`[data-block-id="${activeBlockId}"]`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [activeBlockId]);

  const toggleStage = useCallback((stageId: string) => {
    setExpandedStages((prev) => {
      const next = new Set(prev);
      if (next.has(stageId)) next.delete(stageId);
      else next.add(stageId);
      return next;
    });
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        padding: '12px',
        overflowY: 'auto',
        height: '100%',
      }}
    >
      {data.stages.map((stage) => {
        const isExpanded = expandedStages.has(stage.id);
        const bg = STAGE_TYPE_COLORS[stage.type] ?? STAGE_TYPE_COLORS.default;
        const border = STAGE_TYPE_BORDER[stage.type] ?? STAGE_TYPE_BORDER.default;

        return (
          <div
            key={stage.id}
            style={{
              border: `1px solid ${border}`,
              borderRadius: '10px',
              overflow: 'hidden',
              background: bg,
            }}
          >
            {/* Stage header */}
            <div
              role="button"
              tabIndex={0}
              onClick={() => toggleStage(stage.id)}
              onKeyDown={(e) => e.key === 'Enter' && toggleStage(stage.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '10px 12px',
                cursor: 'pointer',
                userSelect: 'none',
              }}
            >
              <span style={{ color: 'var(--text-muted, #6b7280)', flexShrink: 0 }}>
                {isExpanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
              </span>

              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: '13px',
                  fontWeight: 700,
                  color: 'var(--text-primary, #e2e8f0)',
                }}>
                  {stage.name}
                </div>
                {stage.description && !isExpanded && (
                  <div style={{
                    fontSize: '11px',
                    color: 'var(--text-muted, #6b7280)',
                    marginTop: 2,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}>
                    {stage.description}
                  </div>
                )}
              </div>

              {/* Stage shapes */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
                <ShapePill shape={stage.input_shape} compact />
                <span style={{ color: 'var(--text-muted, #6b7280)', fontSize: 10 }}>→</span>
                <ShapePill shape={stage.output_shape} compact />
              </div>

              <FLOPsBadge flops={stage.flops_mflops} params={stage.params_M} compact />
            </div>

            {/* Blocks (Level 2) */}
            {isExpanded && (
              <div style={{
                borderTop: `1px solid ${border}`,
                padding: '8px',
                display: 'flex',
                flexDirection: 'column',
                gap: '5px',
              }}>
                {stage.description && (
                  <p style={{
                    fontSize: '11px',
                    color: 'var(--text-muted, #6b7280)',
                    margin: '0 0 6px 0',
                    lineHeight: 1.5,
                    fontStyle: 'italic',
                  }}>
                    {stage.description}
                  </p>
                )}
                {stage.blocks.map((block) => (
                  <div key={block.id} data-block-id={block.id}>
                    <BlockBox
                      block={block}
                      isSelected={selectedBlock?.id === block.id}
                      isActive={activeBlockId === block.id}
                      onSelect={onSelectBlock}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
});
