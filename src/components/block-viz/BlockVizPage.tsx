'use client';

import React, { useState, useCallback, useMemo } from 'react';
import useSWR from 'swr';
import { Cpu, Activity, Layers, BarChart2, AlertCircle, RefreshCw } from 'lucide-react';
import { BlockGraph, type HierarchyData } from './BlockGraph';
import { ForwardPassPlayer, type ForwardPassStep } from './ForwardPassPlayer';
import { ShapePill } from './ShapePill';
import { FLOPsBadge } from './FLOPsBadge';
import type { BlockNode, LayerNode } from './BlockBox';

const ARCHITECTURES = [
  { id: 'resnet50', label: 'ResNet-50', subtitle: 'He et al. 2015', icon: '🧱' },
  { id: 'vit', label: 'ViT-B/16', subtitle: 'Dosovitskiy et al. 2020', icon: '🔲' },
  { id: 'unet', label: 'U-Net', subtitle: 'Ronneberger et al. 2015', icon: '↕' },
  { id: 'transformer', label: 'Transformer-Base', subtitle: 'Vaswani et al. 2017', icon: '✦' },
];

const fetcher = (url: string) => fetch(url).then((r) => {
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
});

export function BlockVizPage() {
  const [archId, setArchId] = useState('resnet50');
  const [selectedBlock, setSelectedBlock] = useState<BlockNode | null>(null);
  const [activeBlockId, setActiveBlockId] = useState<string | null>(null);
  const [showPlayer, setShowPlayer] = useState(false);

  const { data: hierarchy, error: hierError, isLoading: hierLoading } = useSWR<HierarchyData>(
    `/api/papers/${archId}/block-hierarchy`,
    fetcher,
    { revalidateOnFocus: false },
  );

  const { data: fwdPass, isLoading: fwdLoading } = useSWR<{
    paper_id: string;
    name: string;
    total_steps: number;
    steps: ForwardPassStep[];
  }>(
    showPlayer ? `/api/papers/${archId}/forward-pass` : null,
    fetcher,
    { revalidateOnFocus: false },
  );

  const handleStepChange = useCallback((step: ForwardPassStep | null) => {
    setActiveBlockId(step?.node_id ?? null);
  }, []);

  const handleSelectBlock = useCallback((block: BlockNode) => {
    setSelectedBlock((prev) => (prev?.id === block.id ? null : block));
  }, []);

  const selectedBlockDetail = useMemo(() => {
    if (!selectedBlock || !hierarchy) return null;
    // Find within hierarchy to get layers
    for (const stage of hierarchy.stages) {
      const found = stage.blocks.find((b: BlockNode) => b.id === selectedBlock.id);
      if (found) return found;
    }
    return selectedBlock;
  }, [selectedBlock, hierarchy]);

  return (
    <div style={{
      display: 'flex',
      height: '100%',
      width: '100%',
      overflow: 'hidden',
      background: 'var(--bg-body, #0f1117)',
      fontFamily: 'var(--font-sans, system-ui)',
    }}>
      {/* ── Left column: arch selector + summary ── */}
      <div style={{
        width: 260,
        flexShrink: 0,
        borderRight: '1px solid rgba(255,255,255,0.08)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{ padding: '16px 16px 12px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Layers size={16} style={{ color: 'var(--accent-primary, #3b82f6)' }} />
            <span style={{ fontSize: '13px', fontWeight: 700, color: '#e2e8f0' }}>
              Block Explorer
            </span>
          </div>
          <p style={{ fontSize: '11px', color: 'var(--text-muted, #6b7280)', margin: '4px 0 0' }}>
            Real architecture hierarchies from PyTorch blocks
          </p>
        </div>

        {/* Architecture selector */}
        <div style={{ padding: '10px 8px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          {ARCHITECTURES.map((arch) => (
            <button
              key={arch.id}
              onClick={() => {
                setArchId(arch.id);
                setSelectedBlock(null);
                setActiveBlockId(null);
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                width: '100%',
                padding: '8px 10px',
                borderRadius: '8px',
                border: `1px solid ${archId === arch.id ? 'rgba(59,130,246,0.4)' : 'transparent'}`,
                background: archId === arch.id ? 'rgba(59,130,246,0.1)' : 'transparent',
                cursor: 'pointer',
                textAlign: 'left',
                marginBottom: 2,
              }}
            >
              <span style={{ fontSize: '18px', flexShrink: 0 }}>{arch.icon}</span>
              <div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: archId === arch.id ? '#93c5fd' : '#cbd5e1' }}>
                  {arch.label}
                </div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted, #6b7280)' }}>
                  {arch.subtitle}
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* Architecture stats */}
        {hierarchy && (
          <div style={{ padding: '12px 16px', flex: 1, overflowY: 'auto' }}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted, #6b7280)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
              Stats
            </div>
            <StatRow label="Total FLOPs" value={<FLOPsBadge flops={hierarchy.total_flops_mflops} />} />
            <StatRow label="Parameters" value={
              <span style={{ fontSize: '11px', fontFamily: 'monospace', color: '#e2e8f0' }}>
                {hierarchy.total_params_M.toFixed(1)}M
              </span>
            } />
            <StatRow label="Stages" value={
              <span style={{ fontSize: '11px', color: '#e2e8f0' }}>{hierarchy.stages.length}</span>
            } />
            <StatRow label="Input" value={<ShapePill shape={hierarchy.input_shape} compact />} />

            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted, #6b7280)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
                Description
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-muted, #9ca3af)', lineHeight: 1.6, margin: 0 }}>
                {hierarchy.description}
              </p>
            </div>
          </div>
        )}

        {/* Forward pass toggle */}
        <div style={{ padding: '10px 12px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
          <button
            onClick={() => setShowPlayer((v) => !v)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              width: '100%',
              padding: '8px 10px',
              borderRadius: '8px',
              border: `1px solid ${showPlayer ? 'rgba(59,130,246,0.4)' : 'rgba(255,255,255,0.12)'}`,
              background: showPlayer ? 'rgba(59,130,246,0.1)' : 'transparent',
              color: showPlayer ? '#93c5fd' : 'var(--text-secondary, #94a3b8)',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: 600,
            }}
          >
            <Activity size={14} />
            {showPlayer ? 'Hide Forward Pass' : 'Animate Forward Pass'}
          </button>
        </div>
      </div>

      {/* ── Middle column: block hierarchy ── */}
      <div style={{
        flex: 1,
        minWidth: 0,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}>
        {/* Column header */}
        <div style={{
          padding: '12px 16px',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          flexShrink: 0,
        }}>
          <BarChart2 size={14} style={{ color: 'var(--text-muted, #6b7280)' }} />
          <span style={{ fontSize: '12px', fontWeight: 600, color: '#cbd5e1' }}>
            {hierarchy?.name ?? 'Architecture'} — Block Hierarchy
          </span>
          {hierLoading && (
            <RefreshCw size={12} style={{ color: 'var(--text-muted, #6b7280)', animation: 'spin 1s linear infinite' }} />
          )}
          <span style={{ flex: 1 }} />
          <span style={{ fontSize: '10px', color: 'var(--text-muted, #6b7280)' }}>
            Click a block to inspect · Expand chevrons for layers
          </span>
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          {hierError ? (
            <ErrorState message={hierError.message} />
          ) : hierLoading ? (
            <LoadingState />
          ) : hierarchy ? (
            <>
              <div style={{ flex: 1, overflow: 'hidden' }}>
                <BlockGraph
                  data={hierarchy}
                  selectedBlock={selectedBlock}
                  activeBlockId={activeBlockId}
                  onSelectBlock={handleSelectBlock}
                />
              </div>
              {/* Forward pass player */}
              {showPlayer && (
                <div style={{
                  flexShrink: 0,
                  borderTop: '1px solid rgba(255,255,255,0.06)',
                  padding: '10px',
                }}>
                  {fwdLoading ? (
                    <div style={{ textAlign: 'center', padding: '12px', color: 'var(--text-muted, #6b7280)', fontSize: 12 }}>
                      Loading forward pass…
                    </div>
                  ) : fwdPass?.steps ? (
                    <ForwardPassPlayer steps={fwdPass.steps} onStepChange={handleStepChange} />
                  ) : null}
                </div>
              )}
            </>
          ) : null}
        </div>
      </div>

      {/* ── Right column: node inspector ── */}
      <div style={{
        width: 320,
        flexShrink: 0,
        borderLeft: '1px solid rgba(255,255,255,0.08)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}>
        <div style={{ padding: '12px 16px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Cpu size={14} style={{ color: 'var(--text-muted, #6b7280)' }} />
            <span style={{ fontSize: '12px', fontWeight: 600, color: '#cbd5e1' }}>
              Node Inspector
            </span>
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
          {selectedBlockDetail ? (
            <NodeInspector block={selectedBlockDetail} />
          ) : (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: 'var(--text-muted, #6b7280)',
              textAlign: 'center',
              padding: 24,
              gap: 8,
            }}>
              <Cpu size={32} style={{ opacity: 0.3 }} />
              <p style={{ fontSize: '12px', margin: 0, lineHeight: 1.6 }}>
                Click any block in the hierarchy to inspect its shapes, FLOPs, and layer breakdown.
              </p>
            </div>
          )}
        </div>
      </div>

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}


// ── Helpers ──────────────────────────────────────────────────────────────

function StatRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '4px 0',
      borderBottom: '1px solid rgba(255,255,255,0.04)',
    }}>
      <span style={{ fontSize: '11px', color: 'var(--text-muted, #6b7280)' }}>{label}</span>
      {value}
    </div>
  );
}

function NodeInspector({ block }: { block: BlockNode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Block identity */}
      <div>
        <div style={{ fontSize: '14px', fontWeight: 700, color: '#e2e8f0', marginBottom: 2 }}>
          {block.name}
        </div>
        <div style={{ fontSize: '11px', color: 'var(--text-muted, #6b7280)', marginBottom: 8 }}>
          {block.type} · <code style={{ fontSize: '10px' }}>{block.id}</code>
        </div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <ShapePill shape={block.input_shape} label="in" />
          <ShapePill shape={block.output_shape} label="out" />
        </div>
      </div>

      {/* Cost summary */}
      <div style={{
        padding: '10px',
        background: 'rgba(255,255,255,0.03)',
        borderRadius: '8px',
        border: '1px solid rgba(255,255,255,0.08)',
      }}>
        <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted, #6b7280)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
          Compute Cost
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <FLOPsBadge flops={block.flops_mflops} params={block.params_M} />
        </div>
        <div style={{ fontSize: '11px', color: 'var(--text-muted, #9ca3af)', marginTop: 4 }}>
          {block.flops_mflops.toFixed(2)} MFLOPs · {block.params_M.toFixed(4)}M params
        </div>
      </div>

      {/* Layer breakdown */}
      {block.layers.length > 0 && (
        <div>
          <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted, #6b7280)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
            Layer Breakdown ({block.layers.length})
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {block.layers.map((layer) => (
              <LayerDetail key={layer.id} layer={layer} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function LayerDetail({ layer }: { layer: LayerNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div
      style={{
        borderRadius: 6,
        border: '1px solid rgba(255,255,255,0.07)',
        overflow: 'hidden',
        background: 'rgba(255,255,255,0.02)',
      }}
    >
      <div
        role="button"
        tabIndex={0}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => e.key === 'Enter' && setOpen((o) => !o)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '6px 8px',
          cursor: 'pointer',
        }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: '#cbd5e1', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {layer.name}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
          <ShapePill shape={layer.output_shape} compact />
          <FLOPsBadge flops={layer.flops_mflops} compact />
        </div>
      </div>
      {open && layer.formula && (
        <div style={{
          padding: '6px 8px',
          borderTop: '1px solid rgba(255,255,255,0.06)',
          fontSize: '10px',
          color: 'var(--text-muted, #9ca3af)',
          fontFamily: 'var(--font-mono, monospace)',
          whiteSpace: 'pre-wrap',
          lineHeight: 1.6,
        }}>
          {layer.formula}
          {layer.complexity && (
            <div style={{ marginTop: 2, color: '#60a5fa' }}>{layer.complexity}</div>
          )}
        </div>
      )}
    </div>
  );
}

function LoadingState() {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      gap: 12,
      color: 'var(--text-muted, #6b7280)',
    }}>
      <RefreshCw size={24} style={{ animation: 'spin 1s linear infinite' }} />
      <div style={{ fontSize: '13px' }}>Loading architecture…</div>
      <div style={{ fontSize: '11px', opacity: 0.7 }}>Running shape propagation via PyTorch hooks</div>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      gap: 8,
      color: '#f87171',
      padding: 24,
      textAlign: 'center',
    }}>
      <AlertCircle size={24} />
      <div style={{ fontSize: '13px', fontWeight: 600 }}>Failed to load architecture</div>
      <div style={{ fontSize: '11px', opacity: 0.8, fontFamily: 'monospace' }}>{message}</div>
    </div>
  );
}
