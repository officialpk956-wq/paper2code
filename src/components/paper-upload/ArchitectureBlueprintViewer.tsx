'use client';

import { useState, useRef } from 'react';
import { ZoomIn, ZoomOut, RefreshCw, ExternalLink } from 'lucide-react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BPComponent {
  id: string;
  name: string;
  type: string;
  shape: number[];
  metadata: Record<string, unknown>;
}

export interface BPConnection {
  source: string;
  target: string;
  connection_type: 'sequential' | 'residual' | 'concat' | 'cross_attention';
}

export interface TensorFlowStep {
  component_id: string;
  component_name: string;
  type: string;
  input_shape: number[];
  output_shape: number[];
  flops_mflops: number;
  params_M: number;
}

export interface ArchitectureBlueprint {
  id: string;
  name: string;
  paper_id: number | null;
  components: BPComponent[];
  connections: BPConnection[];
  input_shape: number[];
  output_shape: number[];
  estimated_parameters: number;
  estimated_flops: number;
  confidence_score: number;
  tensor_flow: TensorFlowStep[];
  architecture_href: string | null;
}

interface ArchitectureBlueprintViewerProps {
  blueprint: ArchitectureBlueprint | null;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CANVAS_W = 700;
const COMP_W   = 320;
const COMP_H   = 76;
const COMP_X   = (CANVAS_W - COMP_W) / 2;  // 190
const ROW_H    = 110;
const SKIP_X   = COMP_X + COMP_W + 72;

const TYPE_COLORS: Record<string, string> = {
  conv:            'var(--accent-cyan)',
  attention:       'var(--accent-primary)',
  embedding:       'var(--accent-transformer)',
  pooling:         'var(--color-text-tertiary)',
  mlp:             'var(--color-warning)',
  normalization:   'var(--color-success)',
  activation:      'var(--color-text-tertiary)',
  skip_connection: 'var(--color-text-tertiary)',
  upsample:        'var(--accent-cyan)',
  downsample:      'var(--accent-cyan)',
  head:            'var(--accent-primary)',
};

const CONN_COLORS: Record<string, string> = {
  sequential:     'var(--color-border)',
  residual:       'var(--accent-primary)',
  concat:         'var(--color-success)',
  cross_attention:'var(--accent-transformer)',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatShape(shape: number[]): string {
  if (!shape || shape.length === 0) return '?';
  const dims = shape.slice(1);  // drop batch dim
  return `[${dims.join('×')}]`;
}

function formatFlops(f: number): string {
  if (f === 0) return '—';
  if (f >= 1000) return `${(f / 1000).toFixed(1)} GFLOPs`;
  return `${f.toFixed(1)} MFLOPs`;
}

function formatParams(p: number): string {
  if (p === 0) return '—';
  if (p >= 1000) return `${(p / 1000).toFixed(2)} B`;
  if (p >= 1) return `${p.toFixed(2)} M`;
  return `${(p * 1000).toFixed(0)} K`;
}

function formatParamsTotal(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B params`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M params`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K params`;
  return `${n} params`;
}

function compY(i: number): number { return 20 + i * ROW_H; }

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ArchitectureBlueprintViewer({ blueprint }: ArchitectureBlueprintViewerProps) {
  const [zoom, setZoom]       = useState(1.0);
  const [pan,  setPan]        = useState({ x: 0, y: 0 });
  const [selected, setSelected] = useState<string | null>(null);
  const dragging  = useRef(false);
  const dragStart = useRef({ x: 0, y: 0, px: 0, py: 0 });

  if (!blueprint) {
    return (
      <div
        className="flex items-center justify-center rounded-2xl border border-dashed border-[--color-border] bg-[--bg-body] p-12 text-sm text-[--color-text-tertiary]"
        role="status"
      >
        No architecture blueprint available for this paper.
      </div>
    );
  }

  const { components, connections, tensor_flow, confidence_score, architecture_href } = blueprint;
  const compIndex: Record<string, number> = {};
  components.forEach((c, i) => { compIndex[c.id] = i; });
  const flowByCompId: Record<string, TensorFlowStep> = {};
  tensor_flow.forEach(s => { flowByCompId[s.component_id] = s; });

  const CANVAS_H = components.length * ROW_H + 40;
  const confidenceColor =
    confidence_score >= 0.70 ? 'var(--color-success)' :
    confidence_score >= 0.40 ? 'var(--color-warning)' :
    'var(--color-text-tertiary)';

  // Pan/zoom handlers
  function onMouseDown(e: React.MouseEvent) {
    dragging.current  = true;
    dragStart.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
  }
  function onMouseMove(e: React.MouseEvent) {
    if (!dragging.current) return;
    setPan({
      x: dragStart.current.px + e.clientX - dragStart.current.x,
      y: dragStart.current.py + e.clientY - dragStart.current.y,
    });
  }
  function onMouseUp() { dragging.current = false; }

  function zoomIn()   { setZoom(z => Math.min(z + 0.2, 3.0)); }
  function zoomOut()  { setZoom(z => Math.max(z - 0.2, 0.4)); }
  function resetView(){ setZoom(1.0); setPan({ x: 0, y: 0 }); }

  function handleSelect(id: string) {
    setSelected(prev => prev === id ? null : id);
  }
  function handleKey(e: React.KeyboardEvent, id: string) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleSelect(id); }
  }

  const selectedStep = selected ? flowByCompId[selected] : null;
  const selectedComp = selected ? components.find(c => c.id === selected) : null;

  return (
    <div className="flex flex-col gap-3">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-[--color-border] bg-[--bg-surface] px-4 py-2.5">
        {/* Confidence */}
        <div className="flex items-center gap-2" role="group" aria-label="Architecture confidence">
          <span className="text-xs text-[--color-text-tertiary]">Confidence</span>
          <div className="relative h-1.5 w-20 overflow-hidden rounded-full bg-[--bg-body]" aria-hidden="true">
            <div
              className="absolute inset-y-0 left-0 rounded-full transition-all"
              style={{ width: `${confidence_score * 100}%`, background: confidenceColor }}
            />
          </div>
          <span className="text-xs font-semibold" style={{ color: confidenceColor }}>
            {Math.round(confidence_score * 100)}%
          </span>
        </div>

        {/* Name + totals */}
        <span className="hidden text-xs text-[--color-text-secondary] sm:block">
          <span className="font-medium text-[--color-text-primary]">{blueprint.name}</span>
          {' · '}
          {formatParamsTotal(blueprint.estimated_parameters)}
          {' · '}
          {formatFlops(blueprint.estimated_flops)}
        </span>

        {/* Controls */}
        <div className="flex items-center gap-2">
          {architecture_href && (
            <a
              href={architecture_href}
              className="flex items-center gap-1 rounded-lg border border-[--color-border] bg-[--bg-body] px-2 py-1 text-xs text-[--accent-cyan] transition hover:border-[--accent-cyan]"
              aria-label="Open Architecture"
            >
              Open Architecture
              <ExternalLink className="h-3 w-3" />
            </a>
          )}
          <span className="text-xs text-[--color-text-tertiary]">
            {components.length} nodes · {connections.length} edges
          </span>
          <div className="flex items-center gap-1 rounded-xl border border-[--color-border] bg-[--bg-body] px-1">
            <button
              onClick={zoomOut}
              aria-label="Zoom out"
              className="flex h-6 w-6 items-center justify-center rounded-lg text-[--color-text-secondary] transition hover:text-[--color-text-primary]"
            >
              <ZoomOut className="h-3 w-3" />
            </button>
            <span className="min-w-[2.5rem] text-center text-xs font-medium text-[--color-text-primary]">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={zoomIn}
              aria-label="Zoom in"
              className="flex h-6 w-6 items-center justify-center rounded-lg text-[--color-text-secondary] transition hover:text-[--color-text-primary]"
            >
              <ZoomIn className="h-3 w-3" />
            </button>
            <button
              onClick={resetView}
              aria-label="Reset view"
              className="flex h-6 w-6 items-center justify-center rounded-lg text-[--color-text-secondary] transition hover:text-[--color-text-primary]"
            >
              <RefreshCw className="h-3 w-3" />
            </button>
          </div>
        </div>
      </div>

      {/* SVG canvas */}
      <div
        className="overflow-hidden rounded-2xl border border-[--color-border] bg-[--bg-body]"
        style={{ height: Math.min(CANVAS_H * zoom + 40, 600), cursor: dragging.current ? 'grabbing' : 'grab' }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        <svg
          role="img"
          aria-label={`Architecture blueprint for ${blueprint.name}: ${components.length} components`}
          width="100%"
          viewBox={`0 0 ${CANVAS_W} ${CANVAS_H}`}
          style={{ display: 'block' }}
        >
          <defs>
            <pattern id="bp-grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="var(--color-border)" strokeWidth="0.3" />
            </pattern>
            <marker id="bp-arrow-seq" markerWidth="7" markerHeight="7" refX="3.5" refY="2" orient="auto">
              <path d="M 0 0 L 7 2 L 0 4 Z" fill="var(--color-border)" />
            </marker>
            <marker id="bp-arrow-concat" markerWidth="7" markerHeight="7" refX="3.5" refY="2" orient="auto">
              <path d="M 0 0 L 7 2 L 0 4 Z" fill="var(--color-success)" />
            </marker>
            <marker id="bp-arrow-residual" markerWidth="7" markerHeight="7" refX="3.5" refY="2" orient="auto">
              <path d="M 0 0 L 7 2 L 0 4 Z" fill="var(--accent-primary)" />
            </marker>
            <marker id="bp-arrow-xattn" markerWidth="7" markerHeight="7" refX="3.5" refY="2" orient="auto">
              <path d="M 0 0 L 7 2 L 0 4 Z" fill="var(--accent-transformer)" />
            </marker>
          </defs>

          {/* Grid background */}
          <rect width={CANVAS_W} height={CANVAS_H} fill="url(#bp-grid)" />

          {/* Pan/zoom group */}
          <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
            {/* Connection edges */}
            {connections.map((conn, ci) => {
              const si = compIndex[conn.source];
              const ti = compIndex[conn.target];
              if (si === undefined || ti === undefined) return null;

              const color = CONN_COLORS[conn.connection_type] ?? 'var(--color-border)';

              if (conn.connection_type === 'sequential') {
                const x  = COMP_X + COMP_W / 2;
                const y1 = compY(si) + COMP_H + 2;
                const y2 = compY(ti) - 2;
                if (y2 <= y1) return null;
                return (
                  <line
                    key={`conn-${ci}`}
                    x1={x} y1={y1} x2={x} y2={y2}
                    stroke={color}
                    strokeWidth="1.5"
                    markerEnd="url(#bp-arrow-seq)"
                  />
                );
              }

              // Skip / concat / residual / cross_attention — bezier on right
              const markerId =
                conn.connection_type === 'concat'         ? 'bp-arrow-concat' :
                conn.connection_type === 'cross_attention'? 'bp-arrow-xattn'  :
                'bp-arrow-residual';
              const strokeDash =
                conn.connection_type === 'concat'          ? 'none' :
                conn.connection_type === 'cross_attention' ? '4 3'  :
                '3 2';

              const sx = COMP_X + COMP_W + 2;
              const sy = compY(si) + COMP_H / 2;
              const tx = COMP_X + COMP_W + 2;
              const ty = compY(ti) + COMP_H / 2;
              const cx = SKIP_X;
              const d  = `M ${sx} ${sy} C ${cx} ${sy}, ${cx} ${ty}, ${tx} ${ty}`;

              return (
                <path
                  key={`conn-${ci}`}
                  d={d}
                  fill="none"
                  stroke={color}
                  strokeWidth="1.5"
                  strokeDasharray={strokeDash}
                  markerEnd={`url(#${markerId})`}
                />
              );
            })}

            {/* Component nodes */}
            {components.map((comp, i) => {
              const step       = flowByCompId[comp.id];
              const isSelected = selected === comp.id;
              const typeColor  = TYPE_COLORS[comp.type] ?? 'var(--color-text-tertiary)';
              const cx         = COMP_X;
              const cy         = compY(i);

              return (
                <g
                  key={comp.id}
                  transform={`translate(${cx},${cy})`}
                  role="button"
                  tabIndex={0}
                  aria-pressed={isSelected}
                  aria-label={`${comp.name} — ${comp.type}`}
                  style={{ cursor: 'pointer', outline: 'none' }}
                  onClick={(e) => { e.stopPropagation(); handleSelect(comp.id); }}
                  onKeyDown={(e) => handleKey(e, comp.id)}
                >
                  {/* Background */}
                  <rect
                    width={COMP_W}
                    height={COMP_H}
                    rx="10"
                    fill={isSelected ? 'var(--bg-panel)' : 'var(--bg-surface)'}
                    stroke={isSelected ? typeColor : 'var(--color-border)'}
                    strokeWidth={isSelected ? 1.5 : 1}
                  />

                  {/* Type badge */}
                  <rect x="8" y="9" width="48" height="16" rx="5"
                        fill={typeColor} fillOpacity="0.15" />
                  <text x="32" y="21" textAnchor="middle"
                        fill={typeColor} fontSize="9" fontWeight="700" fontFamily="monospace">
                    {comp.type.toUpperCase().slice(0, 7)}
                  </text>

                  {/* Has-residual marker */}
                  {!!comp.metadata.has_residual && (
                    <>
                      <rect x="62" y="9" width="20" height="16" rx="5"
                            fill="var(--accent-primary)" fillOpacity="0.12" />
                      <text x="72" y="21" textAnchor="middle"
                            fill="var(--accent-primary)" fontSize="9" fontWeight="700">↩</text>
                    </>
                  )}

                  {/* Component name */}
                  <text x="8" y="43" fill="var(--color-text-primary)"
                        fontSize="12" fontWeight="600">
                    {comp.name.length > 38 ? comp.name.slice(0, 36) + '…' : comp.name}
                  </text>

                  {/* Shape */}
                  {step && (
                    <text x="8" y="62"
                          fill="var(--color-text-tertiary)" fontSize="10" fontFamily="monospace">
                      {formatShape(step.input_shape)} → {formatShape(step.output_shape)}
                    </text>
                  )}

                  {/* FLOPs (right side) */}
                  {step && step.flops_mflops > 0 && (
                    <text x={COMP_W - 8} y="26" textAnchor="end"
                          fill="var(--color-warning)" fontSize="10" fontWeight="500">
                      {formatFlops(step.flops_mflops)}
                    </text>
                  )}

                  {/* Params (right side) */}
                  {step && step.params_M > 0 && (
                    <text x={COMP_W - 8} y="40" textAnchor="end"
                          fill="var(--color-text-tertiary)" fontSize="10">
                      {formatParams(step.params_M)}
                    </text>
                  )}

                  {/* Repeat badge */}
                  {((comp.metadata.repeat_count as number) ?? 1) > 1 && (
                    <text x={COMP_W - 8} y="62" textAnchor="end"
                          fill="var(--accent-cyan)" fontSize="10" fontFamily="monospace">
                      ×{String(comp.metadata.repeat_count)}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {/* Connection legend */}
      <div className="flex flex-wrap gap-4 px-1 text-xs text-[--color-text-tertiary]">
        {[
          { type: 'sequential',      label: 'Sequential',       dash: 'none' },
          { type: 'residual',        label: 'Residual',         dash: '3 2' },
          { type: 'concat',          label: 'Skip/Concat',      dash: 'none' },
          { type: 'cross_attention', label: 'Cross-Attention',  dash: '4 3' },
        ].map(({ type, label, dash }) => (
          <div key={type} className="flex items-center gap-1.5">
            <svg width="24" height="8" aria-hidden="true">
              <line x1="0" y1="4" x2="24" y2="4"
                    stroke={CONN_COLORS[type]} strokeWidth="1.5"
                    strokeDasharray={dash === 'none' ? undefined : dash} />
            </svg>
            <span>{label}</span>
          </div>
        ))}
      </div>

      {/* Selected node info panel */}
      {selectedStep && selectedComp && (
        <div
          role="region"
          aria-label="Selected node details"
          className="rounded-2xl border border-[--color-border] bg-[--bg-panel] p-4"
        >
          <div className="flex items-start justify-between">
            <div>
              <div className="text-sm font-semibold text-[--color-text-primary]">
                {selectedComp.name}
              </div>
              <div className="mt-0.5 text-xs text-[--color-text-tertiary]">
                Type: <span style={{ color: TYPE_COLORS[selectedComp.type] ?? 'inherit' }}>
                  {selectedComp.type}
                </span>
              </div>
            </div>
            <button
              onClick={() => setSelected(null)}
              aria-label="Close details panel"
              className="text-xs text-[--color-text-tertiary] transition hover:text-[--color-text-primary]"
            >
              ✕
            </button>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            <div className="rounded-xl bg-[--bg-body] p-2">
              <div className="text-[--color-text-tertiary]">Input Shape</div>
              <div className="mt-0.5 font-mono text-[--color-text-primary]">
                {formatShape(selectedStep.input_shape)}
              </div>
            </div>
            <div className="rounded-xl bg-[--bg-body] p-2">
              <div className="text-[--color-text-tertiary]">Output Shape</div>
              <div className="mt-0.5 font-mono text-[--color-text-primary]">
                {formatShape(selectedStep.output_shape)}
              </div>
            </div>
            <div className="rounded-xl bg-[--bg-body] p-2">
              <div className="text-[--color-text-tertiary]">FLOPs</div>
              <div className="mt-0.5 font-medium text-[--color-warning]">
                {formatFlops(selectedStep.flops_mflops)}
              </div>
            </div>
            <div className="rounded-xl bg-[--bg-body] p-2">
              <div className="text-[--color-text-tertiary]">Parameters</div>
              <div className="mt-0.5 font-medium text-[--color-text-primary]">
                {formatParams(selectedStep.params_M)}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
