'use client';

import { useState } from 'react';
import { ShapePill } from '@/components/block-viz/ShapePill';
import { FLOPsBadge } from '@/components/block-viz/FLOPsBadge';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface EGNode {
  id: string;
  name: string;
  type: string;
  shape: number[];
  params: number;
  flops: number;
  metadata: Record<string, unknown>;
}

export interface EGEdge {
  source: string;
  target: string;
  edge_type: string;
}

export interface ValidationReport {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface ExecutableGraph {
  id: string;
  nodes: EGNode[];
  edges: EGEdge[];
  input_spec: number[];
  output_spec: number[];
  parameter_count: number;
  flops: number;
  validation_status: string;
  validation_report: ValidationReport;
}

// ---------------------------------------------------------------------------
// Layout constants (consistent with ArchitectureBlueprintViewer)
// ---------------------------------------------------------------------------

const CANVAS_W = 700;
const NODE_W = 320;
const NODE_H = 76;
const NODE_X = 190;
const ROW_H = 110;
const SKIP_X = NODE_X + NODE_W + 72; // 562

const nodeY = (i: number) => 20 + i * ROW_H;

// ---------------------------------------------------------------------------
// Type colour palette
// ---------------------------------------------------------------------------

const TYPE_COLORS: Record<string, string> = {
  conv:          'var(--accent-cyan, #22d3ee)',
  attention:     'var(--accent-primary, #818cf8)',
  embedding:     'var(--accent-transformer, #a78bfa)',
  pooling:       'var(--accent-tertiary, #6366f1)',
  mlp:           'var(--accent-warning, #fbbf24)',
  normalization: 'var(--accent-success, #34d399)',
  activation:    'var(--accent-success, #34d399)',
  head:          'var(--accent-primary, #818cf8)',
  upsample:      'var(--accent-cyan, #22d3ee)',
  downsample:    'var(--accent-cyan, #22d3ee)',
  skip_connection: 'var(--accent-warning, #fbbf24)',
};

const EDGE_COLORS: Record<string, string> = {
  sequential:    'var(--color-border, rgba(255,255,255,0.12))',
  residual:      'var(--accent-primary, #818cf8)',
  concat:        'var(--accent-success, #34d399)',
  cross_attention: 'var(--accent-transformer, #a78bfa)',
};

// ---------------------------------------------------------------------------
// Client-side export helpers
// ---------------------------------------------------------------------------

function buildMermaid(graph: ExecutableGraph): string {
  const sanitize = (id: string) => id.replace(/[^a-zA-Z0-9_]/g, '_');
  const shapeLabel = (s: number[]) =>
    s.length > 1 ? s.slice(1).join('×') : (s[0]?.toString() ?? '?');

  const lines = ['flowchart TD'];
  for (const n of graph.nodes) {
    lines.push(`  ${sanitize(n.id)}["${n.name}\\n[${shapeLabel(n.shape)}]"]`);
  }
  for (const e of graph.edges) {
    const arrow =
      e.edge_type === 'sequential' ? '-->' : '-.->';
    lines.push(`  ${sanitize(e.source)} ${arrow} ${sanitize(e.target)}`);
  }
  return lines.join('\n');
}

function buildDot(graph: ExecutableGraph): string {
  const sanitize = (id: string) => id.replace(/[^a-zA-Z0-9_]/g, '_');
  const shapeLabel = (s: number[]) =>
    s.length > 1 ? s.slice(1).join('×') : (s[0]?.toString() ?? '?');

  const lines = [
    'digraph ExecutableGraph {',
    '  rankdir=TB;',
    '  node [shape=box];',
  ];
  for (const n of graph.nodes) {
    lines.push(`  ${sanitize(n.id)} [label="${n.name}\\n[${shapeLabel(n.shape)}]"];`);
  }
  for (const e of graph.edges) {
    const style =
      e.edge_type !== 'sequential' ? ' [style=dashed]' : '';
    lines.push(`  ${sanitize(e.source)} -> ${sanitize(e.target)}${style};`);
  }
  lines.push('}');
  return lines.join('\n');
}

function downloadText(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface ExecutableGraphViewerProps {
  graph: ExecutableGraph | null;
}

export function ExecutableGraphViewer({ graph }: ExecutableGraphViewerProps) {
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // --- Empty state ---
  if (!graph) {
    return (
      <div
        role="status"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '3rem',
          color: 'var(--color-text-secondary)',
          fontSize: '0.875rem',
        }}
      >
        No executable graph available for this paper.
      </div>
    );
  }

  const n = graph.nodes.length;
  const svgH = Math.max(200, n * ROW_H + 40);

  const selectedNode = selectedId
    ? graph.nodes.find((nd) => nd.id === selectedId) ?? null
    : null;

  // ── Zoom helpers ──
  const zoomIn = () => setZoom((z) => Math.min(3.0, +(z + 0.2).toFixed(1)));
  const zoomOut = () => setZoom((z) => Math.max(0.4, +(z - 0.2).toFixed(1)));
  const resetView = () => { setZoom(1.0); setPan({ x: 0, y: 0 }); };

  // ── Pan helpers ──
  const onMouseDown = (e: React.MouseEvent) => {
    setDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!dragging) return;
    setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  };
  const onMouseUp = () => setDragging(false);

  // ── Edge rendering helpers ──
  const indexById = Object.fromEntries(graph.nodes.map((nd, i) => [nd.id, i]));

  function renderEdge(e: EGEdge, key: string) {
    const si = indexById[e.source];
    const ti = indexById[e.target];
    if (si === undefined || ti === undefined) return null;

    const color = EDGE_COLORS[e.edge_type] ?? EDGE_COLORS.sequential;
    const markerId = `eg-arrow-${e.edge_type}`;

    if (e.edge_type === 'sequential') {
      const x = NODE_X + NODE_W / 2;
      const sy = nodeY(si) + NODE_H;
      const ty = nodeY(ti);
      return (
        <line
          key={key}
          x1={x} y1={sy} x2={x} y2={ty}
          stroke={color}
          strokeWidth={1.5}
          markerEnd={`url(#${markerId})`}
        />
      );
    }

    // Non-sequential: bezier on the right side
    const sx = NODE_X + NODE_W;
    const sy = nodeY(si) + NODE_H / 2;
    const tx = NODE_X + NODE_W;
    const ty = nodeY(ti) + NODE_H / 2;
    const strokeDash = e.edge_type === 'cross_attention' ? '4 3' : '';
    return (
      <path
        key={key}
        d={`M ${sx} ${sy} C ${SKIP_X} ${sy}, ${SKIP_X} ${ty}, ${tx} ${ty}`}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeDasharray={strokeDash || undefined}
        markerEnd={`url(#${markerId})`}
      />
    );
  }

  // ── Node rendering ──
  function renderNode(nd: EGNode, i: number) {
    const y = nodeY(i);
    const isSelected = nd.id === selectedId;
    const color = TYPE_COLORS[nd.type] ?? 'var(--color-text-secondary)';
    const inShape = nd.metadata.input_shape;
    const inShapeArr = Array.isArray(inShape) ? (inShape as number[]) : null;

    const toggle = () =>
      setSelectedId((prev) => (prev === nd.id ? null : nd.id));

    return (
      <g
        key={nd.id}
        role="button"
        aria-pressed={isSelected}
        aria-label={`${nd.name} — ${nd.type}`}
        tabIndex={0}
        style={{ cursor: 'pointer' }}
        onClick={toggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') toggle();
        }}
      >
        {/* Node background */}
        <rect
          x={NODE_X}
          y={y}
          width={NODE_W}
          height={NODE_H}
          rx={10}
          fill={isSelected ? 'rgba(129,140,248,0.14)' : 'rgba(14,18,29,0.85)'}
          stroke={isSelected ? 'var(--accent-primary, #818cf8)' : 'rgba(255,255,255,0.10)'}
          strokeWidth={isSelected ? 1.5 : 1}
        />
        {/* Type badge */}
        <rect
          x={NODE_X + 10}
          y={y + 10}
          width={68}
          height={18}
          rx={4}
          fill="rgba(255,255,255,0.05)"
          stroke={color}
          strokeWidth={0.75}
          opacity={0.8}
        />
        <text
          x={NODE_X + 44}
          y={y + 23}
          textAnchor="middle"
          fill={color}
          fontSize={9}
          fontFamily="var(--font-mono, monospace)"
        >
          {nd.type}
        </text>
        {/* Name */}
        <text
          x={NODE_X + 88}
          y={y + 22}
          fill="var(--color-text-primary, #f1f5f9)"
          fontSize={12}
          fontWeight={500}
          fontFamily="var(--font-sans, sans-serif)"
        >
          {nd.name.length > 28 ? nd.name.slice(0, 27) + '…' : nd.name}
        </text>
        {/* Input → Output shape */}
        <text
          x={NODE_X + 88}
          y={y + 40}
          fill="var(--color-text-secondary, #94a3b8)"
          fontSize={10}
          fontFamily="var(--font-mono, monospace)"
        >
          {inShapeArr
            ? `${inShapeArr.slice(1).join('×')} → ${nd.shape.slice(1).join('×')}`
            : nd.shape.slice(1).join('×') || '—'}
        </text>
        {/* FLOPs */}
        {nd.flops > 0 && (
          <text
            x={NODE_X + NODE_W - 10}
            y={y + 58}
            textAnchor="end"
            fill="var(--color-text-tertiary, #64748b)"
            fontSize={9}
            fontFamily="var(--font-mono, monospace)"
          >
            {nd.flops >= 1000
              ? `${(nd.flops / 1000).toFixed(1)}G FLOPs`
              : `${nd.flops.toFixed(0)}M FLOPs`}
          </text>
        )}
      </g>
    );
  }

  // ── Format helpers ──
  const fmtParams = (p: number) => {
    if (p >= 1e9) return `${(p / 1e9).toFixed(2)}B`;
    if (p >= 1e6) return `${(p / 1e6).toFixed(1)}M`;
    if (p >= 1e3) return `${(p / 1e3).toFixed(0)}K`;
    return p.toString();
  };
  const fmtFlops = (f: number) => {
    if (f >= 1000) return `${(f / 1000).toFixed(1)}G`;
    return `${f.toFixed(0)}M`;
  };

  // ── Export actions ──
  const doExport = (fmt: 'json' | 'mermaid' | 'dot') => {
    if (fmt === 'json') {
      downloadText(
        JSON.stringify(graph, null, 2),
        `${graph.id}.json`,
        'application/json',
      );
    } else if (fmt === 'mermaid') {
      downloadText(buildMermaid(graph), `${graph.id}.mmd`, 'text/plain');
    } else {
      downloadText(buildDot(graph), `${graph.id}.dot`, 'text/plain');
    }
  };

  const validStatus = graph.validation_status;
  const statusColor =
    validStatus === 'valid'
      ? 'var(--accent-success, #34d399)'
      : validStatus === 'warnings'
        ? 'var(--accent-warning, #fbbf24)'
        : 'var(--accent-error, #f87171)';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* ── Toolbar ── */}
      <div
        role="region"
        aria-label="Executable graph toolbar"
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: '0.75rem',
          padding: '0.75rem 1rem',
          borderRadius: '12px',
          border: '1px solid var(--color-border, rgba(255,255,255,0.1))',
          background: 'var(--bg-surface, rgba(14,18,29,0.7))',
          fontSize: '0.8125rem',
        }}
      >
        {/* Validation badge */}
        <span
          aria-label={`Validation status: ${validStatus}`}
          style={{
            padding: '2px 10px',
            borderRadius: '6px',
            border: `1px solid ${statusColor}`,
            color: statusColor,
            fontWeight: 600,
            fontSize: '0.75rem',
            letterSpacing: '0.04em',
            textTransform: 'uppercase',
          }}
        >
          {validStatus}
        </span>

        {/* Title */}
        <span style={{ color: 'var(--color-text-primary)', fontWeight: 500 }}>
          {graph.id}
        </span>

        <span style={{ color: 'var(--color-text-tertiary)' }}>
          {graph.nodes.length} nodes · {graph.edges.length} edges
        </span>

        {/* Stats */}
        <span style={{ color: 'var(--color-text-secondary)' }}>
          {fmtParams(graph.parameter_count)} params
        </span>
        <span style={{ color: 'var(--color-text-secondary)' }}>
          {fmtFlops(graph.flops)} FLOPs
        </span>

        <span style={{ flex: 1 }} />

        {/* Zoom controls */}
        <button
          aria-label="Zoom out"
          onClick={zoomOut}
          style={btnStyle}
        >
          −
        </button>
        <span
          style={{
            minWidth: '3rem',
            textAlign: 'center',
            color: 'var(--color-text-secondary)',
            fontSize: '0.8125rem',
          }}
        >
          {Math.round(zoom * 100)}%
        </span>
        <button
          aria-label="Zoom in"
          onClick={zoomIn}
          style={btnStyle}
        >
          +
        </button>
        <button
          aria-label="Reset view"
          onClick={resetView}
          style={btnStyle}
        >
          ↺
        </button>

        {/* Export buttons */}
        <button
          aria-label="Export as JSON"
          onClick={() => doExport('json')}
          style={{ ...btnStyle, paddingLeft: '0.75rem', paddingRight: '0.75rem' }}
        >
          JSON
        </button>
        <button
          aria-label="Export as Mermaid"
          onClick={() => doExport('mermaid')}
          style={{ ...btnStyle, paddingLeft: '0.75rem', paddingRight: '0.75rem' }}
        >
          Mermaid
        </button>
        <button
          aria-label="Export as DOT"
          onClick={() => doExport('dot')}
          style={{ ...btnStyle, paddingLeft: '0.75rem', paddingRight: '0.75rem' }}
        >
          DOT
        </button>
      </div>

      {/* ── SVG canvas ── */}
      <div
        style={{
          overflow: 'hidden',
          borderRadius: '12px',
          border: '1px solid var(--color-border, rgba(255,255,255,0.1))',
          background: 'var(--bg-body, #0a0c14)',
          cursor: dragging ? 'grabbing' : 'grab',
          userSelect: 'none',
        }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        <svg
          role="img"
          aria-label={`Executable graph for ${graph.id}`}
          viewBox={`0 0 ${CANVAS_W} ${svgH}`}
          width="100%"
          style={{ display: 'block', maxHeight: '480px' }}
        >
          {/* Arrowhead markers */}
          <defs>
            {['sequential', 'residual', 'concat', 'cross_attention'].map((et) => (
              <marker
                key={et}
                id={`eg-arrow-${et}`}
                markerWidth={8}
                markerHeight={6}
                refX={7}
                refY={3}
                orient="auto"
              >
                <path
                  d="M0,0 L0,6 L8,3 z"
                  fill={EDGE_COLORS[et] ?? EDGE_COLORS.sequential}
                />
              </marker>
            ))}
          </defs>

          {/* Grid pattern */}
          <defs>
            <pattern id="eg-grid" width={24} height={24} patternUnits="userSpaceOnUse">
              <path
                d="M 24 0 L 0 0 0 24"
                fill="none"
                stroke="rgba(255,255,255,0.03)"
                strokeWidth={0.5}
              />
            </pattern>
          </defs>
          <rect width={CANVAS_W} height={svgH} fill="url(#eg-grid)" />

          <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
            {/* Edges first (beneath nodes) */}
            {graph.edges.map((e, i) => renderEdge(e, `edge-${i}`))}

            {/* Nodes */}
            {graph.nodes.map((nd, i) => renderNode(nd, i))}
          </g>
        </svg>
      </div>

      {/* ── Selected node inspector ── */}
      {selectedNode && (
        <div
          role="region"
          aria-label="Selected node details"
          style={{
            padding: '1rem',
            borderRadius: '12px',
            border: '1px solid var(--color-border, rgba(255,255,255,0.1))',
            background: 'var(--bg-surface, rgba(14,18,29,0.7))',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span
              style={{
                fontSize: '0.9375rem',
                fontWeight: 600,
                color: 'var(--color-text-primary)',
              }}
            >
              {selectedNode.name}
            </span>
            <span
              style={{
                fontSize: '0.75rem',
                color: TYPE_COLORS[selectedNode.type] ?? 'var(--color-text-secondary)',
                border: `1px solid ${TYPE_COLORS[selectedNode.type] ?? 'rgba(255,255,255,0.15)'}`,
                padding: '1px 7px',
                borderRadius: '4px',
              }}
            >
              {selectedNode.type}
            </span>
            <span style={{ flex: 1 }} />
            <button
              aria-label="Close details panel"
              onClick={() => setSelectedId(null)}
              style={{ ...btnStyle, fontWeight: 700 }}
            >
              ×
            </button>
          </div>

          {/* Shapes */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--color-text-tertiary)' }}>Input:</span>
            <ShapePill
              shape={
                Array.isArray(selectedNode.metadata.input_shape)
                  ? (selectedNode.metadata.input_shape as number[])
                  : null
              }
            />
            <span style={{ color: 'var(--color-text-tertiary)', fontSize: '0.8rem' }}>→</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--color-text-tertiary)' }}>Output:</span>
            <ShapePill shape={selectedNode.shape} />
          </div>

          {/* FLOPs + Params */}
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
            <FLOPsBadge
              flops={selectedNode.flops}
              params={selectedNode.params / 1_000_000}
            />
            <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>
              {fmtParams(selectedNode.params)} parameters
            </span>
          </div>
        </div>
      )}

      {/* ── Validation panel ── */}
      {(graph.validation_report.errors.length > 0 ||
        graph.validation_report.warnings.length > 0) && (
        <div
          role="region"
          aria-label="Graph validation report"
          style={{
            padding: '0.875rem 1rem',
            borderRadius: '12px',
            border: `1px solid ${
              graph.validation_report.errors.length > 0
                ? 'rgba(248,113,113,0.35)'
                : 'rgba(251,191,36,0.35)'
            }`,
            background:
              graph.validation_report.errors.length > 0
                ? 'rgba(248,113,113,0.06)'
                : 'rgba(251,191,36,0.06)',
          }}
        >
          <div
            style={{
              fontSize: '0.8125rem',
              fontWeight: 600,
              color:
                graph.validation_report.errors.length > 0
                  ? 'var(--accent-error, #f87171)'
                  : 'var(--accent-warning, #fbbf24)',
              marginBottom: '0.5rem',
            }}
          >
            Validation Report
          </div>
          {graph.validation_report.errors.map((msg, i) => (
            <div
              key={`err-${i}`}
              style={{
                fontSize: '0.8125rem',
                color: 'var(--accent-error, #f87171)',
                marginBottom: '0.25rem',
              }}
            >
              ✕ {msg}
            </div>
          ))}
          {graph.validation_report.warnings.map((msg, i) => (
            <div
              key={`warn-${i}`}
              style={{
                fontSize: '0.8125rem',
                color: 'var(--accent-warning, #fbbf24)',
                marginBottom: '0.25rem',
              }}
            >
              ⚠ {msg}
            </div>
          ))}
        </div>
      )}

      {/* ── Legend ── */}
      <div
        style={{
          display: 'flex',
          gap: '1.25rem',
          flexWrap: 'wrap',
          fontSize: '0.75rem',
          color: 'var(--color-text-tertiary)',
        }}
      >
        {Object.entries(EDGE_COLORS).map(([type, color]) => (
          <span key={type} style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span
              style={{
                display: 'inline-block',
                width: '22px',
                height: '2px',
                background: color,
                borderRadius: '1px',
              }}
            />
            {type.charAt(0).toUpperCase() + type.slice(1).replace('_', ' ')}
          </span>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared button style
// ---------------------------------------------------------------------------

const btnStyle: React.CSSProperties = {
  padding: '4px 8px',
  borderRadius: '6px',
  border: '1px solid var(--color-border, rgba(255,255,255,0.12))',
  background: 'var(--bg-surface, rgba(14,18,29,0.7))',
  color: 'var(--color-text-secondary, #94a3b8)',
  cursor: 'pointer',
  fontSize: '0.8125rem',
  lineHeight: 1,
};
