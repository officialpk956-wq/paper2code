'use client';

import { useState, useRef } from 'react';
import { ZoomIn, ZoomOut, RefreshCw } from 'lucide-react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type KGNodeType = 'paper' | 'architecture' | 'concept' | 'dataset' | 'metric' | 'equation';

export interface KGNode {
  id: string;
  name: string;
  type: KGNodeType;
  href: string | null;
}

export interface KGEdge {
  source: string;
  target: string;
  relation: string;
}

interface LayoutNode extends KGNode {
  x: number;
  y: number;
  r: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NODE_COLORS: Record<KGNodeType, string> = {
  paper:        'var(--accent-primary)',
  architecture: 'var(--accent-cyan)',
  concept:      'var(--accent-transformer)',
  dataset:      'var(--color-warning)',
  metric:       'var(--color-success)',
  equation:     'var(--color-text-tertiary)',
};

// ---------------------------------------------------------------------------
// Concept-to-learn mapping (Phase 16D)
// ---------------------------------------------------------------------------

type ConceptMeta = {
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  explanation: string;
  learnUrl: string | null;
};

const CONCEPT_META: Record<string, ConceptMeta> = {
  'attention':                     { difficulty: 'intermediate', explanation: 'Computes relevance weights between all positions in a sequence.', learnUrl: '/learn/deep-learning/attention' },
  'multi-head attention':          { difficulty: 'intermediate', explanation: 'Runs multiple attention heads in parallel, each learning different relationships.', learnUrl: '/learn/deep-learning/attention' },
  'multi head attention':          { difficulty: 'intermediate', explanation: 'Runs multiple attention heads in parallel, each learning different relationships.', learnUrl: '/learn/deep-learning/attention' },
  'self-attention':                { difficulty: 'intermediate', explanation: 'Queries, keys, and values all come from the same sequence.', learnUrl: '/learn/deep-learning/attention' },
  'self attention':                { difficulty: 'intermediate', explanation: 'Queries, keys, and values all come from the same sequence.', learnUrl: '/learn/deep-learning/attention' },
  'scaled dot-product attention':  { difficulty: 'intermediate', explanation: 'Attention(Q,K,V) = softmax(QKᵀ/√d)V. The core attention formula.', learnUrl: '/learn/deep-learning/attention' },
  'transformer':                   { difficulty: 'intermediate', explanation: 'Stacked self-attention and FFN layers. Foundation for BERT, GPT, and modern LLMs.', learnUrl: '/architectures/transformer' },
  'residual connection':           { difficulty: 'intermediate', explanation: 'Skip connections that add input to output, enabling very deep networks.', learnUrl: '/architectures/resnet' },
  'residual connections':          { difficulty: 'intermediate', explanation: 'Skip connections that add input to output, enabling very deep networks.', learnUrl: '/architectures/resnet' },
  'resnet':                        { difficulty: 'intermediate', explanation: 'Deep residual network using skip connections.', learnUrl: '/architectures/resnet' },
  'bert':                          { difficulty: 'advanced',     explanation: 'Bidirectional transformer encoder pre-trained on masked language modeling.', learnUrl: '/architectures/bert' },
  'gpt':                           { difficulty: 'advanced',     explanation: 'Autoregressive transformer decoder for language generation.', learnUrl: '/architectures/gpt' },
  'vision transformer':            { difficulty: 'advanced',     explanation: 'Applies self-attention to image patches, bypassing convolutions.', learnUrl: '/architectures/vit' },
  'vit':                           { difficulty: 'advanced',     explanation: 'Vision Transformer — each image is split into patches treated as tokens.', learnUrl: '/architectures/vit' },
  'layer normalization':           { difficulty: 'beginner',     explanation: 'Normalizes activations per sample. Stabilizes training in transformers.', learnUrl: null },
  'feed-forward network':          { difficulty: 'beginner',     explanation: 'Two linear layers with ReLU applied position-wise in each transformer block.', learnUrl: null },
  'positional encoding':           { difficulty: 'beginner',     explanation: 'Adds position information to token embeddings using sinusoids.', learnUrl: null },
  'softmax':                       { difficulty: 'beginner',     explanation: 'Converts logits to a probability distribution. eˣ / Σeˣ.', learnUrl: null },
};

const DIFFICULTY_COLORS = {
  beginner:     { text: '#10b981', bg: 'rgba(16,185,129,0.12)' },
  intermediate: { text: '#f59e0b', bg: 'rgba(245,158,11,0.12)' },
  advanced:     { text: '#ef4444', bg: 'rgba(239,68,68,0.12)' },
};

function getConceptMeta(name: string): ConceptMeta | null {
  return CONCEPT_META[name.toLowerCase()] ?? null;
}

const NODE_RADII: Record<KGNodeType, number> = {
  paper:        14,
  architecture: 10,
  concept:      8,
  dataset:      8,
  metric:       8,
  equation:     6,
};

const TYPE_ORDER: KGNodeType[] = ['paper', 'architecture', 'concept', 'dataset', 'metric', 'equation'];

const CANVAS_W = 800;
const CANVAS_H = 480;
const ROW_Y: Record<KGNodeType, number> = {
  paper:        60,
  architecture: 150,
  concept:      240,
  dataset:      330,
  metric:       330,
  equation:     415,
};

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

function computeLayout(nodes: KGNode[]): LayoutNode[] {
  const byType: Record<string, KGNode[]> = {};
  for (const n of nodes) {
    (byType[n.type] = byType[n.type] ?? []).push(n);
  }

  // Datasets and metrics share y=330; split left/right
  const datasetNodes = byType['dataset'] ?? [];
  const metricNodes  = byType['metric']  ?? [];

  const placed: LayoutNode[] = [];

  for (const type of TYPE_ORDER) {
    if (type === 'dataset' || type === 'metric') continue;
    const group = byType[type] ?? [];
    const y = ROW_Y[type];
    const totalW = CANVAS_W * 0.8;
    const startX = CANVAS_W * 0.1;
    group.forEach((n, i) => {
      placed.push({
        ...n,
        x: group.length === 1
          ? CANVAS_W / 2
          : startX + (totalW / (group.length - 1 || 1)) * i,
        y,
        r: NODE_RADII[n.type as KGNodeType],
      });
    });
  }

  // Dataset: left half, Metric: right half
  const dmY = ROW_Y['dataset'];
  const leftW  = CANVAS_W * 0.45;
  const rightW = CANVAS_W * 0.45;
  datasetNodes.forEach((n, i) => {
    placed.push({
      ...n,
      x: 0.04 * CANVAS_W + (datasetNodes.length > 1 ? (leftW / (datasetNodes.length - 1)) * i : leftW / 2),
      y: dmY,
      r: NODE_RADII['dataset'],
    });
  });
  metricNodes.forEach((n, i) => {
    placed.push({
      ...n,
      x: CANVAS_W * 0.51 + (metricNodes.length > 1 ? (rightW / (metricNodes.length - 1)) * i : rightW / 2),
      y: dmY,
      r: NODE_RADII['metric'],
    });
  });

  return placed;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface PaperKnowledgeGraphProps {
  nodes: KGNode[];
  edges: KGEdge[];
}

export function PaperKnowledgeGraph({ nodes, edges }: PaperKnowledgeGraphProps) {
  const [zoom, setZoom]             = useState(1);
  const [pan, setPan]               = useState({ x: 0, y: 0 });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStart                   = useRef({ x: 0, y: 0 });

  const layout       = computeLayout(nodes);
  const nodeMap      = new Map(layout.map((n) => [n.id, n]));
  const selected     = selectedId ? nodeMap.get(selectedId) ?? null : null;
  const selectedMeta = selected ? getConceptMeta(selected.name) : null;

  const handleZoom = (dir: 'in' | 'out') =>
    setZoom((z) => (dir === 'in' ? Math.min(z + 0.2, 3) : Math.max(z - 0.2, 0.4)));

  const handleReset = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  const onMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    dragStart.current = { x: e.clientX, y: e.clientY };
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan((p) => ({ x: p.x + e.clientX - dragStart.current.x, y: p.y + e.clientY - dragStart.current.y }));
    dragStart.current = { x: e.clientX, y: e.clientY };
  };
  const onMouseUp = () => setIsDragging(false);

  if (nodes.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-[--color-text-secondary]">
        No knowledge graph data extracted from this paper.
      </div>
    );
  }

  const uniqueTypes = [...new Set(nodes.map((n) => n.type))] as KGNodeType[];

  return (
    <div className="flex h-full flex-col rounded-2xl border border-[--color-border] bg-[--bg-panel] overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-[--color-border] bg-[--bg-body] px-4 py-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-[--color-text-primary]">Knowledge Graph</span>
          <span className="rounded-full bg-[--bg-surface] px-2 py-0.5 text-xs text-[--color-text-secondary]">
            {nodes.length} nodes · {edges.length} edges
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => handleZoom('in')}
            aria-label="Zoom in"
            className="rounded p-1.5 text-[--color-text-tertiary] hover:bg-[--bg-surface] hover:text-[--color-text-primary] transition-colors"
          >
            <ZoomIn size={14} />
          </button>
          <button
            onClick={() => handleZoom('out')}
            aria-label="Zoom out"
            className="rounded p-1.5 text-[--color-text-tertiary] hover:bg-[--bg-surface] hover:text-[--color-text-primary] transition-colors"
          >
            <ZoomOut size={14} />
          </button>
          <button
            onClick={handleReset}
            aria-label="Reset view"
            className="rounded p-1.5 text-[--color-text-tertiary] hover:bg-[--bg-surface] hover:text-[--color-text-primary] transition-colors"
          >
            <RefreshCw size={14} />
          </button>
          <span className="ml-1 text-xs text-[--color-text-tertiary]">{Math.round(zoom * 100)}%</span>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 border-b border-[--color-border] bg-[--bg-body] px-4 py-1.5">
        {uniqueTypes.map((t) => (
          <div key={t} className="flex items-center gap-1.5">
            <div className="h-2.5 w-2.5 rounded-full" style={{ background: NODE_COLORS[t] }} />
            <span className="text-xs capitalize text-[--color-text-secondary]">{t}</span>
          </div>
        ))}
      </div>

      {/* Canvas */}
      <div className="relative flex-1 overflow-hidden bg-gradient-to-br from-[--bg-body] via-[--bg-panel] to-[--bg-body]">
        <svg
          role="img"
          aria-label="Knowledge graph showing relationships between paper, architectures, concepts, datasets, and metrics"
          width="100%"
          height="100%"
          viewBox={`0 0 ${CANVAS_W} ${CANVAS_H}`}
          preserveAspectRatio="xMidYMid meet"
          className="cursor-grab active:cursor-grabbing"
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
        >
          <defs>
            <pattern id="kg-grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--color-border)" strokeWidth="0.5" opacity="0.15" />
            </pattern>
            <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L0,6 L6,3 z" fill="var(--color-border)" opacity="0.6" />
            </marker>
          </defs>
          <rect width={CANVAS_W} height={CANVAS_H} fill="url(#kg-grid)" />

          <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
            {/* Edges */}
            {edges.map((edge, i) => {
              const src = nodeMap.get(edge.source);
              const tgt = nodeMap.get(edge.target);
              if (!src || !tgt) return null;

              const dx = tgt.x - src.x;
              const dy = tgt.y - src.y;
              const len = Math.hypot(dx, dy) || 1;
              const ex = tgt.x - (dx / len) * (tgt.r + 6);
              const ey = tgt.y - (dy / len) * (tgt.r + 6);

              const midX = (src.x + tgt.x) / 2;
              const midY = (src.y + tgt.y) / 2;

              return (
                <g key={i} className="pointer-events-none">
                  <line
                    x1={src.x} y1={src.y} x2={ex} y2={ey}
                    stroke="var(--color-border)" strokeWidth="1" opacity="0.5"
                    markerEnd="url(#arrow)"
                  />
                  <text
                    x={midX} y={midY - 4}
                    textAnchor="middle"
                    fontSize="7"
                    fontFamily="var(--font-body)"
                    fill="var(--color-text-tertiary)"
                    opacity="0.75"
                    className="select-none"
                  >
                    {edge.relation}
                  </text>
                </g>
              );
            })}

            {/* Nodes */}
            {layout.map((node) => {
              const isSelected = selectedId === node.id;
              const color = NODE_COLORS[node.type];
              const label = node.name.length > 20 ? node.name.slice(0, 18) + '…' : node.name;

              return (
                <g
                  key={node.id}
                  role="button"
                  tabIndex={0}
                  aria-label={`${node.name} — ${node.type}`}
                  aria-pressed={isSelected}
                  onClick={() => setSelectedId(isSelected ? null : node.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setSelectedId(isSelected ? null : node.id);
                    }
                  }}
                  className="cursor-pointer focus:outline-none"
                >
                  {isSelected && (
                    <circle cx={node.x} cy={node.y} r={node.r * 2.4} fill={color} opacity="0.12" className="pointer-events-none" aria-hidden="true" />
                  )}
                  <circle
                    cx={node.x} cy={node.y} r={node.r}
                    fill={color} opacity={isSelected ? 1 : 0.8}
                    stroke={isSelected ? 'white' : 'none'} strokeWidth="2"
                    className="hover:opacity-100 transition-opacity"
                    aria-hidden="true"
                  />
                  <text
                    x={node.x} y={node.y + node.r + 10}
                    textAnchor="middle"
                    fontSize="8"
                    fontFamily="var(--font-body)"
                    fill={color}
                    aria-hidden="true"
                    className="pointer-events-none select-none"
                  >
                    {label}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        {/* Info panel */}
        {selected && (
          <div
            role="region"
            aria-label="Selected node details"
            className="absolute bottom-4 left-4 max-w-[240px] rounded-2xl border border-[--color-border] bg-[--bg-panel] p-3 shadow-lg"
          >
            <div className="flex items-center gap-2 mb-1">
              <div className="h-3 w-3 rounded-full flex-shrink-0" style={{ background: NODE_COLORS[selected.type] }} aria-hidden="true" />
              <span className="text-xs uppercase tracking-wider text-[--color-text-tertiary]">{selected.type}</span>
              {selectedMeta && (
                <span
                  className="ml-auto text-[10px] font-semibold rounded px-1.5 py-0.5 flex-shrink-0"
                  style={{ color: DIFFICULTY_COLORS[selectedMeta.difficulty].text, background: DIFFICULTY_COLORS[selectedMeta.difficulty].bg }}
                >
                  {selectedMeta.difficulty}
                </span>
              )}
            </div>
            <p className="text-sm font-semibold text-[--color-text-primary] break-words">{selected.name}</p>
            {selectedMeta?.explanation && (
              <p className="text-[11px] mt-1 leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                {selectedMeta.explanation}
              </p>
            )}
            <div className="mt-2 flex flex-col gap-1.5">
              {selected.href && (
                <a
                  href={selected.href}
                  className="block w-full rounded-lg bg-[--accent-primary] px-2 py-1 text-center text-xs font-medium text-white hover:opacity-90 transition-opacity"
                >
                  Open →
                </a>
              )}
              {selectedMeta?.learnUrl && (
                <a
                  href={selectedMeta.learnUrl}
                  className="block w-full rounded-lg px-2 py-1 text-center text-xs font-medium hover:opacity-90 transition-opacity"
                  style={{ background: 'rgba(6,182,212,0.15)', color: 'var(--accent-cyan)' }}
                >
                  Learn This Concept →
                </a>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
