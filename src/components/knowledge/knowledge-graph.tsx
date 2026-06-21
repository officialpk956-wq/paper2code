'use client';

import { useState, useRef, useEffect, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { ZoomIn, ZoomOut, Search, Download, Lock, CheckCircle } from 'lucide-react';
import { EVOLUTION_NODES } from '@/data/evolution';
import { isCompleted } from '@/lib/progress';
import { isUnlocked } from '@/lib/learning-graph';

// ---------------------------------------------------------------------------
// Build graph from real EVOLUTION_NODES data
// ---------------------------------------------------------------------------

interface GraphNode {
  id: string;
  label: string;
  type: 'paper' | 'architecture';
  x: number;
  y: number;
  size: number;
  connections: string[];
  href: string;
}

interface GraphEdge {
  from: string;
  to: string;
  strength: number;
}

// Lay nodes out in a loose timeline-aware grid
function buildNodes(): GraphNode[] {
  const ERA_X: Record<string, number> = {
    // Deep Learning Era → left
    alexnet: 60, vgg: 120, resnet: 180, 'seq2seq-paper': 80,
    // Attention Era → center-left
    'attention-is-all-you-need': 260, bert: 320, gpt: 300,
    // Scaling Era → center
    'gpt-2': 360, 'gpt-3': 420, chinchilla: 480, llama: 500, instructgpt: 440,
    // Vision / Multimodal → right column
    'vision-transformer': 380, clip: 440, 'segment-anything': 520,
    // Generative → right
    gan: 560, 'ddpm-paper': 600, 'latent-diffusion-models': 640, 'stable-diffusion': 680,
    // Reasoning → far right
    'chain-of-thought': 700, 'rlhf-paper': 720,
  };
  const ERA_Y: Record<string, number> = {
    alexnet: 80, vgg: 140, resnet: 200, 'seq2seq-paper': 320,
    'attention-is-all-you-need': 220, bert: 160, gpt: 280,
    'gpt-2': 300, 'gpt-3': 300, chinchilla: 360, llama: 400, instructgpt: 240,
    'vision-transformer': 180, clip: 220, 'segment-anything': 260,
    gan: 380, 'ddpm-paper': 420, 'latent-diffusion-models': 460, 'stable-diffusion': 500,
    'chain-of-thought': 200, 'rlhf-paper': 240,
  };

  return EVOLUTION_NODES.map((evo, i) => {
    const col = i % 8;
    const row = Math.floor(i / 8);
    return {
      id: evo.slug,
      label: evo.title,
      type: evo.type,
      x: ERA_X[evo.slug] ?? 40 + col * 90,
      y: ERA_Y[evo.slug] ?? 60 + row * 90,
      size: evo.children.length >= 3 ? 9 : evo.children.length >= 1 ? 7 : 5,
      connections: evo.children,
      href: evo.type === 'architecture'
        ? `/architectures/${evo.slug}`
        : `/papers/${evo.slug}`,
    };
  });
}

const NODES = buildNodes();
const EDGES: GraphEdge[] = NODES.flatMap((node) =>
  node.connections.map((connId) => ({
    from: node.id,
    to: connId,
    strength: 0.55,
  })),
);

const TYPE_COLORS: Record<GraphNode['type'], string> = {
  paper: '#7C3AED',
  architecture: '#06B6D4',
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface KnowledgeGraphProps {
  onNodeClick?: (nodeId: string) => void;
}

export function KnowledgeGraph({ onNodeClick }: KnowledgeGraphProps) {
  const router = useRouter();
  const canvasRef = useRef<SVGSVGElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filter, setFilter] = useState<GraphNode['type'] | 'all'>('all');
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });

  // Live completion state — read from localStorage after mount
  const [completedSlugs, setCompletedSlugs] = useState<Set<string>>(new Set());
  const [unlockedSlugs, setUnlockedSlugs] = useState<Set<string>>(new Set());

  useEffect(() => {
    const completed = new Set(
      NODES.filter((n) => isCompleted('paper', n.id)).map((n) => n.id),
    );
    const unlocked = new Set(
      NODES.filter((n) => isUnlocked('paper', n.id)).map((n) => n.id),
    );
    setCompletedSlugs(completed);
    setUnlockedSlugs(unlocked);
  }, []);

  const filteredNodes = useMemo(
    () =>
      NODES.filter(
        (node) =>
          (filter === 'all' || node.type === filter) &&
          node.label.toLowerCase().includes(searchQuery.toLowerCase()),
      ),
    [filter, searchQuery],
  );

  const filteredNodeMap = useMemo(
    () => new Map(filteredNodes.map((n) => [n.id, n])),
    [filteredNodes],
  );

  const visibleEdges = useMemo(
    () => EDGES.filter((e) => filteredNodeMap.has(e.from) && filteredNodeMap.has(e.to)),
    [filteredNodeMap],
  );

  const handleZoom = (direction: 'in' | 'out') => {
    setZoom((z) =>
      direction === 'in' ? Math.min(z + 0.2, 3) : Math.max(z - 0.2, 0.5),
    );
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;
    setPan((p) => ({ x: p.x + dx, y: p.y + dy }));
    dragStartRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleNodeActivate = (node: GraphNode) => {
    setSelectedNode(node.id);
    onNodeClick?.(node.id);
  };

  const handleNodeNavigate = (node: GraphNode) => {
    router.push(node.href);
  };

  const selectedNodeData = NODES.find((n) => n.id === selectedNode);

  // Analytics
  const completedCount = completedSlugs.size;
  const unlockedCount = unlockedSlugs.size;

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border border-[--color-border] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[--color-border] bg-[--bg-body]">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-[--color-text-primary]">
            Knowledge Graph
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleZoom('in')}
              aria-label="Zoom in"
              className="p-1.5 rounded hover:bg-[--bg-surface] text-[--color-text-tertiary] hover:text-[--color-text-primary] transition-colors"
            >
              <ZoomIn size={14} />
            </button>
            <button
              onClick={() => handleZoom('out')}
              aria-label="Zoom out"
              className="p-1.5 rounded hover:bg-[--bg-surface] text-[--color-text-tertiary] hover:text-[--color-text-primary] transition-colors"
            >
              <ZoomOut size={14} />
            </button>
            <span className="text-xs text-[--color-text-tertiary] px-2">
              {Math.round(zoom * 100)}%
            </span>
          </div>
        </div>

        {/* Search & Filter */}
        <div className="space-y-2">
          <div className="relative">
            <Search
              size={14}
              className="absolute left-2 top-2 text-[--color-text-tertiary]"
            />
            <input
              type="text"
              placeholder="Search nodes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-6 pr-3 py-1.5 bg-[--bg-panel] border border-[--color-border] rounded text-xs text-[--color-text-primary] placeholder-[--color-text-tertiary] focus:border-[--accent-primary] focus:outline-none"
            />
          </div>
          <div className="flex gap-1 flex-wrap">
            {(['all', 'paper', 'architecture'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setFilter(t)}
                className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                  filter === t
                    ? 'bg-[--accent-primary] text-white'
                    : 'bg-[--bg-surface] text-[--color-text-secondary] hover:text-[--color-text-primary]'
                }`}
              >
                {t === 'all' ? 'All' : t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Graph Canvas */}
      <div className="flex-1 overflow-hidden bg-gradient-to-br from-[--bg-body] via-[--bg-panel] to-[--bg-body] relative">
        <svg
          ref={canvasRef}
          width="100%"
          height="100%"
          className="cursor-grab active:cursor-grabbing"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <defs>
            <pattern
              id="grid"
              width="40"
              height="40"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M 40 0 L 0 0 0 40"
                fill="none"
                stroke="var(--color-border)"
                strokeWidth="0.5"
                opacity="0.2"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />

          <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
            {/* Edges */}
            {visibleEdges.map((edge) => {
              const from = filteredNodeMap.get(edge.from)!;
              const to = filteredNodeMap.get(edge.to)!;
              const isCompleteEdge =
                completedSlugs.has(from.id) && completedSlugs.has(to.id);
              return (
                <line
                  key={`${edge.from}-${edge.to}`}
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  stroke={
                    isCompleteEdge
                      ? 'var(--color-easy)'
                      : TYPE_COLORS[from.type]
                  }
                  strokeWidth="1"
                  opacity={isCompleteEdge ? 0.7 : edge.strength}
                  className="pointer-events-none"
                />
              );
            })}

            {/* Nodes */}
            {filteredNodes.map((node) => {
              const isSelected = selectedNode === node.id;
              const completed = completedSlugs.has(node.id);
              const unlocked = unlockedSlugs.has(node.id);
              const color = TYPE_COLORS[node.type];

              return (
                <g
                  key={node.id}
                  role="button"
                  tabIndex={0}
                  aria-label={`${node.label} — ${node.type}, ${node.connections.length} connections${completed ? ', completed' : ''}`}
                  aria-pressed={isSelected}
                  onClick={() => handleNodeActivate(node)}
                  onDoubleClick={() => handleNodeNavigate(node)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleNodeNavigate(node);
                    } else if (e.key === ' ') {
                      e.preventDefault();
                      handleNodeActivate(node);
                    }
                  }}
                  className="cursor-pointer focus:outline-none"
                >
                  {/* Glow for selected */}
                  {isSelected && (
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={node.size * 2.5}
                      fill={color}
                      opacity="0.12"
                      aria-hidden="true"
                      className="pointer-events-none"
                    />
                  )}

                  {/* Completion ring */}
                  {completed && (
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={node.size + 3}
                      fill="none"
                      stroke="var(--color-easy)"
                      strokeWidth="1.5"
                      aria-hidden="true"
                      className="pointer-events-none"
                    />
                  )}

                  {/* Lock ring for locked content */}
                  {!unlocked && !completed && (
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={node.size + 2}
                      fill="none"
                      stroke="var(--color-text-tertiary)"
                      strokeWidth="1"
                      strokeDasharray="3 2"
                      aria-hidden="true"
                      className="pointer-events-none"
                      opacity="0.5"
                    />
                  )}

                  {/* Node circle */}
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={node.size}
                    fill={color}
                    opacity={completed ? 1 : unlocked ? 0.8 : 0.35}
                    className="hover:opacity-100 transition-opacity"
                    stroke={isSelected ? 'white' : 'none'}
                    strokeWidth="2"
                    aria-hidden="true"
                  />

                  {/* Label */}
                  <text
                    x={node.x}
                    y={node.y + node.size + 11}
                    textAnchor="middle"
                    fontSize="9"
                    fontFamily="var(--font-sans)"
                    fill={completed ? 'var(--color-easy)' : color}
                    opacity={unlocked || completed ? 1 : 0.5}
                    aria-hidden="true"
                    className="pointer-events-none select-none"
                  >
                    {node.label}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        {/* Info Panel */}
        {selectedNodeData && (
          <div className="absolute bottom-4 left-4 max-w-xs bg-[--bg-panel] border border-[--accent-primary] rounded-lg p-3 shadow-lg">
            <div className="flex items-center gap-2 mb-2">
              <div
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{
                  backgroundColor: TYPE_COLORS[selectedNodeData.type],
                }}
              />
              {completedSlugs.has(selectedNodeData.id) && (
                <CheckCircle size={12} className="text-[--color-easy]" />
              )}
              {!unlockedSlugs.has(selectedNodeData.id) &&
                !completedSlugs.has(selectedNodeData.id) && (
                  <Lock size={12} className="text-[--color-text-tertiary]" />
                )}
            </div>
            <h4 className="text-sm font-semibold text-[--color-text-primary]">
              {selectedNodeData.label}
            </h4>
            <p className="text-xs text-[--color-text-tertiary] mt-1">
              {selectedNodeData.type} •{' '}
              {selectedNodeData.connections.length} successor
              {selectedNodeData.connections.length !== 1 ? 's' : ''}
              {completedSlugs.has(selectedNodeData.id) ? ' • Completed ✓' : ''}
            </p>
            <a
              href={selectedNodeData.href}
              className="block w-full mt-2 px-2 py-1.5 rounded text-xs font-medium bg-[--accent-primary] hover:opacity-90 text-white transition-opacity text-center"
            >
              Open →
            </a>
          </div>
        )}
      </div>

      {/* Footer Stats */}
      <div className="px-4 py-3 border-t border-[--color-border] bg-[--bg-body] text-xs text-[--color-text-secondary]">
        <div className="grid grid-cols-4 gap-2">
          <div>
            <div className="font-semibold text-[--color-text-primary]">
              {filteredNodes.length}
            </div>
            <div className="text-[--color-text-tertiary]">Nodes</div>
          </div>
          <div>
            <div className="font-semibold text-[--color-easy]">
              {completedCount}
            </div>
            <div className="text-[--color-text-tertiary]">Completed</div>
          </div>
          <div>
            <div className="font-semibold text-[--accent-primary]">
              {unlockedCount}
            </div>
            <div className="text-[--color-text-tertiary]">Unlocked</div>
          </div>
          <div>
            <button
              aria-label="Export graph"
              className="flex items-center gap-1 px-2 py-1 rounded hover:bg-[--bg-surface] transition-colors"
            >
              <Download size={12} />
              Export
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
