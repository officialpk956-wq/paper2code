'use client';

import { useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Network, ArrowRight } from 'lucide-react';

interface KGNode {
  id: string;
  label: string;
  x: number;
  y: number;
  r: number;
  color: string;
}

interface KGEdge {
  from: string;
  to: string;
}

const KG_NODES: KGNode[] = [
  { id: 'math',         label: 'Math',         x: 60,  y: 110, r: 18, color: '#8B5CF6' },
  { id: 'ml',          label: 'ML',           x: 140, y: 60,  r: 20, color: '#06B6D4' },
  { id: 'dl',          label: 'Deep Learning',x: 240, y: 100, r: 22, color: '#0EA5E9' },
  { id: 'transformers',label: 'Transformers', x: 340, y: 55,  r: 24, color: '#7C3AED' },
  { id: 'llms',        label: 'LLMs',         x: 440, y: 100, r: 22, color: '#A78BFA' },
  { id: 'rag',         label: 'RAG',          x: 530, y: 55,  r: 18, color: '#EC4899' },
  { id: 'agents',      label: 'Agents',       x: 530, y: 145, r: 18, color: '#F59E0B' },
];

const KG_EDGES: KGEdge[] = [
  { from: 'math',         to: 'ml' },
  { from: 'ml',          to: 'dl' },
  { from: 'dl',          to: 'transformers' },
  { from: 'transformers',to: 'llms' },
  { from: 'llms',        to: 'rag' },
  { from: 'llms',        to: 'agents' },
  { from: 'rag',         to: 'agents' },
];

function getNeighbors(nodeId: string): Set<string> {
  const neighbors = new Set<string>();
  for (const e of KG_EDGES) {
    if (e.from === nodeId) neighbors.add(e.to);
    if (e.to === nodeId) neighbors.add(e.from);
  }
  return neighbors;
}

function curve(ax: number, ay: number, bx: number, by: number): string {
  const mx = (ax + bx) / 2, my = (ay + by) / 2;
  const dx = bx - ax, dy = by - ay;
  return `M${ax},${ay} Q${mx - dy * 0.25},${my + dx * 0.25} ${bx},${by}`;
}

export function KnowledgeGraphPreview() {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const nodeMap = Object.fromEntries(KG_NODES.map(n => [n.id, n]));
  const neighbors = hoveredId ? getNeighbors(hoveredId) : new Set<string>();

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
      className="mb-10"
      aria-labelledby="kg-preview-heading"
    >
      <div
        className="rounded-2xl overflow-hidden"
        style={{
          background: 'var(--bg-surface)',
          border: '1px solid rgba(124,58,237,0.18)',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <div className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-xl flex items-center justify-center"
              style={{ background: 'rgba(124,58,237,0.15)' }}
              aria-hidden="true"
            >
              <Network size={15} style={{ color: 'var(--accent-primary)' }} />
            </div>
            <div>
              <h2 id="kg-preview-heading" className="text-[13px] font-bold"
                style={{ color: 'var(--color-text-primary)' }}>
                Knowledge Graph
              </h2>
              <p className="text-[11px]" style={{ color: 'var(--color-text-tertiary)' }}>
                Hover a node to explore relationships
              </p>
            </div>
          </div>
          <Link
            href="/explorer"
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold text-white transition-opacity hover:opacity-80"
            style={{ background: 'var(--accent-primary)' }}
            aria-label="Open the full interactive Knowledge Graph explorer"
          >
            Open Full Graph <ArrowRight size={11} aria-hidden="true" />
          </Link>
        </div>

        {/* SVG graph */}
        <div className="px-4 py-5" style={{ height: 220 }}>
          <svg
            viewBox="0 0 600 200"
            className="w-full h-full"
            role="img"
            aria-label="Knowledge graph showing connections between AI topics"
          >
            <defs>
              <radialGradient id="kgp-ambient" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="rgba(124,58,237,0.08)" />
                <stop offset="100%" stopColor="transparent" />
              </radialGradient>
              {KG_NODES.map(n => (
                <radialGradient key={`kgp-g-${n.id}`} id={`kgp-g-${n.id}`} cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor={n.color} stopOpacity="0.9" />
                  <stop offset="100%" stopColor={n.color} stopOpacity="0.4" />
                </radialGradient>
              ))}
            </defs>

            <rect x="0" y="0" width="600" height="200" fill="url(#kgp-ambient)" />

            {/* Edges */}
            {KG_EDGES.map((e, i) => {
              const A = nodeMap[e.from];
              const B = nodeMap[e.to];
              if (!A || !B) return null;
              const isHighlighted = hoveredId && (e.from === hoveredId || e.to === hoveredId);
              const d = curve(A.x, A.y, B.x, B.y);
              return (
                <path
                  key={`kgp-edge-${i}`}
                  d={d}
                  fill="none"
                  stroke={isHighlighted ? A.color : 'rgba(255,255,255,0.12)'}
                  strokeWidth={isHighlighted ? 2 : 1}
                  opacity={isHighlighted ? 0.9 : (hoveredId ? 0.15 : 0.35)}
                  style={{ transition: 'all 0.2s ease' }}
                />
              );
            })}

            {/* Nodes */}
            {KG_NODES.map(node => {
              const isHovered = hoveredId === node.id;
              const isNeighbor = neighbors.has(node.id);
              const isDimmed = hoveredId && !isHovered && !isNeighbor;

              return (
                <g
                  key={node.id}
                  role="button"
                  aria-label={`${node.label} - click to explore`}
                  tabIndex={0}
                  style={{ cursor: 'pointer', outline: 'none' }}
                  onMouseEnter={() => setHoveredId(node.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  onFocus={() => setHoveredId(node.id)}
                  onBlur={() => setHoveredId(null)}
                  onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') setHoveredId(node.id); }}
                >
                  {/* Outer glow (visible on hover) */}
                  <circle
                    cx={node.x} cy={node.y}
                    r={isHovered ? node.r + 10 : node.r + 4}
                    fill="none"
                    stroke={node.color}
                    strokeWidth="1"
                    opacity={isHovered ? 0.35 : 0.1}
                    style={{ transition: 'all 0.2s ease' }}
                  />

                  {/* Node body */}
                  <circle
                    cx={node.x} cy={node.y}
                    r={isHovered ? node.r + 3 : node.r}
                    fill={`url(#kgp-g-${node.id})`}
                    opacity={isDimmed ? 0.25 : 1}
                    style={{
                      transition: 'all 0.2s ease',
                      filter: isHovered ? `drop-shadow(0 0 ${node.r}px ${node.color})` : 'none',
                    }}
                  />

                  {/* Label */}
                  <text
                    x={node.x} y={node.y + node.r + 13}
                    textAnchor="middle"
                    fontSize={isHovered ? 10 : 9}
                    fontWeight="700"
                    fill="white"
                    opacity={isDimmed ? 0.2 : (isHovered ? 1 : 0.75)}
                    fontFamily="system-ui, sans-serif"
                    style={{ transition: 'all 0.2s ease' }}
                  >
                    {node.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Footer hint */}
        <div className="px-5 pb-4">
          <p className="text-[11px] text-center" style={{ color: 'var(--color-text-muted)' }}>
            This is a simplified view —{' '}
            <Link href="/explorer" className="transition-colors hover:opacity-80"
              style={{ color: 'var(--accent-light)' }}>
              open the full graph
            </Link>{' '}
            to explore all {KG_NODES.length * 10}+ connections
          </p>
        </div>
      </div>
    </motion.section>
  );
}
