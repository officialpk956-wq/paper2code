'use client';

import { useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Network, ArrowRight } from 'lucide-react';
import type { DomainKGNode, DomainKGEdge } from '@/data/domains/types';

interface DomainKnowledgeGraphProps {
  nodes: DomainKGNode[];
  edges: DomainKGEdge[];
  domainSlug: string;
  color: string;
}

function getNeighbors(nodeId: string, edges: DomainKGEdge[]): Set<string> {
  const set = new Set<string>();
  for (const e of edges) {
    if (e.from === nodeId) set.add(e.to);
    if (e.to === nodeId) set.add(e.from);
  }
  return set;
}

function curve(ax: number, ay: number, bx: number, by: number): string {
  const mx = (ax + bx) / 2, my = (ay + by) / 2;
  const dx = bx - ax, dy = by - ay;
  return `M${ax},${ay} Q${mx - dy * 0.22},${my + dx * 0.22} ${bx},${by}`;
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

export function DomainKnowledgeGraph({ nodes, edges, color }: DomainKnowledgeGraphProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
  const neighbors = hoveredId ? getNeighbors(hoveredId, edges) : new Set<string>();
  const rgb = hexToRgb(color);

  return (
    <motion.section
      aria-labelledby="domain-kg-heading"
      className="mb-10"
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
    >
      <div
        className="rounded-2xl overflow-hidden"
        style={{
          background: 'var(--bg-surface)',
          border: `1px solid rgba(${rgb},0.18)`,
        }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-5 py-4"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}
        >
          <div className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-xl flex items-center justify-center"
              style={{ background: `rgba(${rgb},0.14)` }}
              aria-hidden="true"
            >
              <Network size={15} style={{ color }} />
            </div>
            <div>
              <h2 id="domain-kg-heading" className="text-[13px] font-bold" style={{ color: 'var(--color-text-primary)' }}>
                Knowledge Graph
              </h2>
              <p className="text-[11px]" style={{ color: 'var(--color-text-tertiary)' }}>
                Concept relationships in this domain
              </p>
            </div>
          </div>
          <Link
            href="/explorer"
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold text-white transition-opacity hover:opacity-80"
            style={{ background: 'var(--accent-primary)' }}
            aria-label="Open the full knowledge graph explorer"
          >
            Full Graph <ArrowRight size={11} aria-hidden="true" />
          </Link>
        </div>

        {/* SVG */}
        <div className="px-4 py-5" style={{ height: 230 }}>
          <svg
            viewBox="0 0 600 210"
            className="w-full h-full"
            role="img"
            aria-label={`Knowledge graph showing concept relationships within this domain`}
          >
            <defs>
              <radialGradient id="dkg-ambient" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor={`rgba(${rgb},0.08)`} />
                <stop offset="100%" stopColor="transparent" />
              </radialGradient>
              {nodes.map(n => (
                <radialGradient key={`dkg-g-${n.id}`} id={`dkg-g-${n.id}`} cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor={n.color} stopOpacity="0.9" />
                  <stop offset="100%" stopColor={n.color} stopOpacity="0.38" />
                </radialGradient>
              ))}
            </defs>

            <rect x="0" y="0" width="600" height="210" fill="url(#dkg-ambient)" />

            {/* Edges */}
            {edges.map((e, i) => {
              const A = nodeMap[e.from];
              const B = nodeMap[e.to];
              if (!A || !B) return null;
              const isHighlighted = hoveredId && (e.from === hoveredId || e.to === hoveredId);
              const d = curve(A.x, A.y, B.x, B.y);
              return (
                <g key={`dkg-edge-${i}`}>
                  <path
                    d={d} fill="none"
                    stroke={isHighlighted ? A.color : 'rgba(255,255,255,0.12)'}
                    strokeWidth={isHighlighted ? 2 : 1}
                    opacity={isHighlighted ? 0.85 : hoveredId ? 0.12 : 0.3}
                    style={{ transition: 'all 0.2s ease' }}
                  />
                  {/* Edge label */}
                  {isHighlighted && e.label && (() => {
                    const mx = (A.x + B.x) / 2;
                    const my = (A.y + B.y) / 2;
                    return (
                      <text x={mx} y={my - 6} textAnchor="middle" fontSize="8" fill={A.color} opacity="0.8" fontFamily="system-ui">
                        {e.label}
                      </text>
                    );
                  })()}
                </g>
              );
            })}

            {/* Nodes */}
            {nodes.map(node => {
              const isHovered = hoveredId === node.id;
              const isNeighbor = neighbors.has(node.id);
              const isDimmed = !!hoveredId && !isHovered && !isNeighbor;

              return (
                <g
                  key={node.id}
                  role="button"
                  aria-label={`${node.label}: hover to explore connections`}
                  tabIndex={0}
                  style={{ cursor: 'pointer', outline: 'none' }}
                  onMouseEnter={() => setHoveredId(node.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  onFocus={() => setHoveredId(node.id)}
                  onBlur={() => setHoveredId(null)}
                  onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') setHoveredId(node.id); }}
                >
                  {/* Glow */}
                  <circle
                    cx={node.x} cy={node.y}
                    r={isHovered ? node.r + 10 : node.r + 3}
                    fill="none" stroke={node.color} strokeWidth="1"
                    opacity={isHovered ? 0.4 : 0.1}
                    style={{ transition: 'all 0.2s ease' }}
                  />
                  {/* Pulse ring on hover */}
                  {isHovered && (
                    <circle cx={node.x} cy={node.y} r={node.r + 16}
                      fill="none" stroke={node.color} strokeWidth="0.5" opacity="0.15">
                      <animate attributeName="r" values={`${node.r+12};${node.r+20};${node.r+12}`} dur="1.5s" repeatCount="indefinite" />
                      <animate attributeName="opacity" values="0.15;0.04;0.15" dur="1.5s" repeatCount="indefinite" />
                    </circle>
                  )}
                  {/* Node body */}
                  <circle
                    cx={node.x} cy={node.y}
                    r={isHovered ? node.r + 2.5 : node.r}
                    fill={`url(#dkg-g-${node.id})`}
                    opacity={isDimmed ? 0.22 : 1}
                    style={{
                      transition: 'all 0.2s ease',
                      filter: isHovered ? `drop-shadow(0 0 ${node.r * 0.8}px ${node.color})` : 'none',
                    }}
                  />
                  {/* Label */}
                  <text
                    x={node.x} y={node.y + node.r + 13}
                    textAnchor="middle"
                    fontSize={isHovered ? 10 : 9}
                    fontWeight="700"
                    fill="white"
                    opacity={isDimmed ? 0.18 : isHovered ? 1 : 0.75}
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

        <div className="px-5 pb-4">
          <p className="text-[11px] text-center" style={{ color: 'var(--color-text-muted)' }}>
            Hover any node to highlight its connections ·{' '}
            <Link href="/explorer" className="transition-opacity hover:opacity-80" style={{ color: 'var(--accent-light)' }}>
              open the full graph
            </Link>{' '}
            to see all domains
          </p>
        </div>
      </div>
    </motion.section>
  );
}
