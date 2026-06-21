'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ArchNode, ArchEdge } from '@/data/topics/types';

interface ArchitectureSectionProps {
  nodes: ArchNode[];
  edges: ArchEdge[];
  color: string;
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

export function ArchitectureSection({ nodes, edges, color }: ArchitectureSectionProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const rgb = hexToRgb(color);

  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
  const hoveredNode = hoveredId ? nodeMap[hoveredId] : null;

  // Vertical layout: position nodes evenly
  const svgH = Math.max(nodes.length * 80 + 20, 200);
  const nodeX = 300;
  const layoutNodes = nodes.map((n, i) => ({
    ...n,
    cx: nodeX,
    cy: 30 + i * 80,
  }));
  const layoutMap = Object.fromEntries(layoutNodes.map(n => [n.id, n]));

  return (
    <section id="architecture" aria-labelledby="arch-heading" className="mb-8 scroll-mt-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <h2 id="arch-heading" className="text-lg font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
          Visual Architecture
        </h2>
        <p className="text-xs mb-5" style={{ color: 'var(--color-text-secondary)' }}>
          Hover any block to see its shape, parameters, and purpose.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.45, delay: 0.1 }}
        className="rounded-2xl overflow-hidden"
        style={{ background: 'var(--bg-surface)', border: `1px solid rgba(${rgb},0.16)` }}
      >
        <div className="flex flex-col lg:flex-row">
          {/* SVG diagram */}
          <div className="flex-1 p-4" style={{ minWidth: 0 }}>
            <svg
              viewBox={`0 0 600 ${svgH}`}
              className="w-full"
              role="img"
              aria-label="Architecture flow diagram"
              style={{ maxHeight: '520px' }}
            >
              <defs>
                <marker id="arch-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(255,255,255,0.25)" />
                </marker>
                <filter id="arch-glow">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
              </defs>

              {/* Edges */}
              {edges.map((edge, i) => {
                const from = layoutMap[edge.from];
                const to = layoutMap[edge.to];
                if (!from || !to) return null;
                const isHighlighted = hoveredId === edge.from || hoveredId === edge.to;
                return (
                  <g key={i}>
                    <line
                      x1={from.cx}
                      y1={from.cy + 18}
                      x2={to.cx}
                      y2={to.cy - 18}
                      stroke={isHighlighted ? color : 'rgba(255,255,255,0.12)'}
                      strokeWidth={isHighlighted ? 2 : 1}
                      strokeDasharray={isHighlighted ? '0' : '4 3'}
                      markerEnd="url(#arch-arrow)"
                      style={{ transition: 'all 0.2s' }}
                    />
                    {edge.label && isHighlighted && (
                      <text
                        x={(from.cx + to.cx) / 2 + 18}
                        y={(from.cy + to.cy) / 2}
                        fontSize="9"
                        fill={color}
                        opacity="0.9"
                        fontFamily="var(--font-mono)"
                      >
                        {edge.label}
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Nodes */}
              {layoutNodes.map((node) => {
                const isHovered = hoveredId === node.id;
                const isDimmed = hoveredId !== null && !isHovered;
                const nrgb = hexToRgb(node.color);

                return (
                  <g
                    key={node.id}
                    role="button"
                    tabIndex={0}
                    aria-label={`${node.label}: ${node.purpose}`}
                    style={{ cursor: 'pointer', outline: 'none' }}
                    onMouseEnter={() => setHoveredId(node.id)}
                    onMouseLeave={() => setHoveredId(null)}
                    onFocus={() => setHoveredId(node.id)}
                    onBlur={() => setHoveredId(null)}
                    onKeyDown={e => {
                      if (e.key === 'Enter' || e.key === ' ') setHoveredId(node.id);
                    }}
                  >
                    {/* Glow */}
                    {isHovered && (
                      <rect
                        x={node.cx - 110}
                        y={node.cy - 24}
                        width={220}
                        height={48}
                        rx={10}
                        fill={`rgba(${nrgb},0.08)`}
                        filter="url(#arch-glow)"
                      />
                    )}
                    {/* Node rect */}
                    <rect
                      x={node.cx - 100}
                      y={node.cy - 18}
                      width={200}
                      height={36}
                      rx={8}
                      fill={`rgba(${nrgb},${isHovered ? 0.22 : isDimmed ? 0.05 : 0.12})`}
                      stroke={isHovered ? node.color : `rgba(${nrgb},0.3)`}
                      strokeWidth={isHovered ? 1.5 : 1}
                      style={{ transition: 'all 0.2s' }}
                    />
                    {/* Label */}
                    <text
                      x={node.cx}
                      y={node.cy + 1}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fontSize={isHovered ? 12 : 11}
                      fontWeight="700"
                      fill={isHovered ? 'white' : isDimmed ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.8)'}
                      fontFamily="var(--font-heading)"
                      style={{ transition: 'all 0.2s' }}
                    >
                      {node.label}
                    </text>
                    {/* Params tag */}
                    {isHovered && (
                      <text
                        x={node.cx + 108}
                        y={node.cy - 4}
                        fontSize="8"
                        fill={node.color}
                        opacity="0.9"
                        fontFamily="var(--font-mono)"
                      >
                        {node.params}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Info panel */}
          <div
            className="lg:w-64 flex-none p-4"
            style={{ borderLeft: '1px solid rgba(255,255,255,0.06)', minHeight: '200px' }}
          >
            <AnimatePresence mode="wait">
              {hoveredNode ? (
                <motion.div
                  key={hoveredNode.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.18 }}
                >
                  <div className="text-[11px] font-bold uppercase tracking-wider mb-3" style={{ color: 'var(--color-text-muted)' }}>
                    Selected Block
                  </div>
                  <div className="text-[14px] font-black mb-2" style={{ color: hoveredNode.color }}>
                    {hoveredNode.label}
                  </div>
                  <div className="flex flex-col gap-3">
                    <div>
                      <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Shape</span>
                      <code
                        className="block mt-1 text-xs px-2 py-1 rounded"
                        style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--accent-light)', fontFamily: 'var(--font-mono)' }}
                      >
                        {hoveredNode.params}
                      </code>
                    </div>
                    <div>
                      <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Purpose</span>
                      <p className="text-xs mt-1 leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                        {hoveredNode.purpose}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full flex items-center justify-center text-center"
                >
                  <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    Hover a block to see details
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </motion.div>
    </section>
  );
}
