'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { IntuitionData } from '@/data/topics/types';

interface IntuitionSectionProps {
  intuition: IntuitionData;
  color: string;
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

export function IntuitionSection({ intuition, color }: IntuitionSectionProps) {
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);
  const [showAnimation, setShowAnimation] = useState(false);
  const rgb = hexToRgb(color);

  // Find which tokens are highlighted for a given hover
  const getHighlightedTokens = (tokenId: number): Set<number> => {
    const set = new Set<number>();
    set.add(tokenId);
    for (const conn of intuition.connections) {
      if (conn.fromId === tokenId) set.add(conn.toId);
      if (conn.toId === tokenId) set.add(conn.fromId);
    }
    return set;
  };

  const getConnectionsForToken = (tokenId: number) =>
    intuition.connections.filter(c => c.fromId === tokenId || c.toId === tokenId);

  const highlighted = hoveredToken !== null ? getHighlightedTokens(hoveredToken) : new Set<number>();
  const activeConnections = hoveredToken !== null ? getConnectionsForToken(hoveredToken) : [];

  return (
    <section id="intuition" aria-labelledby="intuition-heading" className="mb-8 scroll-mt-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <h2 id="intuition-heading" className="text-lg font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
          Intuition
        </h2>
        <p className="text-xs mb-5" style={{ color: 'var(--color-text-secondary)' }}>
          Hover any word to see how attention connects it to related tokens.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.45, delay: 0.1 }}
        className="rounded-2xl p-5"
        style={{ background: 'var(--bg-surface)', border: `1px solid rgba(${rgb},0.18)` }}
      >
        {/* Token visualization */}
        <div
          className="flex flex-wrap gap-2 mb-5 p-4 rounded-xl"
          style={{ background: 'var(--bg-hover)' }}
          aria-label="Sentence with interactive attention tokens"
          role="group"
        >
          {intuition.tokens.map(token => {
            const isHighlighted = highlighted.has(token.id);
            const isHovered = hoveredToken === token.id;
            const isDimmed = hoveredToken !== null && !isHighlighted;
            const conns = getConnectionsForToken(token.id);
            const hasConnections = conns.length > 0;

            return (
              <button
                key={token.id}
                onMouseEnter={() => { if (hasConnections) setHoveredToken(token.id); }}
                onMouseLeave={() => setHoveredToken(null)}
                onFocus={() => { if (hasConnections) setHoveredToken(token.id); }}
                onBlur={() => setHoveredToken(null)}
                className="px-2.5 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2"
                style={{
                  background: isHovered ? `rgba(${rgb},0.25)` : isHighlighted ? `rgba(${rgb},0.15)` : 'transparent',
                  border: `1px solid ${isHighlighted ? `rgba(${rgb},0.45)` : 'rgba(255,255,255,0.06)'}`,
                  color: isHovered ? color : isHighlighted ? 'var(--color-text-primary)' : isDimmed ? 'var(--color-text-muted)' : 'var(--color-text-secondary)',
                  cursor: hasConnections ? 'pointer' : 'default',
                  transform: isHovered ? 'translateY(-1px)' : 'none',
                  boxShadow: isHovered ? `0 4px 12px rgba(${rgb},0.3)` : 'none',
                }}
                aria-label={`Token: ${token.text}${hasConnections ? ' — hover to see attention connections' : ''}`}
                aria-pressed={isHovered}
              >
                {token.text}
              </button>
            );
          })}
        </div>

        {/* Connection labels */}
        <AnimatePresence>
          {hoveredToken !== null && activeConnections.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -6, height: 0 }}
              animate={{ opacity: 1, y: 0, height: 'auto' }}
              exit={{ opacity: 0, y: -6, height: 0 }}
              transition={{ duration: 0.2 }}
              className="mb-4"
            >
              <div className="flex flex-wrap gap-2">
                {activeConnections.map((conn, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold"
                    style={{
                      background: `rgba(${rgb},0.12)`,
                      border: `1px solid rgba(${rgb},0.3)`,
                      color,
                    }}
                  >
                    <span className="font-black">{conn.label}</span>
                    <span
                      className="px-1.5 py-0.5 rounded text-[10px] tabular-nums"
                      style={{ background: `rgba(${rgb},0.2)`, color }}
                    >
                      {(conn.weight * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Explanation */}
        <div
          className="rounded-xl px-4 py-3 text-sm leading-relaxed"
          style={{
            background: `rgba(${rgb},0.06)`,
            border: `1px solid rgba(${rgb},0.12)`,
            color: 'var(--color-text-secondary)',
          }}
        >
          <span className="font-bold" style={{ color }}>How it works: </span>
          {intuition.explanation}
        </div>

        {/* Animate button */}
        <div className="mt-4 flex justify-center">
          <button
            onClick={() => setShowAnimation(v => !v)}
            className="px-4 py-2 rounded-xl text-xs font-semibold transition-all focus-visible:outline-none focus-visible:ring-2"
            style={{
              background: showAnimation ? `rgba(${rgb},0.2)` : 'var(--bg-hover)',
              border: `1px solid rgba(${rgb},0.25)`,
              color: showAnimation ? color : 'var(--color-text-secondary)',
            }}
          >
            {showAnimation ? '◼ Stop animation' : '▶ Animate attention flow'}
          </button>
        </div>

        {/* Animated attention flow */}
        <AnimatePresence>
          {showAnimation && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="mt-4"
              aria-live="polite"
              aria-label="Animated attention flow visualization"
            >
              <svg
                viewBox="0 0 600 80"
                className="w-full"
                aria-hidden="true"
              >
                <defs>
                  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
                  </marker>
                </defs>
                {intuition.connections.map((conn, i) => {
                  const fromToken = intuition.tokens[conn.fromId];
                  const toToken = intuition.tokens[conn.toId];
                  if (!fromToken || !toToken) return null;
                  const fromX = (conn.fromId / (intuition.tokens.length - 1)) * 560 + 20;
                  const toX = (conn.toId / (intuition.tokens.length - 1)) * 560 + 20;
                  const arcH = 40 + conn.weight * 20;
                  return (
                    <g key={i}>
                      <motion.path
                        d={`M${fromX},60 C${fromX},${60-arcH} ${toX},${60-arcH} ${toX},60`}
                        fill="none"
                        stroke={color}
                        strokeWidth={conn.weight * 3}
                        strokeDasharray="4 2"
                        markerEnd="url(#arrow)"
                        opacity={conn.weight}
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: 1.2, delay: i * 0.3, ease: 'easeInOut' }}
                      />
                      <text x={(fromX + toX) / 2} y={60 - arcH - 5} textAnchor="middle" fontSize="9" fill={color} opacity="0.8">
                        {conn.label}
                      </text>
                    </g>
                  );
                })}
                {intuition.tokens.map((token, i) => {
                  const x = (i / (intuition.tokens.length - 1)) * 560 + 20;
                  return (
                    <g key={token.id}>
                      <circle cx={x} cy={60} r={3} fill={color} opacity="0.6" />
                      <text x={x} y={74} textAnchor="middle" fontSize="8" fill="white" opacity="0.6">
                        {token.text.slice(0, 4)}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </section>
  );
}
