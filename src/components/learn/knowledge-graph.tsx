'use client';

import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { KG_NODES, KG_EDGES, DOMAINS } from './learn-data';

const VW = 640, VH = 400;

function bezierPath(ax: number, ay: number, bx: number, by: number) {
  const mx = (ax + bx) / 2, my = (ay + by) / 2;
  const dx = bx - ax, dy = by - ay;
  return `M${ax},${ay} Q${mx - dy * 0.22},${my + dx * 0.22} ${bx},${by}`;
}

export function KnowledgeGraph() {
  const [selected, setSelected] = useState<string | null>(null);

  const connectedIds = useCallback((id: string) => {
    const ids = new Set<string>();
    KG_EDGES.forEach(e => {
      if (e.from === id) ids.add(e.to);
      if (e.to === id) ids.add(e.from);
    });
    return ids;
  }, []);

  const isActive = (id: string) => !selected || id === selected || connectedIds(selected).has(id);
  const isEdgeActive = (from: string, to: string) =>
    !selected || (selected === from || selected === to) ||
    (connectedIds(selected).has(from) && connectedIds(selected).has(to));

  const selectedNode = selected ? KG_NODES.find(n => n.id === selected) : null;
  const selectedDomain = selected ? DOMAINS.find(d => d.id === selected) : null;

  return (
    <section className="mb-8">
      <div className="flex items-end justify-between mb-5">
        <div>
          <h2 className="text-lg font-black text-white">Knowledge Graph</h2>
          <p className="text-xs text-slate-500 mt-0.5">Click any node to explore connections</p>
        </div>
        {selected && (
          <button onClick={() => setSelected(null)}
            className="text-[11px] font-semibold text-slate-500 hover:text-white transition-colors">
            Clear selection ×
          </button>
        )}
      </div>

      <div className="rounded-2xl overflow-hidden relative"
        style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>

        {/* Info panel */}
        {selectedNode && selectedDomain && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-3 left-3 z-10 rounded-xl p-3 max-w-xs"
            style={{
              background: 'rgba(7,11,24,0.92)',
              border: `1px solid ${selectedNode.color}40`,
              boxShadow: `0 0 20px ${selectedNode.glow}`,
              backdropFilter: 'blur(12px)',
            }}
          >
            <div className="text-[13px] font-bold text-white mb-0.5">{selectedDomain.label}</div>
            <div className="text-[11px] text-slate-500 mb-2 leading-snug">{selectedDomain.description}</div>
            <div className="flex items-center gap-3 text-[10px]">
              <span style={{ color: selectedNode.color }}>{selectedNode.lessonCount} lessons</span>
              <span className="text-slate-700">·</span>
              <span className="text-slate-600">{selectedDomain.estimatedHours}h</span>
            </div>
            {selectedDomain.progress > 0 && (
              <div className="mt-2">
                <div className="h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.08)' }}>
                  <div className="h-full rounded-full" style={{ background: selectedNode.color, width: `${selectedDomain.progress}%` }} />
                </div>
                <div className="text-[9px] text-slate-600 mt-0.5">{selectedDomain.progress}% complete</div>
              </div>
            )}
            <Link href={selectedDomain.href}
              className="inline-flex items-center gap-1 mt-2 text-[11px] font-semibold"
              style={{ color: selectedNode.color }}>
              Explore domain →
            </Link>
          </motion.div>
        )}

        <svg viewBox={`0 0 ${VW} ${VH}`} className="w-full" style={{ height: 400, cursor: 'default' }}>
          <defs>
            <radialGradient id="kgBg" cx="50%" cy="50%" r="60%">
              <stop offset="0%" stopColor="rgba(139,92,246,0.06)" />
              <stop offset="100%" stopColor="rgba(7,11,24,0)" />
            </radialGradient>
          </defs>
          <rect width={VW} height={VH} fill="url(#kgBg)" />

          {/* Edges */}
          {KG_EDGES.map((e, i) => {
            const A = KG_NODES.find(n => n.id === e.from)!;
            const B = KG_NODES.find(n => n.id === e.to)!;
            const active = isEdgeActive(e.from, e.to);
            const d = bezierPath(A.x, A.y, B.x, B.y);
            return (
              <g key={i}>
                <path d={d} fill="none"
                  stroke={A.color}
                  strokeWidth={e.weight * 0.7}
                  opacity={active ? 0.35 : 0.06}
                  style={{ transition: 'opacity 0.3s' }}
                />
                {active && (
                  <circle r={e.weight + 1.5} fill={A.color} opacity="0.85">
                    <animateMotion dur={`${3.5 + i * 0.2}s`} repeatCount="indefinite">
                      <mpath xlinkHref={`#kg-path-${i}`} />
                    </animateMotion>
                  </circle>
                )}
                <path id={`kg-path-${i}`} d={d} fill="none" stroke="none" />
              </g>
            );
          })}

          {/* Nodes */}
          {KG_NODES.map((node, i) => {
            const active = isActive(node.id);
            const isSel = node.id === selected;
            return (
              <g key={node.id} style={{ cursor: 'pointer' }}
                onClick={() => setSelected(prev => prev === node.id ? null : node.id)}>
                {/* Glow halo */}
                <circle cx={node.x} cy={node.y} r={node.r + (isSel ? 14 : 9)}
                  fill="none" stroke={node.color}
                  strokeWidth={isSel ? 2 : 1}
                  opacity={active ? (isSel ? 0.5 : 0.2) : 0.04}
                  style={{ transition: 'all 0.3s' }}>
                  {isSel && (
                    <animate attributeName="r" values={`${node.r+12};${node.r+18};${node.r+12}`}
                      dur="2s" repeatCount="indefinite" />
                  )}
                </circle>

                {/* Node body */}
                <motion.circle
                  cx={node.x} cy={node.y} r={node.r}
                  fill={node.color}
                  opacity={active ? (isSel ? 1 : 0.75) : 0.15}
                  style={{
                    filter: active ? `drop-shadow(0 0 ${isSel ? 12 : 5}px ${node.color})` : 'none',
                    transition: 'all 0.3s',
                  }}
                  whileHover={{ scale: 1.15 }}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: i * 0.05, duration: 0.4, type: 'spring', stiffness: 200 }}
                />

                {/* Inner bright */}
                <circle cx={node.x} cy={node.y} r={node.r * 0.4}
                  fill="white" opacity={active ? 0.2 : 0.04} style={{ transition: 'opacity 0.3s' }} />

                {/* Label */}
                <text x={node.x} y={node.y + node.r + 14}
                  textAnchor="middle" fontSize={10} fontWeight="700"
                  fill="white" opacity={active ? 0.9 : 0.2}
                  fontFamily="system-ui, sans-serif"
                  style={{ transition: 'opacity 0.3s', pointerEvents: 'none' }}>
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </section>
  );
}
