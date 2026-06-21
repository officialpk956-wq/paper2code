'use client';

import { motion } from 'framer-motion';

/* Deterministic node positions — no Math.random() */
const NODES = [
  { id: 'math',      label: 'Mathematics',    x: 50,  y: 50,  r: 22, color: '#8B5CF6', done: true },
  { id: 'stat',      label: 'Statistics',     x: 130, y: 30,  r: 18, color: '#A78BFA', done: true },
  { id: 'ml',        label: 'ML',             x: 200, y: 55,  r: 20, color: '#06B6D4', done: true },
  { id: 'dl',        label: 'Deep Learning',  x: 280, y: 30,  r: 22, color: '#22D3EE', done: false },
  { id: 'nlp',       label: 'NLP',            x: 355, y: 55,  r: 18, color: '#10B981', done: false },
  { id: 'cv',        label: 'CV',             x: 280, y: 100, r: 16, color: '#34D399', done: false },
  { id: 'llm',       label: 'LLMs',           x: 415, y: 30,  r: 20, color: '#F59E0B', done: false },
  { id: 'diffusion', label: 'Diffusion',      x: 420, y: 90,  r: 16, color: '#EC4899', done: false },
];

const EDGES = [
  ['math', 'stat'], ['math', 'ml'], ['stat', 'ml'], ['ml', 'dl'],
  ['dl', 'nlp'], ['dl', 'cv'], ['nlp', 'llm'], ['cv', 'diffusion'],
];

function nodeById(id: string) {
  return NODES.find(n => n.id === id)!;
}

export function KnowledgeMap() {
  return (
    <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
      <div className="px-4 pt-4 pb-2 flex items-center justify-between">
        <div>
          <div className="text-[13px] font-bold text-white">Knowledge Map</div>
          <div className="text-[11px] text-slate-600 mt-0.5">Your AI domain graph</div>
        </div>
        <div className="flex items-center gap-3 text-[11px] text-slate-500">
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full inline-block" style={{ background: '#8B5CF6' }} />Mastered</span>
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full inline-block" style={{ background: 'rgba(255,255,255,0.15)' }} />Locked</span>
        </div>
      </div>

      <div className="px-2 pb-3">
        <svg viewBox="0 0 470 130" className="w-full" style={{ height: 130 }} aria-label="Knowledge map graph">
          {/* Edges */}
          {EDGES.map(([a, b], i) => {
            const na = nodeById(a), nb = nodeById(b);
            return (
              <line key={i}
                x1={na.x} y1={na.y} x2={nb.x} y2={nb.y}
                stroke={na.done && nb.done ? '#8B5CF640' : 'rgba(255,255,255,0.06)'}
                strokeWidth="1.5"
              />
            );
          })}

          {/* Nodes */}
          {NODES.map((node, i) => (
            <motion.g key={node.id} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: i * 0.07, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}>
              <circle
                cx={node.x} cy={node.y} r={node.r}
                fill={node.done ? `${node.color}22` : 'rgba(255,255,255,0.04)'}
                stroke={node.done ? node.color : 'rgba(255,255,255,0.12)'}
                strokeWidth="1.5"
              />
              {node.done && (
                <circle cx={node.x} cy={node.y} r={node.r * 0.4}
                  fill={node.color} opacity="0.8">
                  <animate attributeName="r" values={`${node.r * 0.38};${node.r * 0.45};${node.r * 0.38}`}
                    dur={`${2.5 + i * 0.3}s`} repeatCount="indefinite" />
                </circle>
              )}
              <text x={node.x} y={node.y + node.r + 10}
                textAnchor="middle" fontSize="8"
                fill={node.done ? '#94A3B8' : '#475569'}
                fontFamily="sans-serif"
              >
                {node.label}
              </text>
            </motion.g>
          ))}
        </svg>
      </div>
    </div>
  );
}
