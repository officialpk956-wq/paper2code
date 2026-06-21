'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { ArrowRight, Zap } from 'lucide-react';

/* ─── Floating label chips ─────────────────────────────────── */
const CHIPS = [
  { label: 'Transformer', color: '#8B5CF6', x: '62%', y: '18%', delay: 0 },
  { label: 'CNN', color: '#06B6D4', x: '74%', y: '52%', delay: 0.8 },
  { label: 'LLM', color: '#10B981', x: '58%', y: '72%', delay: 1.4 },
  { label: 'Research', color: '#F59E0B', x: '80%', y: '32%', delay: 0.4 },
  { label: 'Code', color: '#EC4899', x: '68%', y: '86%', delay: 1.1 },
];

/* ─── Mini neural SVG ──────────────────────────────────────── */
function NeuralSVG() {
  const nodes = [
    { x: 20, y: 50 }, { x: 40, y: 25 }, { x: 40, y: 75 },
    { x: 60, y: 40 }, { x: 60, y: 65 }, { x: 80, y: 50 },
  ];
  const edges = [[0,1],[0,2],[1,3],[2,4],[3,5],[4,5],[1,4],[2,3]];
  return (
    <svg viewBox="0 0 100 100" className="w-full h-full" aria-hidden="true">
      {edges.map(([a, b], i) => (
        <line key={i}
          x1={`${nodes[a].x}%`} y1={`${nodes[a].y}%`}
          x2={`${nodes[b].x}%`} y2={`${nodes[b].y}%`}
          stroke="#8B5CF6" strokeWidth="0.4" opacity="0.3">
          <animate attributeName="opacity" values="0.3;0.1;0.3"
            dur={`${2.5 + i * 0.3}s`} repeatCount="indefinite" />
        </line>
      ))}
      {nodes.map((n, i) => (
        <circle key={i} cx={`${n.x}%`} cy={`${n.y}%`} r="2.2%"
          fill={i % 2 === 0 ? '#8B5CF6' : '#06B6D4'} opacity="0.7">
          <animate attributeName="r" values="2.2%;3%;2.2%"
            dur={`${2 + i * 0.4}s`} repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.7;0.3;0.7"
            dur={`${2 + i * 0.4}s`} repeatCount="indefinite" />
        </circle>
      ))}
    </svg>
  );
}

/* ─── Component ────────────────────────────────────────────── */
interface DashboardHeroProps {
  name?: string;
  track?: string;
  progress?: number;
  goal?: string;
}

export function DashboardHero({
  name = 'Researcher',
  track = 'Transformers and Attention Mechanisms',
  progress = 34,
  goal = 'Complete Multi-Head Attention module',
}: DashboardHeroProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="relative mx-6 mt-5 mb-1 rounded-2xl overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, rgba(124,58,237,0.18) 0%, rgba(7,11,24,0.9) 50%, rgba(6,182,212,0.1) 100%)',
        border: '1px solid rgba(139,92,246,0.25)',
        boxShadow: '0 0 40px rgba(139,92,246,0.08)',
        minHeight: 180,
      }}
    >
      {/* Gradient top line */}
      <div className="absolute top-0 left-0 right-0 h-px"
        style={{ background: 'linear-gradient(to right, transparent, #8B5CF6, #06B6D4, transparent)' }} />

      {/* Neural SVG background */}
      <div className="absolute right-0 top-0 bottom-0 w-[45%] opacity-25 pointer-events-none">
        <NeuralSVG />
      </div>

      {/* Floating chips */}
      {CHIPS.map((chip) => (
        <motion.div
          key={chip.label}
          className="absolute hidden lg:block text-[11px] font-semibold px-2.5 py-1 rounded-full pointer-events-none"
          style={{
            left: chip.x, top: chip.y,
            background: `${chip.color}18`,
            border: `1px solid ${chip.color}40`,
            color: chip.color,
          }}
          animate={{ y: [0, -5, 0] }}
          transition={{ duration: 3.5 + chip.delay * 0.4, repeat: Infinity, ease: 'easeInOut', delay: chip.delay }}
        >
          {chip.label}
        </motion.div>
      ))}

      {/* Content */}
      <div className="relative z-10 p-6">
        <div className="flex items-center gap-2 mb-2">
          <Zap size={14} style={{ color: '#8B5CF6' }} />
          <span className="text-xs font-semibold" style={{ color: '#A78BFA' }}>AI Learning OS</span>
        </div>

        <h1 className="text-2xl font-black text-white mb-1">
          Welcome back, <span style={{ background: 'linear-gradient(135deg, #A78BFA, #22D3EE)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>{name}</span>
        </h1>
        <p className="text-sm text-slate-400 mb-5">Continue your AI journey — you&apos;re on a roll.</p>

        <div className="max-w-md space-y-3">
          {/* Track */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-slate-500">Current Track</span>
              <span className="text-xs font-semibold" style={{ color: '#A78BFA' }}>{progress}%</span>
            </div>
            <div className="text-sm font-semibold text-white mb-2">{track}</div>
            <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.08)' }}>
              <motion.div
                className="h-full rounded-full"
                style={{ background: 'linear-gradient(to right, #7C3AED, #06B6D4)' }}
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 1, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
              />
            </div>
          </div>

          {/* Goal */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" style={{ boxShadow: '0 0 6px #10B981' }} />
              <span className="text-xs text-slate-400">{goal}</span>
            </div>
          </div>

          {/* CTA */}
          <Link
            href="/learn"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold text-white transition-all duration-200 group"
            style={{
              background: 'linear-gradient(135deg, #7C3AED, #8B5CF6)',
              boxShadow: '0 0 20px rgba(139,92,246,0.3)',
            }}
            onMouseEnter={e => (e.currentTarget as HTMLElement).style.boxShadow = '0 0 28px rgba(139,92,246,0.5)'}
            onMouseLeave={e => (e.currentTarget as HTMLElement).style.boxShadow = '0 0 20px rgba(139,92,246,0.3)'}
          >
            Continue Learning
            <ArrowRight size={14} className="group-hover:translate-x-0.5 transition-transform" />
          </Link>
        </div>
      </div>
    </motion.div>
  );
}
