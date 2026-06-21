'use client';

import { useRef } from 'react';
import Link from 'next/link';
import { motion, useMotionValue, useMotionTemplate, useSpring } from 'framer-motion';
import { ArrowRight, BookOpen, Layers, Code2, FileText } from 'lucide-react';

/* ─── Hero graph nodes (deterministic, no Math.random at module level) ─── */
const GRAPH_NODES = [
  { id: 'math',    x: 80,  y: 200, r: 22, label: 'Math',         color: '#8B5CF6', delay: 0.0 },
  { id: 'stats',   x: 160, y: 100, r: 18, label: 'Statistics',   color: '#7C3AED', delay: 0.4 },
  { id: 'ml',      x: 230, y: 260, r: 24, label: 'ML',           color: '#06B6D4', delay: 0.8 },
  { id: 'dl',      x: 320, y: 150, r: 28, label: 'Deep Learning',color: '#0EA5E9', delay: 0.2 },
  { id: 'nlp',     x: 390, y: 280, r: 20, label: 'NLP',          color: '#10B981', delay: 1.0 },
  { id: 'cv',      x: 460, y: 120, r: 20, label: 'Vision',       color: '#34D399', delay: 0.6 },
  { id: 'llm',     x: 510, y: 230, r: 26, label: 'LLMs',         color: '#A78BFA', delay: 0.3 },
  { id: 'agents',  x: 430, y: 340, r: 18, label: 'Agents',       color: '#F59E0B', delay: 1.2 },
  { id: 'rag',     x: 570, y: 320, r: 16, label: 'RAG',          color: '#EC4899', delay: 0.9 },
  { id: 'mlops',   x: 260, y: 360, r: 15, label: 'MLOps',        color: '#64748B', delay: 1.4 },
] as const;

const GRAPH_EDGES = [
  { from: 'math',   to: 'ml',     weight: 2.5 },
  { from: 'math',   to: 'stats',  weight: 1.8 },
  { from: 'stats',  to: 'ml',     weight: 2.0 },
  { from: 'ml',     to: 'dl',     weight: 3.0 },
  { from: 'dl',     to: 'nlp',    weight: 2.2 },
  { from: 'dl',     to: 'cv',     weight: 2.2 },
  { from: 'dl',     to: 'llm',    weight: 3.5 },
  { from: 'nlp',    to: 'llm',    weight: 2.8 },
  { from: 'llm',    to: 'agents', weight: 2.0 },
  { from: 'llm',    to: 'rag',    weight: 2.0 },
  { from: 'ml',     to: 'mlops',  weight: 1.5 },
  { from: 'cv',     to: 'llm',    weight: 1.6 },
] as const;

export interface RealStats {
  paperCount: number;
  archCount: number;
  problemCount: number;
  topicCount: number;
}

function curve(ax: number, ay: number, bx: number, by: number): string {
  const mx = (ax + bx) / 2, my = (ay + by) / 2;
  const dx = bx - ax, dy = by - ay;
  return `M${ax},${ay} Q${mx - dy * 0.2},${my + dx * 0.2} ${bx},${by}`;
}

function HeroGraph() {
  const nodeMap = Object.fromEntries(GRAPH_NODES.map(n => [n.id, n]));

  return (
    <svg
      viewBox="0 0 640 420"
      className="w-full h-full"
      aria-hidden="true"
      style={{ overflow: 'visible' }}
    >
      <defs>
        <radialGradient id="hg-ambient" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(124,58,237,0.12)" />
          <stop offset="100%" stopColor="rgba(7,11,24,0)" />
        </radialGradient>
        {GRAPH_NODES.map(n => (
          <radialGradient key={`hg-grd-${n.id}`} id={`hg-grd-${n.id}`} cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={n.color} stopOpacity="0.9" />
            <stop offset="100%" stopColor={n.color} stopOpacity="0.35" />
          </radialGradient>
        ))}
      </defs>

      <ellipse cx={320} cy={210} rx={340} ry={220} fill="url(#hg-ambient)" />

      {/* Edges */}
      {GRAPH_EDGES.map((e, i) => {
        const A = nodeMap[e.from];
        const B = nodeMap[e.to];
        if (!A || !B) return null;
        const d = curve(A.x, A.y, B.x, B.y);
        return (
          <g key={`edge-${i}`}>
            <path id={`hg-path-${i}`} d={d} fill="none" stroke="none" />
            <path d={d} fill="none" stroke={A.color} strokeWidth={e.weight * 0.6} opacity={0.15} />
            <circle r={e.weight + 0.5} fill={A.color} opacity={0.85}
              style={{ filter: `drop-shadow(0 0 3px ${A.color})` }}>
              <animateMotion
                dur={`${3.5 + i * 0.28}s`}
                repeatCount="indefinite"
                keyPoints="0;1"
                keyTimes="0;1"
                calcMode="linear"
              >
                <mpath href={`#hg-path-${i}`} />
              </animateMotion>
            </circle>
          </g>
        );
      })}

      {/* Nodes */}
      {GRAPH_NODES.map((node, i) => (
        <motion.g
          key={node.id}
          animate={{ y: [0, -6, 0] }}
          transition={{
            duration: 3.2 + node.delay * 0.35,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: node.delay,
          }}
        >
          {/* Outer glow ring */}
          <circle cx={node.x} cy={node.y} r={node.r + 8} fill="none"
            stroke={node.color} strokeWidth="1" opacity={0.12}>
            <animate
              attributeName="r"
              values={`${node.r + 5};${node.r + 12};${node.r + 5}`}
              dur={`${3 + i * 0.22}s`}
              repeatCount="indefinite"
            />
            <animate
              attributeName="opacity"
              values="0.12;0.06;0.12"
              dur={`${3 + i * 0.22}s`}
              repeatCount="indefinite"
            />
          </circle>

          {/* Node body */}
          <circle cx={node.x} cy={node.y} r={node.r}
            fill={`url(#hg-grd-${node.id})`}
            style={{ filter: `drop-shadow(0 0 ${node.r * 0.55}px ${node.color})` }}
          />
          <circle cx={node.x} cy={node.y} r={node.r * 0.32} fill="white" opacity={0.16} />

          {/* Label */}
          <text
            x={node.x} y={node.y + node.r + 13}
            textAnchor="middle"
            fontSize={node.r > 22 ? 10.5 : 9}
            fontWeight="700"
            fill="white"
            opacity={0.82}
            fontFamily="system-ui, sans-serif"
          >
            {node.label}
          </text>
        </motion.g>
      ))}
    </svg>
  );
}

/* ─── Hero Section ──────────────────────────────────────────── */
export function LearnHero({ stats }: { stats: RealStats }) {
  const dynamicStats = [
    { icon: BookOpen, value: stats.topicCount.toString(),  label: 'Lessons',          color: 'var(--accent-primary)' },
    { icon: Layers,   value: stats.archCount.toString(),  label: 'Architectures',    color: 'var(--accent-cyan)' },
    { icon: Code2,    value: stats.problemCount.toString(),label: 'Problems',         color: 'var(--accent-emerald)' },
    { icon: FileText, value: stats.paperCount.toString(),  label: 'Research Papers',  color: 'var(--accent-amber)' },
  ];

  const sectionRef = useRef<HTMLElement>(null);
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  const smoothX = useSpring(mouseX, { stiffness: 60, damping: 20 });
  const smoothY = useSpring(mouseY, { stiffness: 60, damping: 20 });
  const glowBg = useMotionTemplate`radial-gradient(700px at ${smoothX}px ${smoothY}px, rgba(124,58,237,0.09), transparent 70%)`;

  return (
    <motion.section
      ref={sectionRef}
      aria-label="Learn home hero"
      className="relative overflow-hidden px-6 pt-8 pb-12"
      style={{
        background: 'linear-gradient(160deg, rgba(124,58,237,0.07) 0%, transparent 55%, rgba(6,182,212,0.05) 100%)',
      }}
      onMouseMove={e => {
        const r = sectionRef.current?.getBoundingClientRect();
        if (r) { mouseX.set(e.clientX - r.left); mouseY.set(e.clientY - r.top); }
      }}
    >
      {/* Mouse-follow glow */}
      <motion.div className="absolute inset-0 pointer-events-none" style={{ background: glowBg }} />

      {/* Top accent line */}
      <div className="absolute top-0 left-0 right-0 h-px"
        style={{ background: 'linear-gradient(to right, transparent, var(--accent-primary), var(--accent-cyan), transparent)' }} />

      <div className="flex flex-col lg:flex-row items-center gap-10 max-w-full">
        {/* ── Left: text + stats + CTAs ── */}
        <div className="flex-1 min-w-0">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45 }}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold mb-5"
            style={{
              background: 'rgba(124,58,237,0.12)',
              border: '1px solid rgba(124,58,237,0.3)',
              color: 'var(--accent-light)',
            }}
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: 'var(--accent-light)', boxShadow: '0 0 6px var(--accent-primary)' }}
            />
            AI Engineering University
          </motion.div>

          {/* Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, delay: 0.08 }}
            className="text-4xl xl:text-5xl font-black leading-[1.07] mb-4"
          >
            <span className="block" style={{ color: 'var(--color-text-primary)' }}>Learn AI</span>
            <span className="block gradient-text">Like A Researcher</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, delay: 0.18 }}
            className="text-base leading-relaxed mb-7 max-w-lg"
            style={{ color: 'var(--color-text-secondary)' }}
          >
            Master mathematics, deep learning, LLMs, system design and AI engineering through
            interactive lessons, paper-to-code projects and a live knowledge graph.
          </motion.p>

          {/* Stats grid */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.28 }}
            className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-7"
          >
            {dynamicStats.map((s, i) => {
              const Icon = s.icon;
              return (
                <motion.div
                  key={s.label}
                  initial={{ opacity: 0, scale: 0.88 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.32 + i * 0.07, duration: 0.38 }}
                  className="flex flex-col gap-1 px-3 py-2.5 rounded-xl"
                  style={{
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid rgba(255,255,255,0.06)',
                  }}
                >
                  <Icon size={13} style={{ color: s.color }} />
                  <div className="text-lg font-black leading-none" style={{ color: 'var(--color-text-primary)' }}>
                    {s.value}
                  </div>
                  <div className="text-[10px] font-medium uppercase tracking-widest"
                    style={{ color: 'var(--color-text-muted)' }}>
                    {s.label}
                  </div>
                </motion.div>
              );
            })}
          </motion.div>

          {/* CTAs */}
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.48 }}
            className="flex flex-wrap gap-3"
          >
            <Link
              href="/learn/deep-learning/multi-head-attention"
              className="btn-primary inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold text-white"
              aria-label="Continue learning where you left off"
            >
              Continue Learning <ArrowRight size={14} aria-hidden="true" />
            </Link>
            <Link
              href="#domains"
              className="btn-secondary inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold"
              aria-label="Explore all learning domains"
            >
              Explore Domains
            </Link>
          </motion.div>
        </div>

        {/* ── Right: animated graph ── */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.75, delay: 0.12, ease: [0.22, 1, 0.36, 1] }}
          className="flex-none w-full lg:w-[480px] xl:w-[560px]"
          style={{ height: 380 }}
          aria-hidden="true"
        >
          <HeroGraph />
        </motion.div>
      </div>
    </motion.section>
  );
}
