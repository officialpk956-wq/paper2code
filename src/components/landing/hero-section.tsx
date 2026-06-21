'use client';

import { useEffect, useRef } from 'react';
import { motion, useMotionValue, useMotionTemplate } from 'framer-motion';
import Link from 'next/link';
import { ArrowRight, Sparkles, BookOpen, Code2, FlaskConical, Map, Mic2, Layers3, Brain } from 'lucide-react';

/* ─── Neural network background ────────────────────────────────────────── */
const NN_NODES = [
  { cx: 8, cy: 18, r: 3, c: '#8B5CF6', d: '3.2s', delay: '0s' },
  { cx: 22, cy: 8,  r: 2, c: '#06B6D4', d: '4.1s', delay: '0.5s' },
  { cx: 38, cy: 22, r: 4, c: '#8B5CF6', d: '3.8s', delay: '1s' },
  { cx: 55, cy: 12, r: 2, c: '#22D3EE', d: '5s',   delay: '0.3s' },
  { cx: 72, cy: 28, r: 3, c: '#8B5CF6', d: '3.5s', delay: '0.8s' },
  { cx: 88, cy: 10, r: 2, c: '#06B6D4', d: '4.5s', delay: '1.2s' },
  { cx: 14, cy: 42, r: 2, c: '#22D3EE', d: '4.8s', delay: '0.6s' },
  { cx: 30, cy: 55, r: 3, c: '#8B5CF6', d: '3.6s', delay: '1.4s' },
  { cx: 48, cy: 45, r: 2, c: '#06B6D4', d: '4.2s', delay: '0.2s' },
  { cx: 65, cy: 58, r: 4, c: '#8B5CF6', d: '3.9s', delay: '0.9s' },
  { cx: 82, cy: 44, r: 2, c: '#22D3EE', d: '5.1s', delay: '1.6s' },
  { cx: 95, cy: 55, r: 3, c: '#8B5CF6', d: '4.3s', delay: '0.4s' },
  { cx: 6,  cy: 68, r: 2, c: '#06B6D4', d: '3.7s', delay: '1.1s' },
  { cx: 25, cy: 75, r: 3, c: '#8B5CF6', d: '4.6s', delay: '0.7s' },
  { cx: 42, cy: 80, r: 2, c: '#22D3EE', d: '5.2s', delay: '1.5s' },
  { cx: 60, cy: 72, r: 4, c: '#8B5CF6', d: '3.4s', delay: '0.1s' },
  { cx: 78, cy: 85, r: 2, c: '#06B6D4', d: '4.7s', delay: '1.3s' },
  { cx: 92, cy: 78, r: 3, c: '#8B5CF6', d: '3.3s', delay: '0.8s' },
];

const NN_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
  [0, 6], [6, 7], [7, 8], [8, 2], [8, 9],
  [9, 10], [10, 11], [4, 10], [5, 11],
  [6, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17],
  [7, 13], [9, 15], [11, 17],
];

function NeuralNetworkBG() {
  return (
    <svg
      className="absolute inset-0 w-full h-full"
      viewBox="0 0 100 100"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      {/* Edges */}
      {NN_EDGES.map(([a, b], i) => {
        const na = NN_NODES[a], nb = NN_NODES[b];
        return (
          <line
            key={i}
            x1={`${na.cx}%`} y1={`${na.cy}%`}
            x2={`${nb.cx}%`} y2={`${nb.cy}%`}
            stroke={na.c}
            strokeWidth="0.08"
            opacity="0.25"
          >
            <animate attributeName="opacity" values="0.25;0.1;0.25" dur={na.d} begin={na.delay} repeatCount="indefinite" />
          </line>
        );
      })}
      {/* Nodes */}
      {NN_NODES.map((n, i) => (
        <circle key={i} cx={`${n.cx}%`} cy={`${n.cy}%`} r="0.35%" fill={n.c} opacity="0.6">
          <animate attributeName="opacity" values="0.6;0.2;0.6" dur={n.d} begin={n.delay} repeatCount="indefinite" />
          <animate attributeName="r" values="0.35%;0.55%;0.35%" dur={n.d} begin={n.delay} repeatCount="indefinite" />
        </circle>
      ))}
    </svg>
  );
}

/* ─── Orbital brain visualization ──────────────────────────────────────── */
function OrbitalBrain() {
  return (
    <div className="relative w-[340px] h-[340px] md:w-[420px] md:h-[420px]" aria-hidden="true">
      {/* Outer ambient glow */}
      <div className="absolute inset-[-40px] rounded-full bg-gradient-to-br from-violet-900/30 to-cyan-900/20 blur-3xl" />

      {/* Ring 3 — outermost */}
      <motion.div
        className="absolute inset-[10px] rounded-full border border-dashed border-violet-500/15"
        animate={{ rotate: 360 }}
        transition={{ duration: 40, repeat: Infinity, ease: 'linear' }}
      >
        <motion.div
          className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-3 h-3 rounded-full bg-blue-400"
          style={{ boxShadow: '0 0 10px #3B82F6, 0 0 20px rgba(59,130,246,0.4)' }}
          animate={{ rotate: -360 }}
          transition={{ duration: 40, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-pink-400"
          style={{ boxShadow: '0 0 8px #EC4899' }}
          animate={{ rotate: -360 }}
          transition={{ duration: 40, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute top-1/2 -right-1.5 -translate-y-1/2 w-2 h-2 rounded-full bg-emerald-400"
          style={{ boxShadow: '0 0 8px #10B981' }}
          animate={{ rotate: -360 }}
          transition={{ duration: 40, repeat: Infinity, ease: 'linear' }}
        />
      </motion.div>

      {/* Ring 2 — medium */}
      <motion.div
        className="absolute inset-[60px] rounded-full border border-dashed border-cyan-500/20"
        animate={{ rotate: -360 }}
        transition={{ duration: 25, repeat: Infinity, ease: 'linear' }}
      >
        <motion.div
          className="absolute -top-2 left-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-cyan-400"
          style={{ boxShadow: '0 0 14px #22D3EE, 0 0 28px rgba(34,211,238,0.3)' }}
          animate={{ rotate: 360 }}
          transition={{ duration: 25, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-3 h-3 rounded-full bg-violet-400"
          style={{ boxShadow: '0 0 10px #8B5CF6' }}
          animate={{ rotate: 360 }}
          transition={{ duration: 25, repeat: Infinity, ease: 'linear' }}
        />
      </motion.div>

      {/* Ring 1 — innermost */}
      <motion.div
        className="absolute inset-[110px] rounded-full border border-violet-500/40"
        style={{ boxShadow: '0 0 20px rgba(139,92,246,0.15) inset' }}
        animate={{ rotate: 360 }}
        transition={{ duration: 14, repeat: Infinity, ease: 'linear' }}
      >
        <motion.div
          className="absolute -top-2 left-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-violet-500"
          style={{ boxShadow: '0 0 16px #8B5CF6, 0 0 32px rgba(139,92,246,0.5)' }}
          animate={{ rotate: -360 }}
          transition={{ duration: 14, repeat: Infinity, ease: 'linear' }}
        />
      </motion.div>

      {/* Central orb */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative w-28 h-28 md:w-32 md:h-32">
          {/* Glow layers */}
          <div className="absolute inset-0 rounded-full bg-violet-600/30 blur-xl animate-pulse" />
          <div
            className="absolute inset-0 rounded-full"
            style={{ background: 'radial-gradient(circle at 35% 35%, rgba(167,139,250,0.9), rgba(109,40,217,0.9) 60%, rgba(30,10,80,0.95))' }}
          />
          <div className="absolute inset-[4px] rounded-full border border-violet-300/20" />
          <div className="absolute inset-[10px] rounded-full border border-violet-400/15" />
          <div className="absolute inset-0 flex items-center justify-center">
            <Brain className="w-10 h-10 text-violet-200 opacity-80" />
          </div>
        </div>
      </div>

      {/* Animated connection lines from center to ring nodes */}
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 420 420" aria-hidden="true">
        <line x1="210" y1="210" x2="210" y2="72" stroke="#8B5CF6" strokeWidth="0.8" strokeDasharray="4 6" opacity="0.3">
          <animate attributeName="stroke-dashoffset" values="0;-20" dur="2s" repeatCount="indefinite" />
        </line>
        <line x1="210" y1="210" x2="348" y2="210" stroke="#22D3EE" strokeWidth="0.8" strokeDasharray="4 6" opacity="0.25">
          <animate attributeName="stroke-dashoffset" values="0;-20" dur="3s" repeatCount="indefinite" />
        </line>
        <line x1="210" y1="210" x2="100" y2="320" stroke="#8B5CF6" strokeWidth="0.8" strokeDasharray="4 6" opacity="0.2">
          <animate attributeName="stroke-dashoffset" values="0;-20" dur="2.5s" repeatCount="indefinite" />
        </line>
      </svg>

      {/* Floating label chips */}
      <motion.div
        className="absolute -right-6 top-12 px-3 py-1.5 rounded-full text-[11px] font-semibold text-cyan-300 border border-cyan-500/30 bg-cyan-950/50 backdrop-blur-sm whitespace-nowrap"
        animate={{ y: [0, -4, 0] }}
        transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
        style={{ boxShadow: '0 0 12px rgba(34,211,238,0.15)' }}
      >
        Transformers
      </motion.div>
      <motion.div
        className="absolute -left-8 top-1/2 -translate-y-1/2 px-3 py-1.5 rounded-full text-[11px] font-semibold text-violet-300 border border-violet-500/30 bg-violet-950/50 backdrop-blur-sm whitespace-nowrap"
        animate={{ y: [0, 4, 0] }}
        transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
        style={{ boxShadow: '0 0 12px rgba(139,92,246,0.15)' }}
      >
        LLMs
      </motion.div>
      <motion.div
        className="absolute -right-4 bottom-16 px-3 py-1.5 rounded-full text-[11px] font-semibold text-emerald-300 border border-emerald-500/30 bg-emerald-950/50 backdrop-blur-sm whitespace-nowrap"
        animate={{ y: [0, -3, 0] }}
        transition={{ duration: 4.5, repeat: Infinity, ease: 'easeInOut', delay: 2 }}
        style={{ boxShadow: '0 0 12px rgba(16,185,129,0.15)' }}
      >
        MLOps
      </motion.div>
      <motion.div
        className="absolute left-2 bottom-8 px-3 py-1.5 rounded-full text-[11px] font-semibold text-pink-300 border border-pink-500/30 bg-pink-950/50 backdrop-blur-sm whitespace-nowrap"
        animate={{ y: [0, 3, 0] }}
        transition={{ duration: 5.5, repeat: Infinity, ease: 'easeInOut', delay: 0.5 }}
        style={{ boxShadow: '0 0 12px rgba(236,72,153,0.15)' }}
      >
        Diffusion
      </motion.div>
    </div>
  );
}

/* ─── Feature pills marquee ─────────────────────────────────────────────── */
const PILLS = [
  { icon: <BookOpen size={12} />, label: 'Interactive Learning' },
  { icon: <FlaskConical size={12} />, label: 'Research Papers' },
  { icon: <Layers3 size={12} />, label: 'System Design' },
  { icon: <Code2 size={12} />, label: 'Practice Problems' },
  { icon: <Sparkles size={12} />, label: 'Projects' },
  { icon: <Map size={12} />, label: 'Roadmaps' },
  { icon: <Mic2 size={12} />, label: 'Interview Prep' },
  { icon: <Brain size={12} />, label: 'Architectures' },
];

function FeaturePills() {
  return (
    <div className="relative overflow-hidden mt-10 py-1" aria-label="Platform features">
      <div className="absolute left-0 top-0 bottom-0 w-16 z-10 pointer-events-none"
        style={{ background: 'linear-gradient(to right, #050816, transparent)' }} />
      <div className="absolute right-0 top-0 bottom-0 w-16 z-10 pointer-events-none"
        style={{ background: 'linear-gradient(to left, #050816, transparent)' }} />
      <motion.div
        className="flex gap-3 w-max"
        animate={{ x: ['0%', '-50%'] }}
        transition={{ duration: 28, repeat: Infinity, ease: 'linear' }}
      >
        {[...PILLS, ...PILLS].map((pill, i) => (
          <div
            key={i}
            className="flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium text-slate-300 border border-white/8 bg-white/4 backdrop-blur-sm whitespace-nowrap"
            style={{ background: 'rgba(255,255,255,0.04)' }}
          >
            <span className="text-violet-400">{pill.icon}</span>
            {pill.label}
          </div>
        ))}
      </motion.div>
    </div>
  );
}

/* ─── Main Hero Section ─────────────────────────────────────────────────── */
export interface RealStats {
  paperCount: number;
  archCount: number;
  sysDesignCount: number;
  problemCount: number;
  topicCount: number;
}

export function HeroSection({ stats }: { stats: RealStats }) {
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  const mouseGlow = useMotionTemplate`radial-gradient(550px at ${mouseX}px ${mouseY}px, rgba(139,92,246,0.10) 0%, transparent 70%)`;

  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouseX.set(e.clientX);
      mouseY.set(e.clientY);
    };
    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [mouseX, mouseY]);

  const heroStats = [
    { value: stats.topicCount,     label: 'Topics' },
    { value: stats.paperCount,     label: 'Research Papers' },
    { value: stats.archCount,      label: 'Architectures' },
    { value: stats.problemCount,   label: 'Practice Problems' },
    { value: stats.sysDesignCount, label: 'System Designs' },
  ];

  return (
    <section
      ref={sectionRef}
      className="relative min-h-screen flex flex-col overflow-hidden"
      style={{ background: '#050816' }}
    >
      {/* Grid overlay */}
      <div
        className="absolute inset-0 opacity-[0.35] pointer-events-none"
        style={{
          backgroundImage: 'linear-gradient(rgba(139,92,246,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,0.08) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />

      {/* Neural network background */}
      <div className="absolute inset-0 opacity-50 pointer-events-none">
        <NeuralNetworkBG />
      </div>

      {/* Ambient gradient blobs */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div
          className="absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full opacity-20"
          style={{ background: 'radial-gradient(circle, rgba(139,92,246,0.5) 0%, transparent 70%)' }}
        />
        <div
          className="absolute top-20 right-[-100px] w-[500px] h-[500px] rounded-full opacity-15"
          style={{ background: 'radial-gradient(circle, rgba(6,182,212,0.5) 0%, transparent 70%)' }}
        />
        <div
          className="absolute bottom-[-100px] left-1/3 w-[400px] h-[400px] rounded-full opacity-10"
          style={{ background: 'radial-gradient(circle, rgba(139,92,246,0.4) 0%, transparent 70%)' }}
        />
      </div>

      {/* Mouse-tracking glow */}
      <motion.div
        className="absolute inset-0 pointer-events-none"
        style={{ background: mouseGlow }}
      />

      {/* Content */}
      <div className="relative z-10 flex-1">
        <div className="w-full max-w-7xl mx-auto px-6 md:px-10 pt-24 pb-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-8 items-center">

            {/* Left: Text content */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
            >
              {/* Badge */}
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1, duration: 0.6 }}
                className="inline-flex items-center gap-2.5 px-4 py-2 rounded-full mb-8 text-xs font-semibold"
                style={{
                  background: 'rgba(139,92,246,0.12)',
                  border: '1px solid rgba(139,92,246,0.35)',
                  color: '#C4B5FD',
                  boxShadow: '0 0 20px rgba(139,92,246,0.1)',
                }}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
                The Complete Data Science &amp; AI Learning Platform
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              </motion.div>

              {/* Headline */}
              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.8 }}
                className="text-5xl md:text-6xl lg:text-[72px] font-heading font-black leading-[1.05] tracking-tight mb-6"
              >
                <span
                  className="block"
                  style={{ background: 'linear-gradient(135deg, #A78BFA 0%, #8B5CF6 40%, #22D3EE 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}
                >
                  Learn.
                </span>
                <span
                  className="block"
                  style={{ background: 'linear-gradient(135deg, #C4B5FD 0%, #8B5CF6 50%, #06B6D4 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}
                >
                  Practice.
                </span>
                <span
                  className="block"
                  style={{ background: 'linear-gradient(135deg, #E2E8F0 0%, #CBD5E1 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}
                >
                  Build.
                </span>
                <span className="block text-white mt-1">Master Data Science &amp; AI</span>
              </motion.h1>

              {/* Subheadline */}
              <motion.p
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.35, duration: 0.7 }}
                className="text-lg text-slate-400 mb-10 leading-relaxed max-w-xl"
              >
                From fundamentals to research papers, from models to production systems —
                everything you need to become an AI Engineer.
              </motion.p>

              {/* CTA Buttons */}
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.45, duration: 0.7 }}
                className="flex flex-col sm:flex-row gap-4"
              >
                <Link
                  href="/dashboard"
                  className="inline-flex items-center justify-center gap-2.5 px-8 py-4 rounded-2xl text-sm font-bold text-white transition-all duration-200 group"
                  style={{
                    background: 'linear-gradient(135deg, #7C3AED 0%, #8B5CF6 50%, #06B6D4 100%)',
                    boxShadow: '0 0 30px rgba(139,92,246,0.4), 0 4px 20px rgba(0,0,0,0.4)',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.boxShadow = '0 0 45px rgba(139,92,246,0.6), 0 4px 24px rgba(0,0,0,0.4)')}
                  onMouseLeave={e => (e.currentTarget.style.boxShadow = '0 0 30px rgba(139,92,246,0.4), 0 4px 20px rgba(0,0,0,0.4)')}
                >
                  Start Learning
                  <ArrowRight size={16} className="group-hover:translate-x-0.5 transition-transform" />
                </Link>
                <Link
                  href="/learn"
                  className="inline-flex items-center justify-center gap-2.5 px-8 py-4 rounded-2xl text-sm font-bold transition-all duration-200"
                  style={{
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.12)',
                    color: '#E2E8F0',
                  }}
                  onMouseEnter={e => {
                    (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.09)';
                    (e.currentTarget as HTMLElement).style.borderColor = 'rgba(139,92,246,0.5)';
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.05)';
                    (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.12)';
                  }}
                >
                  Explore Platform
                  <ArrowRight size={16} />
                </Link>
                <Link
                  href="/papers/upload"
                  className="inline-flex items-center justify-center gap-2.5 px-8 py-4 rounded-2xl text-sm font-bold transition-all duration-200"
                  style={{
                    background: 'rgba(6,182,212,0.08)',
                    border: '1px solid rgba(6,182,212,0.25)',
                    color: '#67E8F9',
                  }}
                  onMouseEnter={e => {
                    (e.currentTarget as HTMLElement).style.background = 'rgba(6,182,212,0.14)';
                    (e.currentTarget as HTMLElement).style.borderColor = 'rgba(6,182,212,0.45)';
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLElement).style.background = 'rgba(6,182,212,0.08)';
                    (e.currentTarget as HTMLElement).style.borderColor = 'rgba(6,182,212,0.25)';
                  }}
                >
                  Upload a Paper
                  <ArrowRight size={16} />
                </Link>
              </motion.div>

              {/* Feature pills */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6, duration: 0.8 }}
              >
                <FeaturePills />
              </motion.div>
            </motion.div>

            {/* Right: Orbital brain */}
            <motion.div
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3, duration: 1, ease: [0.22, 1, 0.36, 1] }}
              className="flex items-center justify-center lg:justify-end"
            >
              <OrbitalBrain />
            </motion.div>
          </div>

          {/* Stats row — real derived counts, no inflation */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.7 }}
            className="mt-20 pt-10 border-t border-white/6 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-8"
          >
            {heroStats.map((stat) => (
              <div key={stat.label} className="text-center">
                <div
                  className="text-3xl md:text-4xl font-black mb-1"
                  style={{ background: 'linear-gradient(135deg, #A78BFA, #22D3EE)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}
                >
                  {stat.value}
                </div>
                <p className="text-sm text-slate-500">{stat.label}</p>
              </div>
            ))}
          </motion.div>
        </div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        animate={{ y: [0, 6, 0], opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
        aria-hidden="true"
      >
        <span className="text-[10px] text-slate-600 uppercase tracking-[0.2em]">Scroll</span>
        <div className="w-px h-8 bg-gradient-to-b from-violet-500/60 to-transparent" />
      </motion.div>
    </section>
  );
}
