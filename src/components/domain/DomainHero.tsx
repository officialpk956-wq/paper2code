'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { ArrowLeft, BookOpen, Layers2, Code2, FolderOpen } from 'lucide-react';
import type { DomainData } from '@/data/domains/types';

interface DomainHeroProps {
  data: DomainData;
}

const PROGRESSION_COLORS = {
  beginner:     { bg: 'rgba(16,185,129,0.12)',  border: 'rgba(16,185,129,0.3)',  text: '#10B981' },
  intermediate: { bg: 'rgba(245,158,11,0.12)',  border: 'rgba(245,158,11,0.3)',  text: '#F59E0B' },
  advanced:     { bg: 'rgba(239,68,68,0.12)',   border: 'rgba(239,68,68,0.3)',   text: '#EF4444' },
};

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

export function DomainHero({ data }: DomainHeroProps) {
  const rgb = hexToRgb(data.color);

  return (
    <motion.section
      aria-labelledby="domain-hero-heading"
      className="relative overflow-hidden px-6 pt-6 pb-10"
      style={{
        background: `linear-gradient(160deg, rgba(${rgb},0.09) 0%, transparent 60%, rgba(124,58,237,0.04) 100%)`,
      }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      {/* Top accent line */}
      <div
        className="absolute top-0 left-0 right-0 h-px"
        style={{ background: `linear-gradient(to right, transparent, ${data.color}, rgba(124,58,237,0.5), transparent)` }}
        aria-hidden="true"
      />

      {/* Breadcrumb */}
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
        className="flex items-center gap-2 mb-6"
      >
        <Link
          href="/learn"
          className="inline-flex items-center gap-1.5 text-xs font-medium transition-colors hover:opacity-80 focus-visible:outline-none focus-visible:ring-2"
          style={{ color: 'var(--color-text-tertiary)' }}
          aria-label="Back to Learn Home"
        >
          <ArrowLeft size={12} aria-hidden="true" />
          Learn
        </Link>
        <span style={{ color: 'var(--color-text-muted)' }} aria-hidden="true">/</span>
        <span className="text-xs font-semibold" style={{ color: data.color }}>{data.name}</span>
      </motion.div>

      <div className="flex flex-col lg:flex-row items-start gap-8">
        {/* Left: headline + stats */}
        <div className="flex-1 min-w-0">
          {/* Domain badge */}
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05, duration: 0.35 }}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold mb-4"
            style={{ background: `rgba(${rgb},0.12)`, border: `1px solid rgba(${rgb},0.28)`, color: data.color }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: data.color }} aria-hidden="true" />
            {data.name}
          </motion.div>

          {/* Headline */}
          <motion.h1
            id="domain-hero-heading"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08, duration: 0.5 }}
            className="text-3xl xl:text-4xl font-black leading-[1.08] mb-4"
            style={{ color: 'var(--color-text-primary)' }}
          >
            {data.name}
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.14, duration: 0.45 }}
            className="text-base leading-relaxed mb-2 max-w-xl"
            style={{ color: 'var(--color-text-secondary)' }}
          >
            {data.tagline}
          </motion.p>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.22, duration: 0.4 }}
            className="flex flex-wrap items-center gap-5 mt-5"
            aria-label="Domain statistics"
          >
            {[
              { icon: BookOpen, value: data.stats.lessons,  label: 'Lessons' },
              { icon: Layers2,  value: data.stats.tracks,   label: 'Tracks' },
              { icon: Code2,    value: data.stats.problems,  label: 'Problems' },
              { icon: FolderOpen,value: data.stats.projects, label: 'Projects' },
            ].map((s, i) => {
              const Icon = s.icon;
              return (
                <motion.div
                  key={s.label}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.25 + i * 0.06 }}
                  className="flex items-center gap-2"
                >
                  <Icon size={13} style={{ color: data.color }} aria-hidden="true" />
                  <span className="text-sm font-black" style={{ color: 'var(--color-text-primary)' }}>
                    {s.value}
                  </span>
                  <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>{s.label}</span>
                </motion.div>
              );
            })}
          </motion.div>
        </div>

        {/* Right: progression ladder */}
        <motion.div
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="flex-none w-full lg:w-72"
          aria-label="Learning progression levels"
        >
          <div
            className="rounded-2xl overflow-hidden"
            style={{
              background: 'var(--bg-surface)',
              border: `1px solid rgba(${rgb},0.18)`,
            }}
          >
            <div
              className="px-4 py-3 text-xs font-bold uppercase tracking-wider"
              style={{
                background: `rgba(${rgb},0.08)`,
                borderBottom: '1px solid rgba(255,255,255,0.06)',
                color: data.color,
              }}
            >
              Learning Progression
            </div>
            {data.progression.map((level, i) => {
              const colors = PROGRESSION_COLORS[level.level];
              return (
                <div key={level.level} className="relative">
                  {/* Connector line between levels */}
                  {i < data.progression.length - 1 && (
                    <div
                      className="absolute left-[27px] top-[44px] w-px h-[20px]"
                      style={{ background: 'rgba(255,255,255,0.08)' }}
                      aria-hidden="true"
                    />
                  )}
                  <div className="flex items-start gap-3 px-4 py-3">
                    <div
                      className="flex-none w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-black mt-0.5"
                      style={{ background: colors.bg, border: `1px solid ${colors.border}`, color: colors.text }}
                      aria-hidden="true"
                    >
                      {i + 1}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-0.5">
                        <span className="text-[12px] font-bold" style={{ color: 'var(--color-text-primary)' }}>
                          {level.label}
                        </span>
                        <span
                          className="text-[9px] font-bold uppercase px-1.5 py-0.5 rounded-full"
                          style={{ background: colors.bg, color: colors.text }}
                        >
                          {level.level}
                        </span>
                      </div>
                      <p className="text-[11px] leading-relaxed mb-1.5" style={{ color: 'var(--color-text-secondary)' }}>
                        {level.description}
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {level.topicSamples.map(t => (
                          <span
                            key={t}
                            className="text-[9px] px-1.5 py-0.5 rounded-md"
                            style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--color-text-muted)' }}
                          >
                            {t}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </motion.div>
      </div>
    </motion.section>
  );
}
