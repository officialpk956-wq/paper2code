'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { ArrowLeft, Clock, BookOpen, Tag, ChevronRight } from 'lucide-react';
import type { TopicMeta } from '@/data/topics/types';

interface TopicHeroProps {
  meta: TopicMeta;
}

const DIFFICULTY_CONFIG = {
  beginner:     { label: 'Beginner',     className: 'badge-easy' },
  intermediate: { label: 'Intermediate', className: 'badge-medium' },
  advanced:     { label: 'Advanced',     className: 'badge-hard' },
  expert:       { label: 'Expert',       className: 'badge-expert' },
};

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

export function TopicHero({ meta }: TopicHeroProps) {
  const diff = DIFFICULTY_CONFIG[meta.difficulty];
  const rgb = hexToRgb(meta.domainColor);

  return (
    <div
      className="relative overflow-hidden px-6 pt-5 pb-8"
      style={{
        background: `linear-gradient(160deg, rgba(${rgb},0.08) 0%, transparent 55%, rgba(124,58,237,0.04) 100%)`,
        borderBottom: '1px solid rgba(255,255,255,0.06)',
      }}
      aria-labelledby="topic-hero-heading"
    >
      {/* Top accent */}
      <div
        className="absolute top-0 left-0 right-0 h-px"
        style={{ background: `linear-gradient(to right, transparent, ${meta.domainColor}, rgba(124,58,237,0.4), transparent)` }}
        aria-hidden="true"
      />

      {/* Breadcrumb */}
      <motion.nav
        aria-label="Breadcrumb"
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex items-center gap-1.5 mb-5 text-xs"
      >
        <Link
          href="/learn"
          className="transition-opacity hover:opacity-80"
          style={{ color: 'var(--color-text-muted)' }}
        >
          Learn
        </Link>
        <ChevronRight size={10} style={{ color: 'var(--color-text-muted)' }} aria-hidden="true" />
        <Link
          href={`/learn/${meta.domainSlug}`}
          className="transition-opacity hover:opacity-80"
          style={{ color: 'var(--color-text-tertiary)' }}
        >
          {meta.domain}
        </Link>
        <ChevronRight size={10} style={{ color: 'var(--color-text-muted)' }} aria-hidden="true" />
        <span style={{ color: meta.domainColor }}>{meta.title}</span>
      </motion.nav>

      <div className="flex flex-col lg:flex-row items-start gap-6">
        <div className="flex-1 min-w-0">
          {/* Domain badge */}
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05, duration: 0.3 }}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold mb-3"
            style={{ background: `rgba(${rgb},0.12)`, border: `1px solid rgba(${rgb},0.28)`, color: meta.domainColor }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: meta.domainColor }} aria-hidden="true" />
            {meta.domain}
          </motion.div>

          {/* Title */}
          <motion.h1
            id="topic-hero-heading"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08, duration: 0.45 }}
            className="text-2xl xl:text-3xl font-black leading-tight mb-2"
            style={{ color: 'var(--color-text-primary)', letterSpacing: '-0.02em' }}
          >
            {meta.title}
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.12, duration: 0.4 }}
            className="text-sm leading-relaxed mb-4 max-w-2xl"
            style={{ color: 'var(--color-text-secondary)' }}
          >
            {meta.subtitle}
          </motion.p>

          {/* Meta row */}
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.18, duration: 0.35 }}
            className="flex flex-wrap items-center gap-3"
          >
            <span className={diff.className}>{diff.label}</span>

            <div className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
              <Clock size={11} aria-hidden="true" />
              <span>{meta.durationMinutes} min read</span>
            </div>

            <div className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
              <BookOpen size={11} aria-hidden="true" />
              <span>{meta.prerequisites.length} prerequisites</span>
            </div>

            {meta.tags.slice(0, 2).map(tag => (
              <div
                key={tag}
                className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full"
                style={{ background: `rgba(${rgb},0.1)`, color: meta.domainColor }}
              >
                <Tag size={8} aria-hidden="true" />
                {tag}
              </div>
            ))}
          </motion.div>
        </div>

        {/* Back button */}
        <motion.div
          initial={{ opacity: 0, x: 8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2, duration: 0.35 }}
          className="flex-none"
        >
          <Link
            href={`/learn/${meta.domainSlug}`}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold transition-all hover:opacity-80 focus-visible:outline-none focus-visible:ring-2"
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid rgba(255,255,255,0.08)',
              color: 'var(--color-text-secondary)',
            }}
            aria-label={`Back to ${meta.domain} domain`}
          >
            <ArrowLeft size={12} aria-hidden="true" />
            Back to {meta.domain}
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
