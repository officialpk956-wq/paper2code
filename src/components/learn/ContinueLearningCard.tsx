'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { ArrowRight, Clock, BookOpen } from 'lucide-react';
import type { ContinueLearningData } from '@/data/learn/recommendations';

interface ContinueLearningCardProps {
  data: ContinueLearningData;
}

export function ContinueLearningCard({ data }: ContinueLearningCardProps) {
  const pct = Math.max(0, Math.min(100, data.progress));

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.1 }}
      className="relative rounded-2xl overflow-hidden mb-8"
      style={{
        background: `linear-gradient(135deg, rgba(${hexToRgb(data.color)},0.12) 0%, var(--bg-surface) 60%)`,
        border: `1px solid rgba(${hexToRgb(data.color)},0.22)`,
      }}
      aria-label={`Continue learning: ${data.title}, ${pct}% complete`}
    >
      {/* Accent line */}
      <div
        className="absolute top-0 left-0 right-0 h-0.5"
        style={{ background: `linear-gradient(to right, ${data.color}, transparent)` }}
      />

      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 px-5 py-4">
        {/* Icon */}
        <div
          className="flex-none w-11 h-11 rounded-xl flex items-center justify-center"
          style={{ background: `rgba(${hexToRgb(data.color)},0.15)` }}
          aria-hidden="true"
        >
          <BookOpen size={18} style={{ color: data.color }} />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider"
              style={{ color: data.color }}>
              Continue Learning
            </span>
          </div>
          <p className="text-[15px] font-bold truncate" style={{ color: 'var(--color-text-primary)' }}>
            {data.title}
          </p>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-tertiary)' }}>
            {data.subtitle}
          </p>

          {/* Progress bar */}
          <div className="mt-3 flex items-center gap-3">
            <div className="flex-1 h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.07)' }}>
              <motion.div
                className="h-full rounded-full"
                style={{ background: data.color }}
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.8, delay: 0.3, ease: 'easeOut' }}
              />
            </div>
            <span className="text-xs font-semibold tabular-nums flex-none"
              style={{ color: data.color }}>
              {pct}%
            </span>
          </div>
        </div>

        {/* Meta + CTA */}
        <div className="flex sm:flex-col items-center sm:items-end gap-3 sm:gap-1.5">
          <div className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
            <Clock size={11} aria-hidden="true" />
            <span>{data.minutesRemaining} min left</span>
          </div>
          <Link
            href={data.href}
            className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-bold text-white transition-opacity hover:opacity-80"
            style={{ background: data.color }}
            aria-label={`Resume ${data.title}`}
          >
            Resume <ArrowRight size={11} aria-hidden="true" />
          </Link>
        </div>
      </div>
    </motion.section>
  );
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `${r},${g},${b}`;
}
