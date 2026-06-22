'use client';

import { useRef } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import {
  Cpu, Workflow, Microscope, MessageSquare, Eye, Server, ChevronRight,
} from 'lucide-react';
import type { LearningPath } from '@/data/learn/paths';

const ICON_MAP: Record<string, React.ElementType> = {
  Cpu, Workflow, Microscope, MessageSquare, Eye, Server,
};

const DIFFICULTY_STYLES: Record<LearningPath['difficulty'], { label: string; className: string }> = {
  beginner:     { label: 'Beginner',     className: 'badge-easy' },
  intermediate: { label: 'Intermediate', className: 'badge-medium' },
  advanced:     { label: 'Advanced',     className: 'badge-hard' },
  expert:       { label: 'Expert',       className: 'badge-expert' },
};

interface LearningPathsProps {
  paths: LearningPath[];
}

function PathCard({ path, index }: { path: LearningPath; index: number }) {
  const Icon = ICON_MAP[path.icon] ?? Cpu;
  const diff = DIFFICULTY_STYLES[path.difficulty];
  const href = `/learn/paths/${path.id}`;
  
  // Real paths progress doesn't map directly to a single array yet. Force honesty (0%) to new users.
  const completionPercent = 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: index * 0.06 }}
      className="flex-none w-64 sm:w-72"
    >
      <Link
        href={href}
        className="block h-full rounded-2xl p-5 transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 group"
        style={{
          background: 'var(--bg-surface)',
          border: `1px solid rgba(${hexToRgb(path.color)},0.18)`,
        }}
        aria-label={`${path.title}: ${path.difficulty}, ${path.lessonCount} lessons, ${path.estimatedHours} hours`}
        onFocus={e => {
          (e.currentTarget as HTMLElement).style.borderColor = `rgba(${hexToRgb(path.color)},0.45)`;
        }}
        onBlur={e => {
          (e.currentTarget as HTMLElement).style.borderColor = `rgba(${hexToRgb(path.color)},0.18)`;
        }}
        onMouseEnter={e => {
          (e.currentTarget as HTMLElement).style.borderColor = `rgba(${hexToRgb(path.color)},0.45)`;
          (e.currentTarget as HTMLElement).style.background = 'var(--bg-hover)';
        }}
        onMouseLeave={e => {
          (e.currentTarget as HTMLElement).style.borderColor = `rgba(${hexToRgb(path.color)},0.18)`;
          (e.currentTarget as HTMLElement).style.background = 'var(--bg-surface)';
        }}
      >
        {/* Icon */}
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center mb-3"
          style={{ background: `rgba(${hexToRgb(path.color)},0.14)` }}
          aria-hidden="true"
        >
          <Icon size={18} style={{ color: path.color }} />
        </div>

        {/* Title + subtitle */}
        <div className="mb-1">
          <div className="text-[14px] font-bold leading-snug" style={{ color: 'var(--color-text-primary)' }}>
            {path.title}
          </div>
          <div className="text-xs mt-0.5" style={{ color: 'var(--color-text-tertiary)' }}>
            {path.subtitle}
          </div>
        </div>

        {/* Description */}
        <p className="text-xs leading-relaxed mb-4 line-clamp-2"
          style={{ color: 'var(--color-text-secondary)' }}>
          {path.description}
        </p>

        {/* Meta row */}
        <div className="flex items-center gap-2 mb-3 flex-wrap">
          <span className={diff.className}>{diff.label}</span>
          <span className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
            {path.lessonCount} lessons · {path.estimatedHours}h
          </span>
        </div>

        {/* Progress */}
        {completionPercent > 0 && (
          <div className="mb-2">
            <div className="flex items-center justify-between text-[10px] mb-1"
              style={{ color: 'var(--color-text-muted)' }}>
              <span>Progress</span>
              <span style={{ color: path.color }}>{completionPercent}%</span>
            </div>
            <div className="h-1 rounded-full" style={{ background: 'rgba(255,255,255,0.07)' }}>
                <div
                  className="h-full rounded-full"
                  style={{ width: `${completionPercent}%`, background: path.color }}
                />
              </div>
            </div>
          )}

        {/* CTA */}
        <div className="flex items-center gap-1 text-xs font-semibold mt-3"
          style={{ color: path.color }}>
          <span>{completionPercent > 0 ? 'Continue' : 'Start Path'}</span>
          <ChevronRight size={12} aria-hidden="true" />
        </div>
      </Link>
    </motion.div>
  );
}

export function LearningPaths({ paths }: LearningPathsProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  return (
    <section aria-labelledby="paths-heading" className="mb-10">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 id="paths-heading" className="text-lg font-black" style={{ color: 'var(--color-text-primary)' }}>
            Learning Paths
          </h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
            Structured curricula for every AI engineering role
          </p>
        </div>
        <Link
          href="/learn/paths"
          className="text-xs font-semibold transition-colors hover:opacity-80"
          style={{ color: 'var(--accent-primary)' }}
        >
          View all →
        </Link>
      </div>

      <div
        ref={scrollRef}
        className="flex gap-4 overflow-x-auto pb-3"
        style={{ scrollbarWidth: 'none' }}
        role="list"
        aria-label="Learning paths"
      >
        {paths.map((path, i) => (
          <div key={path.id} role="listitem">
            <PathCard path={path} index={i} />
          </div>
        ))}
      </div>
    </section>
  );
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `${r},${g},${b}`;
}
