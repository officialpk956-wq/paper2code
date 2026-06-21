'use client';

import { motion } from 'framer-motion';
import { BookOpen, Clock, BarChart2 } from 'lucide-react';
import { useState, useEffect } from 'react';
import { isCompleted } from '@/lib/progress';

export const CHAPTER_SECTIONS = [
  { id: 'motivation',    label: 'Motivation' },
  { id: 'intuition',     label: 'Intuition' },
  { id: 'mathematics',   label: 'Mathematics' },
  { id: 'architecture',  label: 'Architecture' },
  { id: 'code',          label: 'Code Walkthrough' },
  { id: 'applications',  label: 'Applications' },
  { id: 'timeline',      label: 'Research Timeline' },
  { id: 'papers',        label: 'Related Papers' },
  { id: 'interview',     label: 'Interview Notes' },
  { id: 'practice',      label: 'Practice' },
  { id: 'summary',       label: 'Summary' },
] as const;

export type SectionId = (typeof CHAPTER_SECTIONS)[number]['id'];

interface TopicSidebarProps {
  activeSection: SectionId | null;
  topicSlug: string;
  durationMinutes: number;
  difficulty: string;
  onNavigate: (id: SectionId) => void;
}

const DIFF_COLOR: Record<string, string> = {
  beginner:     '#10B981',
  intermediate: '#F59E0B',
  advanced:     '#EF4444',
  expert:       '#EC4899',
};

export function TopicSidebar({
  activeSection,
  topicSlug,
  durationMinutes,
  difficulty,
  onNavigate,
}: TopicSidebarProps) {
  const diffColor = DIFF_COLOR[difficulty] ?? 'var(--accent-primary)';
  
  const [loaded, setLoaded] = useState(false);
  const [completionPercent, setCompletionPercent] = useState(0);

  useEffect(() => {
    setCompletionPercent(isCompleted('topic', topicSlug) ? 100 : 0);
    setLoaded(true);
  }, [topicSlug]);

  return (
    <nav
      aria-label="Chapter navigation"
      className="flex flex-col gap-4"
    >
      {/* Progress card */}
      <div
        className="rounded-2xl p-4"
        style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
      >
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] font-semibold" style={{ color: 'var(--color-text-secondary)' }}>
            Progress
          </span>
          <span className="text-[12px] font-black tabular-nums" style={{ color: 'var(--accent-primary)' }}>
            {loaded ? completionPercent : '-'}%
          </span>
        </div>
        {/* Progress bar */}
        <div className="h-1 rounded-full mb-3" style={{ background: 'rgba(255,255,255,0.07)', opacity: loaded ? 1 : 0.5 }}>
          <motion.div
            className="h-full rounded-full"
            style={{ background: 'linear-gradient(to right, var(--accent-primary), var(--accent-cyan))' }}
            initial={{ width: 0 }}
            animate={{ width: `${loaded ? completionPercent : 0}%` }}
            transition={{ duration: 0.7, ease: 'easeOut' }}
          />
        </div>

        <div className="flex items-center justify-between text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
          <span className="flex items-center gap-1">
            <Clock size={9} aria-hidden="true" />
            {durationMinutes} min
          </span>
          <span className="flex items-center gap-1">
            <BarChart2 size={9} aria-hidden="true" />
            <span style={{ color: diffColor }}>{difficulty}</span>
          </span>
        </div>
      </div>

      {/* Chapter list */}
      <div
        className="rounded-2xl overflow-hidden"
        style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
      >
        <div
          className="px-4 py-3 flex items-center gap-2"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}
        >
          <BookOpen size={12} style={{ color: 'var(--accent-primary)' }} aria-hidden="true" />
          <span className="text-[11px] font-bold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>
            Chapters
          </span>
        </div>

        <ol className="py-1" aria-label="Topic sections">
          {CHAPTER_SECTIONS.map((section, index) => {
            const isActive = activeSection === section.id;
            return (
              <li key={section.id}>
                <button
                  onClick={() => onNavigate(section.id)}
                  className="w-full flex items-center gap-3 px-4 py-2.5 text-left transition-all focus-visible:outline-none focus-visible:ring-inset focus-visible:ring-1"
                  style={{
                    background: isActive ? 'rgba(124,58,237,0.1)' : 'transparent',
                    borderLeft: `2px solid ${isActive ? 'var(--accent-primary)' : 'transparent'}`,
                    color: isActive ? 'var(--accent-light)' : 'var(--color-text-secondary)',
                  }}
                  aria-current={isActive ? 'true' : undefined}
                  aria-label={`Go to ${section.label} section`}
                >
                  <span
                    className="flex-none text-[10px] font-bold tabular-nums w-4"
                    style={{ color: isActive ? 'var(--accent-primary)' : 'var(--color-text-muted)' }}
                    aria-hidden="true"
                  >
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <span className="text-[12px] font-medium">{section.label}</span>
                </button>
              </li>
            );
          })}
        </ol>
      </div>
    </nav>
  );
}
