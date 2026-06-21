'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Clock, ArrowRight, CheckCircle2, Sparkles } from 'lucide-react';
import type { DomainLesson } from '@/data/domains/types';
import { isCompleted } from '@/lib/progress';
import { resolveTopicPath } from '@/data/topics';
import { useState, useEffect } from 'react';

interface FeaturedLessonsProps {
  lessons: DomainLesson[];
  domainSlug: string;
  color: string;
}

const DIFFICULTY_CONFIG = {
  beginner:     { className: 'badge-easy',   label: 'Beginner' },
  intermediate: { className: 'badge-medium', label: 'Intermediate' },
  advanced:     { className: 'badge-hard',   label: 'Advanced' },
};

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

function LessonCard({
  lesson,
  color,
  index,
}: {
  lesson: DomainLesson;
  color: string;
  index: number;
}) {
  const diff = DIFFICULTY_CONFIG[lesson.difficulty];
  const rgb = hexToRgb(color);

  const [isDone, setIsDone] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setIsDone(isCompleted('topic', lesson.topicSlug));
    setLoaded(true);
  }, [lesson.topicSlug]);

  const displayPercent = isDone ? 100 : 0;

  const validHref = resolveTopicPath(lesson.topicSlug);
  const isDisabled = !validHref;

  const innerContent = (
    <>
      {/* Index / completion indicator */}
      <div
        className="flex-none w-9 h-9 rounded-xl flex items-center justify-center text-xs font-black mt-0.5"
        style={{
          background: isDone ? 'rgba(16,185,129,0.14)' : `rgba(${rgb},0.12)`,
          color: isDone ? '#10B981' : color,
        }}
        aria-hidden="true"
      >
        {isDone
          ? <CheckCircle2 size={16} />
          : <span>{String(index + 1).padStart(2, '0')}</span>
        }
      </div>

      <div className="flex-1 min-w-0">
        {/* Badges row */}
        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
          <span className={diff.className}>{diff.label}</span>
          {lesson.isNew && (
            <span
              className="inline-flex items-center gap-1 text-[9px] font-bold px-1.5 py-0.5 rounded-full uppercase tracking-wider"
              style={{ background: `rgba(${rgb},0.15)`, color }}
            >
              <Sparkles size={8} aria-hidden="true" /> New
            </span>
          )}
          {isDisabled && (
            <span className="inline-flex items-center gap-1 text-[9px] font-bold px-1.5 py-0.5 rounded-full uppercase tracking-wider" style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--color-text-muted)' }}>
              Coming soon
            </span>
          )}
        </div>

        {/* Title */}
        <div className="text-[13px] font-bold leading-snug mb-1" style={{ color: 'var(--color-text-primary)' }}>
          {lesson.title}
        </div>

        {/* Description */}
        <p className="text-[11px] leading-relaxed line-clamp-2 mb-3" style={{ color: 'var(--color-text-secondary)' }}>
          {lesson.description}
        </p>

        {/* Footer: time + progress + CTA */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
              <Clock size={9} aria-hidden="true" /> {lesson.durationMinutes} min
            </span>
            {loaded && displayPercent > 0 && !isDone && (
              <div className="flex items-center gap-1.5">
                <div className="w-16 h-1 rounded-full" style={{ background: 'rgba(255,255,255,0.08)' }}>
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${displayPercent}%`, background: color }}
                  />
                </div>
                <span className="text-[10px] tabular-nums" style={{ color }}>{displayPercent}%</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-1 text-[11px] font-semibold" style={{ color: isDisabled ? 'var(--color-text-muted)' : color }}>
            {loaded ? (isDisabled ? 'Coming soon' : isDone ? 'Review' : displayPercent > 0 ? 'Continue' : 'Start') : '...'}
            {!isDisabled && <ArrowRight size={11} aria-hidden="true" />}
          </div>
        </div>
      </div>
    </>
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.38, delay: index * 0.06 }}
    >
      {isDisabled ? (
        <div
          className="flex items-start gap-4 rounded-2xl p-4 transition-all duration-200 block"
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid rgba(255,255,255,0.06)',
            opacity: 0.5,
            cursor: 'not-allowed'
          }}
          aria-label={`${lesson.title}: ${lesson.difficulty}, ${lesson.durationMinutes} minutes`}
        >
          {innerContent}
        </div>
      ) : (
        <Link
          href={validHref!}
          className="flex items-start gap-4 rounded-2xl p-4 transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 block"
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid rgba(255,255,255,0.06)',
            opacity: loaded ? 1 : 0.5,
          }}
          aria-label={`${lesson.title}: ${lesson.difficulty}, ${lesson.durationMinutes} minutes${isDone ? ', completed' : displayPercent > 0 ? `, ${displayPercent}% done` : ''}`}
          onMouseEnter={e => {
            const el = e.currentTarget as HTMLElement;
            el.style.borderColor = `rgba(${rgb},0.32)`;
            el.style.background = 'var(--bg-hover)';
          }}
          onMouseLeave={e => {
            const el = e.currentTarget as HTMLElement;
            el.style.borderColor = 'rgba(255,255,255,0.06)';
            el.style.background = 'var(--bg-surface)';
          }}
          onFocus={e => {
            (e.currentTarget as HTMLElement).style.borderColor = `rgba(${rgb},0.45)`;
          }}
          onBlur={e => {
            (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.06)';
          }}
        >
          {innerContent}
        </Link>
      )}
    </motion.div>
  );
}

export function FeaturedLessons({ lessons, domainSlug, color }: FeaturedLessonsProps) {
  return (
    <section aria-labelledby="lessons-heading" className="mb-10">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 id="lessons-heading" className="text-lg font-black" style={{ color: 'var(--color-text-primary)' }}>
            Featured Lessons
          </h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
            Curated lessons that define this domain
          </p>
        </div>
        <Link
          href={`/learn/${domainSlug}/lessons`}
          className="text-xs font-semibold transition-opacity hover:opacity-80"
          style={{ color: 'var(--accent-primary)' }}
        >
          View all →
        </Link>
      </div>

      <div
        className="grid grid-cols-1 sm:grid-cols-2 gap-3"
        role="list"
        aria-label="Featured lessons"
      >
        {lessons.map((lesson, i) => (
          <div key={lesson.id} role="listitem">
            <LessonCard lesson={lesson} color={color} index={i} />
          </div>
        ))}
      </div>
    </section>
  );
}
