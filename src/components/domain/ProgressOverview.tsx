'use client';

import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { BookOpen, Flame, Clock, TrendingUp } from 'lucide-react';
import type { DomainProgress } from '@/data/domains/types';

interface ProgressOverviewProps {
  progress: DomainProgress;
  color: string;
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

function AnimatedCounter({ target, suffix = '' }: { target: number; suffix?: string }) {
  const [value, setValue] = useState(0);
  const hasRun = useRef(false);

  useEffect(() => {
    if (hasRun.current) return;
    hasRun.current = true;
    const duration = 900;
    const start = performance.now();
    const tick = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.round(eased * target));
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target]);

  return <span>{value}{suffix}</span>;
}

export function ProgressOverview({ progress, color }: ProgressOverviewProps) {
  const rgb = hexToRgb(color);
  const pct = Math.round((progress.lessonsCompleted / progress.totalLessons) * 100);

  const cards = [
    {
      icon: BookOpen,
      label: 'Lessons Completed',
      value: progress.lessonsCompleted,
      subtitle: `of ${progress.totalLessons} total`,
      suffix: '',
      color,
    },
    {
      icon: Flame,
      label: 'Current Streak',
      value: progress.currentStreak,
      subtitle: 'days in a row',
      suffix: ' day',
      color: '#F59E0B',
    },
    {
      icon: Clock,
      label: 'Hours Remaining',
      value: progress.estimatedHoursRemaining,
      subtitle: 'at current pace',
      suffix: 'h',
      color: '#0EA5E9',
    },
    {
      icon: TrendingUp,
      label: 'Domain Mastery',
      value: progress.masteryPercent,
      subtitle: 'mastery score',
      suffix: '%',
      color: '#10B981',
    },
  ];

  return (
    <section aria-labelledby="progress-overview-heading" className="mb-10">
      <motion.h2
        id="progress-overview-heading"
        initial={{ opacity: 0, y: 8 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.35 }}
        className="text-lg font-black mb-2"
        style={{ color: 'var(--color-text-primary)' }}
      >
        Your Progress
      </motion.h2>

      {/* Overall progress bar */}
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.35, delay: 0.05 }}
        className="mb-5"
        aria-label={`Overall progress: ${pct}%`}
      >
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span style={{ color: 'var(--color-text-secondary)' }}>Overall completion</span>
          <span className="font-bold tabular-nums" style={{ color }}>
            {pct}%
          </span>
        </div>
        <div className="h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.07)' }}>
          <motion.div
            className="h-full rounded-full"
            style={{ background: `linear-gradient(to right, ${color}, rgba(${rgb},0.6))` }}
            initial={{ width: 0 }}
            whileInView={{ width: `${pct}%` }}
            viewport={{ once: true }}
            transition={{ duration: 0.9, ease: 'easeOut' }}
          />
        </div>
      </motion.div>

      {/* Metric cards */}
      <div
        className="grid grid-cols-2 lg:grid-cols-4 gap-3"
        role="list"
        aria-label="Progress metrics"
      >
        {cards.map((card, i) => {
          const Icon = card.icon;
          const cardRgb = hexToRgb(card.color);
          return (
            <motion.div
              key={card.label}
              role="listitem"
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.35, delay: i * 0.07 }}
              className="rounded-2xl px-4 py-4"
              style={{
                background: 'var(--bg-surface)',
                border: `1px solid rgba(${cardRgb},0.16)`,
              }}
              aria-label={`${card.label}: ${card.value}${card.suffix}`}
            >
              <div
                className="w-8 h-8 rounded-xl flex items-center justify-center mb-3"
                style={{ background: `rgba(${cardRgb},0.14)` }}
                aria-hidden="true"
              >
                <Icon size={15} style={{ color: card.color }} />
              </div>
              <div className="text-2xl font-black tabular-nums mb-0.5" style={{ color: 'var(--color-text-primary)' }}>
                <AnimatedCounter target={card.value} suffix={card.suffix} />
              </div>
              <div className="text-[11px] font-semibold mb-0.5" style={{ color: card.color }}>
                {card.label}
              </div>
              <div className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
                {card.subtitle}
              </div>
            </motion.div>
          );
        })}
      </div>
    </section>
  );
}
