'use client';

import { useRef } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { TrendingUp, Flame, ArrowUpRight, Users } from 'lucide-react';
import type { TrendingTopic } from '@/data/learn/topics';

const TREND_ICON: Record<TrendingTopic['trend'], React.ElementType> = {
  hot:    Flame,
  rising: TrendingUp,
  stable: TrendingUp,
};

interface TrendingTopicsProps {
  topics: TrendingTopic[];
}

function TopicCard({ topic, index }: { topic: TrendingTopic; index: number }) {
  const TrendIcon = TREND_ICON[topic.trend];
  const href = `/learn/topics/${topic.id}`;

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: index * 0.07 }}
      className="flex-none w-64 sm:w-72"
    >
      <Link
        href={href}
        className="block rounded-2xl p-4 transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 h-full"
        style={{
          background: 'var(--bg-surface)',
          border: `1px solid rgba(${hexToRgb(topic.color)},0.18)`,
        }}
        aria-label={`${topic.title}: ${topic.popularity}% popularity, ${topic.learnerCount.toLocaleString()} learners`}
        onMouseEnter={e => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = `rgba(${hexToRgb(topic.color)},0.4)`;
          el.style.background = 'var(--bg-hover)';
        }}
        onMouseLeave={e => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = `rgba(${hexToRgb(topic.color)},0.18)`;
          el.style.background = 'var(--bg-surface)';
        }}
        onFocus={e => {
          (e.currentTarget as HTMLElement).style.borderColor = `rgba(${hexToRgb(topic.color)},0.45)`;
        }}
        onBlur={e => {
          (e.currentTarget as HTMLElement).style.borderColor = `rgba(${hexToRgb(topic.color)},0.18)`;
        }}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ background: `rgba(${hexToRgb(topic.color)},0.15)` }}
              aria-hidden="true"
            >
              <TrendIcon size={15} style={{ color: topic.color }} />
            </div>
            {topic.isNew && (
              <span
                className="px-1.5 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider"
                style={{ background: `rgba(${hexToRgb(topic.color)},0.18)`, color: topic.color }}
              >
                New
              </span>
            )}
          </div>
          <ArrowUpRight size={13} style={{ color: 'var(--color-text-muted)' }} aria-hidden="true" />
        </div>

        {/* Title */}
        <div className="text-[13px] font-bold mb-1 leading-snug"
          style={{ color: 'var(--color-text-primary)' }}>
          {topic.title}
        </div>

        {/* Description */}
        <p className="text-[11px] leading-relaxed line-clamp-2 mb-4"
          style={{ color: 'var(--color-text-secondary)' }}>
          {topic.description}
        </p>

        {/* Popularity bar */}
        <div className="mb-3">
          <div className="flex items-center justify-between text-[10px] mb-1.5">
            <span style={{ color: 'var(--color-text-muted)' }}>Popularity</span>
            <span className="font-bold tabular-nums" style={{ color: topic.color }}>
              {topic.popularity}%
            </span>
          </div>
          <div className="h-1 rounded-full" style={{ background: 'rgba(255,255,255,0.07)' }}>
            <motion.div
              className="h-full rounded-full"
              style={{ background: `linear-gradient(to right, ${topic.color}, ${topic.color}aa)` }}
              initial={{ width: 0 }}
              whileInView={{ width: `${topic.popularity}%` }}
              viewport={{ once: true }}
              transition={{ duration: 0.7, delay: 0.1 + index * 0.05, ease: 'easeOut' }}
            />
          </div>
        </div>

        {/* Learners + growth */}
        <div className="flex items-center justify-between text-[10px]"
          style={{ color: 'var(--color-text-muted)' }}>
          <div className="flex items-center gap-1">
            <Users size={10} aria-hidden="true" />
            <span>{topic.learnerCount.toLocaleString()} learners</span>
          </div>
          <span style={{ color: '#10B981' }}>+{topic.weeklyGrowth}% this week</span>
        </div>
      </Link>
    </motion.div>
  );
}

export function TrendingTopics({ topics }: TrendingTopicsProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  return (
    <section aria-labelledby="trending-heading" className="mb-10">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 id="trending-heading" className="text-lg font-black" style={{ color: 'var(--color-text-primary)' }}>
            Trending Topics
          </h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
            What the AI engineering community is studying right now
          </p>
        </div>
        <Link
          href="/learn/topics"
          className="text-xs font-semibold transition-opacity hover:opacity-80"
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
        aria-label="Trending topics"
      >
        {topics.map((topic, i) => (
          <div key={topic.id} role="listitem">
            <TopicCard topic={topic} index={i} />
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
