'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { CheckCircle2, Circle, Lock, PlayCircle, ChevronRight, Clock } from 'lucide-react';
import type { TopicCluster } from '@/data/domains/types';
import { resolveTopicPath } from '@/data/topics';

interface TopicClustersProps {
  clusters: TopicCluster[];
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

const STATUS_ICON = {
  completed:   { Icon: CheckCircle2, color: '#10B981' },
  in_progress: { Icon: PlayCircle,   color: '#F59E0B' },
  available:   { Icon: Circle,       color: '#94A3B8' },
  locked:      { Icon: Lock,         color: '#475569' },
};

const DIFFICULTY_BADGE = {
  beginner:     'badge-easy',
  intermediate: 'badge-medium',
  advanced:     'badge-hard',
};

function TopicRow({
  topic,
  accentColor,
}: {
  topic: TopicCluster['topics'][0];
  accentColor: string;
}) {
  const { Icon, color } = STATUS_ICON[topic.status];
  const isLocked = topic.status === 'locked';

  const validHref = resolveTopicPath(topic.slug);
  const isDisabled = isLocked || !validHref;
  
  const innerContent = (
    <>
      <Icon size={14} style={{ color, flexShrink: 0 }} aria-hidden="true" />
      <div className="flex-1 min-w-0 flex items-center gap-2">
        <span className="text-[13px] font-medium truncate" style={{ color: 'var(--color-text-primary)' }}>
          {topic.title}
        </span>
        {!validHref && !isLocked && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider" style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--color-text-muted)' }}>
            <Clock size={8} aria-hidden="true" /> Coming soon
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 flex-none">
        <span className={DIFFICULTY_BADGE[topic.difficulty]}>{topic.difficulty}</span>
        <span className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
          {topic.lessonCount} lessons
        </span>
        {!isDisabled && <ChevronRight size={11} style={{ color: 'var(--color-text-muted)' }} aria-hidden="true" />}
      </div>
    </>
  );

  if (isDisabled) {
    return (
      <div
        aria-label={`${topic.title}: ${topic.status}, ${topic.lessonCount} lessons, ${topic.difficulty}`}
        className="flex items-center gap-3 px-3 py-2.5 rounded-xl"
        style={{ opacity: 0.45, cursor: 'not-allowed' }}
      >
        {innerContent}
      </div>
    );
  }

  return (
    <Link
      href={validHref}
      aria-label={`${topic.title}: ${topic.status}, ${topic.lessonCount} lessons, ${topic.difficulty}`}
      className="flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-150 focus-visible:outline-none focus-visible:ring-1"
      onMouseEnter={e => {
        (e.currentTarget as HTMLElement).style.background = `rgba(${hexToRgb(accentColor)},0.07)`;
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.background = 'transparent';
      }}
    >
      {innerContent}
    </Link>
  );
}

function ClusterCard({ cluster, index }: { cluster: TopicCluster; index: number; }) {
  const rgb = hexToRgb(cluster.color);
  const completedCount = cluster.topics.filter(t => t.status === 'completed').length;
  const pct = Math.round((completedCount / cluster.topics.length) * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: index * 0.08 }}
      className="rounded-2xl overflow-hidden"
      style={{
        background: 'var(--bg-surface)',
        border: `1px solid rgba(${rgb},0.16)`,
      }}
    >
      {/* Header */}
      <div
        className="px-5 py-4"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', background: `rgba(${rgb},0.06)` }}
      >
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-[14px] font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
              {cluster.title}
            </h3>
            <p className="text-[11px] leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
              {cluster.description}
            </p>
          </div>
          <div className="flex-none text-right">
            <div className="text-xs font-bold tabular-nums" style={{ color: cluster.color }}>
              {pct}%
            </div>
            <div className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
              {completedCount}/{cluster.topics.length} done
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div className="mt-3 h-1 rounded-full" style={{ background: 'rgba(255,255,255,0.07)' }}>
          <motion.div
            className="h-full rounded-full"
            style={{ background: cluster.color }}
            initial={{ width: 0 }}
            whileInView={{ width: `${pct}%` }}
            viewport={{ once: true }}
            transition={{ duration: 0.7, ease: 'easeOut' }}
          />
        </div>
      </div>

      {/* Topics */}
      <div className="px-3 py-2" role="list" aria-label={`${cluster.title} topics`}>
        {cluster.topics.map(topic => (
          <div key={topic.id} role="listitem">
            <TopicRow topic={topic} accentColor={cluster.color} />
          </div>
        ))}
      </div>
    </motion.div>
  );
}

export function TopicClusters({ clusters }: TopicClustersProps) {
  return (
    <section aria-labelledby="clusters-heading" className="mb-10">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 id="clusters-heading" className="text-lg font-black" style={{ color: 'var(--color-text-primary)' }}>
            Topic Clusters
          </h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
            All topics grouped by theme — click any topic to open the lesson
          </p>
        </div>
      </div>

      <div
        className="grid grid-cols-1 md:grid-cols-2 gap-4"
        role="list"
        aria-label="Topic clusters"
      >
        {clusters.map((cluster, i) => (
          <div key={cluster.id} role="listitem">
            <ClusterCard cluster={cluster} index={i} />
          </div>
        ))}
      </div>
    </section>
  );
}
