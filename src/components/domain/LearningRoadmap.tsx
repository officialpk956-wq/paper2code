'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import { CheckCircle2, Circle, Lock, ChevronDown, ChevronUp, PlayCircle } from 'lucide-react';
import type { RoadmapStage } from '@/data/domains/types';

interface LearningRoadmapProps {
  stages: RoadmapStage[];
  domainSlug: string;
  color: string;
}

const STATUS_CONFIG = {
  completed:   { icon: CheckCircle2, color: '#10B981', bg: 'rgba(16,185,129,0.14)',  border: 'rgba(16,185,129,0.3)',  label: 'Completed' },
  in_progress: { icon: PlayCircle,   color: '#F59E0B', bg: 'rgba(245,158,11,0.14)',  border: 'rgba(245,158,11,0.3)',  label: 'In Progress' },
  available:   { icon: Circle,       color: '#94A3B8', bg: 'rgba(148,163,184,0.1)',   border: 'rgba(148,163,184,0.25)', label: 'Available' },
  locked:      { icon: Lock,         color: '#475569', bg: 'rgba(71,85,105,0.1)',     border: 'rgba(71,85,105,0.2)',   label: 'Locked' },
};

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

function RoadmapNode({
  stage,
  index,
  isExpanded,
  onToggle,
  domainSlug,
  isLast,
  accentColor,
}: {
  stage: RoadmapStage;
  index: number;
  isExpanded: boolean;
  onToggle: () => void;
  domainSlug: string;
  isLast: boolean;
  accentColor: string;
}) {
  const cfg = STATUS_CONFIG[stage.status];
  const Icon = cfg.icon;
  const isClickable = stage.status !== 'locked';

  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.38, delay: Math.min(index * 0.07, 0.45) }}
      className="relative flex gap-4"
    >
      {/* Vertical connector line */}
      {!isLast && (
        <div
          className="absolute left-[19px] top-[42px] w-px"
          style={{
            height: isExpanded ? 'calc(100% - 12px)' : '28px',
            background: stage.status === 'completed'
              ? `rgba(${hexToRgb(accentColor)},0.3)`
              : 'rgba(255,255,255,0.07)',
            transition: 'height 0.3s ease',
          }}
          aria-hidden="true"
        />
      )}

      {/* Node icon */}
      <div className="flex-none relative z-10">
        <div
          className="w-10 h-10 rounded-full flex items-center justify-center"
          style={{ background: cfg.bg, border: `2px solid ${cfg.border}` }}
          aria-hidden="true"
        >
          <Icon size={17} style={{ color: cfg.color }} />
        </div>
      </div>

      {/* Card */}
      <div className="flex-1 min-w-0 pb-7">
        <button
          onClick={onToggle}
          disabled={!isClickable}
          className="w-full text-left rounded-2xl px-4 py-3.5 transition-all duration-200 focus-visible:outline-none focus-visible:ring-2"
          style={{
            background: isExpanded ? 'var(--bg-hover)' : 'var(--bg-surface)',
            border: isExpanded
              ? `1px solid rgba(${hexToRgb(accentColor)},0.3)`
              : `1px solid rgba(255,255,255,${stage.status === 'locked' ? '0.04' : '0.07'})`,
            opacity: stage.status === 'locked' ? 0.5 : 1,
            cursor: isClickable ? 'pointer' : 'default',
          }}
          aria-expanded={isExpanded}
          aria-label={`${stage.title}: ${cfg.label}. ${isClickable ? (isExpanded ? 'Collapse' : 'Expand') : 'Locked'}`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 min-w-0">
              <span
                className="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-md flex-none"
                style={{ background: cfg.bg, color: cfg.color }}
              >
                Stage {stage.order}
              </span>
              <div className="min-w-0">
                <div className="text-[14px] font-bold truncate" style={{ color: 'var(--color-text-primary)' }}>
                  {stage.title}
                </div>
                <div className="text-[11px] truncate" style={{ color: 'var(--color-text-secondary)' }}>
                  {stage.description}
                </div>
              </div>
            </div>
            {isClickable && (
              <div className="flex-none ml-3" aria-hidden="true">
                {isExpanded
                  ? <ChevronUp size={14} style={{ color: 'var(--color-text-muted)' }} />
                  : <ChevronDown size={14} style={{ color: 'var(--color-text-muted)' }} />
                }
              </div>
            )}
          </div>

          {/* Expanded topic list */}
          <AnimatePresence initial={false}>
            {isExpanded && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.25, ease: 'easeInOut' }}
                className="overflow-hidden"
              >
                <div className="pt-3 mt-3 flex flex-wrap gap-2" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
                  {stage.topics.map(topic => (
                    <Link
                      key={topic}
                      href={`/learn/${domainSlug}/${topic.toLowerCase().replace(/ /g, '-')}`}
                      onClick={e => e.stopPropagation()}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-1"
                      style={{
                        background: `rgba(${hexToRgb(accentColor)},0.1)`,
                        border: `1px solid rgba(${hexToRgb(accentColor)},0.2)`,
                        color: 'var(--color-text-secondary)',
                      }}
                      aria-label={`Study topic: ${topic}`}
                      onMouseEnter={e => {
                        (e.currentTarget as HTMLElement).style.color = 'var(--color-text-primary)';
                        (e.currentTarget as HTMLElement).style.borderColor = `rgba(${hexToRgb(accentColor)},0.4)`;
                      }}
                      onMouseLeave={e => {
                        (e.currentTarget as HTMLElement).style.color = 'var(--color-text-secondary)';
                        (e.currentTarget as HTMLElement).style.borderColor = `rgba(${hexToRgb(accentColor)},0.2)`;
                      }}
                    >
                      {topic}
                    </Link>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </button>
      </div>
    </motion.div>
  );
}

export function LearningRoadmap({ stages, domainSlug, color }: LearningRoadmapProps) {
  const [expandedId, setExpandedId] = useState<string | null>(
    stages.find(s => s.status === 'in_progress')?.id ?? null
  );

  const toggle = (id: string) => setExpandedId(prev => prev === id ? null : id);

  return (
    <section aria-labelledby="roadmap-heading" className="mb-10">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 id="roadmap-heading" className="text-lg font-black" style={{ color: 'var(--color-text-primary)' }}>
            Learning Roadmap
          </h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
            The structured path through this domain — click any stage to explore topics
          </p>
        </div>
      </div>

      <div
        className="rounded-2xl p-5"
        style={{
          background: 'var(--bg-surface)',
          border: '1px solid rgba(255,255,255,0.06)',
        }}
        role="list"
        aria-label="Learning roadmap stages"
      >
        {stages.map((stage, i) => (
          <div key={stage.id} role="listitem">
            <RoadmapNode
              stage={stage}
              index={i}
              isExpanded={expandedId === stage.id}
              onToggle={() => toggle(stage.id)}
              domainSlug={domainSlug}
              isLast={i === stages.length - 1}
              accentColor={color}
            />
          </div>
        ))}
      </div>
    </section>
  );
}
