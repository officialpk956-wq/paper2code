'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Clock, BookOpen, FileText, Layers, Code2 } from 'lucide-react';
import type { Recommendation } from '@/data/learn/recommendations';

const TYPE_ICON: Record<Recommendation['type'], React.ElementType> = {
  lesson:       BookOpen,
  paper:        FileText,
  architecture: Layers,
  project:      Code2,
};

const TYPE_LABEL: Record<Recommendation['type'], string> = {
  lesson:       'Lesson',
  paper:        'Paper',
  architecture: 'Architecture',
  project:      'Project',
};

interface RecommendationsProps {
  items: Recommendation[];
}

function ConfidenceRing({ confidence, color }: { confidence: number; color: string }) {
  const size = 44;
  const r = (size - 6) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, confidence));

  return (
    <svg width={size} height={size} aria-hidden="true" className="flex-none">
      <circle cx={size / 2} cy={size / 2} r={r}
        fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="3.5" />
      <motion.circle
        cx={size / 2} cy={size / 2} r={r}
        fill="none" stroke={color} strokeWidth="3.5"
        strokeLinecap="round"
        strokeDasharray={c}
        initial={{ strokeDashoffset: c }}
        whileInView={{ strokeDashoffset: c - (pct / 100) * c }}
        viewport={{ once: true }}
        transition={{ duration: 0.75, ease: 'easeOut' }}
        style={{ transform: 'rotate(-90deg)', transformOrigin: `${size / 2}px ${size / 2}px` }}
      />
      <text
        x={size / 2} y={size / 2 + 4}
        textAnchor="middle"
        fontSize="9"
        fontWeight="800"
        fill={color}
        fontFamily="system-ui, sans-serif"
      >
        {pct}%
      </text>
    </svg>
  );
}

function RecommendationCard({ item, index }: { item: Recommendation; index: number }) {
  const Icon = TYPE_ICON[item.type];
  const typeLabel = TYPE_LABEL[item.type];

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.38, delay: index * 0.07 }}
    >
      <Link
        href={item.href}
        className="block rounded-2xl p-4 transition-all duration-200 focus-visible:outline-none focus-visible:ring-2"
        style={{
          background: 'var(--bg-surface)',
          border: `1px solid rgba(${hexToRgb(item.color)},0.16)`,
        }}
        aria-label={`${item.title}: ${typeLabel}, ${item.confidence}% match, ${item.estimatedMinutes} minutes`}
        onMouseEnter={e => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = `rgba(${hexToRgb(item.color)},0.4)`;
          el.style.background = 'var(--bg-hover)';
        }}
        onMouseLeave={e => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = `rgba(${hexToRgb(item.color)},0.16)`;
          el.style.background = 'var(--bg-surface)';
        }}
        onFocus={e => {
          (e.currentTarget as HTMLElement).style.borderColor = `rgba(${hexToRgb(item.color)},0.45)`;
        }}
        onBlur={e => {
          (e.currentTarget as HTMLElement).style.borderColor = `rgba(${hexToRgb(item.color)},0.16)`;
        }}
      >
        <div className="flex items-start gap-3">
          {/* Icon */}
          <div
            className="flex-none w-9 h-9 rounded-xl flex items-center justify-center mt-0.5"
            style={{ background: `rgba(${hexToRgb(item.color)},0.14)` }}
            aria-hidden="true"
          >
            <Icon size={16} style={{ color: item.color }} />
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            {/* Type badge */}
            <div
              className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[9px] font-bold uppercase tracking-wider mb-1"
              style={{ background: `rgba(${hexToRgb(item.color)},0.15)`, color: item.color }}
            >
              {typeLabel}
            </div>

            {/* Title */}
            <div className="text-[13px] font-bold leading-snug mb-1"
              style={{ color: 'var(--color-text-primary)' }}>
              {item.title}
            </div>

            {/* Why recommended */}
            <div className="flex items-center gap-1 text-[11px] mb-2"
              style={{ color: 'var(--color-text-secondary)' }}>
              <span>✦</span>
              <span>{item.reason}</span>
            </div>

            {/* Meta */}
            <div className="flex items-center gap-3 text-[10px]"
              style={{ color: 'var(--color-text-muted)' }}>
              <span className="flex items-center gap-1">
                <Clock size={9} aria-hidden="true" />
                {item.estimatedMinutes} min
              </span>
              <span>{item.domain}</span>
            </div>
          </div>

          {/* Confidence ring */}
          <ConfidenceRing confidence={item.confidence} color={item.color} />
        </div>
      </Link>
    </motion.div>
  );
}

export function Recommendations({ items }: RecommendationsProps) {
  return (
    <section aria-labelledby="rec-heading" className="mb-10">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 id="rec-heading" className="text-lg font-black" style={{ color: 'var(--color-text-primary)' }}>
            Recommended For You
          </h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
            Personalized based on your progress and interests
          </p>
        </div>
      </div>

      {items.length === 0 ? (
        <div
          className="rounded-2xl px-6 py-10 text-center"
          style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
        >
          <p className="text-sm" style={{ color: 'var(--color-text-tertiary)' }}>
            Complete a few lessons to unlock personalized recommendations.
          </p>
        </div>
      ) : (
        <div
          className="grid grid-cols-1 sm:grid-cols-2 gap-3"
          role="list"
          aria-label="Recommended content"
        >
          {items.map((item, i) => (
            <div key={item.id} role="listitem">
              <RecommendationCard item={item} index={i} />
            </div>
          ))}
        </div>
      )}
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
