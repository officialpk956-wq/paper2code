'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import {
  Sigma, BarChart2, Brain, Layers3, MessageSquare, Eye,
  Sparkles, Bot, Database, Target, Settings2, Microscope, Cpu,
} from 'lucide-react';
import type { Domain } from '@/data/learn/domains';

const ICON_MAP: Record<string, React.ElementType> = {
  Sigma, BarChart2, Brain, Layers3, MessageSquare, Eye,
  Sparkles, Bot, Database, Target, Settings2, Microscope, Cpu,
};

interface DomainGridProps {
  domains: Domain[];
}

function ProgressRing({ progress, color, size = 36 }: { progress: number; color: string; size?: number }) {
  const r = (size - 6) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, progress));
  return (
    <svg width={size} height={size} aria-hidden="true">
      <circle cx={size / 2} cy={size / 2} r={r}
        fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="3" />
      <motion.circle
        cx={size / 2} cy={size / 2} r={r}
        fill="none" stroke={color} strokeWidth="3"
        strokeLinecap="round"
        strokeDasharray={c}
        initial={{ strokeDashoffset: c }}
        whileInView={{ strokeDashoffset: c - (pct / 100) * c }}
        viewport={{ once: true }}
        transition={{ duration: 0.7, ease: 'easeOut' }}
        style={{ transform: 'rotate(-90deg)', transformOrigin: `${size / 2}px ${size / 2}px` }}
      />
      <text
        x={size / 2} y={size / 2 + 3.5}
        textAnchor="middle"
        fontSize="8"
        fontWeight="700"
        fill={color}
        fontFamily="system-ui, sans-serif"
      >
        {pct > 0 ? `${pct}%` : '—'}
      </text>
    </svg>
  );
}

function DomainCard({ domain, index }: { domain: Domain; index: number }) {
  const Icon = ICON_MAP[domain.icon] ?? Brain;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.35, delay: Math.min(index * 0.045, 0.4) }}
    >
      <Link
        href={domain.href}
        className="block rounded-2xl p-4 h-full transition-all duration-200 focus-visible:outline-none focus-visible:ring-2"
        style={{
          background: 'var(--bg-surface)',
          border: '1px solid rgba(255,255,255,0.06)',
        }}
        aria-label={`${domain.name}: ${domain.lessonCount} lessons, ${domain.progress}% complete`}
        onMouseEnter={e => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = `rgba(${hexToRgb(domain.color)},0.35)`;
          el.style.background = 'var(--bg-hover)';
        }}
        onMouseLeave={e => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = 'rgba(255,255,255,0.06)';
          el.style.background = 'var(--bg-surface)';
        }}
        onFocus={e => {
          (e.currentTarget as HTMLElement).style.borderColor = `rgba(${hexToRgb(domain.color)},0.45)`;
        }}
        onBlur={e => {
          (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.06)';
        }}
      >
        {/* Header: icon + progress ring */}
        <div className="flex items-start justify-between mb-3">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: `rgba(${hexToRgb(domain.color)},0.14)` }}
            aria-hidden="true"
          >
            <Icon size={16} style={{ color: domain.color }} />
          </div>
          <ProgressRing progress={domain.progress} color={domain.color} size={36} />
        </div>

        {/* Name + description */}
        <div className="text-[13px] font-bold mb-1 leading-snug"
          style={{ color: 'var(--color-text-primary)' }}>
          {domain.name}
        </div>
        <p className="text-[11px] leading-relaxed line-clamp-2"
          style={{ color: 'var(--color-text-secondary)' }}>
          {domain.description}
        </p>

        {/* Footer: topic count */}
        <div className="mt-3 text-[10px] font-medium" style={{ color: 'var(--color-text-muted)' }}>
          {domain.topicCount} topics · {domain.lessonCount} lessons
        </div>
      </Link>
    </motion.div>
  );
}

export function DomainGrid({ domains }: DomainGridProps) {
  return (
    <section id="domains" aria-labelledby="domains-heading" className="mb-10">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 id="domains-heading" className="text-lg font-black" style={{ color: 'var(--color-text-primary)' }}>
            Learning Domains
          </h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
            Explore the full AI engineering curriculum
          </p>
        </div>
      </div>

      <div
        className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3"
        role="list"
        aria-label="Learning domains"
      >
        {domains.map((domain, i) => (
          <div key={domain.id} role="listitem">
            <DomainCard domain={domain} index={i} />
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
