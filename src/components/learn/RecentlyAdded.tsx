'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { BookOpen, FileText, Layers, Code2, Clock } from 'lucide-react';
import type { RecentlyAddedItem } from '@/data/learn/recommendations';

const TYPE_CONFIG: Record<
  RecentlyAddedItem['type'],
  { icon: React.ElementType; label: string; className: string }
> = {
  lesson:       { icon: BookOpen, label: 'Lesson',       className: 'badge-easy' },
  paper:        { icon: FileText, label: 'Paper',        className: 'badge-hard' },
  architecture: { icon: Layers,   label: 'Architecture', className: 'badge-medium' },
  project:      { icon: Code2,    label: 'Project',      className: 'badge-expert' },
};

function formatRelativeDate(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7)  return `${days} days ago`;
  if (days < 14) return '1 week ago';
  return `${Math.floor(days / 7)} weeks ago`;
}

interface RecentlyAddedProps {
  items: RecentlyAddedItem[];
}

export function RecentlyAdded({ items }: RecentlyAddedProps) {
  return (
    <section aria-labelledby="recent-heading" className="mb-10">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 id="recent-heading" className="text-lg font-black" style={{ color: 'var(--color-text-primary)' }}>
            Recently Added
          </h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
            Fresh content added to the curriculum
          </p>
        </div>
        <Link
          href="/learn/new"
          className="text-xs font-semibold transition-opacity hover:opacity-80"
          style={{ color: 'var(--accent-primary)' }}
        >
          View all →
        </Link>
      </div>

      {items.length === 0 ? (
        <div
          className="rounded-2xl px-6 py-10 text-center"
          style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
        >
          <p className="text-sm" style={{ color: 'var(--color-text-tertiary)' }}>
            No new content has been added recently.
          </p>
        </div>
      ) : (
        <div
          className="relative"
          role="list"
          aria-label="Recently added content"
        >
          {/* Timeline line */}
          <div
            className="absolute left-[17px] top-0 bottom-0 w-px"
            style={{ background: 'rgba(255,255,255,0.07)' }}
            aria-hidden="true"
          />

          <div className="flex flex-col gap-3">
            {items.map((item, i) => {
              const { icon: Icon, label, className } = TYPE_CONFIG[item.type];
              return (
                <motion.div
                  key={item.id}
                  role="listitem"
                  initial={{ opacity: 0, x: -10 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.35, delay: i * 0.06 }}
                  className="flex gap-4 items-start"
                >
                  {/* Timeline dot */}
                  <div
                    className="relative flex-none w-[35px] flex items-center justify-center pt-3"
                    aria-hidden="true"
                  >
                    <div
                      className="w-[22px] h-[22px] rounded-full flex items-center justify-center"
                      style={{ background: `rgba(${hexToRgb(item.color)},0.2)`, border: `2px solid ${item.color}` }}
                    >
                      <Icon size={10} style={{ color: item.color }} />
                    </div>
                  </div>

                  {/* Card */}
                  <Link
                    href={item.href}
                    className="flex-1 min-w-0 rounded-2xl p-4 transition-all duration-200 focus-visible:outline-none focus-visible:ring-2"
                    style={{
                      background: 'var(--bg-surface)',
                      border: '1px solid rgba(255,255,255,0.06)',
                    }}
                    aria-label={`${item.title}: ${label} in ${item.domain}, ${item.readMinutes} min read`}
                    onMouseEnter={e => {
                      const el = e.currentTarget as HTMLElement;
                      el.style.borderColor = `rgba(${hexToRgb(item.color)},0.32)`;
                      el.style.background = 'var(--bg-hover)';
                    }}
                    onMouseLeave={e => {
                      const el = e.currentTarget as HTMLElement;
                      el.style.borderColor = 'rgba(255,255,255,0.06)';
                      el.style.background = 'var(--bg-surface)';
                    }}
                    onFocus={e => {
                      (e.currentTarget as HTMLElement).style.borderColor = `rgba(${hexToRgb(item.color)},0.45)`;
                    }}
                    onBlur={e => {
                      (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.06)';
                    }}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        {/* Badges */}
                        <div className="flex items-center gap-2 mb-1.5">
                          <span className={className}>{label}</span>
                          <span className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
                            {item.domain}
                          </span>
                        </div>

                        {/* Title */}
                        <div className="text-[13px] font-bold leading-snug mb-1"
                          style={{ color: 'var(--color-text-primary)' }}>
                          {item.title}
                        </div>

                        {/* Description */}
                        <p className="text-[11px] leading-relaxed line-clamp-1"
                          style={{ color: 'var(--color-text-secondary)' }}>
                          {item.description}
                        </p>
                      </div>

                      {/* Meta */}
                      <div className="flex-none flex flex-col items-end gap-1 text-[10px]"
                        style={{ color: 'var(--color-text-muted)' }}>
                        <span>{formatRelativeDate(item.addedAt)}</span>
                        <div className="flex items-center gap-1">
                          <Clock size={9} aria-hidden="true" />
                          <span>{item.readMinutes} min</span>
                        </div>
                      </div>
                    </div>
                  </Link>
                </motion.div>
              );
            })}
          </div>
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
