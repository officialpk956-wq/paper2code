'use client';

import { motion } from 'framer-motion';
import type { SummaryPoint } from '@/data/topics/types';

interface SummarySectionProps {
  points: SummaryPoint[];
  color: string;
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

export function SummarySection({ points, color }: SummarySectionProps) {
  const rgb = hexToRgb(color);

  return (
    <section id="summary" aria-labelledby="summary-heading" className="mb-8 scroll-mt-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <h2 id="summary-heading" className="text-lg font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
          Summary
        </h2>
        <p className="text-xs mb-5" style={{ color: 'var(--color-text-secondary)' }}>
          Key takeaways from this chapter.
        </p>
      </motion.div>

      {/* Gradient banner */}
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.45, delay: 0.05 }}
        className="rounded-2xl p-5 mb-4"
        style={{
          background: `linear-gradient(135deg, rgba(${rgb},0.16) 0%, rgba(${rgb},0.04) 100%)`,
          border: `1px solid rgba(${rgb},0.22)`,
        }}
      >
        <div className="text-[13px] font-black mb-1" style={{ color }}>
          Chapter Complete
        </div>
        <p className="text-xs leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
          You&apos;ve covered motivation, intuition, mathematics, architecture, code, and applications.
          Here are the core ideas to carry forward.
        </p>
      </motion.div>

      {/* Takeaway cards */}
      <div
        className="flex flex-col gap-3"
        role="list"
        aria-label="Key takeaways"
      >
        {points.map((point, i) => (
          <motion.div
            key={i}
            role="listitem"
            initial={{ opacity: 0, x: -10 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.35, delay: i * 0.08 }}
            className="flex items-start gap-4 p-4 rounded-2xl"
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid rgba(255,255,255,0.07)',
            }}
          >
            <div
              className="flex-none w-9 h-9 rounded-xl flex items-center justify-center text-lg"
              style={{ background: `rgba(${rgb},0.12)` }}
              aria-hidden="true"
            >
              {point.icon}
            </div>
            <p className="text-sm leading-relaxed pt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
              {point.emphasis && (
                <strong style={{ color: 'var(--color-text-primary)', fontWeight: 700 }}>
                  {point.emphasis}{' '}
                </strong>
              )}
              {point.text}
            </p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
