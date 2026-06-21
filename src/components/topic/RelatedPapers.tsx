'use client';

import { motion } from 'framer-motion';
import { ExternalLink } from 'lucide-react';
import type { TopicPaper, PaperImpact } from '@/data/topics/types';

interface RelatedPapersProps {
  papers: TopicPaper[];
}

const IMPACT_CONFIG: Record<PaperImpact, { label: string; bg: string; color: string }> = {
  landmark: { label: 'Landmark', bg: 'rgba(245,158,11,0.14)', color: '#F59E0B' },
  high:     { label: 'High Impact', bg: 'rgba(16,185,129,0.12)', color: '#10B981' },
  medium:   { label: 'Notable', bg: 'rgba(124,58,237,0.12)', color: '#A78BFA' },
};

function formatCitations(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(n % 1000 >= 100 ? 1 : 0)}k`;
  return String(n);
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

export function RelatedPapers({ papers }: RelatedPapersProps) {
  return (
    <section id="papers" aria-labelledby="papers-heading" className="mb-8 scroll-mt-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <h2 id="papers-heading" className="text-lg font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
          Related Papers
        </h2>
        <p className="text-xs mb-5" style={{ color: 'var(--color-text-secondary)' }}>
          Seminal and recent research to deepen your understanding.
        </p>
      </motion.div>

      <div className="flex flex-col gap-3" role="list" aria-label="Related research papers">
        {papers.map((paper, i) => {
          const impact = IMPACT_CONFIG[paper.impact];
          const rgb = hexToRgb(paper.color);

          return (
            <motion.div
              key={paper.id}
              role="listitem"
              initial={{ opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.35, delay: i * 0.07 }}
              className="rounded-2xl p-4"
              style={{
                background: 'var(--bg-surface)',
                border: `1px solid rgba(${rgb},0.14)`,
              }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  {/* Badges row */}
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <span
                      className="text-[10px] font-bold px-2 py-0.5 rounded-full"
                      style={{ background: impact.bg, color: impact.color }}
                    >
                      {impact.label}
                    </span>
                    <span className="text-[10px] font-mono tabular-nums" style={{ color: 'var(--color-text-muted)' }}>
                      {paper.year}
                    </span>
                    <span
                      className="text-[10px] font-semibold tabular-nums"
                      style={{ color: 'var(--accent-amber)' }}
                      aria-label={`${paper.citations} citations`}
                    >
                      ⭐ {formatCitations(paper.citations)}
                    </span>
                  </div>

                  {/* Title */}
                  <div className="text-[13px] font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
                    {paper.title}
                  </div>

                  {/* Authors */}
                  <div className="text-[11px] mb-2" style={{ color: 'var(--color-text-muted)' }}>
                    {paper.authors}
                  </div>

                  {/* Description */}
                  <p className="text-xs leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                    {paper.description}
                  </p>
                </div>

                {/* CTA */}
                <a
                  href={paper.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex-none flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 whitespace-nowrap"
                  style={{
                    background: `rgba(${rgb},0.12)`,
                    color: paper.color,
                    border: `1px solid rgba(${rgb},0.2)`,
                  }}
                  aria-label={`Study paper: ${paper.title} (opens in new tab)`}
                >
                  Study
                  <ExternalLink size={10} aria-hidden="true" />
                </a>
              </div>
            </motion.div>
          );
        })}
      </div>
    </section>
  );
}
