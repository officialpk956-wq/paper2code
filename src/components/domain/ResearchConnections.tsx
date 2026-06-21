'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { FileText, ArrowRight, Star } from 'lucide-react';
import type { DomainPaper } from '@/data/domains/types';

interface ResearchConnectionsProps {
  papers: DomainPaper[];
}

const IMPACT_CONFIG = {
  landmark: { label: 'Landmark', className: 'badge-expert' },
  high:     { label: 'High Impact', className: 'badge-hard' },
  medium:   { label: 'Notable', className: 'badge-medium' },
};

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

function formatCitations(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(0)}k`;
  return String(n);
}

function PaperCard({ paper, index }: { paper: DomainPaper; index: number }) {
  const impact = IMPACT_CONFIG[paper.impact];
  const rgb = hexToRgb(paper.color);

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.38, delay: index * 0.07 }}
    >
      <div
        className="rounded-2xl p-4 h-full transition-all duration-200"
        style={{
          background: 'var(--bg-surface)',
          border: `1px solid rgba(${rgb},0.16)`,
        }}
        onMouseEnter={e => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = `rgba(${rgb},0.38)`;
          el.style.background = 'var(--bg-hover)';
          el.style.boxShadow = `0 0 20px rgba(${rgb},0.1)`;
        }}
        onMouseLeave={e => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = `rgba(${rgb},0.16)`;
          el.style.background = 'var(--bg-surface)';
          el.style.boxShadow = 'none';
        }}
      >
        {/* Header */}
        <div className="flex items-start gap-3 mb-3">
          <div
            className="flex-none w-9 h-9 rounded-xl flex items-center justify-center mt-0.5"
            style={{ background: `rgba(${rgb},0.14)` }}
            aria-hidden="true"
          >
            <FileText size={16} style={{ color: paper.color }} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1 flex-wrap">
              <span className={impact.className}>{impact.label}</span>
              <span className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
                {paper.year}
              </span>
            </div>
            <div className="text-[13px] font-bold leading-snug" style={{ color: 'var(--color-text-primary)' }}>
              {paper.title}
            </div>
          </div>
        </div>

        {/* Authors */}
        <div className="text-[11px] mb-2" style={{ color: paper.color }}>
          {paper.authors}
        </div>

        {/* Description */}
        <p className="text-[11px] leading-relaxed line-clamp-2 mb-4" style={{ color: 'var(--color-text-secondary)' }}>
          {paper.description}
        </p>

        {/* Footer */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
            <Star size={9} aria-hidden="true" />
            <span>{formatCitations(paper.citations)} citations</span>
          </div>
          <Link
            href={paper.href}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-bold text-white transition-opacity hover:opacity-80 focus-visible:outline-none focus-visible:ring-2"
            style={{ background: paper.color }}
            aria-label={`Study paper: ${paper.title}`}
          >
            Study Paper <ArrowRight size={10} aria-hidden="true" />
          </Link>
        </div>
      </div>
    </motion.div>
  );
}

export function ResearchConnections({ papers }: ResearchConnectionsProps) {
  return (
    <section aria-labelledby="research-heading" className="mb-10">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 id="research-heading" className="text-lg font-black" style={{ color: 'var(--color-text-primary)' }}>
            Research Connections
          </h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
            Foundational and landmark papers in this domain
          </p>
        </div>
        <Link
          href="/papers"
          className="text-xs font-semibold transition-opacity hover:opacity-80"
          style={{ color: 'var(--accent-primary)' }}
        >
          Browse all papers →
        </Link>
      </div>

      {papers.length === 0 ? (
        <div
          className="rounded-2xl px-6 py-10 text-center"
          style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
        >
          <p className="text-sm" style={{ color: 'var(--color-text-tertiary)' }}>
            No papers linked to this domain yet.
          </p>
        </div>
      ) : (
        <div
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
          role="list"
          aria-label="Research papers"
        >
          {papers.map((paper, i) => (
            <div key={paper.id} role="listitem">
              <PaperCard paper={paper} index={i} />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
