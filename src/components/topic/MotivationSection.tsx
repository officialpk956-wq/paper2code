'use client';

import { motion } from 'framer-motion';
import { AlertTriangle, TrendingDown, Zap, ArrowDown } from 'lucide-react';
import type { MotivationCard } from '@/data/topics/types';

interface MotivationSectionProps {
  cards: MotivationCard[];
}

const CARD_CONFIG = {
  problem: {
    border: 'rgba(239,68,68,0.25)',
    bg: 'rgba(239,68,68,0.07)',
    accent: '#EF4444',
    label: 'Problem',
  },
  limitation: {
    border: 'rgba(245,158,11,0.25)',
    bg: 'rgba(245,158,11,0.07)',
    accent: '#F59E0B',
    label: 'Limitation',
  },
  solution: {
    border: 'rgba(16,185,129,0.25)',
    bg: 'rgba(16,185,129,0.07)',
    accent: '#10B981',
    label: 'Solution',
  },
};

const ICON_MAP: Record<string, React.FC<{ size: number; style?: React.CSSProperties }>> = {
  AlertTriangle,
  TrendingDown,
  Zap,
};

function SectionDivider() {
  return (
    <div className="flex justify-center" aria-hidden="true">
      <ArrowDown size={16} style={{ color: 'var(--color-text-muted)' }} />
    </div>
  );
}

export function MotivationSection({ cards }: MotivationSectionProps) {
  return (
    <section id="motivation" aria-labelledby="motivation-heading" className="mb-8 scroll-mt-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <h2 id="motivation-heading" className="text-lg font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
          Motivation
        </h2>
        <p className="text-xs mb-5" style={{ color: 'var(--color-text-secondary)' }}>
          Why was this invented? What problems does it solve?
        </p>
      </motion.div>

      <div className="flex flex-col gap-3" role="list" aria-label="Problem, limitation, and solution">
        {cards.map((card, i) => {
          const cfg = CARD_CONFIG[card.type];
          const IconComp = ICON_MAP[card.icon];

          return (
            <div key={card.type} role="listitem">
              {i > 0 && <SectionDivider />}
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.1 }}
                className="rounded-2xl p-5"
                style={{ background: cfg.bg, border: `1px solid ${cfg.border}` }}
              >
                <div className="flex items-start gap-4">
                  {IconComp && (
                    <div
                      className="flex-none w-9 h-9 rounded-xl flex items-center justify-center mt-0.5"
                      style={{ background: `${cfg.accent}20` }}
                      aria-hidden="true"
                    >
                      <IconComp size={16} style={{ color: cfg.accent }} />
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1.5">
                      <span
                        className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full"
                        style={{ background: `${cfg.accent}20`, color: cfg.accent }}
                      >
                        {cfg.label}
                      </span>
                      <span className="text-[13px] font-bold" style={{ color: 'var(--color-text-primary)' }}>
                        {card.title}
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                      {card.body}
                    </p>
                  </div>
                </div>
              </motion.div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
