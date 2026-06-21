'use client';

import { motion } from 'framer-motion';
import type { ApplicationCard } from '@/data/topics/types';

interface ApplicationsSectionProps {
  applications: ApplicationCard[];
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

export function ApplicationsSection({ applications }: ApplicationsSectionProps) {
  return (
    <section id="applications" aria-labelledby="applications-heading" className="mb-8 scroll-mt-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <h2 id="applications-heading" className="text-lg font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
          Real-World Applications
        </h2>
        <p className="text-xs mb-5" style={{ color: 'var(--color-text-secondary)' }}>
          Where this concept is used in production systems.
        </p>
      </motion.div>

      <div
        className="grid grid-cols-1 sm:grid-cols-2 gap-3"
        role="list"
        aria-label="Application examples"
      >
        {applications.map((app, i) => {
          const rgb = hexToRgb(app.color);
          return (
            <motion.div
              key={app.name}
              role="listitem"
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.35, delay: i * 0.06 }}
              className="rounded-2xl p-4 flex flex-col gap-3"
              style={{
                background: 'var(--bg-surface)',
                border: `1px solid rgba(${rgb},0.16)`,
              }}
            >
              {/* Header */}
              <div className="flex items-start gap-3">
                <div
                  className="w-9 h-9 rounded-xl flex items-center justify-center text-lg flex-none"
                  style={{ background: `rgba(${rgb},0.14)` }}
                  aria-hidden="true"
                >
                  {app.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-[13px] font-black" style={{ color: 'var(--color-text-primary)' }}>
                    {app.name}
                  </div>
                  <div className="text-[11px] mt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
                    {app.description}
                  </div>
                </div>
              </div>

              {/* Where / Why */}
              <div className="flex flex-col gap-2">
                <div
                  className="flex items-start gap-2 p-2.5 rounded-xl text-xs"
                  style={{ background: `rgba(${rgb},0.06)` }}
                >
                  <span className="font-bold flex-none" style={{ color: app.color }}>Where:</span>
                  <span style={{ color: 'var(--color-text-secondary)' }}>{app.whereUsed}</span>
                </div>
                <div
                  className="flex items-start gap-2 p-2.5 rounded-xl text-xs"
                  style={{ background: 'rgba(255,255,255,0.03)' }}
                >
                  <span className="font-bold flex-none" style={{ color: 'var(--accent-emerald)' }}>Why:</span>
                  <span style={{ color: 'var(--color-text-secondary)' }}>{app.whyUseful}</span>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </section>
  );
}
