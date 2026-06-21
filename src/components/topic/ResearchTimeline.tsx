'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { TimelineEvent } from '@/data/topics/types';

interface ResearchTimelineProps {
  events: TimelineEvent[];
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

export function ResearchTimeline({ events }: ResearchTimelineProps) {
  const [selectedId, setSelectedId] = useState<string | null>(
    events.find(e => e.isHighlighted)?.id ?? events[0]?.id ?? null
  );

  const selectedEvent = events.find(e => e.id === selectedId) ?? null;

  return (
    <section id="timeline" aria-labelledby="timeline-heading" className="mb-8 scroll-mt-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <h2 id="timeline-heading" className="text-lg font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
          Research Timeline
        </h2>
        <p className="text-xs mb-5" style={{ color: 'var(--color-text-secondary)' }}>
          Click any milestone to read the summary.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.45, delay: 0.1 }}
        className="rounded-2xl overflow-hidden"
        style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.07)' }}
      >
        <div className="flex flex-col lg:flex-row">
          {/* Timeline column */}
          <div
            className="lg:w-52 flex-none p-4"
            style={{ borderRight: '1px solid rgba(255,255,255,0.06)' }}
          >
            <div className="relative" role="list" aria-label="Timeline milestones">
              {/* Vertical line */}
              <div
                className="absolute left-[18px] top-4 bottom-4 w-px"
                style={{ background: 'rgba(255,255,255,0.08)' }}
                aria-hidden="true"
              />

              {events.map((event, i) => {
                const isSelected = selectedId === event.id;
                const rgb = hexToRgb(event.color);

                return (
                  <motion.button
                    key={event.id}
                    role="listitem"
                    initial={{ opacity: 0, x: -8 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.3, delay: i * 0.07 }}
                    onClick={() => setSelectedId(event.id)}
                    className="relative w-full flex items-start gap-3 text-left py-2 pl-0 pr-2 rounded-xl transition-all focus-visible:outline-none focus-visible:ring-1"
                    style={{
                      background: isSelected ? `rgba(${rgb},0.1)` : 'transparent',
                    }}
                    aria-selected={isSelected}
                    aria-label={`${event.year}: ${event.title}${event.isHighlighted ? ' — landmark' : ''}`}
                  >
                    {/* Dot */}
                    <div
                      className="relative z-10 flex-none w-9 flex justify-center pt-1"
                      aria-hidden="true"
                    >
                      <div
                        className="rounded-full transition-all"
                        style={{
                          width: event.isHighlighted ? 14 : 10,
                          height: event.isHighlighted ? 14 : 10,
                          background: isSelected ? event.color : `rgba(${rgb},0.45)`,
                          border: `2px solid ${isSelected ? event.color : `rgba(${rgb},0.25)`}`,
                          boxShadow: isSelected ? `0 0 8px rgba(${rgb},0.5)` : 'none',
                        }}
                      />
                    </div>

                    <div className="flex-1 min-w-0 pb-1">
                      <div
                        className="text-[11px] font-black tabular-nums"
                        style={{ color: isSelected ? event.color : 'var(--color-text-muted)' }}
                      >
                        {event.year}
                      </div>
                      <div
                        className="text-[12px] font-semibold leading-tight mt-0.5"
                        style={{ color: isSelected ? 'var(--color-text-primary)' : 'var(--color-text-secondary)' }}
                      >
                        {event.title}
                      </div>
                      {event.isHighlighted && (
                        <span
                          className="inline-block mt-1 text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full"
                          style={{ background: `rgba(${rgb},0.18)`, color: event.color }}
                        >
                          Landmark
                        </span>
                      )}
                    </div>
                  </motion.button>
                );
              })}
            </div>
          </div>

          {/* Detail panel */}
          <div className="flex-1 p-5 min-w-0">
            <AnimatePresence mode="wait">
              {selectedEvent ? (
                <motion.div
                  key={selectedEvent.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                  aria-live="polite"
                  aria-atomic="true"
                >
                  <div className="flex items-center gap-3 mb-4">
                    <span
                      className="text-2xl font-black tabular-nums"
                      style={{ color: selectedEvent.color }}
                    >
                      {selectedEvent.year}
                    </span>
                    {selectedEvent.isHighlighted && (
                      <span
                        className="text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-full"
                        style={{
                          background: `rgba(${hexToRgb(selectedEvent.color)},0.12)`,
                          color: selectedEvent.color,
                        }}
                      >
                        Landmark paper
                      </span>
                    )}
                  </div>
                  <h3
                    className="text-[15px] font-black mb-3"
                    style={{ color: 'var(--color-text-primary)' }}
                  >
                    {selectedEvent.title}
                  </h3>
                  <p
                    className="text-sm leading-relaxed"
                    style={{ color: 'var(--color-text-secondary)' }}
                  >
                    {selectedEvent.summary}
                  </p>
                </motion.div>
              ) : (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex items-center justify-center h-full"
                >
                  <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    Select a milestone to read more
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </motion.div>
    </section>
  );
}
