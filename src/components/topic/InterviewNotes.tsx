'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, MessageSquare } from 'lucide-react';
import type { InterviewQuestion } from '@/data/topics/types';

interface InterviewNotesProps {
  questions: InterviewQuestion[];
}

export function InterviewNotes({ questions }: InterviewNotesProps) {
  const [openId, setOpenId] = useState<string | null>(null);

  const toggle = (id: string) => setOpenId(prev => prev === id ? null : id);

  return (
    <section id="interview" aria-labelledby="interview-heading" className="mb-8 scroll-mt-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <h2 id="interview-heading" className="text-lg font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
          Interview Notes
        </h2>
        <p className="text-xs mb-5" style={{ color: 'var(--color-text-secondary)' }}>
          Common interview questions with model answers. Click to reveal.
        </p>
      </motion.div>

      <div className="flex flex-col gap-2" role="list" aria-label="Interview questions">
        {questions.map((q, i) => {
          const isOpen = openId === q.id;

          return (
            <motion.div
              key={q.id}
              role="listitem"
              initial={{ opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.35, delay: i * 0.06 }}
              className="rounded-2xl overflow-hidden"
              style={{
                background: 'var(--bg-surface)',
                border: `1px solid ${isOpen ? 'rgba(124,58,237,0.25)' : 'rgba(255,255,255,0.07)'}`,
              }}
            >
              {/* Question (toggle) */}
              <button
                onClick={() => toggle(q.id)}
                className="w-full flex items-start gap-3 p-4 text-left transition-colors focus-visible:outline-none focus-visible:ring-inset focus-visible:ring-2"
                style={{ background: isOpen ? 'rgba(124,58,237,0.06)' : 'transparent' }}
                aria-expanded={isOpen}
                aria-controls={`interview-answer-${q.id}`}
              >
                <div
                  className="flex-none w-7 h-7 rounded-lg flex items-center justify-center mt-0.5"
                  style={{ background: isOpen ? 'rgba(124,58,237,0.18)' : 'rgba(255,255,255,0.06)' }}
                  aria-hidden="true"
                >
                  <MessageSquare size={12} style={{ color: isOpen ? 'var(--accent-primary)' : 'var(--color-text-muted)' }} />
                </div>
                <span
                  className="flex-1 text-[13px] font-semibold leading-relaxed"
                  style={{ color: isOpen ? 'var(--color-text-primary)' : 'var(--color-text-secondary)' }}
                >
                  {q.question}
                </span>
                <ChevronDown
                  size={14}
                  style={{
                    color: 'var(--color-text-muted)',
                    transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s',
                    flexShrink: 0,
                    marginTop: 2,
                  }}
                  aria-hidden="true"
                />
              </button>

              {/* Answer */}
              <AnimatePresence initial={false}>
                {isOpen && (
                  <motion.div
                    id={`interview-answer-${q.id}`}
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.22, ease: 'easeInOut' }}
                    className="overflow-hidden"
                  >
                    <div
                      className="px-4 pb-4"
                      style={{ borderTop: '1px solid rgba(124,58,237,0.12)' }}
                    >
                      <p
                        className="text-sm leading-relaxed pt-3 mb-3"
                        style={{ color: 'var(--color-text-secondary)' }}
                      >
                        {q.answer}
                      </p>

                      {/* Tags */}
                      {q.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                          {q.tags.map(tag => (
                            <span
                              key={tag}
                              className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
                              style={{ background: 'rgba(167,139,250,0.1)', color: 'var(--accent-light)' }}
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>
    </section>
  );
}
