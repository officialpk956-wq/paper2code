'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, CheckCircle, XCircle } from 'lucide-react';
import Link from 'next/link';
import type { PracticeQuestion, QuestionType } from '@/data/topics/types';

interface PracticePreviewProps {
  questions: PracticeQuestion[];
  domainSlug: string;
  topicSlug: string;
}

const TYPE_CONFIG: Record<QuestionType, { label: string; bg: string; color: string }> = {
  mcq:      { label: 'MCQ',      bg: 'rgba(6,182,212,0.12)',   color: 'var(--accent-cyan)' },
  math:     { label: 'Math',     bg: 'rgba(167,139,250,0.12)', color: 'var(--accent-light)' },
  coding:   { label: 'Coding',   bg: 'rgba(16,185,129,0.12)',  color: 'var(--accent-emerald)' },
  research: { label: 'Research', bg: 'rgba(245,158,11,0.12)',  color: 'var(--accent-amber)' },
};

function MCQQuestion({ q }: { q: PracticeQuestion }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);

  return (
    <div>
      <p className="text-sm font-semibold mb-3" style={{ color: 'var(--color-text-primary)' }}>
        {q.prompt}
      </p>
      <div className="flex flex-col gap-2" role="radiogroup" aria-label="Answer options">
        {q.options?.map(opt => {
          const isSelected = selected === opt.id;
          const isCorrect = opt.id === q.correctAnswer;
          let border = 'rgba(255,255,255,0.08)';
          let bg = 'transparent';
          let color = 'var(--color-text-secondary)';

          if (isSelected && !revealed) {
            border = 'rgba(124,58,237,0.4)';
            bg = 'rgba(124,58,237,0.1)';
            color = 'var(--color-text-primary)';
          } else if (revealed && isCorrect) {
            border = 'rgba(16,185,129,0.45)';
            bg = 'rgba(16,185,129,0.1)';
            color = '#10B981';
          } else if (revealed && isSelected && !isCorrect) {
            border = 'rgba(239,68,68,0.4)';
            bg = 'rgba(239,68,68,0.08)';
            color = '#EF4444';
          }

          return (
            <button
              key={opt.id}
              role="radio"
              aria-checked={isSelected}
              onClick={() => { setSelected(opt.id); setRevealed(false); }}
              className="flex items-center gap-3 px-3 py-2 rounded-xl text-left text-xs font-medium transition-all focus-visible:outline-none focus-visible:ring-1"
              style={{ background: bg, border: `1px solid ${border}`, color }}
            >
              <span
                className="flex-none w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-black"
                style={{ background: isSelected ? 'rgba(124,58,237,0.2)' : 'rgba(255,255,255,0.06)', color: isSelected ? 'var(--accent-light)' : 'var(--color-text-muted)' }}
                aria-hidden="true"
              >
                {opt.id.toUpperCase()}
              </span>
              {opt.text}
              {revealed && isCorrect && <CheckCircle size={12} style={{ color: '#10B981', marginLeft: 'auto' }} aria-hidden="true" />}
              {revealed && isSelected && !isCorrect && <XCircle size={12} style={{ color: '#EF4444', marginLeft: 'auto' }} aria-hidden="true" />}
            </button>
          );
        })}
      </div>
      {selected && (
        <button
          onClick={() => setRevealed(true)}
          className="mt-3 text-xs font-semibold px-3 py-1.5 rounded-lg transition-all focus-visible:outline-none focus-visible:ring-1"
          style={{ background: 'rgba(124,58,237,0.14)', color: 'var(--accent-primary)' }}
          aria-label="Reveal correct answer"
        >
          Check answer
        </button>
      )}
      {revealed && q.hint && (
        <div
          className="mt-2 text-xs px-3 py-2 rounded-xl leading-relaxed"
          style={{ background: 'rgba(245,158,11,0.08)', color: 'var(--accent-amber)' }}
          role="alert"
        >
          💡 {q.hint}
        </div>
      )}
    </div>
  );
}

function TextQuestion({ q }: { q: PracticeQuestion }) {
  const [showHint, setShowHint] = useState(false);

  return (
    <div>
      <p className="text-sm font-semibold mb-3" style={{ color: 'var(--color-text-primary)' }}>
        {q.prompt}
      </p>
      {q.hint && (
        <>
          <button
            onClick={() => setShowHint(v => !v)}
            className="text-xs font-semibold px-3 py-1.5 rounded-lg transition-all focus-visible:outline-none focus-visible:ring-1"
            style={{ background: 'rgba(245,158,11,0.1)', color: 'var(--accent-amber)' }}
            aria-expanded={showHint}
          >
            {showHint ? 'Hide hint' : 'Show hint'}
          </button>
          {showHint && (
            <div
              className="mt-2 text-xs px-3 py-2 rounded-xl leading-relaxed"
              style={{ background: 'rgba(245,158,11,0.08)', color: 'var(--accent-amber)' }}
              role="alert"
            >
              💡 {q.hint}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function PracticePreview({ questions, domainSlug, topicSlug }: PracticePreviewProps) {
  return (
    <section id="practice" aria-labelledby="practice-heading" className="mb-8 scroll-mt-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <div className="flex items-end justify-between mb-5">
          <div>
            <h2 id="practice-heading" className="text-lg font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
              Practice
            </h2>
            <p className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
              A preview of question types — {questions.length} shown here.
            </p>
          </div>
          <Link
            href={`/dojo?domain=${domainSlug}&topic=${topicSlug}`}
            className="flex items-center gap-1.5 text-xs font-semibold px-3 py-2 rounded-xl transition-all focus-visible:outline-none focus-visible:ring-2"
            style={{ background: 'rgba(124,58,237,0.14)', color: 'var(--accent-primary)' }}
            aria-label="Open full practice in coding dojo"
          >
            Full Practice
            <ArrowRight size={11} aria-hidden="true" />
          </Link>
        </div>
      </motion.div>

      <div className="flex flex-col gap-4" role="list" aria-label="Practice questions preview">
        {questions.map((q, i) => {
          const typeCfg = TYPE_CONFIG[q.type];

          return (
            <motion.div
              key={q.id}
              role="listitem"
              initial={{ opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.35, delay: i * 0.07 }}
              className="rounded-2xl p-4"
              style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.07)' }}
            >
              {/* Type badge + question number */}
              <div className="flex items-center gap-2 mb-3">
                <span
                  className="text-[10px] font-bold px-2 py-0.5 rounded-full"
                  style={{ background: typeCfg.bg, color: typeCfg.color }}
                >
                  {typeCfg.label}
                </span>
                <span className="text-[10px] font-mono" style={{ color: 'var(--color-text-muted)' }}>
                  Q{i + 1}
                </span>
              </div>

              {q.type === 'mcq' ? (
                <MCQQuestion q={q} />
              ) : (
                <TextQuestion q={q} />
              )}
            </motion.div>
          );
        })}
      </div>
    </section>
  );
}
