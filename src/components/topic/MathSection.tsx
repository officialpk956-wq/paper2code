'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import type { Formula } from '@/data/topics/types';

interface MathSectionProps {
  formulas: Formula[];
}

function renderKaTeX(latex: string): string {
  try {
    // Dynamic import not needed since katex is synchronous
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const katex = require('katex') as typeof import('katex');
    return katex.renderToString(latex, { throwOnError: false, displayMode: true });
  } catch {
    return `<span>${latex}</span>`;
  }
}

function FormulaCard({ formula, index }: { formula: Formula; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const [derivationOpen, setDerivationOpen] = useState(false);
  const [rendered, setRendered] = useState('');

  useEffect(() => {
    setRendered(renderKaTeX(formula.latex));
  }, [formula.latex]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: index * 0.08 }}
      className="rounded-2xl overflow-hidden"
      style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.07)' }}
    >
      {/* Header */}
      <button
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center justify-between px-5 py-4 text-left focus-visible:outline-none focus-visible:ring-inset focus-visible:ring-2 transition-colors"
        style={{ background: expanded ? 'var(--bg-hover)' : 'transparent' }}
        aria-expanded={expanded}
        aria-controls={`formula-body-${formula.id}`}
      >
        <div>
          <div className="text-[13px] font-bold" style={{ color: 'var(--color-text-primary)' }}>
            {formula.name}
          </div>
          <div className="text-[11px] mt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
            {formula.intuition.slice(0, 60)}…
          </div>
        </div>
        <ChevronDown
          size={14}
          style={{
            color: 'var(--color-text-muted)',
            transform: expanded ? 'rotate(180deg)' : 'rotate(0)',
            transition: 'transform 0.2s',
          }}
          aria-hidden="true"
        />
      </button>

      {/* LaTeX display (always visible) */}
      <div
        className="px-5 py-4 text-center overflow-x-auto"
        style={{ borderTop: '1px solid rgba(255,255,255,0.06)', borderBottom: '1px solid rgba(255,255,255,0.06)' }}
        aria-label={`Formula: ${formula.name}`}
      >
        {rendered ? (
          <div
            className="katex-display-override"
            dangerouslySetInnerHTML={{ __html: rendered }}
            style={{ color: 'var(--color-text-primary)' }}
          />
        ) : (
          <code className="text-sm" style={{ color: 'var(--accent-light)' }}>{formula.latex}</code>
        )}
      </div>

      {/* Expanded content */}
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            id={`formula-body-${formula.id}`}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="px-5 py-4 flex flex-col gap-4">
              {/* Intuition */}
              <div>
                <div className="text-[10px] font-bold uppercase tracking-wider mb-1.5" style={{ color: 'var(--color-text-muted)' }}>
                  Intuition
                </div>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                  {formula.intuition}
                </p>
              </div>

              {/* Variable breakdown */}
              <div>
                <div className="text-[10px] font-bold uppercase tracking-wider mb-2" style={{ color: 'var(--color-text-muted)' }}>
                  Variable Breakdown
                </div>
                <div className="flex flex-col gap-2" role="list" aria-label="Formula variables">
                  {formula.variables.map(v => (
                    <div key={v.symbol} role="listitem" className="flex items-start gap-3">
                      <code
                        className="flex-none px-2 py-0.5 rounded text-xs font-bold"
                        style={{ background: 'rgba(167,139,250,0.12)', color: 'var(--accent-light)', fontFamily: 'var(--font-mono)' }}
                        aria-label={`Variable: ${v.symbol}`}
                      >
                        {v.symbol}
                      </code>
                      <span className="text-xs leading-relaxed pt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
                        {v.meaning}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Derivation */}
              {formula.derivationSteps && formula.derivationSteps.length > 0 && (
                <div>
                  <button
                    onClick={() => setDerivationOpen(v => !v)}
                    className="flex items-center gap-2 text-[11px] font-semibold mb-2 focus-visible:outline-none focus-visible:ring-1 rounded"
                    style={{ color: 'var(--accent-cyan)' }}
                    aria-expanded={derivationOpen}
                  >
                    <ChevronDown
                      size={12}
                      style={{ transform: derivationOpen ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform 0.2s' }}
                      aria-hidden="true"
                    />
                    Derivation steps
                  </button>
                  <AnimatePresence initial={false}>
                    {derivationOpen && (
                      <motion.ol
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden flex flex-col gap-2"
                        aria-label="Derivation steps"
                      >
                        {formula.derivationSteps.map((step, i) => (
                          <li key={i} className="flex items-start gap-2 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                            <span
                              className="flex-none w-5 h-5 rounded-full text-[9px] font-black flex items-center justify-center mt-0.5"
                              style={{ background: 'rgba(6,182,212,0.14)', color: 'var(--accent-cyan)' }}
                              aria-hidden="true"
                            >
                              {i + 1}
                            </span>
                            <code style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}>{step}</code>
                          </li>
                        ))}
                      </motion.ol>
                    )}
                  </AnimatePresence>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export function MathSection({ formulas }: MathSectionProps) {
  return (
    <section id="mathematics" aria-labelledby="math-heading" className="mb-8 scroll-mt-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <h2 id="math-heading" className="text-lg font-black mb-1" style={{ color: 'var(--color-text-primary)' }}>
          Mathematics
        </h2>
        <p className="text-xs mb-5" style={{ color: 'var(--color-text-secondary)' }}>
          Click any formula to expand its variable breakdown and derivation.
        </p>
      </motion.div>

      <div className="flex flex-col gap-3" role="list" aria-label="Mathematical formulas">
        {formulas.map((formula, i) => (
          <div key={formula.id} role="listitem">
            <FormulaCard formula={formula} index={i} />
          </div>
        ))}
      </div>
    </section>
  );
}
