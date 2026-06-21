'use client';

import { motion } from 'framer-motion';
import { Check, X } from 'lucide-react';

const ROWS = [
  { feature: 'Theory & Foundations', traditional: true, p2c: true },
  { feature: 'Research Paper Understanding', traditional: false, p2c: true },
  { feature: 'Implementation from Scratch', traditional: false, p2c: true },
  { feature: 'Interactive Visual Learning', traditional: false, p2c: true },
  { feature: 'Production System Design', traditional: false, p2c: true },
  { feature: 'Architecture Exploration', traditional: false, p2c: true },
  { feature: 'Coding Practice', traditional: true, p2c: true },
  { feature: 'Interview Preparation', traditional: true, p2c: true },
  { feature: 'Guided Learning Roadmaps', traditional: false, p2c: true },
  { feature: 'Math & Equation Derivations', traditional: false, p2c: true },
];

export function ComparisonTable() {
  return (
    <section className="py-28 relative overflow-hidden" style={{ background: '#060912' }}>
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-violet-500/40 to-transparent" />

      <div className="max-w-4xl mx-auto px-6 md:px-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="text-center mb-16"
        >
          <div
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold mb-6"
            style={{ background: 'rgba(139,92,246,0.12)', border: '1px solid rgba(139,92,246,0.3)', color: '#C4B5FD' }}
          >
            Why Paper2Code?
          </div>
          <h2 className="text-4xl md:text-5xl font-heading font-black text-white mb-4">
            Not just another learning platform
          </h2>
          <p className="text-lg text-slate-400 max-w-xl mx-auto">
            Traditional courses stop at theory. We take you all the way to production.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="rounded-2xl overflow-hidden"
          style={{ border: '1px solid rgba(255,255,255,0.08)' }}
        >
          {/* Header */}
          <div
            className="grid grid-cols-3 px-6 py-4 text-xs font-bold uppercase tracking-wider"
            style={{ background: 'rgba(255,255,255,0.04)', borderBottom: '1px solid rgba(255,255,255,0.06)' }}
          >
            <div className="text-slate-400">Feature</div>
            <div className="text-center text-slate-500">Traditional</div>
            <div className="text-center" style={{ color: '#A78BFA' }}>Paper2Code</div>
          </div>

          {/* Rows */}
          {ROWS.map((row, i) => (
            <motion.div
              key={row.feature}
              initial={{ opacity: 0, x: -10 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.04, duration: 0.4 }}
              className="grid grid-cols-3 px-6 py-4 items-center transition-colors hover:bg-white/[0.02]"
              style={{
                borderBottom: i < ROWS.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none',
              }}
            >
              <div className="text-sm text-slate-300">{row.feature}</div>
              <div className="flex justify-center">
                {row.traditional ? (
                  <div className="w-6 h-6 rounded-full flex items-center justify-center" style={{ background: 'rgba(16,185,129,0.15)' }}>
                    <Check size={12} className="text-emerald-400" />
                  </div>
                ) : (
                  <div className="w-6 h-6 rounded-full flex items-center justify-center" style={{ background: 'rgba(239,68,68,0.12)' }}>
                    <X size={12} className="text-red-400" />
                  </div>
                )}
              </div>
              <div className="flex justify-center">
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center"
                  style={{ background: 'rgba(139,92,246,0.2)', boxShadow: '0 0 8px rgba(139,92,246,0.2)' }}
                >
                  <Check size={12} style={{ color: '#A78BFA' }} />
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
