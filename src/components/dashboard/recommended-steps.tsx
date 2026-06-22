'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Sparkles, ArrowRight } from 'lucide-react';

const RECS = [
  {
    priority: 'Next Up',
    priorityColor: '#8B5CF6',
    title: 'Complete Multi-Head Attention',
    reason: 'You\'re 34% through Transformers — this is the core module.',
    href: '/learn/deep-learning/attention',
    xp: 80,
  },
  {
    priority: 'Recommended',
    priorityColor: '#06B6D4',
    title: 'Read "Attention is All You Need"',
    reason: 'Pairs directly with your current track.',
    href: '/papers/attention-is-all-you-need',
    xp: 30,
  },
  {
    priority: 'Challenge',
    priorityColor: '#F59E0B',
    title: 'Implement Transformer in PyTorch',
    reason: 'Convert your reading into working code.',
    href: '/problems',
    xp: 150,
  },
  {
    priority: 'Boost Score',
    priorityColor: '#10B981',
    title: 'Practice System Design: ML Serving',
    reason: 'Your system design score is lowest — quick win.',
    href: '/system-design',
    xp: 60,
  },
];

export function RecommendedSteps() {
  return (
    <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
      <div className="px-4 pt-4 pb-3 flex items-center gap-2">
        <Sparkles size={13} style={{ color: '#A78BFA' }} />
        <div>
          <div className="text-[13px] font-bold text-white">AI Recommendations</div>
          <div className="text-[11px] text-slate-600">Personalized next steps</div>
        </div>
      </div>

      <div className="px-4 pb-4 space-y-2">
        {RECS.map((rec, i) => (
          <motion.div
            key={rec.title}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.09, duration: 0.4 }}
          >
            <Link
              href={rec.href}
              className="group flex items-start gap-3 p-3 rounded-xl transition-all duration-200 block"
              style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.background = `${rec.priorityColor}08`;
                (e.currentTarget as HTMLElement).style.borderColor = `${rec.priorityColor}28`;
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.02)';
                (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.06)';
              }}
            >
              {/* Priority badge */}
              <div className="flex-shrink-0 mt-0.5">
                <span className="text-[9px] font-black uppercase tracking-wider px-2 py-0.5 rounded-full"
                  style={{ background: `${rec.priorityColor}18`, color: rec.priorityColor }}>
                  {rec.priority}
                </span>
              </div>

              <div className="flex-1 min-w-0">
                <div className="text-[12px] font-semibold text-white leading-snug">{rec.title}</div>
                <div className="text-[11px] text-slate-600 mt-0.5 leading-snug">{rec.reason}</div>
              </div>

              <div className="flex items-center gap-1.5 flex-shrink-0">
                <span className="text-[10px] font-bold" style={{ color: '#F59E0B' }}>+{rec.xp}</span>
                <ArrowRight size={11} style={{ color: '#334155' }}
                  className="group-hover:translate-x-0.5 transition-transform" />
              </div>
            </Link>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
