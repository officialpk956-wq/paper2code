'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  Sigma, Brain, Layers3, Eye, MessageSquare, Sparkles,
  Bot, Database, Target, Settings2, Microscope, Cpu,
} from 'lucide-react';
import { DOMAINS } from './learn-data';

const ICON_MAP: Record<string, React.ElementType> = {
  Sigma, Brain, Layers3, Eye, MessageSquare, Sparkles,
  Bot, Database, Target, Settings2, Microscope, Cpu,
};

const CATEGORY_LABELS: Record<string, string> = {
  foundation: 'Foundation',
  core: 'Core',
  advanced: 'Advanced',
  specialized: 'Specialized',
};
const CATEGORY_COLORS: Record<string, string> = {
  foundation: '#8B5CF6',
  core: '#06B6D4',
  advanced: '#F59E0B',
  specialized: '#10B981',
};

export function LearningDomains() {
  return (
    <section className="mb-8">
      {/* Header */}
      <div className="flex items-end justify-between mb-5">
        <div>
          <h2 className="text-lg font-black text-white">Learning Domains</h2>
          <p className="text-xs text-slate-500 mt-0.5">12 domains · 554 total lessons</p>
        </div>
        <Link href="/learn/all" className="text-[11px] font-semibold transition-colors"
          style={{ color: '#8B5CF6' }}
          onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = '#A78BFA'}
          onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = '#8B5CF6'}>
          View all →
        </Link>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-3">
        {DOMAINS.map((domain, i) => {
          const Icon = ICON_MAP[domain.icon] ?? Brain;
          const catColor = CATEGORY_COLORS[domain.category];
          return (
            <motion.div
              key={domain.id}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-40px' }}
              transition={{ delay: i * 0.04, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
            >
              <Link
                href={domain.href}
                className="group flex flex-col p-4 rounded-2xl h-full transition-all duration-200 block"
                style={{
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid rgba(255,255,255,0.07)',
                }}
                onMouseEnter={e => {
                  const el = e.currentTarget as HTMLElement;
                  el.style.background = `${domain.color}0c`;
                  el.style.borderColor = `${domain.color}40`;
                  el.style.boxShadow = `0 0 24px ${domain.glow}`;
                  el.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={e => {
                  const el = e.currentTarget as HTMLElement;
                  el.style.background = 'rgba(255,255,255,0.02)';
                  el.style.borderColor = 'rgba(255,255,255,0.07)';
                  el.style.boxShadow = 'none';
                  el.style.transform = 'translateY(0)';
                }}
              >
                {/* Icon + category */}
                <div className="flex items-start justify-between mb-3">
                  <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{ background: `${domain.color}18`, color: domain.color }}>
                    <Icon size={16} />
                  </div>
                  <span className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded"
                    style={{ background: `${catColor}15`, color: catColor }}>
                    {CATEGORY_LABELS[domain.category]}
                  </span>
                </div>

                {/* Name + description */}
                <div className="text-[13px] font-bold text-white leading-tight mb-1">{domain.label}</div>
                <div className="text-[11px] text-slate-600 leading-snug mb-3 line-clamp-2 flex-1">{domain.description}</div>

                {/* Stats */}
                <div className="flex items-center justify-between text-[10px] text-slate-600 mb-2.5">
                  <span>{domain.lessonCount} lessons</span>
                  <span>{domain.estimatedHours}h</span>
                </div>

                {/* Progress */}
                {domain.progress > 0 ? (
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] text-slate-600">Progress</span>
                      <span className="text-[10px] font-semibold" style={{ color: domain.color }}>{domain.progress}%</span>
                    </div>
                    <div className="h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.07)' }}>
                      <motion.div className="h-full rounded-full" style={{ background: domain.color }}
                        initial={{ width: 0 }}
                        whileInView={{ width: `${domain.progress}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, delay: 0.3 + i * 0.04, ease: [0.22, 1, 0.36, 1] }}
                      />
                    </div>
                  </div>
                ) : (
                  <div className="h-1 rounded-full" style={{ background: 'rgba(255,255,255,0.05)' }}>
                    <div className="text-[10px] text-slate-700 mt-1">Not started</div>
                  </div>
                )}
              </Link>
            </motion.div>
          );
        })}
      </div>
    </section>
  );
}
