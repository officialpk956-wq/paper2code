'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Code2, Trophy, CalendarDays, Target } from 'lucide-react';

const CHALLENGES = [
  {
    type: 'Daily',
    icon: CalendarDays,
    title: 'Implement Scaled Dot-Product Attention',
    difficulty: 'Medium',
    diffColor: '#F59E0B',
    xp: 50,
    href: '/problems',
    tags: ['attention', 'transformers'],
  },
  {
    type: 'Weekly',
    icon: Target,
    title: 'Build a Mini GPT from Scratch',
    difficulty: 'Hard',
    diffColor: '#EF4444',
    xp: 200,
    href: '/problems',
    tags: ['LLM', 'autoregressive'],
  },
];

const STATS = [
  { label: 'Solved', value: '47', color: '#8B5CF6' },
  { label: 'Streak', value: '3d',  color: '#F59E0B' },
  { label: 'Rank',   value: '#124', color: '#06B6D4' },
];

export function PracticeCenter() {
  return (
    <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
      <div className="px-4 pt-4 pb-3 flex items-center justify-between">
        <div>
          <div className="text-[13px] font-bold text-white">Practice Center</div>
          <div className="text-[11px] text-slate-600 mt-0.5">Daily & weekly challenges</div>
        </div>
        <Link href="/problems" className="text-[11px] font-semibold" style={{ color: '#8B5CF6' }}>All problems →</Link>
      </div>

      {/* Stats row */}
      <div className="px-4 pb-3 flex gap-3">
        {STATS.map(s => (
          <div key={s.label} className="flex-1 rounded-lg py-2 text-center"
            style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
            <div className="text-base font-black" style={{ color: s.color }}>{s.value}</div>
            <div className="text-[10px] text-slate-600 mt-0.5">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Challenges */}
      <div className="px-4 pb-4 space-y-2.5">
        {CHALLENGES.map((ch, i) => {
          const Icon = ch.icon;
          return (
            <motion.div
              key={ch.type}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1, duration: 0.4 }}
            >
              <Link href={ch.href}
                className="block p-3 rounded-xl transition-all duration-200"
                style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(139,92,246,0.3)';
                  (e.currentTarget as HTMLElement).style.background = 'rgba(139,92,246,0.06)';
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.07)';
                  (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.03)';
                }}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-start gap-2.5">
                    <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
                      style={{ background: 'rgba(139,92,246,0.15)', color: '#8B5CF6' }}>
                      <Icon size={12} />
                    </div>
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: '#8B5CF6' }}>{ch.type}</span>
                        <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded" style={{ background: `${ch.diffColor}15`, color: ch.diffColor }}>
                          {ch.difficulty}
                        </span>
                      </div>
                      <div className="text-[12px] font-semibold text-white leading-tight">{ch.title}</div>
                      <div className="flex items-center gap-1.5 mt-1.5">
                        {ch.tags.map(t => (
                          <span key={t} className="text-[10px] px-1.5 py-0.5 rounded text-slate-500"
                            style={{ background: 'rgba(255,255,255,0.05)' }}>{t}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1 flex-shrink-0">
                    <Trophy size={11} style={{ color: '#F59E0B' }} />
                    <span className="text-[11px] font-bold" style={{ color: '#F59E0B' }}>+{ch.xp} XP</span>
                  </div>
                </div>
              </Link>
            </motion.div>
          );
        })}
      </div>

      {/* Dojo CTA */}
      <div className="px-4 pb-4">
        <Link href="/dojo"
          className="flex items-center gap-2.5 w-full p-3 rounded-xl transition-all duration-200"
          style={{ background: 'rgba(6,182,212,0.08)', border: '1px solid rgba(6,182,212,0.2)' }}
          onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = 'rgba(6,182,212,0.14)'}
          onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = 'rgba(6,182,212,0.08)'}
        >
          <Code2 size={14} style={{ color: '#06B6D4' }} />
          <div>
            <div className="text-[12px] font-semibold text-white">Dojo Challenge</div>
            <div className="text-[11px] text-slate-600">Interactive coding practice</div>
          </div>
          <div className="ml-auto text-[10px] font-bold px-2 py-1 rounded-lg text-white" style={{ background: '#06B6D4' }}>
            Start
          </div>
        </Link>
      </div>
    </div>
  );
}
