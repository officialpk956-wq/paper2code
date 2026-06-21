'use client';

import { motion } from 'framer-motion';
import { Flame, Trophy, CheckSquare, Target, Zap, Star } from 'lucide-react';
import { useUserStats } from '@/hooks/use-user-stats';

/* ─── Streak Heatmap (7×6 grid) ─────────────────────────────── */
const HEATMAP = [
  [0,1,1,0,2,2,1],
  [1,0,1,2,1,0,2],
  [2,1,2,1,1,2,1],
  [0,1,0,2,2,1,0],
  [1,2,1,1,0,2,1],
  [1,1,2,1,2,1,2],
];
const DAYS = ['M','T','W','T','F','S','S'];
const HEAT_COLORS = ['rgba(255,255,255,0.06)', 'rgba(139,92,246,0.35)', 'rgba(139,92,246,0.8)'];

/* ─── Daily tasks ────────────────────────────────────────────── */
const TASKS = [
  { label: 'Read 1 paper',         done: true },
  { label: 'Solve 1 problem',       done: true },
  { label: 'Review flashcards',     done: false },
  { label: 'Watch architecture vid',done: false },
];

/* ─── Goals ──────────────────────────────────────────────────── */
const GOALS = [
  { label: 'Finish Transformers track',  progress: 34, color: '#8B5CF6' },
  { label: 'Solve 100 problems',         progress: 47, color: '#06B6D4' },
  { label: 'Read 20 papers',             progress: 60, color: '#10B981' },
];

/* ─── Achievements ───────────────────────────────────────────── */
const ACHIEVEMENTS = [
  { label: 'First Paper',    color: '#F59E0B', icon: Star },
  { label: '7-Day Streak',   color: '#EC4899', icon: Flame },
  { label: 'Code Runner',    color: '#8B5CF6', icon: Zap },
];

export function RightInsightsPanel() {
  const stats = useUserStats();
  
  const xpCurrent = stats.xp;
  const xpForNextLevel = stats.level * 200;
  const xpProgress = xpCurrent > 0 ? (xpCurrent / xpForNextLevel) * 100 : 0;
  const xpRemaining = xpForNextLevel - xpCurrent;
  return (
    <aside
      className="w-[300px] flex-shrink-0 h-full overflow-y-auto flex flex-col gap-4 py-5 px-4"
      style={{
        borderLeft: '1px solid rgba(255,255,255,0.07)',
        background: 'rgba(7,11,24,0.5)',
        scrollbarWidth: 'none',
      }}
      aria-label="Insights panel"
    >
      {/* ── Streak banner ─────────────── */}
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="rounded-xl p-4 flex items-center gap-3"
        style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)', opacity: stats.loaded ? 1 : 0.5 }}
      >
        <div className="w-10 h-10 rounded-xl flex items-center justify-center"
          style={{ background: 'rgba(245,158,11,0.15)', color: '#F59E0B' }}>
          <Flame size={18} />
        </div>
        <div>
          <div className="text-xl font-black text-white">{stats.loaded ? stats.streak : '-'}</div>
          <div className="text-[11px] text-amber-400">{stats.streak === 1 ? 'Day streak — keep going!' : 'Day streak — keep going!'}</div>
        </div>
      </motion.div>

      {/* ── Activity Heatmap ──────────── */}
      <div className="rounded-xl p-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="text-[12px] font-bold text-white mb-3">Activity</div>
        <div className="flex gap-0.5">
          {DAYS.map((d, di) => (
            <div key={di} className="flex flex-col gap-0.5 flex-1">
              <div className="text-[9px] text-slate-700 text-center mb-0.5">{d}</div>
              {HEATMAP.map((row, ri) => (
                <div key={ri} className="rounded-[2px] w-full aspect-square"
                  style={{ background: HEAT_COLORS[row[di]] }} />
              ))}
            </div>
          ))}
        </div>
        <div className="flex items-center justify-between mt-2">
          <span className="text-[10px] text-slate-700">Less</span>
          <div className="flex gap-1">
            {HEAT_COLORS.map((c, i) => (
              <div key={i} className="w-2.5 h-2.5 rounded-[2px]" style={{ background: c }} />
            ))}
          </div>
          <span className="text-[10px] text-slate-700">More</span>
        </div>
      </div>

      {/* ── XP Tracker ───────────────── */}
      <div className="rounded-xl p-4" style={{ background: 'rgba(139,92,246,0.06)', border: '1px solid rgba(139,92,246,0.18)', opacity: stats.loaded ? 1 : 0.5 }}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <Zap size={12} style={{ color: '#8B5CF6' }} />
            <span className="text-[12px] font-bold text-white">XP Tracker</span>
          </div>
          <span className="text-[11px] font-bold" style={{ color: '#A78BFA' }}>Level {stats.loaded ? stats.level : '-'}</span>
        </div>
        <div className="flex items-end justify-between mb-1.5">
          <span className="text-2xl font-black text-white">{stats.loaded ? stats.xp.toLocaleString() : '-'}</span>
          <span className="text-[11px] text-slate-600">/ {stats.loaded ? xpForNextLevel.toLocaleString() : '-'} XP</span>
        </div>
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.08)' }}>
          <motion.div className="h-full rounded-full"
            style={{ background: 'linear-gradient(to right, #7C3AED, #06B6D4)' }}
            initial={{ width: 0 }}
            animate={{ width: `${stats.loaded ? xpProgress : 0}%` }}
            transition={{ duration: 1, delay: 0.4, ease: [0.22, 1, 0.36, 1] }}
          />
        </div>
        <div className="text-[10px] text-slate-600 mt-1">{stats.loaded ? xpRemaining.toLocaleString() : '-'} XP to Level {stats.loaded ? stats.level + 1 : '-'}</div>
      </div>

      {/* ── Daily Tasks ──────────────── */}
      <div className="rounded-xl p-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="flex items-center gap-1.5 mb-3">
          <CheckSquare size={12} style={{ color: '#06B6D4' }} />
          <span className="text-[12px] font-bold text-white">Today&apos;s Tasks</span>
          <span className="ml-auto text-[10px] font-semibold px-2 py-0.5 rounded-full"
            style={{ background: 'rgba(6,182,212,0.12)', color: '#06B6D4' }}>
            2/4
          </span>
        </div>
        <div className="space-y-2">
          {TASKS.map((t, i) => (
            <div key={i} className="flex items-center gap-2.5">
              <div className="w-4 h-4 rounded flex items-center justify-center flex-shrink-0"
                style={{
                  background: t.done ? 'rgba(16,185,129,0.15)' : 'rgba(255,255,255,0.05)',
                  border: `1.5px solid ${t.done ? '#10B981' : 'rgba(255,255,255,0.12)'}`,
                }}>
                {t.done && <span className="text-[8px]" style={{ color: '#10B981' }}>✓</span>}
              </div>
              <span className="text-[12px]" style={{ color: t.done ? '#475569' : '#94A3B8', textDecoration: t.done ? 'line-through' : 'none' }}>
                {t.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Weekly Goals ─────────────── */}
      <div className="rounded-xl p-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="flex items-center gap-1.5 mb-3">
          <Target size={12} style={{ color: '#10B981' }} />
          <span className="text-[12px] font-bold text-white">Weekly Goals</span>
        </div>
        <div className="space-y-3">
          {GOALS.map((g, i) => (
            <div key={i}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[11px] text-slate-500 truncate pr-2">{g.label}</span>
                <span className="text-[11px] font-bold flex-shrink-0" style={{ color: g.color }}>{g.progress}%</span>
              </div>
              <div className="h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.07)' }}>
                <motion.div className="h-full rounded-full" style={{ background: g.color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${g.progress}%` }}
                  transition={{ duration: 0.8, delay: 0.5 + i * 0.1, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Achievements ─────────────── */}
      <div className="rounded-xl p-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="flex items-center gap-1.5 mb-3">
          <Trophy size={12} style={{ color: '#F59E0B' }} />
          <span className="text-[12px] font-bold text-white">Recent Achievements</span>
        </div>
        <div className="flex gap-2.5">
          {ACHIEVEMENTS.map((a, i) => {
            const Icon = a.icon;
            return (
              <motion.div key={a.label}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.3 + i * 0.1, type: 'spring', stiffness: 200, damping: 15 }}
                className="flex-1 flex flex-col items-center gap-1.5 p-2.5 rounded-xl"
                style={{ background: `${a.color}10`, border: `1px solid ${a.color}25` }}
              >
                <div className="w-8 h-8 rounded-full flex items-center justify-center"
                  style={{ background: `${a.color}20`, color: a.color, boxShadow: `0 0 10px ${a.color}30` }}>
                  <Icon size={14} />
                </div>
                <span className="text-[10px] font-semibold text-center leading-tight" style={{ color: '#94A3B8' }}>{a.label}</span>
              </motion.div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}
