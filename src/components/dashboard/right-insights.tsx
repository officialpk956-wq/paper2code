'use client';

import { motion } from 'framer-motion';
import { Flame, Trophy, CheckSquare, Target, Zap, Star } from 'lucide-react';
import { useUserStats } from '@/hooks/use-user-stats';
import { getContributionGrid, getCompletionRate, getCompletedCount } from '@/lib/progress';
import { getDisplayName, getBio } from '@/lib/persistence';
import { useState, useEffect } from 'react';

const HEAT_COLORS = ['rgba(255,255,255,0.06)', 'rgba(139,92,246,0.35)', 'rgba(139,92,246,0.8)'];

/* ─── Achievements ───────────────────────────────────────────── */
const ACHIEVEMENTS = [
  { label: 'First Paper',    color: '#F59E0B', icon: Star },
  { label: '7-Day Streak',   color: '#EC4899', icon: Flame },
  { label: 'Code Runner',    color: '#8B5CF6', icon: Zap },
];

export function RightInsightsPanel() {
  const stats = useUserStats();
  const [grid, setGrid] = useState<number[]>([]);
  const [bio, setBio] = useState('Loading bio...');
  const [displayName, setDisplayName] = useState('Researcher');
  const [goals, setGoals] = useState([
    { label: 'Finish topics', progress: 0, color: '#8B5CF6' },
    { label: 'Solve problems', progress: 0, color: '#06B6D4' },
    { label: 'Read papers', progress: 0, color: '#10B981' },
  ]);
  const [tasks, setTasks] = useState([
    { label: 'Read 1 paper', done: false },
    { label: 'Solve 1 problem', done: false },
    { label: 'Review flashcards', done: false },
    { label: 'Watch architecture vid', done: false },
  ]);
  
  useEffect(() => {
    setGrid(getContributionGrid(52));
    setBio(getBio());
    setDisplayName(getDisplayName());
    
    setGoals([
      { label: 'Finish topics', progress: getCompletionRate('topic'), color: '#8B5CF6' },
      { label: 'Solve problems', progress: getCompletionRate('problem'), color: '#06B6D4' },
      { label: 'Read papers', progress: getCompletionRate('paper'), color: '#10B981' },
    ]);

    setTasks([
      { label: 'Read 1 paper', done: getCompletedCount('paper') > 0 },
      { label: 'Solve 1 problem', done: getCompletedCount('problem') > 0 },
      { label: 'Review flashcards', done: false },
      { label: 'Watch architecture vid', done: getCompletedCount('architecture') > 0 },
    ]);
  }, []);

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
      {/* ── Profile Card ─────────────── */}
      <div className="rounded-xl p-4 flex flex-col items-center text-center gap-2" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold text-white mb-1 shadow-lg shadow-[#8B5CF6]/20" style={{ background: 'linear-gradient(135deg, #7C3AED, #8B5CF6)' }}>
          {stats.loaded ? displayName.charAt(0).toUpperCase() : 'R'}
        </div>
        <h3 className="text-white font-bold">{stats.loaded ? displayName : 'Researcher'}</h3>
        <div className="text-[11px] font-semibold tracking-wide uppercase" style={{ color: '#A78BFA' }}>Level {stats.loaded ? stats.level : '-'}</div>
        <p className="text-[11px] text-slate-400 mt-1 italic px-2">
          {stats.loaded ? bio : 'Loading bio...'}
        </p>
      </div>

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
        <div className="flex gap-[2px] items-start">
          {/* We have 52 weeks, each week is a column of 7 days */}
          {Array.from({ length: 52 }).map((_, w) => (
            <div key={w} className="flex flex-col gap-[2px]">
              {Array.from({ length: 7 }).map((_, d) => {
                const idx = w * 7 + d;
                const val = grid[idx] ?? 0;
                return (
                  <div key={d} className="w-[3px] h-[3px] rounded-[1px]"
                    style={{ background: HEAT_COLORS[val] }} />
                );
              })}
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
          {tasks.map((t, i) => (
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
          {goals.map((g, i) => (
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
