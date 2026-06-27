'use client';

import { motion } from 'framer-motion';
import { Flame, Zap, Target, BookOpen, Award, CheckSquare } from 'lucide-react';
import { useUserStats } from '@/hooks/use-user-stats';

/* ─── Progress donut ring ──────────────────────────────────── */
function ProgressRing({ value, size = 80 }: { value: number; size?: number }) {
  const r = size / 2 - 6;
  const circ = 2 * Math.PI * r;
  return (
    <svg width={size} height={size} aria-label={`${value}% overall progress`}>
      <circle cx={size/2} cy={size/2} r={r} fill="none"
        stroke="rgba(255,255,255,0.06)" strokeWidth="5" />
      <motion.circle cx={size/2} cy={size/2} r={r} fill="none"
        stroke="url(#ringGrad)" strokeWidth="5"
        strokeLinecap="round"
        strokeDasharray={circ}
        transform={`rotate(-90 ${size/2} ${size/2})`}
        initial={{ strokeDashoffset: circ }}
        animate={{ strokeDashoffset: circ - (circ * value / 100) }}
        transition={{ duration: 1.2, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
      />
      <defs>
        <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#7C3AED" />
          <stop offset="100%" stopColor="#06B6D4" />
        </linearGradient>
      </defs>
      <text x={size/2} y={size/2 + 5} textAnchor="middle"
        fontSize="14" fontWeight="800" fill="white" fontFamily="system-ui, sans-serif">
        {value}%
      </text>
    </svg>
  );
}

/* ─── Weekly goal ring ─────────────────────────────────────── */
function WeeklyGoalBar({ completed, total }: { completed: number; total: number }) {
  const pct = Math.min((completed / total) * 100, 100);
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[11px] text-slate-500">Weekly Goal</span>
        <span className="text-[11px] font-bold text-white">{completed}/{total} lessons</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.07)' }}>
        <motion.div className="h-full rounded-full"
          style={{ background: 'linear-gradient(to right, #7C3AED, #06B6D4)' }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1, delay: 0.5, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
    </div>
  );
}

export function ProgressSidebar() {
  const stats = useUserStats();
  
  // Fabricated tasks and achievements removed
  const tasks: { done: boolean; label: string; xp: number }[] = [];
  const achievements: { color: string; icon: string; label: string; earnedAt: string }[] = [];
  const doneTasks = 0;

  return (
    <aside
      className="hidden xl:flex flex-col gap-4 w-[272px] flex-shrink-0 h-full overflow-y-auto py-5 px-4"
      style={{
        borderLeft: '1px solid rgba(255,255,255,0.07)',
        background: 'rgba(7,11,24,0.6)',
        scrollbarWidth: 'none',
      }}
      aria-label="Learning progress"
    >
      {/* ── Overall progress ─────────── */}
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="rounded-xl p-4"
        style={{ background: 'rgba(139,92,246,0.07)', border: '1px solid rgba(139,92,246,0.18)', opacity: stats.loaded ? 1 : 0.5 }}
      >
        <div className="text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-3">Overall Progress</div>
        <div className="flex items-center gap-4">
          <ProgressRing value={stats.loaded ? stats.overallProgress : 0} />
          <div className="space-y-2 text-[11px]">
            <div>
              <div className="font-bold text-white">{stats.loaded ? stats.completedLessons : '-'}</div>
              <div className="text-slate-600">Lessons done</div>
            </div>
            <div>
              <div className="font-bold text-white">-</div>
              <div className="text-slate-600">Time spent</div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* ── XP & Streak ──────────────── */}
      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-xl p-3"
          style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)', opacity: stats.loaded ? 1 : 0.5 }}>
          <div className="flex items-center gap-1.5 mb-1">
            <Flame size={12} style={{ color: '#F59E0B' }} />
            <span className="text-[10px] font-bold text-amber-400">Streak</span>
          </div>
          <div className="text-xl font-black text-white">{stats.loaded ? stats.streak : '-'}</div>
          <div className="text-[10px] text-amber-700">days</div>
        </div>
        <div className="rounded-xl p-3"
          style={{ background: 'rgba(139,92,246,0.08)', border: '1px solid rgba(139,92,246,0.2)', opacity: stats.loaded ? 1 : 0.5 }}>
          <div className="flex items-center gap-1.5 mb-1">
            <Zap size={12} style={{ color: '#8B5CF6' }} />
            <span className="text-[10px] font-bold" style={{ color: '#A78BFA' }}>Level {stats.loaded ? stats.level : '-'}</span>
          </div>
          <div className="text-xl font-black text-white">{stats.loaded ? stats.xp.toLocaleString() : '-'}</div>
          <div className="text-[10px] text-slate-600">XP total</div>
        </div>
      </div>

      {/* ── Weekly Goal ──────────────── */}
      <div className="rounded-xl p-4"
        style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="flex items-center gap-1.5 mb-3">
          <Target size={12} style={{ color: '#10B981' }} />
          <span className="text-[11px] font-bold text-white">Weekly Goal</span>
        </div>
        <WeeklyGoalBar completed={stats.loaded ? (stats.streak > 0 ? 1 : 0) : 0} total={10} />
      </div>

      {/* ── Today's Tasks ─────────────── */}
      <div className="rounded-xl p-4"
        style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-1.5">
            <CheckSquare size={12} style={{ color: '#06B6D4' }} />
            <span className="text-[11px] font-bold text-white">Today&apos;s Tasks</span>
          </div>
          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full"
            style={{ background: 'rgba(6,182,212,0.12)', color: '#06B6D4' }}>
            {doneTasks}/{tasks.length}
          </span>
        </div>
        <div className="space-y-2">
          {tasks.map((task, i) => (
            <div key={i} className="flex items-center gap-2.5">
              <div className="w-4 h-4 rounded flex items-center justify-center flex-shrink-0"
                style={{
                  background: task.done ? 'rgba(16,185,129,0.15)' : 'rgba(255,255,255,0.05)',
                  border: `1.5px solid ${task.done ? '#10B981' : 'rgba(255,255,255,0.12)'}`,
                }}>
                {task.done && <span className="text-[8px]" style={{ color: '#10B981' }}>✓</span>}
              </div>
              <span className="text-[11px] flex-1" style={{ color: task.done ? '#475569' : '#94A3B8', textDecoration: task.done ? 'line-through' : 'none' }}>
                {task.label}
              </span>
              <span className="text-[10px] font-semibold" style={{ color: '#F59E0B' }}>+{task.xp}</span>
            </div>
          ))}
          {tasks.length === 0 && (
             <div className="text-[11px] text-slate-500">No tasks generated yet.</div>
          )}
        </div>
      </div>

      {/* ── Recent Achievements ───────── */}
      <div className="rounded-xl p-4"
        style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="flex items-center gap-1.5 mb-3">
          <Award size={12} style={{ color: '#F59E0B' }} />
          <span className="text-[11px] font-bold text-white">Achievements</span>
        </div>
        <div className="space-y-2">
          {achievements.map((a, i) => (
            <motion.div key={i}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 + i * 0.1, duration: 0.4 }}
              className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 text-[13px]"
                style={{ background: `${a.color}18`, boxShadow: `0 0 8px ${a.color}30` }}>
                {a.icon === 'BookOpen' && <BookOpen size={12} style={{ color: a.color }} />}
                {a.icon === 'Flame' && <Flame size={12} style={{ color: a.color }} />}
                {a.icon === 'Zap' && <Zap size={12} style={{ color: a.color }} />}
              </div>
              <div>
                <div className="text-[11px] font-semibold text-white">{a.label}</div>
                <div className="text-[10px] text-slate-600">{a.earnedAt}</div>
              </div>
            </motion.div>
          ))}
          {achievements.length === 0 && (
             <div className="text-[11px] text-slate-500">No achievements yet. Keep learning!</div>
          )}
        </div>
      </div>
    </aside>
  );
}
