'use client';

import { TrendingUp, Flame, CheckCircle, BarChart3 } from 'lucide-react';

interface HeatmapDay {
  date: string;
  count: number;
}

const generateHeatmapData = (): HeatmapDay[] => {
  const data: HeatmapDay[] = [];
  // Base date fixed at 2026-06-16 (112 days back from publication date)
  const BASE_DATE = new Date('2026-06-16');
  for (let i = 0; i < 112; i++) {
    const date = new Date(BASE_DATE);
    date.setDate(date.getDate() - (111 - i));
    // Deterministic activity count — no Math.random()
    const count = (i * 7 + 13 * (i % 5) + 3) % 5;
    data.push({
      date: date.toISOString().split('T')[0],
      count,
    });
  }
  return data;
};

interface SkillLevel {
  name: string;
  level: number; // 0-100
  improvement: number; // percentage change
}

const skills: SkillLevel[] = [
  { name: 'Transformers', level: 92, improvement: 15 },
  { name: 'Diffusion Models', level: 78, improvement: 8 },
  { name: 'System Design', level: 85, improvement: 12 },
  { name: 'Reinforcement Learning', level: 65, improvement: 5 },
  { name: 'Neural Networks', level: 88, improvement: 10 },
  { name: 'Mathematics', level: 72, improvement: 7 },
];

const heatmapData = generateHeatmapData();

interface LearningAnalyticsProps {
  weeklyTarget?: number;
}

export function LearningAnalytics({ weeklyTarget = 15 }: LearningAnalyticsProps) {
  const thisWeekCount = 12;
  const currentStreak = 34;
  const totalProblems = 287;
  const completionRate = 73;
  const avgTimePerSession = 45; // minutes

  // Group heatmap data by weeks
  const weeks = [];
  for (let i = 0; i < 16; i++) {
    weeks.push(heatmapData.slice(i * 7, (i + 1) * 7));
  }


  const getHeatmapColor = (count: number) => {
    if (count === 0) return 'bg-[--bg-surface]';
    if (count === 1) return 'bg-green-900/40';
    if (count === 2) return 'bg-green-700/50';
    if (count === 3) return 'bg-green-600/60';
    if (count === 4) return 'bg-green-500/70';
    return 'bg-green-400/80';
  };

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border border-[--color-border] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-body]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-3">
          Learning Analytics
        </h3>

        {/* KPI Cards */}
        <div className="grid grid-cols-4 gap-3">
          <div className="bg-[--bg-panel] rounded-lg p-3 border border-[--color-border]">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-[--color-text-tertiary]">This Week</span>
              <TrendingUp size={14} className="text-green-400" />
            </div>
            <div className="text-lg font-bold text-[--color-text-primary]">
              {thisWeekCount}
              <span className="text-xs text-[--color-text-tertiary] ml-1">/{weeklyTarget}</span>
            </div>
            <div className="text-xs text-green-400 mt-1">+{Math.round((thisWeekCount / weeklyTarget) * 100) - 100}% on target</div>
          </div>

          <div className="bg-[--bg-panel] rounded-lg p-3 border border-[--color-border]">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-[--color-text-tertiary]">Streak</span>
              <Flame size={14} className="text-orange-400" />
            </div>
            <div className="text-lg font-bold text-[--color-text-primary]">{currentStreak}</div>
            <div className="text-xs text-orange-400 mt-1">Days consistent</div>
          </div>

          <div className="bg-[--bg-panel] rounded-lg p-3 border border-[--color-border]">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-[--color-text-tertiary]">Completed</span>
              <CheckCircle size={14} className="text-blue-400" />
            </div>
            <div className="text-lg font-bold text-[--color-text-primary]">{completionRate}%</div>
            <div className="text-xs text-blue-400 mt-1">Of all content</div>
          </div>

          <div className="bg-[--bg-panel] rounded-lg p-3 border border-[--color-border]">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-[--color-text-tertiary]">Avg Session</span>
              <BarChart3 size={14} className="text-purple-400" />
            </div>
            <div className="text-lg font-bold text-[--color-text-primary]">{avgTimePerSession}m</div>
            <div className="text-xs text-purple-400 mt-1">Deep work focus</div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Activity Heatmap */}
        <div>
          <h4 className="text-sm font-semibold text-[--color-text-primary] mb-3">
            16-Week Activity Heatmap
          </h4>
          <div className="bg-[--bg-body] rounded-lg p-4 border border-[--color-border] overflow-x-auto">
            <div className="flex gap-1">
              {weeks.map((week, weekIdx) => (
                <div key={weekIdx} className="flex flex-col gap-1">
                  {/* Week label */}
                  {weekIdx % 4 === 0 && (
                    <div className="text-xs text-[--color-text-tertiary] h-5 flex items-end justify-center">
                      {weekIdx / 4 + 1}w
                    </div>
                  )}
                  {weekIdx % 4 !== 0 && <div className="h-5" />}

                  {/* Days in week */}
                  {week.map((day, dayIdx) => (
                    <div
                      key={`${weekIdx}-${dayIdx}`}
                      className={`w-4 h-4 rounded border border-[--color-border] ${getHeatmapColor(day.count)} hover:border-[--accent-primary] cursor-pointer transition-colors group relative`}
                      title={`${day.date}: ${day.count} activities`}
                    >
                      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-[--bg-panel] text-[--color-text-primary] text-xs px-2 py-1 rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 border border-[--color-border]">
                        {day.date}: {day.count} activities
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
          <div className="mt-2 text-xs text-[--color-text-tertiary]">
            Each square represents one day. Darker = more learning activity.
          </div>
        </div>

        {/* Skill Radar */}
        <div>
          <h4 className="text-sm font-semibold text-[--color-text-primary] mb-3">
            Skill Mastery Levels
          </h4>
          <div className="space-y-3">
            {skills.map((skill) => (
              <div key={skill.name} className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-[--color-text-primary]">
                    {skill.name}
                  </span>
                  <span className="text-xs text-[--color-text-tertiary]">
                    {skill.level}%
                    {skill.improvement > 0 && (
                      <span className="text-green-400 ml-1">+{skill.improvement}%</span>
                    )}
                  </span>
                </div>
                <div className="w-full bg-[--bg-surface] rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] h-full transition-all rounded-full"
                    style={{ width: `${skill.level}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Statistics */}
        <div>
          <h4 className="text-sm font-semibold text-[--color-text-primary] mb-3">
            Learning Statistics
          </h4>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs text-[--color-text-tertiary] mb-1">Total Problems Solved</div>
              <div className="text-2xl font-bold text-[--accent-primary]">{totalProblems}</div>
            </div>
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs text-[--color-text-tertiary] mb-1">Avg Daily Sessions</div>
              <div className="text-2xl font-bold text-[--accent-cyan]">2.4</div>
            </div>
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs text-[--color-text-tertiary] mb-1">Total Study Hours</div>
              <div className="text-2xl font-bold text-green-400">245</div>
            </div>
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs text-[--color-text-tertiary] mb-1">Personal Best</div>
              <div className="text-2xl font-bold text-yellow-400">12 days</div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-3 border-t border-[--color-border] bg-[--bg-body]">
        <button className="w-full px-3 py-2 rounded text-xs font-medium bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          View Detailed Report
        </button>
      </div>
    </div>
  );
}
