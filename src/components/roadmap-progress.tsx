"use client";

import { Roadmap } from "@/data/roadmaps";
import { TrendingUp, Flame, Target, Clock } from "lucide-react";

interface RoadmapProgressProps {
  roadmap: Roadmap;
  stats?: {
    streak: number;
    hoursSpent: number;
    completedToday: number;
  };
}

export function RoadmapProgress({
  roadmap,
  stats = { streak: 12, hoursSpent: 8.5, completedToday: 2 },
}: RoadmapProgressProps) {
  const totalNodes = roadmap.phases.reduce((acc, phase) => acc + phase.nodes.length, 0);
  const completedNodes = roadmap.phases.reduce(
    (acc, phase) => acc + phase.nodes.filter((n) => n.state === "completed").length,
    0
  );

  return (
    <div className="space-y-6">
      {/* Overall progress */}
      <div className="p-6 rounded-lg bg-gradient-to-br from-[rgb(124,58,237,0.1)] to-[rgb(6,182,212,0.1)] border border-[--color-border]">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-lg font-bold mb-1">{roadmap.title}</h3>
            <p className="text-sm text-[--color-text-tertiary]">
              {completedNodes} of {totalNodes} lessons completed
            </p>
          </div>
          <TrendingUp className="w-6 h-6 text-[--accent-primary]" />
        </div>

        <div className="space-y-3">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold">Overall Progress</span>
              <span className="text-lg font-bold text-[--accent-primary]">
                {roadmap.totalProgress}%
              </span>
            </div>
            <div className="h-3 bg-[--bg-body] rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] transition-all duration-500"
                style={{ width: `${roadmap.totalProgress}%` }}
              />
            </div>
          </div>

          {/* Phase breakdown */}
          <div className="space-y-2 pt-2">
            {roadmap.phases.map((phase) => (
              <div key={phase.id}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-[--color-text-secondary]">{phase.name}</span>
                  <span className="font-semibold text-[--color-text-tertiary]">
                    {phase.progress}%
                  </span>
                </div>
                <div className="h-1.5 bg-[--bg-body] rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      phase.progress === 100
                        ? "bg-[#10B981]"
                        : phase.progress === 0
                          ? "bg-[--color-border]"
                          : "bg-[#F59E0B]"
                    }`}
                    style={{ width: `${phase.progress}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-3 gap-3">
        <div className="p-4 rounded-lg bg-[--bg-surface] border border-[--color-border]">
          <div className="flex items-center justify-between mb-2">
            <Flame className="w-5 h-5 text-[#EF4444]" />
            <span className="text-xs font-semibold text-[--color-text-tertiary]">
              Streak
            </span>
          </div>
          <p className="text-2xl font-bold">{stats.streak}</p>
          <p className="text-xs text-[--color-text-tertiary]">days</p>
        </div>

        <div className="p-4 rounded-lg bg-[--bg-surface] border border-[--color-border]">
          <div className="flex items-center justify-between mb-2">
            <Clock className="w-5 h-5 text-[--accent-primary]" />
            <span className="text-xs font-semibold text-[--color-text-tertiary]">
              Hours
            </span>
          </div>
          <p className="text-2xl font-bold">{stats.hoursSpent}</p>
          <p className="text-xs text-[--color-text-tertiary]">spent</p>
        </div>

        <div className="p-4 rounded-lg bg-[--bg-surface] border border-[--color-border]">
          <div className="flex items-center justify-between mb-2">
            <Target className="w-5 h-5 text-[--accent-cyan]" />
            <span className="text-xs font-semibold text-[--color-text-tertiary]">
              Today
            </span>
          </div>
          <p className="text-2xl font-bold">{stats.completedToday}</p>
          <p className="text-xs text-[--color-text-tertiary]">completed</p>
        </div>
      </div>

      {/* Milestone check-in */}
      <div className="p-4 rounded-lg bg-[--bg-surface] border border-[--color-border]">
        <h4 className="font-semibold mb-3">Current Phase</h4>
        <div className="space-y-2">
          {roadmap.phases.map((phase) => {
            const isCurrentPhase = phase.progress > 0 && phase.progress < 100;
            if (!isCurrentPhase && phase.progress !== 0) return null;

            return (
              <div
                key={phase.id}
                className={`p-3 rounded-lg ${
                  isCurrentPhase
                    ? "bg-[rgb(124,58,237,0.1)] border border-[--accent-primary]"
                    : "bg-[--bg-body] border border-[--color-border]"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-sm">{phase.name}</p>
                    <p className="text-xs text-[--color-text-tertiary]">
                      {phase.description}
                    </p>
                  </div>
                  {isCurrentPhase && (
                    <div className="text-right">
                      <p className="text-sm font-bold text-[--accent-primary]">
                        {phase.progress}%
                      </p>
                      <p className="text-xs text-[--color-text-tertiary]">in progress</p>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
