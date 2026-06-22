"use client";

import { LearningTrack } from "@/data/learning-tracks";

interface TrackProgressProps {
  track: LearningTrack;
  completedProblems: Set<string>;
}

export function TrackProgress({
  track,
  completedProblems,
}: TrackProgressProps) {
  const completed = track.problems.filter((p) =>
    completedProblems.has(p.problemId)
  ).length;
  const total = track.problems.length;
  const percentage = Math.round((completed / total) * 100);

  // Estimate remaining time
  const completedMinutes = track.problems
    .filter((p) => completedProblems.has(p.problemId))
    .reduce((sum, p) => sum + p.estimatedMinutes, 0);
  const remainingMinutes = track.estimatedHours * 60 - completedMinutes;
  const remainingHours = Math.round(remainingMinutes / 60 * 10) / 10;

  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-[--color-text-primary]">
            Track Progress
          </h3>
          <span className="text-sm font-mono text-[--color-text-secondary]">
            {completed} of {total} completed
          </span>
        </div>
        <div className="w-full h-3 bg-[--bg-body] rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] transition-all duration-500"
            style={{ width: `${percentage}%` }}
          />
        </div>
        <div className="flex items-center justify-between mt-2">
          <span className="text-xs text-[--color-text-tertiary]">
            {percentage}% complete
          </span>
          {remainingHours > 0 && (
            <span className="text-xs text-[--color-text-tertiary]">
              ~{remainingHours} hours remaining
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="p-3 rounded-lg bg-[--bg-body] border border-[--color-border]">
          <p className="text-xs text-[--color-text-tertiary] mb-1">
            Completed
          </p>
          <p className="text-2xl font-bold text-[#10B981]">{completed}</p>
        </div>
        <div className="p-3 rounded-lg bg-[--bg-body] border border-[--color-border]">
          <p className="text-xs text-[--color-text-tertiary] mb-1">Total</p>
          <p className="text-2xl font-bold text-[--color-text-primary]">
            {total}
          </p>
        </div>
        <div className="p-3 rounded-lg bg-[--bg-body] border border-[--color-border]">
          <p className="text-xs text-[--color-text-tertiary] mb-1">
            Estimated Total
          </p>
          <p className="text-2xl font-bold text-[--accent-primary]">
            {track.estimatedHours}h
          </p>
        </div>
      </div>
    </div>
  );
}
