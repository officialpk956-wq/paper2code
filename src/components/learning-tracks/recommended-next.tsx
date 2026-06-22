"use client";

import Link from "next/link";
import { ChevronRight, Lock } from "lucide-react";
import { LearningTrack } from "@/data/learning-tracks";

interface RecommendedNextProps {
  track: LearningTrack;
  completedProblems: Set<string>;
}

export function RecommendedNext({
  track,
  completedProblems,
}: RecommendedNextProps) {
  // Find the first incomplete problem whose prerequisites are met
  const nextStep = track.problems.find((p) => {
    if (completedProblems.has(p.problemId)) return false; // Already completed
    return p.prerequisites.every((prereqId) => completedProblems.has(prereqId));
  });

  if (!nextStep) {
    // All problems completed
    return (
      <div className="p-6 rounded-lg bg-gradient-to-r from-[#10B981]/10 to-[#10B981]/5 border border-[#10B981]/20">
        <h3 className="font-semibold text-[#10B981] mb-2">
          🎉 Track Completed!
        </h3>
        <p className="text-sm text-[--color-text-secondary] mb-4">
          Congratulations! You&apos;ve completed the entire {track.title} track.
          You now understand how to build transformers from scratch.
        </p>
        <div className="flex flex-wrap gap-2">
          <Link
            href={`/architectures/${track.architecture.slug}`}
            className="inline-flex items-center gap-1 px-3 py-2 rounded-lg bg-[#10B981]/20 text-[#10B981] hover:bg-[#10B981]/30 transition-colors text-sm font-medium"
          >
            View {track.architecture.title} Architecture <ChevronRight className="w-4 h-4" />
          </Link>
          <Link
            href={`/papers/${track.paper.slug}`}
            className="inline-flex items-center gap-1 px-3 py-2 rounded-lg bg-[--accent-primary]/20 text-[--accent-primary] hover:bg-[--accent-primary]/30 transition-colors text-sm font-medium"
          >
            Read {track.paper.title} <ChevronRight className="w-4 h-4" />
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 rounded-lg bg-[--accent-primary]/10 border border-[--accent-primary]/30">
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="text-xs font-semibold text-[--accent-primary] uppercase mb-1">
            Recommended Next
          </p>
          <h3 className="font-semibold text-[--color-text-primary]">
            {nextStep.order}. {nextStep.title}
          </h3>
        </div>
        {nextStep.isCapstone && (
          <span className="px-2 py-1 rounded text-xs font-bold bg-[--accent-primary] text-white">
            Capstone
          </span>
        )}
      </div>

      <p className="text-sm text-[--color-text-secondary] mb-4">
        {nextStep.estimatedMinutes} min • {nextStep.difficulty}
      </p>

      {nextStep.prerequisites.length > 0 && (
        <div className="mb-4 p-3 rounded-lg bg-[--bg-surface]">
          <p className="text-xs font-semibold text-[--color-text-secondary] mb-2">
            Prerequisites to complete first:
          </p>
          <ul className="space-y-1">
            {nextStep.prerequisites.map((prereqId) => {
              const prereq = track.problems.find((p) => p.problemId === prereqId);
              const isCompleted = completedProblems.has(prereqId);
              return (
                <li
                  key={prereqId}
                  className="text-xs text-[--color-text-secondary] flex items-center gap-2"
                >
                  {isCompleted ? (
                    <span className="text-[#10B981]">✓</span>
                  ) : (
                    <Lock className="w-3 h-3 text-[--color-text-tertiary]" />
                  )}
                  {prereq?.title}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <Link
        href={`/problems/${nextStep.problemId}`}
        className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[--accent-primary] text-white font-semibold hover:opacity-90 transition-opacity"
      >
        Start Problem <ChevronRight className="w-4 h-4" />
      </Link>
    </div>
  );
}
