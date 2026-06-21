"use client";

import { motion } from "framer-motion";
import { CheckCircle2, Circle, Lock } from "lucide-react";
import Link from "next/link";
import { LearningTrack, TrackStep } from "@/data/learning-tracks";

interface TrackRoadmapProps {
  track: LearningTrack;
  completedProblems: Set<string>;
}

export function TrackRoadmap({
  track,
  completedProblems,
}: TrackRoadmapProps) {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.05 },
    },
  };

  const item = {
    hidden: { opacity: 0, x: -10 },
    show: { opacity: 1, x: 0 },
  };

  const getStepStatus = (step: TrackStep) => {
    if (completedProblems.has(step.problemId)) return "completed";
    if (
      step.prerequisites.every((prereqId) => completedProblems.has(prereqId))
    )
      return "available";
    return "locked";
  };

  return (
    <motion.div
      className="space-y-3"
      variants={container}
      initial="hidden"
      animate="show"
    >
      {track.problems.map((step, idx) => {
        const status = getStepStatus(step);
        const isCompleted = status === "completed";
        const isAvailable = status === "available";
        const isLocked = status === "locked";

        return (
          <motion.div key={step.problemId} variants={item}>
            <Link href={`/problems/${step.problemId}`}>
              <div
                className={`p-4 rounded-lg border transition-all ${
                  isCompleted
                    ? "bg-[#10B981]/10 border-[#10B981]/30"
                    : isAvailable
                      ? "bg-[--accent-primary]/10 border-[--accent-primary]/50 hover:border-[--accent-primary] cursor-pointer"
                      : "bg-[--bg-body] border-[--color-border] opacity-60"
                }`}
              >
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 pt-1">
                    {isCompleted ? (
                      <CheckCircle2 className="w-6 h-6 text-[#10B981]" />
                    ) : isAvailable ? (
                      <Circle className="w-6 h-6 text-[--accent-primary]" />
                    ) : (
                      <Lock className="w-6 h-6 text-[--color-text-tertiary]" />
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-mono text-[--color-text-tertiary]">
                        Step {step.order} of {track.totalProblems}
                      </span>
                      {step.isCapstone && (
                        <span className="px-2 py-0.5 rounded text-xs font-semibold bg-[--accent-primary]/20 text-[--accent-primary]">
                          Capstone
                        </span>
                      )}
                    </div>

                    <h3
                      className={`font-semibold ${
                        isLocked
                          ? "text-[--color-text-secondary]"
                          : "text-[--color-text-primary]"
                      }`}
                    >
                      {step.title}
                    </h3>

                    <div className="flex items-center gap-3 mt-2">
                      <span className="text-xs text-[--color-text-tertiary]">
                        ⏱️ {step.estimatedMinutes} min
                      </span>
                      <span
                        className={`text-xs font-medium px-2 py-1 rounded ${
                          step.difficulty === "beginner"
                            ? "bg-[#10B981]/20 text-[#10B981]"
                            : step.difficulty === "intermediate"
                              ? "bg-[#F59E0B]/20 text-[#F59E0B]"
                              : "bg-[#EF4444]/20 text-[#EF4444]"
                        }`}
                      >
                        {step.difficulty}
                      </span>
                    </div>

                    {step.prerequisites.length > 0 && (
                      <div className="mt-2 text-xs text-[--color-text-tertiary]">
                        Prerequisites:{" "}
                        {step.prerequisites
                          .map((pId) => {
                            const prereqStep = track.problems.find(
                              (p) => p.problemId === pId
                            );
                            return prereqStep ? (
                              <span key={pId}>
                                {prereqStep.title}
                                {completedProblems.has(pId) && " ✓"}
                              </span>
                            ) : null;
                          })
                          .filter(Boolean)
                          .map((node, i) => (
                            <span key={i}>
                              {i > 0 && ", "}
                              {node}
                            </span>
                          ))}
                      </div>
                    )}

                    {step.conceptsIntroduced.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {step.conceptsIntroduced.map((concept) => (
                          <span
                            key={concept}
                            className="px-2 py-0.5 rounded text-xs bg-[--bg-surface] text-[--color-text-secondary]"
                          >
                            {concept.replace(/-/g, " ")}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </Link>

            {idx < track.problems.length - 1 && (
              <div className="flex justify-center py-1">
                <div className="w-0.5 h-4 bg-[--color-border]" />
              </div>
            )}
          </motion.div>
        );
      })}
    </motion.div>
  );
}
