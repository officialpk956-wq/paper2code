"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  ChevronDown,
  ChevronRight,
  Clock,
  Target,
  BookOpen,
  Lightbulb,
  CheckCircle,
} from "lucide-react";
import Link from "next/link";
import {
  getTrackByProblemId,
  getNextProblemInTrack,
  getPreviousProblemInTrack,
} from "@/data/learning-tracks";
import { Breadcrumb } from "@/components/breadcrumb";
import type { Problem } from "@/data/problems";

const DIFFICULTY_COLOR = {
  beginner: "text-[#10B981]",
  intermediate: "text-[#F59E0B]",
  advanced: "text-[#EF4444]",
};

const DIFFICULTY_BG = {
  beginner: "bg-[#10B981]/10 border-[#10B981]/30",
  intermediate: "bg-[#F59E0B]/10 border-[#F59E0B]/30",
  advanced: "bg-[#EF4444]/10 border-[#EF4444]/30",
};

interface ProblemDetailClientProps {
  problem: Problem;
}

export function ProblemDetailClient({ problem }: ProblemDetailClientProps) {
  const [expandedHintLevel, setExpandedHintLevel] = useState<string | null>(
    null
  );
  const [completedHints, setCompletedHints] = useState<Set<string>>(
    new Set()
  );

  const toggleHint = (level: string) => {
    setExpandedHintLevel(expandedHintLevel === level ? null : level);
    setCompletedHints(new Set([...completedHints, level]));
  };

  const track = getTrackByProblemId(problem.id);
  const trackStep = track?.problems.find((p) => p.problemId === problem.id);
  const nextProblem = track
    ? getNextProblemInTrack(track.slug, problem.id)
    : undefined;
  const prevProblem = track
    ? getPreviousProblemInTrack(track.slug, problem.id)
    : undefined;

  const hints = [
    { level: "hint1", title: "Hint 1", content: problem.hints.hint1 },
    { level: "hint2", title: "Hint 2", content: problem.hints.hint2 },
    { level: "hint3", title: "Hint 3", content: problem.hints.hint3 },
    {
      level: "solutionOutline",
      title: "Solution Outline",
      content: problem.hints.solutionOutline,
    },
    {
      level: "fullSolution",
      title: "Full Solution",
      content: problem.hints.fullSolution,
    },
  ];

  const testCasesVisible = problem.testCases.filter((tc) => tc.visible);
  const testCasesHidden = problem.testCases.filter((tc) => !tc.visible);

  return (
    <div className="pb-20">
      <div className="border-b border-[--color-border] bg-[--bg-surface] sticky top-14 z-10">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Breadcrumb
            items={[
              { label: "Problems", href: "/problems" },
              ...(track
                ? [{ label: track.title, href: `/learning-tracks/${track.slug}` }]
                : []),
              { label: problem.title, current: true },
            ]}
          />

          <div className="mt-4 flex items-start justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2 text-[--color-text-primary]">
                {problem.title}
              </h1>
              <p className="text-[--color-text-secondary]">
                {problem.category.replace(/-/g, " ")} - {problem.tags.join(", ")}
              </p>
            </div>

            <div className="flex items-center gap-3">
              <div
                className={`px-3 py-1 rounded-lg border font-semibold text-sm ${
                  DIFFICULTY_BG[problem.difficulty]
                } ${DIFFICULTY_COLOR[problem.difficulty]}`}
              >
                {problem.difficulty}
              </div>
              <div className="flex items-center gap-1 text-[--color-text-secondary]">
                <Clock className="w-4 h-4" />
                <span className="text-sm">{problem.estimatedTime} min</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {track && trackStep && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 p-4 rounded-lg bg-[--accent-primary]/10 border border-[--accent-primary]/20"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs font-semibold text-[--accent-primary] uppercase mb-1">
                  Part of: {track.title}
                </p>
                <p className="text-sm text-[--color-text-secondary]">
                  Step {trackStep.order} of {track.totalProblems}
                </p>
              </div>
              <Link
                href={`/learning-tracks/${track.slug}`}
                className="inline-flex items-center gap-1 px-3 py-1 rounded text-sm font-medium text-[--accent-primary] hover:bg-[--accent-primary]/10 transition-colors"
              >
                View Track <ChevronRight className="w-4 h-4" />
              </Link>
            </div>
          </motion.div>
        )}

        <div className="grid md:grid-cols-3 gap-8">
          <div className="md:col-span-2 space-y-8">
            <section>
              <h2 className="text-xl font-bold mb-4 text-[--color-text-primary] flex items-center gap-2">
                <Target className="w-5 h-5 text-[--accent-primary]" />
                Problem
              </h2>
              <p className="text-[--color-text-secondary] leading-relaxed">
                {problem.description}
              </p>
            </section>

            <section>
              <h2 className="text-xl font-bold mb-4 text-[--color-text-primary] flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-[--accent-primary]" />
                Learning Points
              </h2>
              <ul className="space-y-2">
                {problem.learningPoints.map((point, idx) => (
                  <li
                    key={idx}
                    className="flex gap-3 text-[--color-text-secondary]"
                  >
                    <CheckCircle className="w-5 h-5 text-[--accent-cyan] flex-shrink-0 mt-0.5" />
                    <span>{point}</span>
                  </li>
                ))}
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-bold mb-4 text-[--color-text-primary]">
                Test Cases ({testCasesVisible.length} visible)
              </h2>
              <div className="space-y-3">
                {testCasesVisible.map((tc, idx) => (
                  <div
                    key={idx}
                    className="p-4 rounded-lg bg-[--bg-body] border border-[--color-border]"
                  >
                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs font-semibold text-[--color-text-tertiary] uppercase mb-2">
                          Input
                        </p>
                        <code className="text-sm text-[--color-text-secondary] font-mono">
                          {JSON.stringify(tc.input, null, 2)}
                        </code>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-[--color-text-tertiary] uppercase mb-2">
                          Expected Output
                        </p>
                        <code className="text-sm text-[--color-text-secondary] font-mono">
                          {JSON.stringify(tc.output, null, 2)}
                        </code>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              {testCasesHidden.length > 0 && (
                <p className="text-xs text-[--color-text-tertiary] mt-3">
                  + {testCasesHidden.length} hidden test
                  {testCasesHidden.length > 1 ? "s" : ""} (run to check)
                </p>
              )}
            </section>

            <section>
              <h2 className="text-xl font-bold mb-4 text-[--color-text-primary] flex items-center gap-2">
                <Lightbulb className="w-5 h-5 text-[--accent-primary]" />
                Hints and Solutions
              </h2>
              <div className="space-y-2">
                {hints.map((hint) => (
                  <motion.div key={hint.level}>
                    <button
                      onClick={() => toggleHint(hint.level)}
                      className={`w-full p-4 rounded-lg border transition-colors text-left ${
                        expandedHintLevel === hint.level
                          ? "bg-[--accent-primary]/10 border-[--accent-primary]"
                          : "bg-[--bg-body] border-[--color-border] hover:border-[--accent-primary]/50"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="text-sm font-semibold text-[--color-text-primary]">
                            {hint.title}
                          </span>
                          {completedHints.has(hint.level) && (
                            <CheckCircle className="w-4 h-4 text-[--accent-cyan]" />
                          )}
                        </div>
                        <ChevronDown
                          className={`w-5 h-5 text-[--accent-primary] transition-transform ${
                            expandedHintLevel === hint.level ? "rotate-180" : ""
                          }`}
                        />
                      </div>
                    </button>

                    {expandedHintLevel === hint.level && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.2 }}
                        className="mt-2 p-4 rounded-lg bg-[--accent-primary]/5 border border-[--accent-primary]/20"
                      >
                        <p className="text-[--color-text-secondary] whitespace-pre-wrap">
                          {hint.content}
                        </p>
                      </motion.div>
                    )}
                  </motion.div>
                ))}
              </div>
            </section>

            <section>
              <h2 className="text-xl font-bold mb-4 text-[--color-text-primary]">
                Explanation
              </h2>
              <div className="space-y-6">
                <div>
                  <h3 className="font-semibold text-[--color-text-primary] mb-2">
                    Step-by-Step Walkthrough
                  </h3>
                  <p className="text-[--color-text-secondary] whitespace-pre-wrap">
                    {problem.explanation.stepByStep}
                  </p>
                </div>

                <div>
                  <h3 className="font-semibold text-[--color-text-primary] mb-2">
                    Complexity Analysis
                  </h3>
                  <p className="text-[--color-text-secondary] whitespace-pre-wrap">
                    {problem.explanation.complexityAnalysis}
                  </p>
                </div>

                <div>
                  <h3 className="font-semibold text-[--color-text-primary] mb-2">
                    Common Mistakes
                  </h3>
                  <p className="text-[--color-text-secondary] whitespace-pre-wrap">
                    {problem.explanation.commonMistakes}
                  </p>
                </div>

                <div>
                  <h3 className="font-semibold text-[--color-text-primary] mb-2">
                    Interview Discussion
                  </h3>
                  <p className="text-[--color-text-secondary] whitespace-pre-wrap">
                    {problem.explanation.interviewDiscussion}
                  </p>
                </div>
              </div>
            </section>
          </div>

          <div className="md:col-span-1 space-y-6">
            {track && (
              <div className="space-y-3 p-4 rounded-lg bg-[--bg-body] border border-[--color-border]">
                <h3 className="font-semibold text-[--color-text-primary] text-sm">
                  Track Navigation
                </h3>

                {prevProblem && (
                  <Link
                    href={`/problems/${prevProblem.problemId}`}
                    className="flex items-center gap-2 p-2 rounded text-sm hover:bg-[--bg-surface] transition-colors text-left"
                  >
                    <ChevronRight className="w-4 h-4 rotate-180" />
                    <div className="min-w-0">
                      <p className="text-xs text-[--color-text-tertiary]">
                        Previous
                      </p>
                      <p className="text-[--color-text-secondary] truncate">
                        {prevProblem.title}
                      </p>
                    </div>
                  </Link>
                )}

                {nextProblem && (
                  <Link
                    href={`/problems/${nextProblem.problemId}`}
                    className="flex items-center gap-2 p-2 rounded text-sm hover:bg-[--bg-surface] transition-colors text-left"
                  >
                    <div className="min-w-0">
                      <p className="text-xs text-[--color-text-tertiary]">
                        Next
                      </p>
                      <p className="text-[--color-text-secondary] truncate">
                        {nextProblem.title}
                      </p>
                    </div>
                    <ChevronRight className="w-4 h-4" />
                  </Link>
                )}
              </div>
            )}

            {problem.relatedArchitectures.length > 0 && (
              <div className="p-4 rounded-lg bg-[--bg-body] border border-[--color-border]">
                <h3 className="font-semibold text-[--color-text-primary] mb-3 text-sm">
                  Related Architectures
                </h3>
                <div className="space-y-2">
                  {problem.relatedArchitectures.map((arch) => (
                    <a
                      key={arch}
                      href={`/architectures/${arch
                        .toLowerCase()
                        .replace(/\s+/g, "-")}`}
                      className="flex items-center gap-2 text-sm text-[--accent-primary] hover:text-[--accent-cyan] transition-colors"
                    >
                      <ChevronRight className="w-4 h-4" />
                      {arch}
                    </a>
                  ))}
                </div>
              </div>
            )}

            {problem.relatedPapers.length > 0 && (
              <div className="p-4 rounded-lg bg-[--bg-body] border border-[--color-border]">
                <h3 className="font-semibold text-[--color-text-primary] mb-3 text-sm">
                  Related Papers
                </h3>
                <div className="space-y-2">
                  {problem.relatedPapers.map((paper) => (
                    <a
                      key={paper}
                      href={`/papers/${paper
                        .toLowerCase()
                        .replace(/\s+/g, "-")}`}
                      className="flex items-center gap-2 text-sm text-[--accent-primary] hover:text-[--accent-cyan] transition-colors"
                    >
                      <ChevronRight className="w-4 h-4" />
                      {paper}
                    </a>
                  ))}
                </div>
              </div>
            )}

            {problem.relatedMath.length > 0 && (
              <div className="p-4 rounded-lg bg-[--bg-body] border border-[--color-border]">
                <h3 className="font-semibold text-[--color-text-primary] mb-3 text-sm">
                  Math Concepts
                </h3>
                <div className="flex flex-wrap gap-2">
                  {problem.relatedMath.map((math) => (
                    <span
                      key={math}
                      className="px-2 py-1 rounded bg-[--accent-primary]/10 text-[--accent-primary] text-xs font-medium"
                    >
                      {math}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
