"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ChevronRight, BookOpen, Award, Clock } from "lucide-react";
import { LEARNING_TRACKS } from "@/data/learning-tracks";

interface ArchitectureLearningTrackProps {
  architectureSlug: string;
}

export function ArchitectureLearningTrack({
  architectureSlug,
}: ArchitectureLearningTrackProps) {
  const track = LEARNING_TRACKS.find((t) => t.architecture.slug === architectureSlug);

  if (!track) return null;

  const container = {
    hidden: { opacity: 0, y: 20 },
    show: {
      opacity: 1,
      y: 0,
      transition: { staggerChildren: 0.05 },
    },
  };

  const item = {
    hidden: { opacity: 0 },
    show: { opacity: 1 },
  };

  return (
    <motion.div
      className="mt-12 py-8 border-t border-[--color-border]"
      initial="hidden"
      whileInView="show"
      viewport={{ once: true }}
      variants={container}
    >
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2 text-[--color-text-primary]">
          {track.title}
        </h2>
        <p className="text-[--color-text-secondary]">
          A structured learning path with {track.totalProblems} problems to master this
          architecture
        </p>
      </div>

      <motion.div
        className="p-6 rounded-lg bg-gradient-to-r from-[--accent-primary]/10 to-[--accent-cyan]/10 border border-[--accent-primary]/20 mb-6"
        variants={item}
      >
        <p className="text-[--color-text-secondary] mb-4">{track.description}</p>

        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="flex items-center gap-2">
            <Award className="w-5 h-5 text-[--accent-primary]" />
            <div>
              <p className="text-xs text-[--color-text-tertiary]">Problems</p>
              <p className="font-bold text-[--color-text-primary]">
                {track.totalProblems}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-[--accent-primary]" />
            <div>
              <p className="text-xs text-[--color-text-tertiary]">Estimated</p>
              <p className="font-bold text-[--color-text-primary]">
                {track.estimatedHours}h
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-[--accent-primary]" />
            <div>
              <p className="text-xs text-[--color-text-tertiary]">Difficulty</p>
              <p className="font-bold text-[--color-text-primary] capitalize">
                {track.difficulty}
              </p>
            </div>
          </div>
        </div>

        <Link
          href={`/learning-tracks/${track.slug}`}
          className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-[--accent-primary] text-white font-semibold hover:opacity-90 transition-opacity"
        >
          Start Learning Path
          <ChevronRight className="w-5 h-5" />
        </Link>
      </motion.div>

      <motion.div
        className="grid md:grid-cols-2 gap-4"
        variants={container}
      >
        <motion.div
          className="p-4 rounded-lg bg-[--bg-body] border border-[--color-border]"
          variants={item}
        >
          <h3 className="font-semibold text-[--color-text-primary] mb-3">
            Key Objectives
          </h3>
          <ul className="space-y-2">
            {track.keyObjectives.map((objective, idx) => (
              <li key={idx} className="text-sm text-[--color-text-secondary] flex gap-2">
                <span className="text-[--accent-cyan]">✓</span>
                <span>{objective}</span>
              </li>
            ))}
          </ul>
        </motion.div>

        <motion.div
          className="p-4 rounded-lg bg-[--bg-body] border border-[--color-border]"
          variants={item}
        >
          <h3 className="font-semibold text-[--color-text-primary] mb-3">
            Problem Sequence
          </h3>
          <ol className="space-y-2 text-sm">
            {track.problems.slice(0, 5).map((step) => (
              <li key={step.problemId} className="text-[--color-text-secondary]">
                <span className="font-mono text-xs text-[--color-text-tertiary]">
                  {step.order}.
                </span>{" "}
                {step.title}
              </li>
            ))}
            {track.problems.length > 5 && (
              <li className="text-[--color-text-tertiary] text-xs">
                + {track.problems.length - 5} more...
              </li>
            )}
          </ol>
        </motion.div>
      </motion.div>
    </motion.div>
  );
}
