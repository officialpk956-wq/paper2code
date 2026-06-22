"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronRight, Award, Clock } from "lucide-react";
import Link from "next/link";
import { LEARNING_TRACKS } from "@/data/learning-tracks";
import { Breadcrumb } from "@/components/breadcrumb";
import { TrackRoadmap } from "@/components/learning-tracks/track-roadmap";
import { TrackProgress } from "@/components/learning-tracks/track-progress";
import { RecommendedNext } from "@/components/learning-tracks/recommended-next";
import type { LearningTrack } from "@/data/learning-tracks";

interface TrackPageClientProps {
  track: LearningTrack;
}

export function TrackPageClient({ track }: TrackPageClientProps) {
  const [completedProblems] = useState<Set<string>>(new Set());

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  };

  return (
    <div className="pb-20">
      {/* Header */}
      <div className="border-b border-[--color-border] bg-[--bg-surface] sticky top-14 z-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Breadcrumb
            items={[
              { label: "Problem Explorer", href: "/problems" },
              { label: "Learning Tracks", href: "/learning-tracks" },
              { label: track.title, current: true },
            ]}
          />
          <div className="mt-4">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="text-sm text-[--color-text-secondary] mb-1">
                  Learning Track
                </p>
                <h1 className="text-3xl font-bold text-[--color-text-primary]">
                  {track.title}
                </h1>
                <p className="text-[--color-text-secondary] mt-2">
                  {track.subtitle}
                </p>
              </div>
              <div className="text-right">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-[--accent-primary]/20 text-[--accent-primary] font-semibold">
                  {track.difficulty}
                </div>
              </div>
            </div>

            <div className="flex flex-wrap gap-4 text-sm">
              <div className="flex items-center gap-2 text-[--color-text-secondary]">
                <Clock className="w-4 h-4" />
                <span>{track.estimatedHours} hours</span>
              </div>
              <div className="flex items-center gap-2 text-[--color-text-secondary]">
                <Award className="w-4 h-4" />
                <span>{track.totalProblems} problems</span>
              </div>
              <Link
                href={`/architectures/${track.architecture.slug}`}
                className="flex items-center gap-2 text-[--accent-primary] hover:text-[--accent-cyan] transition-colors"
              >
                📘 {track.architecture.title}
                <ChevronRight className="w-4 h-4" />
              </Link>
              <Link
                href={`/papers/${track.paper.slug}`}
                className="flex items-center gap-2 text-[--accent-primary] hover:text-[--accent-cyan] transition-colors"
              >
                📄 {track.paper.title} ({track.paper.year})
                <ChevronRight className="w-4 h-4" />
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          className="grid lg:grid-cols-3 gap-8"
          variants={container}
          initial="hidden"
          animate="show"
        >
          {/* Main Content */}
          <motion.div className="lg:col-span-2 space-y-12" variants={item}>
            {/* Description */}
            <section>
              <h2 className="text-2xl font-bold mb-4 text-[--color-text-primary]">
                About This Track
              </h2>
              <p className="text-[--color-text-secondary] leading-relaxed mb-6">
                {track.description}
              </p>

              <div>
                <h3 className="text-sm font-semibold text-[--color-text-primary] mb-3">
                  Learning Objectives
                </h3>
                <ul className="space-y-2">
                  {track.keyObjectives.map((objective, idx) => (
                    <li key={idx} className="flex gap-3 text-[--color-text-secondary]">
                      <span className="text-[--accent-primary] font-bold flex-shrink-0">
                        •
                      </span>
                      <span>{objective}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </section>

            {/* Prerequisites */}
            <section>
              <h2 className="text-2xl font-bold mb-4 text-[--color-text-primary]">
                Prerequisites
              </h2>
              <div className="space-y-3">
                {track.prerequisites.knowledge.length > 0 && (
                  <div>
                    <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                      Knowledge Required
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {track.prerequisites.knowledge.map((prereq) => (
                        <span
                          key={prereq}
                          className="px-3 py-1 rounded-lg bg-[--bg-body] border border-[--color-border] text-sm text-[--color-text-secondary]"
                        >
                          {prereq}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {track.prerequisites.tracks && track.prerequisites.tracks.length > 0 && (
                  <div>
                    <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                      Required Tracks
                    </h3>
                    <div className="space-y-2">
                      {track.prerequisites.tracks.map((trackId) => {
                        const prereqTrack = LEARNING_TRACKS.find((t) => t.id === trackId);
                        return (
                          <Link
                            key={trackId}
                            href={`/learning-tracks/${prereqTrack?.slug}`}
                            className="flex items-center gap-2 p-3 rounded-lg bg-[--bg-body] border border-[--color-border] hover:border-[--accent-primary] transition-colors"
                          >
                            <span>{prereqTrack?.title}</span>
                            <ChevronRight className="w-4 h-4 text-[--color-text-tertiary]" />
                          </Link>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </section>

            {/* Roadmap */}
            <section>
              <h2 className="text-2xl font-bold mb-4 text-[--color-text-primary]">
                Problem Sequence
              </h2>
              <TrackRoadmap
                track={track}
                completedProblems={completedProblems}
              />
            </section>

            {/* Resources */}
            <section>
              <h2 className="text-2xl font-bold mb-4 text-[--color-text-primary]">
                Resources
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                <Link
                  href={`/architectures/${track.architecture.slug}`}
                  className="p-4 rounded-lg bg-[--bg-body] border border-[--color-border] hover:border-[--accent-primary] transition-colors"
                >
                  <h3 className="font-semibold text-[--color-text-primary] mb-2">
                    {track.architecture.title} Architecture
                  </h3>
                  <p className="text-sm text-[--color-text-secondary] mb-3">
                    Explore the architecture diagram, key facts, and interactive
                    modules.
                  </p>
                  <span className="inline-flex items-center gap-1 text-[--accent-primary] font-medium">
                    View Page <ChevronRight className="w-4 h-4" />
                  </span>
                </Link>

                <Link
                  href={`/papers/${track.paper.slug}`}
                  className="p-4 rounded-lg bg-[--bg-body] border border-[--color-border] hover:border-[--accent-primary] transition-colors"
                >
                  <h3 className="font-semibold text-[--color-text-primary] mb-2">
                    {track.paper.title}
                  </h3>
                  <p className="text-sm text-[--color-text-secondary] mb-3">
                    Read the original research paper that introduced these concepts.
                  </p>
                  <span className="inline-flex items-center gap-1 text-[--accent-primary] font-medium">
                    Read Paper <ChevronRight className="w-4 h-4" />
                  </span>
                </Link>
              </div>
            </section>
          </motion.div>

          {/* Sidebar */}
          <motion.div className="space-y-6" variants={item}>
            <TrackProgress track={track} completedProblems={completedProblems} />

            <div className="border-t border-[--color-border] pt-6">
              <RecommendedNext
                track={track}
                completedProblems={completedProblems}
              />
            </div>

            {/* Tips */}
            <div className="p-4 rounded-lg bg-gradient-to-r from-[--accent-primary]/10 to-[--accent-cyan]/10 border border-[--accent-primary]/20">
              <h3 className="font-semibold text-[--color-text-primary] mb-3">
                Tips for Success
              </h3>
              <ul className="space-y-2 text-sm text-[--color-text-secondary]">
                <li>• Complete problems in order to build understanding</li>
                <li>• Read hints before looking at solutions</li>
                <li>• Review explanations after each problem</li>
                <li>• Reference the architecture and paper as needed</li>
              </ul>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
