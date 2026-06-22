"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronRight, BookOpen, Award } from "lucide-react";
import Link from "next/link";
import { PROBLEMS } from "@/data/problems";
import { LEARNING_TRACKS, getTrackByProblemId } from "@/data/learning-tracks";
import { Breadcrumb } from "@/components/breadcrumb";

type FilterType = "all" | "track" | "difficulty" | "paper";

export default function ProblemsPage() {
  const [filterType, setFilterType] = useState<FilterType>("all");
  const [selectedTrack, setSelectedTrack] = useState<string | null>(null);
  const [selectedDifficulty, setSelectedDifficulty] = useState<string | null>(
    null
  );

  // Filter problems based on selection
  const filteredProblems = PROBLEMS.filter((problem) => {
    if (filterType === "track" && selectedTrack) {
      const track = getTrackByProblemId(problem.id);
      return track?.slug === selectedTrack;
    }
    if (filterType === "difficulty" && selectedDifficulty) {
      return problem.difficulty === selectedDifficulty;
    }
    return true;
  });


  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.05 },
    },
  };

  const item = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0 },
  };

  return (
    <div className="pb-20">
      {/* Header */}
      <div className="border-b border-[--color-border] bg-[--bg-surface] sticky top-14 z-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Breadcrumb items={[{ label: "Problems", current: true }]} />
          <h1 className="text-3xl font-bold mt-4 mb-2 text-[--color-text-primary]">
            AI Engineering Problems
          </h1>
          <p className="text-[--color-text-secondary]">
            Structured learning paths to master transformers, CNNs, BERT, and more
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Learning Tracks Showcase */}
        <motion.section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-[--color-text-primary]">
            Learning Tracks
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            {LEARNING_TRACKS.map((track) => (
              <Link
                key={track.slug}
                href={`/learning-tracks/${track.slug}`}
                className="p-6 rounded-lg bg-[--bg-body] border border-[--color-border] hover:border-[--accent-primary] transition-colors group"
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="font-bold text-[--color-text-primary] group-hover:text-[--accent-primary] transition-colors">
                      {track.title}
                    </h3>
                    <p className="text-xs text-[--color-text-tertiary] mt-1">
                      {track.architecture.title} • {track.paper.year}
                    </p>
                  </div>
                  <ChevronRight className="w-5 h-5 text-[--color-text-tertiary] group-hover:text-[--accent-primary] transition-colors" />
                </div>
                <p className="text-sm text-[--color-text-secondary] mb-4">
                  {track.description.substring(0, 100)}...
                </p>
                <div className="flex items-center gap-4 text-xs text-[--color-text-tertiary]">
                  <span>📚 {track.totalProblems} problems</span>
                  <span>⏱️ {track.estimatedHours}h</span>
                </div>
              </Link>
            ))}
          </div>
        </motion.section>

        {/* Browse Section */}
        <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="mb-6">
            <h2 className="text-2xl font-bold mb-4 text-[--color-text-primary]">
              Browse Problems
            </h2>

            {/* Filter Tabs */}
            <div className="flex flex-wrap gap-2 mb-6">
              <button
                onClick={() => {
                  setFilterType("all");
                  setSelectedTrack(null);
                  setSelectedDifficulty(null);
                }}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  filterType === "all"
                    ? "bg-[--accent-primary] text-white"
                    : "bg-[--bg-body] text-[--color-text-secondary] hover:bg-[--bg-surface]"
                }`}
              >
                All ({PROBLEMS.length})
              </button>

              <div className="border-l border-[--color-border]" />

              <div className="flex gap-2">
                <span className="px-4 py-2 text-sm font-medium text-[--color-text-tertiary]">
                  Tracks:
                </span>
                {LEARNING_TRACKS.map((track) => (
                  <button
                    key={track.slug}
                    onClick={() => {
                      setFilterType("track");
                      setSelectedTrack(track.slug);
                      setSelectedDifficulty(null);
                    }}
                    className={`px-3 py-1 rounded text-sm transition-colors ${
                      filterType === "track" && selectedTrack === track.slug
                        ? "bg-[--accent-primary] text-white"
                        : "bg-[--bg-body] text-[--color-text-secondary] hover:bg-[--bg-surface]"
                    }`}
                  >
                    {track.title}
                  </button>
                ))}
              </div>

              <div className="border-l border-[--color-border]" />

              <div className="flex gap-2">
                <span className="px-4 py-2 text-sm font-medium text-[--color-text-tertiary]">
                  Difficulty:
                </span>
                {["beginner", "intermediate", "advanced"].map((diff) => (
                  <button
                    key={diff}
                    onClick={() => {
                      setFilterType("difficulty");
                      setSelectedDifficulty(diff);
                      setSelectedTrack(null);
                    }}
                    className={`px-3 py-1 rounded text-sm transition-colors capitalize ${
                      filterType === "difficulty" && selectedDifficulty === diff
                        ? "bg-[--accent-primary] text-white"
                        : "bg-[--bg-body] text-[--color-text-secondary] hover:bg-[--bg-surface]"
                    }`}
                  >
                    {diff}
                  </button>
                ))}
              </div>
            </div>

            {/* Results */}
            <div className="text-sm text-[--color-text-tertiary] mb-4">
              Showing {filteredProblems.length} of {PROBLEMS.length} problems
            </div>
          </div>

          {/* Problem Grid */}
          <motion.div
            className="grid md:grid-cols-2 gap-4"
            variants={container}
            initial="hidden"
            animate="show"
          >
            {filteredProblems.map((problem) => {
              const track = getTrackByProblemId(problem.id);
              const trackStep = track?.problems.find(
                (p) => p.problemId === problem.id
              );

              return (
                <motion.div key={problem.id} variants={item}>
                  <Link href={`/problems/${problem.slug}`}>
                    <div className="p-5 rounded-lg bg-[--bg-body] border border-[--color-border] hover:border-[--accent-primary] transition-all h-full cursor-pointer group">
                      {/* Track Badge */}
                      {track && trackStep && (
                        <div className="mb-3">
                          <Link
                            href={`/learning-tracks/${track.slug}`}
                            className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-semibold bg-[--accent-primary]/20 text-[--accent-primary] hover:bg-[--accent-primary]/30 transition-colors"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <BookOpen className="w-3 h-3" />
                            {track.title} ({trackStep.order}/{track.totalProblems})
                          </Link>
                        </div>
                      )}

                      <h3 className="font-bold text-[--color-text-primary] mb-2 group-hover:text-[--accent-primary] transition-colors">
                        {problem.title}
                      </h3>

                      <p className="text-sm text-[--color-text-secondary] mb-4 line-clamp-2">
                        {problem.description}
                      </p>

                      <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-3">
                          <span className="text-[--color-text-tertiary]">
                            ⏱️ {problem.estimatedTime} min
                          </span>
                          <span
                            className={`px-2 py-1 rounded font-medium ${
                              problem.difficulty === "beginner"
                                ? "bg-[#10B981]/20 text-[#10B981]"
                                : problem.difficulty === "intermediate"
                                  ? "bg-[#F59E0B]/20 text-[#F59E0B]"
                                  : "bg-[#EF4444]/20 text-[#EF4444]"
                            }`}
                          >
                            {problem.difficulty}
                          </span>
                        </div>
                        <ChevronRight className="w-4 h-4 text-[--color-text-tertiary] group-hover:text-[--accent-primary] transition-colors" />
                      </div>

                      {/* Tags */}
                      {problem.tags.length > 0 && (
                        <div className="mt-4 flex flex-wrap gap-1">
                          {problem.tags.slice(0, 2).map((tag) => (
                            <span
                              key={tag}
                              className="px-2 py-0.5 rounded text-xs bg-[--bg-surface] text-[--color-text-tertiary]"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </Link>
                </motion.div>
              );
            })}
          </motion.div>

          {filteredProblems.length === 0 && (
            <div className="text-center py-12">
              <Award className="w-12 h-12 mx-auto text-[--color-text-tertiary] mb-3 opacity-50" />
              <p className="text-[--color-text-secondary]">
                No problems found for this filter
              </p>
            </div>
          )}
        </motion.section>
      </div>
    </div>
  );
}
