"use client";

import { RoadmapNode } from "@/data/roadmaps";
import { X, Clock, Zap, BookOpen, Lock, CheckCircle2 } from "lucide-react";

interface NodeDetailPanelProps {
  node: RoadmapNode | null;
  onClose: () => void;
  onStart?: () => void;
}

export function NodeDetailPanel({ node, onClose, onStart }: NodeDetailPanelProps) {
  if (!node) return null;

  const isCompleted = node.state === "completed";
  const isLocked = node.state === "locked";
  const isInProgress = node.state === "in_progress";

  return (
    <div className="fixed inset-0 z-40">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="absolute right-0 top-0 bottom-0 w-full max-w-md bg-[--bg-surface] border-l border-[--color-border] overflow-y-auto shadow-2xl animate-in slide-in-from-right-96 duration-300">
        {/* Header */}
        <div className="sticky top-0 border-b border-[--color-border] bg-[--bg-surface] p-6 flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              {isCompleted && (
                <CheckCircle2 className="w-5 h-5 text-[#10B981]" />
              )}
              {isInProgress && (
                <div className="w-4 h-4 rounded-full bg-[--accent-primary] animate-pulse" />
              )}
              {isLocked && <Lock className="w-5 h-5 text-[--color-text-tertiary]" />}
            </div>
            <h2 className="text-2xl font-bold">{node.title}</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-[--bg-panel] rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Status badge */}
          <div className="flex gap-2">
            <span
              className={`text-xs font-semibold px-3 py-1 rounded-full ${
                isCompleted
                  ? "bg-[rgb(16,185,129,0.15)] text-[#6EE7B7]"
                  : isInProgress
                    ? "bg-[rgb(124,58,237,0.15)] text-[--accent-primary]"
                    : isLocked
                      ? "bg-[--bg-body] text-[--color-text-tertiary]"
                      : "bg-[rgb(124,58,237,0.1)] text-[--accent-primary]"
              }`}
            >
              {isCompleted
                ? "✓ Completed"
                : isInProgress
                  ? "In Progress"
                  : isLocked
                    ? "Locked"
                    : "Available"}
            </span>
            <span
              className={`text-xs font-semibold px-3 py-1 rounded-full ${
                node.difficulty === "beginner"
                  ? "bg-[#10B981]/20 text-[#6EE7B7]"
                  : node.difficulty === "intermediate"
                    ? "bg-[#F59E0B]/20 text-[#FCD34D]"
                    : "bg-[#EF4444]/20 text-[#FCA5A5]"
              }`}
            >
              {node.difficulty}
            </span>
          </div>

          {/* Why it matters */}
          <div>
            <h3 className="font-semibold mb-2 flex items-center gap-2">
              <Zap className="w-4 h-4 text-[--accent-primary]" />
              Why It Matters
            </h3>
            <p className="text-sm text-[--color-text-secondary] leading-relaxed">
              {node.whyItMatters}
            </p>
          </div>

          {/* Description */}
          <div>
            <h3 className="font-semibold mb-2">What You&apos;ll Learn</h3>
            <p className="text-sm text-[--color-text-secondary] leading-relaxed">
              {node.description}
            </p>
          </div>

          {/* Meta info */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-[--bg-body] border border-[--color-border]">
              <div className="flex items-center gap-2 mb-1">
                <Clock className="w-4 h-4 text-[--color-text-tertiary]" />
                <span className="text-xs font-semibold text-[--color-text-tertiary]">
                  Estimated Time
                </span>
              </div>
              <p className="text-sm font-semibold">{node.estimatedTime}</p>
            </div>

            <div className="p-3 rounded-lg bg-[--bg-body] border border-[--color-border]">
              <div className="flex items-center gap-2 mb-1">
                <BookOpen className="w-4 h-4 text-[--color-text-tertiary]" />
                <span className="text-xs font-semibold text-[--color-text-tertiary]">
                  Difficulty
                </span>
              </div>
              <p className="text-sm font-semibold capitalize">{node.difficulty}</p>
            </div>
          </div>

          {/* Prerequisites */}
          {node.prerequisites.length > 0 && (
            <div>
              <h3 className="font-semibold mb-2">Prerequisites</h3>
              <div className="space-y-2">
                {node.prerequisites.map((prereq) => (
                  <div
                    key={prereq}
                    className="p-2 rounded-lg bg-[--bg-body] border border-[--color-border] text-sm text-[--color-text-secondary]"
                  >
                    Complete &quot;{prereq}&quot; first
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Linked resources */}
          {node.linkedResources.length > 0 && (
            <div>
              <h3 className="font-semibold mb-2">Connected Resources</h3>
              <div className="space-y-2">
                {node.linkedResources.map((resource, idx) => (
                  <a
                    key={idx}
                    href={resource.url}
                    className="flex items-start gap-3 p-3 rounded-lg bg-[--bg-body] border border-[--color-border] hover:border-[--accent-primary] transition-colors text-sm"
                  >
                    <span className="text-sm font-semibold text-[--color-text-tertiary] bg-[--bg-surface] px-2 py-0.5 rounded whitespace-nowrap">
                      {resource.type}
                    </span>
                    <span className="text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
                      {resource.title}
                    </span>
                  </a>
                ))}
              </div>
            </div>
          )}

          {/* CTA buttons */}
          <div className="space-y-2 pt-4 border-t border-[--color-border]">
            {isLocked ? (
              <button
                disabled
                className="w-full py-3 rounded-lg bg-[--bg-body] text-[--color-text-tertiary] font-semibold cursor-not-allowed opacity-50"
              >
                Complete prerequisites to unlock
              </button>
            ) : isCompleted ? (
              <button className="w-full py-3 rounded-lg bg-[rgb(16,185,129,0.2)] border border-[#10B981] text-[#6EE7B7] font-semibold hover:bg-[rgb(16,185,129,0.3)] transition-colors">
                Review Completed Lesson
              </button>
            ) : !onStart ? (
              <button
                disabled
                className="w-full py-3 rounded-lg bg-[--bg-body] text-[--color-text-tertiary] font-semibold cursor-not-allowed opacity-50 border border-[--color-border]"
              >
                Coming soon
              </button>
            ) : isInProgress ? (
              <button
                onClick={onStart}
                className="w-full py-3 rounded-lg bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] text-white font-semibold hover:opacity-90 transition-opacity"
              >
                Continue Learning
              </button>
            ) : (
              <button
                onClick={onStart}
                className="w-full py-3 rounded-lg bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] text-white font-semibold hover:opacity-90 transition-opacity"
              >
                Start Lesson
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
