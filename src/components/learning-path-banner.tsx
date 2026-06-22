"use client";

import { useState } from "react";
import { ChevronRight, X, Zap, CheckCircle2 } from "lucide-react";

interface PathMilestone {
  id: string;
  title: string;
  status: "completed" | "current" | "upcoming";
  progress?: number;
}

interface LearningPathBannerProps {
  path: string;
  milestones: PathMilestone[];
  onClose?: () => void;
}

export function LearningPathBanner({
  path,
  milestones,
  onClose,
}: LearningPathBannerProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const completedCount = milestones.filter(
    (m) => m.status === "completed"
  ).length;
  const totalCount = milestones.length;
  const progressPercent = (completedCount / totalCount) * 100;

  if (collapsed) {
    return (
      <div className="fixed bottom-6 right-6 z-20">
        <button
          onClick={() => setCollapsed(false)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] text-white text-sm font-medium shadow-lg hover:opacity-90 transition-opacity"
        >
          <Zap className="w-4 h-4" />
          {path}
          <span className="text-xs bg-white/20 px-2 py-0.5 rounded">
            {completedCount}/{totalCount}
          </span>
        </button>
      </div>
    );
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 z-20 border-t border-[--color-border] bg-[--bg-surface]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Zap className="w-5 h-5 text-[--accent-cyan]" />
            <div>
              <h3 className="text-sm font-bold text-[--color-text-primary]">
                {path}
              </h3>
              <p className="text-xs text-[--color-text-tertiary]">
                {completedCount} of {totalCount} milestones complete
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-sm font-medium text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors flex items-center gap-1"
            >
              Details
              <ChevronRight
                className={`w-4 h-4 transition-transform ${
                  showDetails ? "rotate-90" : ""
                }`}
              />
            </button>

            <button
              onClick={() => setCollapsed(true)}
              className="p-1 hover:bg-[--bg-panel] rounded transition-colors"
            >
              <ChevronRight className="w-5 h-5 rotate-180" />
            </button>

            {onClose && (
              <button
                onClick={onClose}
                className="p-1 hover:bg-[--bg-panel] rounded transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>

        {/* Progress bar */}
        <div className="mb-4">
          <div className="h-2 bg-[--bg-body] rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        {/* Milestones */}
        {showDetails && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
            {milestones.map((milestone) => (
              <div
                key={milestone.id}
                className={`p-3 rounded-lg border transition-colors ${
                  milestone.status === "completed"
                    ? "bg-[rgb(16,185,129,0.1)] border-[rgb(16,185,129,0.3)]"
                    : milestone.status === "current"
                      ? "bg-[rgb(124,58,237,0.1)] border-[--accent-primary]"
                      : "bg-[--bg-body] border-[--color-border]"
                }`}
              >
                <div className="flex items-start gap-2">
                  {milestone.status === "completed" && (
                    <CheckCircle2 className="w-4 h-4 text-[#10B981] flex-shrink-0 mt-0.5" />
                  )}
                  {milestone.status === "current" && (
                    <div className="w-4 h-4 rounded-full bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] flex-shrink-0 mt-0.5" />
                  )}
                  {milestone.status === "upcoming" && (
                    <div className="w-4 h-4 rounded-full border border-[--color-border] flex-shrink-0 mt-0.5" />
                  )}
                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-medium text-[--color-text-primary]">
                      {milestone.title}
                    </p>
                    {milestone.progress !== undefined && (
                      <p className="text-xs text-[--color-text-tertiary] mt-1">
                        {milestone.progress}%
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
