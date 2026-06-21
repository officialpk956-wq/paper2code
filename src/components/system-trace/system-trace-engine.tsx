"use client";

import { useState, useEffect } from "react";
import { SystemTrace } from "@/data/system-traces";
import { Breadcrumb } from "@/components/breadcrumb";
import { SystemFlowPanel } from "./system-flow-panel";
import { SystemStepCard } from "./system-step-card";

export function SystemTraceEngine({ trace }: { trace: SystemTrace }) {
  const [currentIdx, setCurrentIdx] = useState(0);
  const step = trace.steps[currentIdx];

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") setCurrentIdx(i => Math.min(i + 1, trace.steps.length - 1));
      if (e.key === "ArrowLeft") setCurrentIdx(i => Math.max(i - 1, 0));
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [trace.steps.length]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-[--color-border] bg-[--bg-surface] px-6 py-4 flex items-center justify-between">
        <div>
          <Breadcrumb items={[
            { label: "System Design", href: "/system-design" },
            { label: trace.title, current: true }
          ]} />
          <h1 className="text-xl font-bold text-[--color-text-primary] mt-2">{trace.title}</h1>
        </div>
        <div className="flex flex-col items-end gap-2 w-64">
          <span className="text-xs font-semibold text-[--color-text-tertiary]">
            Step {currentIdx + 1} of {trace.steps.length}
          </span>
          <div className="w-full bg-[--bg-panel] h-2 rounded-full overflow-hidden">
            <div
              className="bg-emerald-500 h-full rounded-full transition-all duration-300 ease-out"
              style={{ width: `${((currentIdx + 1) / trace.steps.length) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Main Layout */}
      <div className="flex-1 min-h-0 flex overflow-hidden">
        {/* Left: Flow Panel */}
        <div className="w-72 flex-shrink-0 border-r border-[--color-border] bg-[--bg-surface] overflow-y-auto hidden md:block">
          <SystemFlowPanel trace={trace} currentIndex={currentIdx} onSelectStep={setCurrentIdx} />
        </div>

        {/* Center: Step + Nav */}
        <div className="flex-1 flex flex-col min-w-0 bg-[--bg-body] overflow-y-auto">
          <div className="max-w-3xl w-full mx-auto p-6 flex flex-col gap-5">
            {/* Navigation */}
            <div className="flex items-center justify-between p-3 rounded-lg bg-[--bg-surface] border border-[--color-border]">
              <button
                onClick={() => setCurrentIdx(i => Math.max(i - 1, 0))}
                disabled={currentIdx === 0}
                className="flex items-center gap-2 px-4 py-2 rounded-md bg-[--bg-panel] border border-[--color-border] text-sm font-semibold disabled:opacity-40 hover:bg-[--color-border] transition-colors"
              >
                ← Prev
              </button>
              <span className="text-sm font-bold text-[--color-text-secondary]">
                {step.name}
              </span>
              <button
                onClick={() => setCurrentIdx(i => Math.min(i + 1, trace.steps.length - 1))}
                disabled={currentIdx === trace.steps.length - 1}
                className="flex items-center gap-2 px-4 py-2 rounded-md bg-emerald-600 text-white text-sm font-semibold disabled:opacity-40 hover:brightness-110 transition-all"
              >
                Next →
              </button>
            </div>

            <SystemStepCard step={step} />
          </div>
        </div>
      </div>
    </div>
  );
}
