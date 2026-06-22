"use client";

import { SystemTrace } from "@/data/system-traces";
import { useEffect, useRef } from "react";
import { Clock } from "lucide-react";

interface Props {
  trace: SystemTrace;
  currentIndex: number;
  onSelectStep: (i: number) => void;
}

export function SystemFlowPanel({ trace, currentIndex, onSelectStep }: Props) {
  const activeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [currentIndex]);

  return (
    <div className="py-8 px-4">
      <h2 className="text-xs font-bold uppercase tracking-widest text-[--color-text-tertiary] mb-8 text-center">
        Request Flow
      </h2>
      <div className="relative flex flex-col items-center">
        {/* Vertical connector line */}
        <div className="absolute top-0 bottom-0 w-[2px] bg-[--color-border] left-1/2 -translate-x-1/2 -z-10" />

        {trace.steps.map((step, idx) => {
          const isActive = idx === currentIndex;
          const isPast = idx < currentIndex;

          return (
            <button
              key={step.id}
              ref={isActive ? activeRef : null}
              onClick={() => onSelectStep(idx)}
              className="relative w-full flex flex-col items-center mb-6 last:mb-0 group"
            >
              {/* Node dot */}
              <div className={`w-4 h-4 rounded-full border-2 absolute top-4 left-1/2 -translate-x-1/2 z-10 transition-all duration-300 ${
                isActive
                  ? "bg-emerald-500 border-emerald-500 shadow-[0_0_14px_rgba(16,185,129,0.5)]"
                  : isPast
                  ? "bg-emerald-500 border-emerald-500"
                  : "bg-[--bg-surface] border-[--color-border] group-hover:border-[--color-text-secondary]"
              }`} />

              {/* Card */}
              <div className={`w-full max-w-[230px] p-3 rounded-lg border transition-all duration-300 z-20 mt-1 ${
                isActive
                  ? "bg-[rgba(16,185,129,0.08)] border-emerald-500 shadow-md"
                  : "bg-[--bg-surface] border-[--color-border] opacity-70 scale-95 hover:opacity-100 hover:border-[--color-text-tertiary]"
              }`}>
                <div className={`text-xs font-bold mb-1 ${isActive ? "text-emerald-400" : "text-[--color-text-secondary]"}`}>
                  {step.name}
                </div>
                <div className="text-[10px] font-mono text-[--color-text-tertiary] truncate">{step.component}</div>
                {step.latency && (
                  <div className="flex items-center gap-1 mt-1.5 text-[10px] text-[--accent-cyan]">
                    <Clock className="w-2.5 h-2.5" /> {step.latency}
                  </div>
                )}
              </div>

              {idx < trace.steps.length - 1 && (
                <svg className="w-4 h-4 text-[--color-border] mt-1 z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
