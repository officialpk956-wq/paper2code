"use client";

import { ModelTrace } from "@/data/tensor-traces";
import { useEffect, useRef } from "react";
import { TensorShapeBadge } from "./tensor-shape-badge";

interface Props {
  trace: ModelTrace;
  currentIndex: number;
  onSelectStep: (index: number) => void;
}

export function ShapeFlowPanel({ trace, currentIndex, onSelectStep }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const activeStepRef = useRef<HTMLButtonElement>(null);

  // Auto-scroll to active step
  useEffect(() => {
    if (activeStepRef.current && containerRef.current) {
      activeStepRef.current.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [currentIndex]);

  return (
    <div className="py-8 px-4" ref={containerRef}>
      <h2 className="text-xs font-bold uppercase tracking-widest text-[--color-text-tertiary] mb-8 text-center">
        Shape Evolution
      </h2>

      <div className="relative flex flex-col items-center">
        {/* Continuous SVG line behind the nodes */}
        <div className="absolute top-0 bottom-0 w-[2px] bg-[--color-border] left-1/2 -translate-x-1/2 -z-10" />

        {trace.steps.map((step, idx) => {
          const isActive = idx === currentIndex;
          const isPast = idx < currentIndex;
          
          return (
            <button
              key={step.id}
              ref={isActive ? activeStepRef : null}
              onClick={() => onSelectStep(idx)}
              className="relative w-full flex flex-col items-center mb-8 last:mb-0 group"
            >
              {/* Connector dot on the line */}
              <div className={`w-4 h-4 rounded-full border-2 absolute top-4 left-1/2 -translate-x-1/2 z-10 transition-colors duration-300 ${
                isActive 
                  ? "bg-[--accent-primary] border-[--accent-primary] shadow-[0_0_15px_rgba(124,58,237,0.5)]" 
                  : isPast 
                    ? "bg-[--accent-primary] border-[--accent-primary]" 
                    : "bg-[--bg-surface] border-[--color-border] group-hover:border-[--color-text-secondary]"
              }`} />

              {/* Node Card */}
              <div className={`w-full max-w-[240px] p-3 rounded-lg border backdrop-blur-sm transition-all duration-300 z-20 mt-1 ${
                isActive 
                  ? "bg-[rgb(124,58,237,0.1)] border-[--accent-primary] shadow-lg translate-x-0" 
                  : "bg-[--bg-surface] border-[--color-border] opacity-70 hover:opacity-100 hover:border-[--color-text-tertiary] scale-95"
              }`}>
                <div className={`text-sm font-bold mb-2 ${isActive ? "text-[--accent-light]" : "text-[--color-text-secondary]"}`}>
                  {step.name}
                </div>
                <div className="flex justify-center">
                  <TensorShapeBadge 
                    shape={step.outputShape} 
                    className={`text-[10px] ${isActive ? "border-[--accent-primary]/30 bg-[--accent-primary]/10 text-[--accent-light]" : ""}`}
                  />
                </div>
              </div>

              {/* Down Arrow for next step (except last) */}
              {idx < trace.steps.length - 1 && (
                <svg className="w-4 h-6 text-[--color-border] mt-2 z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
