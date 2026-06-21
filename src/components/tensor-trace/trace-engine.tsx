"use client";

import { useState, useEffect } from "react";
import { ModelTrace } from "@/data/tensor-traces";
import { Breadcrumb } from "@/components/breadcrumb";
import { ShapeFlowPanel } from "./shape-flow-panel";
import { StepCard } from "./step-card";
import { OperationExplainer } from "./operation-explainer";
import { StepNav } from "./step-nav";

export function TraceEngine({ trace }: { trace: ModelTrace }) {
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const step = trace.steps[currentStepIdx];

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") {
        setCurrentStepIdx((prev) => Math.min(prev + 1, trace.steps.length - 1));
      } else if (e.key === "ArrowLeft") {
        setCurrentStepIdx((prev) => Math.max(prev - 1, 0));
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [trace.steps.length]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-[--color-border] bg-[--bg-surface] px-6 py-4 flex items-center justify-between">
        <div>
          <Breadcrumb 
            items={[
              { label: "Tensor Trace", href: "/tensor-trace" },
              { label: trace.title, current: true }
            ]} 
          />
          <h1 className="text-xl font-bold text-[--color-text-primary] mt-2">{trace.title}</h1>
        </div>
        
        {/* Progress Bar */}
        <div className="flex flex-col items-end gap-2 w-64">
          <div className="text-xs font-semibold text-[--color-text-tertiary]">
            Step {currentStepIdx + 1} of {trace.steps.length}
          </div>
          <div className="w-full bg-[--bg-panel] h-2 rounded-full overflow-hidden">
            <div 
              className="bg-[--accent-primary] h-full rounded-full transition-all duration-300 ease-out"
              style={{ width: `${((currentStepIdx + 1) / trace.steps.length) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Main Layout */}
      <div className="flex-1 min-h-0 flex overflow-hidden">
        
        {/* Left: Shape Flow (SVG diagram) */}
        <div className="w-80 flex-shrink-0 border-r border-[--color-border] bg-[--bg-surface] overflow-y-auto hidden md:block">
          <ShapeFlowPanel 
            trace={trace} 
            currentIndex={currentStepIdx} 
            onSelectStep={setCurrentStepIdx} 
          />
        </div>

        {/* Center/Right: Interactive Explainer Area */}
        <div className="flex-1 flex flex-col min-w-0 bg-[--bg-body] overflow-y-auto">
          <div className="max-w-4xl w-full mx-auto p-6 flex flex-col gap-6 h-full min-h-max">
            
            <StepNav 
              steps={trace.steps}
              currentIndex={currentStepIdx}
              onSelect={setCurrentStepIdx}
            />

            <StepCard step={step} traceConfig={trace.config} />

            <OperationExplainer step={step} />

          </div>
        </div>
      </div>
    </div>
  );
}
