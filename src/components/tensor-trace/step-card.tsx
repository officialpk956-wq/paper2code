import { TensorStep } from "@/data/tensor-traces";
import { TensorShapeBadge } from "./tensor-shape-badge";
import { ArrowDown } from "lucide-react";

interface Props {
  step: TensorStep;
  traceConfig: Record<string, number>;
}

export function StepCard({ step, traceConfig }: Props) {
  return (
    <div className="bg-[--bg-surface] rounded-xl border border-[--color-border] overflow-hidden shadow-sm animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="px-6 py-4 border-b border-[--color-border] bg-[rgb(124,58,237,0.05)]">
        <h2 className="text-2xl font-bold text-[--color-text-primary] mb-1">{step.name}</h2>
        <div className="font-mono text-sm text-[--accent-primary]">{step.operation}</div>
      </div>

      <div className="p-8 flex flex-col items-center justify-center min-h-[300px] bg-[--bg-panel]">
        {/* Inputs */}
        <div className="flex flex-wrap items-center justify-center gap-4">
          {step.inputShape.length > 0 ? (
            step.inputShape.map((shape, idx) => (
              <TensorShapeBadge key={idx} shape={shape} config={traceConfig} className="text-sm px-4 py-2" />
            ))
          ) : (
            <div className="text-sm font-semibold text-[--color-text-tertiary] italic px-4 py-2 border border-dashed border-[--color-border] rounded-md">Start of sequence</div>
          )}
        </div>

        {/* Arrow & Operation */}
        <div className="flex flex-col items-center my-6 text-[--color-text-tertiary]">
          <ArrowDown className="w-6 h-6 mb-2" />
          <div className="px-3 py-1 rounded-full bg-[--bg-surface] border border-[--color-border] text-xs font-mono">
            {step.operation.split("=")[0].trim()}
          </div>
          <ArrowDown className="w-6 h-6 mt-2" />
        </div>

        {/* Output */}
        <TensorShapeBadge 
          shape={step.outputShape} 
          config={traceConfig} 
          className="text-base px-5 py-2.5 bg-[rgb(16,185,129,0.1)] border-emerald-500/30 text-emerald-400" 
        />
      </div>
    </div>
  );
}
