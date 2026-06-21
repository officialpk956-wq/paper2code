"use client";

import { TensorStep } from "@/data/tensor-traces";
import { ArrowLeft, ArrowRight, ListFilter } from "lucide-react";
import { useState } from "react";

interface Props {
  steps: TensorStep[];
  currentIndex: number;
  onSelect: (index: number) => void;
}

export function StepNav({ steps, currentIndex, onSelect }: Props) {
  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <div className="flex items-center justify-between p-3 rounded-lg bg-[--bg-surface] border border-[--color-border] shadow-sm">
      <button 
        onClick={() => onSelect(Math.max(0, currentIndex - 1))}
        disabled={currentIndex === 0}
        className="flex items-center gap-2 px-4 py-2 rounded-md bg-[--bg-panel] border border-[--color-border] text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[--color-border] transition-colors"
      >
        <ArrowLeft className="w-4 h-4" /> Prev
        <span className="hidden sm:inline text-xs text-[--color-text-tertiary] font-normal ml-2 tracking-widest">←</span>
      </button>

      <div className="relative">
        <button 
          onClick={() => setDropdownOpen(!dropdownOpen)}
          className="flex items-center gap-2 px-4 py-2 text-sm font-bold text-[--color-text-primary] hover:text-[--accent-primary] transition-colors"
        >
          <ListFilter className="w-4 h-4" />
          Jump to Layer
        </button>

        {dropdownOpen && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setDropdownOpen(false)} />
            <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-64 max-h-80 overflow-y-auto bg-[--bg-surface] border border-[--color-border] rounded-xl shadow-xl z-50 py-2">
              {steps.map((step, idx) => (
                <button
                  key={step.id}
                  onClick={() => { onSelect(idx); setDropdownOpen(false); }}
                  className={`w-full text-left px-4 py-2 text-sm flex flex-col transition-colors ${
                    idx === currentIndex 
                      ? "bg-[rgb(124,58,237,0.1)] text-[--accent-light] font-bold border-l-2 border-[--accent-primary]"
                      : "text-[--color-text-secondary] hover:bg-[--bg-panel] hover:text-[--color-text-primary] border-l-2 border-transparent"
                  }`}
                >
                  <span>{step.name}</span>
                  <span className="text-[10px] text-[--color-text-tertiary] font-mono mt-0.5">{step.outputShape}</span>
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      <button 
        onClick={() => onSelect(Math.min(steps.length - 1, currentIndex + 1))}
        disabled={currentIndex === steps.length - 1}
        className="flex items-center gap-2 px-4 py-2 rounded-md bg-[--accent-primary] text-white text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:brightness-110 transition-all"
      >
        Next <ArrowRight className="w-4 h-4" />
        <span className="hidden sm:inline text-xs text-white/50 font-normal ml-2 tracking-widest">→</span>
      </button>
    </div>
  );
}
