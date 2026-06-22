"use client";

import { SystemTraceStep } from "@/data/system-traces";
import { ArrowRight, Clock, Code2, ExternalLink, Info, Cpu } from "lucide-react";
import Link from "next/link";

interface Props {
  step: SystemTraceStep;
}

export function SystemStepCard({ step }: Props) {
  return (
    <div className="flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-4 duration-400">
      {/* Header */}
      <div className="bg-[--bg-surface] rounded-xl border border-[--color-border] overflow-hidden shadow-sm">
        <div className="px-6 py-4 border-b border-[--color-border] bg-[rgba(16,185,129,0.05)]">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-[--color-text-primary]">{step.name}</h2>
            {step.latency && (
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[rgba(6,182,212,0.1)] border border-cyan-500/20 text-xs font-mono font-semibold text-[--accent-cyan]">
                <Clock className="w-3 h-3" /> {step.latency}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2 mt-1.5">
            <Cpu className="w-3.5 h-3.5 text-[--color-text-tertiary]" />
            <span className="text-sm font-mono text-[--color-text-secondary]">{step.component}</span>
          </div>
        </div>

        {/* Input → Output flow */}
        <div className="p-6 bg-[--bg-panel] flex flex-col items-center gap-4">
          <div className="w-full max-w-lg flex items-center gap-3">
            {/* Input */}
            <div className="flex-1 p-3 rounded-lg border border-[--color-border] bg-[--bg-surface] text-center">
              <div className="text-[10px] font-bold uppercase tracking-wider text-[--color-text-tertiary] mb-1">Input</div>
              <div className="text-sm text-[--color-text-secondary] leading-snug">{step.inputDesc}</div>
            </div>

            <ArrowRight className="w-5 h-5 text-emerald-500 flex-shrink-0" />

            {/* Output */}
            <div className="flex-1 p-3 rounded-lg border border-emerald-500/30 bg-[rgba(16,185,129,0.06)] text-center">
              <div className="text-[10px] font-bold uppercase tracking-wider text-emerald-500 mb-1">Output</div>
              <div className="text-sm text-emerald-300 leading-snug">{step.outputDesc}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Explainer */}
      <div className="bg-[--bg-surface] rounded-xl border border-[--color-border] overflow-hidden shadow-sm">
        <div className="p-6 space-y-5">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Info className="w-4 h-4 text-[--accent-primary]" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-[--color-text-tertiary]">What happened?</h3>
            </div>
            <p className="text-[--color-text-primary] leading-relaxed">{step.what}</p>
          </div>

          <div className="border-t border-[--color-border] pt-5">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[--color-text-tertiary] mb-2">Why?</h3>
            <p className="text-[--color-text-secondary] leading-relaxed bg-[--bg-panel] p-4 rounded-lg border border-[--color-border] border-l-4 border-l-emerald-500">
              {step.why}
            </p>
          </div>

          {step.techNote && (
            <div className="border-t border-[--color-border] pt-5">
              <h3 className="text-xs font-bold uppercase tracking-wider text-[--color-text-tertiary] mb-2">Engineering Note</h3>
              <p className="text-sm text-[--accent-cyan] leading-relaxed bg-[rgba(6,182,212,0.06)] p-4 rounded-lg border border-cyan-500/20">
                💡 {step.techNote}
              </p>
            </div>
          )}
        </div>

        {/* Footer link */}
        {step.systemDesignSlug && (
          <div className="bg-[--bg-panel] border-t border-[--color-border] px-6 py-3">
            <Link
              href={`/system-design/${step.systemDesignSlug}`}
              className="inline-flex items-center gap-2 text-xs font-semibold text-[--accent-primary] hover:underline"
            >
              <Code2 className="w-3.5 h-3.5" />
              Full System Design: {step.systemDesignSlug.replace(/-/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
              <ExternalLink className="w-3 h-3" />
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
