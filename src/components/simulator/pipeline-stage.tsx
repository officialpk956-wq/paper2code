"use client";

import { motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { useState } from "react";

interface PipelineStageProps {
  number: number;
  name: string;
  description: string;
  inputShape: string;
  outputShape: string;
  explanation: string;
  interview: string;
  isActive: boolean;
  onClick: () => void;
}

export function PipelineStage({
  number,
  name,
  description,
  inputShape,
  outputShape,
  explanation,
  interview,
  isActive,
  onClick,
}: PipelineStageProps) {
  const [expanded, setExpanded] = useState(isActive);

  const toggleExpand = () => {
    setExpanded(!expanded);
    onClick();
  };

  return (
    <motion.div
      onClick={toggleExpand}
      className={`cursor-pointer rounded-lg border transition-all ${
        isActive
          ? "bg-[rgb(124,58,237,0.1)] border-[--accent-primary] shadow-lg"
          : "bg-[--bg-surface] border-[--color-border] hover:border-[--accent-primary]"
      }`}
    >
      <div className="p-4">
        <div className="flex items-start justify-between gap-3 mb-2">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-[--accent-primary] text-white text-xs font-bold">
              {number}
            </div>
            <div>
              <h3 className="font-semibold text-[--color-text-primary]">
                {name}
              </h3>
              <p className="text-xs text-[--color-text-tertiary]">
                {description}
              </p>
            </div>
          </div>
          <motion.div
            animate={{ rotate: expanded ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronDown className="w-5 h-5 text-[--color-text-tertiary]" />
          </motion.div>
        </div>

        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="mt-4 pt-4 border-t border-[--color-border] space-y-3"
          >
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-xs font-semibold text-[--color-text-secondary] mb-1">
                  Input
                </p>
                <code className="text-xs font-mono bg-[--bg-body] p-2 rounded block text-[--accent-cyan]">
                  {inputShape}
                </code>
              </div>
              <div>
                <p className="text-xs font-semibold text-[--color-text-secondary] mb-1">
                  Output
                </p>
                <code className="text-xs font-mono bg-[--bg-body] p-2 rounded block text-[--accent-cyan]">
                  {outputShape}
                </code>
              </div>
            </div>

            <div>
              <p className="text-xs font-semibold text-[--color-text-secondary] mb-1">
                What Happened
              </p>
              <p className="text-sm text-[--color-text-secondary]">
                {explanation}
              </p>
            </div>

            <div className="bg-[rgb(6,182,212,0.1)] border border-[--accent-cyan] rounded p-3">
              <p className="text-xs font-semibold text-[--accent-cyan] mb-1">
                Interview Question
              </p>
              <p className="text-sm text-[--color-text-secondary]">
                {interview}
              </p>
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
