"use client";

import { motion } from "framer-motion";

interface AttentionInspectorProps {
  queryToken: string;
  keyToken: string;
  score: number;
  interpretation: string;
  reasoning: string;
}

export function AttentionInspector({
  queryToken,
  keyToken,
  score,
  interpretation,
  reasoning,
}: AttentionInspectorProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-4 rounded-lg bg-gradient-to-r from-[rgb(124,58,237,0.1)] to-[rgb(6,182,212,0.1)] border border-[--accent-primary]"
    >
      <div className="space-y-3">
        <div>
          <p className="text-xs font-semibold text-[--color-text-secondary] mb-1">
            Attention Focus
          </p>
          <p className="text-sm">
            <span className="font-mono text-[--accent-primary]">&quot;{queryToken}&quot;</span>
            {" → "}
            <span className="font-mono text-[--accent-cyan]">&quot;{keyToken}&quot;</span>
          </p>
        </div>

        <div>
          <p className="text-xs font-semibold text-[--color-text-secondary] mb-1">
            Attention Score
          </p>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 bg-[--bg-body] rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${score * 100}%` }}
                transition={{ duration: 0.5 }}
                className="h-full bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan]"
              />
            </div>
            <span className="text-sm font-mono text-[--accent-primary]">
              {(score * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        <div>
          <p className="text-xs font-semibold text-[--color-text-secondary] mb-1">
            What This Means
          </p>
          <p className="text-sm text-[--color-text-secondary]">
            {interpretation}
          </p>
        </div>

        <div className="pt-3 border-t border-[--color-border]">
          <p className="text-xs font-semibold text-[--color-text-secondary] mb-1">
            Why The Model Focuses Here
          </p>
          <p className="text-sm text-[--color-text-secondary]">
            {reasoning}
          </p>
        </div>
      </div>
    </motion.div>
  );
}
