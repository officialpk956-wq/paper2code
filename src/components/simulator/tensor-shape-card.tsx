"use client";

import { motion } from "framer-motion";

interface TensorShapeCardProps {
  label: string;
  shape: string;
  description: string;
  value?: string;
}

export function TensorShapeCard({
  label,
  shape,
  description,
  value,
}: TensorShapeCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-4 rounded-lg bg-[--bg-surface] border border-[--color-border] hover:border-[--accent-primary] transition-colors"
    >
      <div className="flex items-start justify-between gap-3 mb-2">
        <h4 className="font-semibold text-sm text-[--color-text-primary]">
          {label}
        </h4>
        <code className="text-xs font-mono px-2 py-1 rounded bg-[--bg-body] text-[--accent-cyan]">
          {shape}
        </code>
      </div>
      <p className="text-xs text-[--color-text-tertiary] mb-2">{description}</p>
      {value && (
        <div className="mt-2 p-2 rounded bg-[--bg-body] border border-[--color-border]">
          <p className="text-xs font-mono text-[--color-text-secondary]">
            {value}
          </p>
        </div>
      )}
    </motion.div>
  );
}
