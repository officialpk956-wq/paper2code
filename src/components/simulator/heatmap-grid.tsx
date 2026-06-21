"use client";

import { useState } from "react";
import { motion } from "framer-motion";

interface HeatmapGridProps {
  tokens: string[];
  attention: number[][];
  onHover?: (row: number, col: number) => void;
}

export function HeatmapGrid({ tokens, attention, onHover }: HeatmapGridProps) {
  const [hoveredCell, setHoveredCell] = useState<[number, number] | null>(null);

  const handleHover = (row: number, col: number) => {
    setHoveredCell([row, col]);
    onHover?.(row, col);
  };

  const getColor = (value: number, isHovered: boolean) => {
    if (isHovered) {
      return `hsl(262, 100%, ${100 - value * 40}%)`;
    }
    return `hsl(262, 60%, ${100 - value * 30}%)`;
  };

  return (
    <div className="overflow-x-auto">
      <div className="inline-block">
        <div className="flex items-start gap-2">
          <div className="flex flex-col gap-1 pt-8">
            {tokens.map((token) => (
              <div
                key={token}
                className="h-8 w-16 flex items-center justify-center text-xs font-mono text-[--color-text-secondary] text-right pr-2"
              >
                {token}
              </div>
            ))}
          </div>

          <div className="border border-[--color-border] rounded overflow-hidden">
            <div className="flex gap-1 p-2 bg-[--bg-body]">
              {tokens.map((token) => (
                <div
                  key={token}
                  className="h-6 w-8 flex items-center justify-center text-xs font-mono text-[--color-text-secondary]"
                >
                  {token}
                </div>
              ))}
            </div>

            <div className="p-2 bg-[--bg-surface] flex flex-col gap-1">
              {attention.map((row, rowIdx) => (
                <div key={rowIdx} className="flex gap-1">
                  {row.map((value, colIdx) => {
                    const isHovered =
                      hoveredCell?.[0] === rowIdx || hoveredCell?.[1] === colIdx;
                    return (
                      <motion.div
                        key={`${rowIdx}-${colIdx}`}
                        onMouseEnter={() => handleHover(rowIdx, colIdx)}
                        onMouseLeave={() => setHoveredCell(null)}
                        whileHover={{ scale: 1.1 }}
                        className="w-8 h-8 rounded cursor-pointer flex items-center justify-center text-xs font-mono transition-colors"
                        style={{
                          backgroundColor: getColor(value, isHovered),
                          color: value > 0.5 ? "#fff" : "var(--color-text-tertiary)",
                        }}
                        title={`${(value * 100).toFixed(1)}%`}
                      >
                        {(value * 100).toFixed(0)}
                      </motion.div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
