"use client";

import { DiagramBlock } from "@/data/architecture-catalog";
import { ArrowDown } from "lucide-react";

interface ArchitectureDiagramProps {
  blocks: DiagramBlock[];
  selectedBlockId: string | null;
  onSelectBlock: (id: string) => void;
}

/**
 * Interactive vertical block diagram. Clicking a block selects it and the
 * explorer's right panel shows its description. Tensor shapes annotate the
 * edges between blocks.
 */
export function ArchitectureDiagram({
  blocks,
  selectedBlockId,
  onSelectBlock,
}: ArchitectureDiagramProps) {
  return (
    <div className="flex flex-col items-center py-8 px-4">
      {/* Input marker */}
      <div className="text-xs font-mono text-[--color-text-tertiary] mb-2">
        input
      </div>
      <ArrowDown className="w-4 h-4 text-[--color-text-tertiary] mb-3" />

      {blocks.map((block, i) => {
        const selected = selectedBlockId === block.id;
        return (
          <div key={block.id} className="flex flex-col items-center w-full max-w-sm">
            {/* Repeat bracket */}
            <div className="relative w-full">
              {block.repeat && (
                <div className="absolute -right-2 top-1/2 -translate-y-1/2 translate-x-full hidden sm:flex items-center gap-1">
                  <div className="h-12 w-2 border-r-2 border-t-2 border-b-2 border-[--color-border] rounded-r" />
                  <span className="text-[10px] font-mono text-[--color-text-tertiary] whitespace-nowrap">
                    {block.repeat}
                  </span>
                </div>
              )}

              <button
                onClick={() => onSelectBlock(block.id)}
                aria-label={`Architecture block: ${block.label}${block.sublabel ? ` - ${block.sublabel}` : ''}`}
                className={`w-full px-5 py-3.5 rounded-lg bg-gradient-to-r ${block.color} text-left transition-all hover:scale-[1.02] hover:shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[--accent-cyan] ${
                  selected
                    ? "ring-2 ring-[--accent-cyan] ring-offset-2 ring-offset-[--bg-body] shadow-lg scale-[1.02]"
                    : "opacity-90 hover:opacity-100"
                }`}
              >
                <p className="text-sm font-bold text-white">{block.label}</p>
                {block.sublabel && (
                  <p className="text-[11px] text-white/70 font-mono mt-0.5">
                    {block.sublabel}
                  </p>
                )}
              </button>
            </div>

            {/* Edge with tensor shape */}
            {i < blocks.length - 1 && (
              <div className="flex items-center gap-2 py-2">
                <ArrowDown className="w-4 h-4 text-[--color-text-tertiary]" />
                {block.outputShape && (
                  <span className="text-[10px] font-mono text-[--accent-cyan]">
                    {block.outputShape}
                  </span>
                )}
              </div>
            )}
          </div>
        );
      })}

      <ArrowDown className="w-4 h-4 text-[--color-text-tertiary] mt-3 mb-2" />
      <div className="text-xs font-mono text-[--color-text-tertiary]">
        {blocks[blocks.length - 1]?.outputShape ?? "output"}
      </div>

      <p className="mt-6 text-[11px] text-[--color-text-tertiary]">
        Click any block to inspect it →
      </p>
    </div>
  );
}
