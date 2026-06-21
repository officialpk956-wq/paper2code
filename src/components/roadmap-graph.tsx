"use client";

import { useState } from "react";
import { RoadmapNode, NodeState } from "@/data/roadmaps";
import { ChevronRight, Lock, CheckCircle2 } from "lucide-react";

interface RoadmapGraphProps {
  phases: Array<{
    name: string;
    nodes: RoadmapNode[];
  }>;
  onNodeClick: (node: RoadmapNode) => void;
}

const STATE_CONFIG: Record<NodeState, { bg: string; border: string; text: string; icon: React.ReactNode }> = {
  completed: {
    bg: "bg-[rgb(16,185,129,0.15)]",
    border: "border-[rgb(16,185,129,0.4)]",
    text: "text-[#6EE7B7]",
    icon: <CheckCircle2 className="w-4 h-4" />,
  },
  in_progress: {
    bg: "bg-[rgb(124,58,237,0.15)]",
    border: "border-[--accent-primary]",
    text: "text-[--accent-primary]",
    icon: <div className="w-3 h-3 rounded-full bg-[--accent-primary] animate-pulse" />,
  },
  available: {
    bg: "bg-[--bg-surface]",
    border: "border-[--color-border] hover:border-[--accent-primary]",
    text: "text-[--color-text-secondary]",
    icon: null,
  },
  locked: {
    bg: "bg-[--bg-body]",
    border: "border-[--color-border] opacity-50",
    text: "text-[--color-text-tertiary]",
    icon: <Lock className="w-4 h-4" />,
  },
};

export function RoadmapGraph({ phases, onNodeClick }: RoadmapGraphProps) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  return (
    <div className="space-y-12">
      {phases.map((phase, phaseIdx) => (
        <div key={phaseIdx} className="relative">
          {/* Phase header */}
          <div className="mb-8">
            <h3 className="text-xl font-bold mb-2">{phase.name}</h3>
            <div className="h-1 bg-[--bg-surface] rounded-full overflow-hidden w-32">
              <div
                className="h-full bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan]"
                style={{
                  width: `${(phase.nodes.filter((n) => n.state === "completed").length / phase.nodes.length) * 100}%`,
                }}
              />
            </div>
          </div>

          {/* Nodes grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 relative">
            {/* Connection lines SVG */}
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ minHeight: "300px" }}
            >
              {/* Draw lines between nodes with prerequisites */}
              {phase.nodes.map((node, idx) => {
                return node.prerequisites.map((prereqId) => {
                  const prereqNode = phases
                    .flatMap((p) => p.nodes)
                    .find((n) => n.id === prereqId);
                  if (!prereqNode) return null;

                  const fromIdx = phase.nodes.indexOf(prereqNode);
                  const toIdx = idx;

                  if (fromIdx === -1) return null;

                  const fromCol = fromIdx % 3;
                  const toCol = toIdx % 3;
                  const fromRow = Math.floor(fromIdx / 3);
                  const toRow = Math.floor(toIdx / 3);

                  const x1 = fromCol * 33.33 + 16.5;
                  const y1 = fromRow * 150 + 80;
                  const x2 = toCol * 33.33 + 16.5;
                  const y2 = toRow * 150 + 10;

                  return (
                    <line
                      key={`${node.id}-${prereqId}`}
                      x1={`${x1}%`}
                      y1={`${y1}px`}
                      x2={`${x2}%`}
                      y2={`${y2}px`}
                      stroke="var(--color-border)"
                      strokeWidth="2"
                      strokeDasharray="5,5"
                      opacity="0.5"
                    />
                  );
                });
              })}
            </svg>

            {/* Node cards */}
            {phase.nodes.map((node) => {
              const config = STATE_CONFIG[node.state];
              const isSelected = selectedNode === node.id;

              return (
                <button
                  key={node.id}
                  onClick={() => {
                    setSelectedNode(node.id);
                    onNodeClick(node);
                  }}
                  disabled={node.state === "locked"}
                  className={`relative p-4 rounded-lg border transition-all transform hover:scale-105 ${
                    node.state === "locked" ? "cursor-not-allowed" : "cursor-pointer"
                  } ${config.bg} ${config.border} ${isSelected ? "ring-2 ring-[--accent-primary]" : ""}`}
                >
                  {/* Lock overlay for locked nodes */}
                  {node.state === "locked" && (
                    <div className="absolute inset-0 bg-black/20 rounded-lg flex items-center justify-center">
                      <Lock className="w-5 h-5 text-[--color-text-tertiary]" />
                    </div>
                  )}

                  {/* Node content */}
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <h4 className={`font-semibold text-sm mb-1 ${config.text}`}>
                        {node.title}
                      </h4>
                      <p className="text-xs text-[--color-text-tertiary]">
                        {node.estimatedTime}
                      </p>
                    </div>
                    {config.icon && <div className={config.text}>{config.icon}</div>}
                  </div>

                  {/* Difficulty badge */}
                  <div className="flex items-center justify-between mt-3">
                    <span
                      className={`text-xs font-semibold px-2 py-1 rounded-full ${
                        node.difficulty === "beginner"
                          ? "bg-[#10B981]/20 text-[#6EE7B7]"
                          : node.difficulty === "intermediate"
                            ? "bg-[#F59E0B]/20 text-[#FCD34D]"
                            : "bg-[#EF4444]/20 text-[#FCA5A5]"
                      }`}
                    >
                      {node.difficulty}
                    </span>
                    {node.prerequisites.length > 0 && (
                      <span className="text-xs text-[--color-text-tertiary]">
                        {node.prerequisites.length} prereq{node.prerequisites.length !== 1 ? "s" : ""}
                      </span>
                    )}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Arrow to next phase */}
          {phaseIdx < phases.length - 1 && (
            <div className="flex justify-center my-8">
              <ChevronRight className="w-6 h-6 rotate-90 text-[--color-border]" />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
