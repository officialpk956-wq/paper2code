"use client";

import { useState } from "react";
import { ArrowRight } from "lucide-react";

interface GraphConnection {
  from: string;
  to: string;
  type: "introduces" | "requires" | "reinforces" | "used_in" | "references";
  fromType: string;
  toType: string;
}

interface GraphNode {
  id: string;
  label: string;
  type: "topic" | "architecture" | "paper" | "math" | "problem" | "system-design";
  color: string;
}

interface LearningGraphProps {
  nodes: GraphNode[];
  connections: GraphConnection[];
}

const TYPE_COLORS: Record<string, string> = {
  topic: "from-blue-500 to-blue-600",
  architecture: "from-purple-500 to-purple-600",
  paper: "from-pink-500 to-pink-600",
  math: "from-green-500 to-green-600",
  problem: "from-orange-500 to-orange-600",
  "system-design": "from-red-500 to-red-600",
};

const CONNECTION_TYPE_LABELS: Record<string, string> = {
  introduces: "introduces",
  requires: "requires",
  reinforces: "reinforces",
  used_in: "used in",
  references: "references",
};

export function LearningGraph({ nodes, connections }: LearningGraphProps) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [hoveredConnection, setHoveredConnection] = useState<string | null>(null);

  const getConnectedNodes = (nodeId: string) => {
    const outgoing = connections.filter((c) => c.from === nodeId).map((c) => c.to);
    const incoming = connections.filter((c) => c.to === nodeId).map((c) => c.from);
    return { outgoing, incoming };
  };

  return (
    <div className="space-y-6">
      {/* Legend */}
      <div className="flex flex-wrap gap-3">
        {Object.entries(TYPE_COLORS).map(([type]) => (
          <div key={type} className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full bg-gradient-to-r ${TYPE_COLORS[type]}`} />
            <span className="text-xs font-medium text-[--color-text-secondary] capitalize">
              {type === "system-design" ? "System Design" : type}
            </span>
          </div>
        ))}
      </div>

      {/* Graph visualization */}
      <div className="overflow-x-auto">
        <svg
          width="100%"
          height="600"
          className="rounded-lg bg-[--bg-body] border border-[--color-border]"
          viewBox="0 0 1000 600"
          preserveAspectRatio="xMidYMid"
        >
          {/* Draw connections */}
          {connections.map((connection, idx) => {
            const fromNode = nodes.find((n) => n.id === connection.from);
            const toNode = nodes.find((n) => n.id === connection.to);

            if (!fromNode || !toNode) return null;

            const fromIdx = nodes.indexOf(fromNode);
            const toIdx = nodes.indexOf(toNode);

            const x1 = ((fromIdx % 3) + 0.5) * 300 + 50;
            const y1 = Math.floor(fromIdx / 3) * 200 + 100;
            const x2 = ((toIdx % 3) + 0.5) * 300 + 50;
            const y2 = Math.floor(toIdx / 3) * 200 + 100;

            const isHovered = hoveredConnection === `${connection.from}-${connection.to}`;
            const isSelected =
              selectedNode === connection.from || selectedNode === connection.to;

            return (
              <g key={idx} opacity={isHovered || !selectedNode || isSelected ? 1 : 0.2}>
                {/* Line */}
                <defs>
                  <marker
                    id={`arrowhead-${idx}`}
                    markerWidth="10"
                    markerHeight="10"
                    refX="9"
                    refY="3"
                    orient="auto"
                  >
                    <polygon
                      points="0 0, 10 3, 0 6"
                      fill={isHovered ? "var(--accent-primary)" : "var(--color-border)"}
                    />
                  </marker>
                </defs>

                <line
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke={isHovered ? "var(--accent-primary)" : "var(--color-border)"}
                  strokeWidth={isHovered ? "2" : "1"}
                  markerEnd={`url(#arrowhead-${idx})`}
                  strokeDasharray={
                    connection.type === "requires" ? "5,5" : undefined
                  }
                  className="transition-all"
                  onMouseEnter={() =>
                    setHoveredConnection(`${connection.from}-${connection.to}`)
                  }
                  onMouseLeave={() => setHoveredConnection(null)}
                  style={{ cursor: "pointer" }}
                />

                {/* Label */}
                <text
                  x={(x1 + x2) / 2}
                  y={(y1 + y2) / 2 - 5}
                  textAnchor="middle"
                  className="text-xs fill-[--color-text-tertiary]"
                  pointerEvents="none"
                >
                  {CONNECTION_TYPE_LABELS[connection.type]}
                </text>
              </g>
            );
          })}

          {/* Draw nodes */}
          {nodes.map((node, idx) => {
            const x = ((idx % 3) + 0.5) * 300 + 50;
            const y = Math.floor(idx / 3) * 200 + 100;
            const isSelected = selectedNode === node.id;
            const { outgoing, incoming } = getConnectedNodes(node.id);
            const isConnected = selectedNode
              ? outgoing.includes(selectedNode) || incoming.includes(selectedNode)
              : false;

            const opacity =
              !selectedNode || isSelected || isConnected ? 1 : 0.3;

            return (
              <g key={node.id} opacity={opacity} className="transition-opacity">
                <defs>
                  <linearGradient
                    id={`gradient-${node.id}`}
                    x1="0%"
                    y1="0%"
                    x2="100%"
                    y2="100%"
                  >
                    <stop offset="0%" stopColor={node.color || "#7C3AED"} />
                    <stop offset="100%" stopColor={node.color || "#06B6D4"} />
                  </linearGradient>
                </defs>

                {/* Circle */}
                <circle
                  cx={x}
                  cy={y}
                  r={isSelected ? 45 : 35}
                  fill={`url(#gradient-${node.id})`}
                  className="transition-all cursor-pointer hover:r-45"
                  onClick={() => setSelectedNode(isSelected ? null : node.id)}
                />

                {/* Border for selected */}
                {isSelected && (
                  <circle
                    cx={x}
                    cy={y}
                    r="50"
                    fill="none"
                    stroke="var(--accent-primary)"
                    strokeWidth="2"
                    opacity="0.5"
                  />
                )}

                {/* Label */}
                <text
                  x={x}
                  y={y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="text-xs font-semibold fill-white pointer-events-none"
                  style={{
                    textShadow: "0 1px 3px rgba(0,0,0,0.5)",
                  }}
                >
                  {node.label.substring(0, 12)}
                  {node.label.length > 12 ? "..." : ""}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Selected node details */}
      {selectedNode && (
        <div className="p-4 rounded-lg bg-[--bg-surface] border border-[--color-border]">
          <h4 className="font-semibold mb-3">
            {nodes.find((n) => n.id === selectedNode)?.label} Connections
          </h4>
          <div className="space-y-2">
            {/* Incoming */}
            {getConnectedNodes(selectedNode).incoming.map((nodeId) => {
              const connection = connections.find(
                (c) => c.from === nodeId && c.to === selectedNode
              );
              const node = nodes.find((n) => n.id === nodeId);
              if (!node || !connection) return null;

              return (
                <div
                  key={nodeId}
                  className="flex items-center gap-2 p-2 rounded bg-[--bg-body] text-xs"
                >
                  <span className="font-medium text-[--color-text-secondary]">
                    {node.label}
                  </span>
                  <span className="text-[--color-text-tertiary]">
                    {CONNECTION_TYPE_LABELS[connection.type]}
                  </span>
                </div>
              );
            })}

            {/* Outgoing */}
            {getConnectedNodes(selectedNode).outgoing.map((nodeId) => {
              const connection = connections.find(
                (c) => c.from === selectedNode && c.to === nodeId
              );
              const node = nodes.find((n) => n.id === nodeId);
              if (!node || !connection) return null;

              return (
                <div
                  key={nodeId}
                  className="flex items-center gap-2 p-2 rounded bg-[--bg-body] text-xs"
                >
                  <ArrowRight className="w-3 h-3 flex-shrink-0" />
                  <span className="font-medium text-[--color-text-secondary]">
                    {node.label}
                  </span>
                  <span className="text-[--color-text-tertiary]">
                    {CONNECTION_TYPE_LABELS[connection.type]}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
