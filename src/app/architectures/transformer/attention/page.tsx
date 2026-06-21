"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Breadcrumb } from "@/components/breadcrumb";
import { HeatmapGrid } from "@/components/simulator/heatmap-grid";
import { AttentionInspector } from "@/components/simulator/attention-inspector";

const MOCK_ATTENTION_DATA = {
  "head-1": [
    [0.95, 0.02, 0.01, 0.01, 0.01, 0.0],
    [0.1, 0.8, 0.05, 0.03, 0.01, 0.01],
    [0.05, 0.1, 0.7, 0.1, 0.04, 0.01],
    [0.02, 0.05, 0.1, 0.75, 0.06, 0.02],
    [0.01, 0.02, 0.05, 0.1, 0.8, 0.02],
    [0.0, 0.01, 0.01, 0.02, 0.06, 0.9],
  ],
  "head-2": [
    [0.7, 0.1, 0.08, 0.07, 0.03, 0.02],
    [0.15, 0.5, 0.15, 0.12, 0.05, 0.03],
    [0.08, 0.15, 0.6, 0.12, 0.04, 0.01],
    [0.1, 0.1, 0.1, 0.6, 0.08, 0.02],
    [0.05, 0.08, 0.08, 0.12, 0.6, 0.07],
    [0.02, 0.03, 0.04, 0.06, 0.15, 0.7],
  ],
  "head-3": [
    [0.3, 0.2, 0.2, 0.15, 0.1, 0.05],
    [0.2, 0.3, 0.2, 0.15, 0.1, 0.05],
    [0.15, 0.2, 0.35, 0.15, 0.1, 0.05],
    [0.15, 0.15, 0.15, 0.35, 0.15, 0.05],
    [0.1, 0.1, 0.1, 0.2, 0.4, 0.1],
    [0.08, 0.08, 0.08, 0.12, 0.18, 0.46],
  ],
};

const TOKENS = ["The", "cat", "sat", "on", "the", "mat"];

const INTERPRETATIONS = {
  "The-The": {
    interpretation: "Each word pays strong attention to itself (diagonal attention)",
    reasoning:
      "The model uses self-attention to understand token relationships. Diagonal dominance is common - tokens need to attend to themselves for context.",
  },
  "cat-mat": {
    interpretation: "Subject attends to related noun",
    reasoning:
      "The model learns that 'cat' and 'mat' are semantically related. The animal (cat) pays attention to the location (mat).",
  },
  "sat-mat": {
    interpretation: "Verb attends to location complement",
    reasoning:
      "Verbs often attend to their complements. 'Sat' needs to know where the cat sat, so it attends to 'mat'.",
  },
};

export default function AttentionExplorer() {
  const [selectedLayer, setSelectedLayer] = useState("0");
  const [selectedHead, setSelectedHead] = useState("head-1");
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
  } | null>(null);

  const attentionMatrix = MOCK_ATTENTION_DATA[
    selectedHead as keyof typeof MOCK_ATTENTION_DATA
  ] || MOCK_ATTENTION_DATA["head-1"];

  const getInterpretation = () => {
    if (!hoveredCell) {
      return {
        interpretation: "Hover over a cell to inspect attention patterns",
        reasoning: "Each cell shows how much one token attends to another",
      };
    }
    const queryToken = TOKENS[hoveredCell.row];
    const keyToken = TOKENS[hoveredCell.col];
    const pair = `${queryToken}-${keyToken}`;
    return (
      INTERPRETATIONS[pair as keyof typeof INTERPRETATIONS] || {
        interpretation: `"${queryToken}" attends to "${keyToken}"`,
        reasoning:
          "This attention weight shows the model is considering the relationship between these tokens.",
      }
    );
  };

  return (
    <div className="pb-20">
      {/* Header */}
      <div className="border-b border-[--color-border] bg-[--bg-surface] sticky top-14 z-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Breadcrumb
            items={[
              { label: "Architectures", href: "/architectures" },
              { label: "Transformer", href: "/architectures/transformer" },
              { label: "Attention Explorer", current: true },
            ]}
          />
          <h1 className="text-3xl font-bold mb-2">Attention Heatmap Explorer</h1>
          <p className="text-[--color-text-secondary]">
            Visualize how Transformer attention flows between tokens
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Controls */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div>
            <label className="block text-sm font-semibold text-[--color-text-primary] mb-3">
              Layer
            </label>
            <div className="flex gap-2">
              {["0", "1", "2"].map((layer) => (
                <button
                  key={layer}
                  onClick={() => setSelectedLayer(layer)}
                  className={`px-4 py-2 rounded-lg border transition-colors ${
                    selectedLayer === layer
                      ? "bg-[--accent-primary] border-[--accent-primary] text-white"
                      : "border-[--color-border] text-[--color-text-secondary] hover:border-[--accent-primary]"
                  }`}
                >
                  Layer {parseInt(layer) + 1}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold text-[--color-text-primary] mb-3">
              Attention Head
            </label>
            <div className="flex gap-2">
              {["head-1", "head-2", "head-3"].map((head) => (
                <button
                  key={head}
                  onClick={() => setSelectedHead(head)}
                  className={`px-4 py-2 rounded-lg border transition-colors ${
                    selectedHead === head
                      ? "bg-[--accent-cyan] border-[--accent-cyan] text-white"
                      : "border-[--color-border] text-[--color-text-secondary] hover:border-[--accent-cyan]"
                  }`}
                >
                  {head}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Heatmap */}
          <motion.div
            key={`${selectedLayer}-${selectedHead}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="lg:col-span-2"
          >
            <div className="p-6 rounded-lg bg-[--bg-surface] border border-[--color-border]">
              <h2 className="text-lg font-semibold text-[--color-text-primary] mb-4">
                Attention Matrix
              </h2>
              <p className="text-xs text-[--color-text-tertiary] mb-4">
                Rows = Query tokens · Columns = Key tokens · Brighter = higher attention · Hover to inspect
              </p>
              <HeatmapGrid
                tokens={TOKENS}
                attention={attentionMatrix}
                onHover={(row, col) => setHoveredCell({ row, col })}
              />
            </div>
          </motion.div>

          {/* Inspector */}
          <motion.div
            key={hoveredCell ? `${hoveredCell.row}-${hoveredCell.col}` : "none"}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {hoveredCell ? (
              <AttentionInspector
                queryToken={TOKENS[hoveredCell.row]}
                keyToken={TOKENS[hoveredCell.col]}
                score={attentionMatrix[hoveredCell.row][hoveredCell.col]}
                interpretation={getInterpretation().interpretation}
                reasoning={getInterpretation().reasoning}
              />
            ) : (
              <div className="p-4 rounded-lg bg-[--bg-surface] border border-[--color-border] h-full flex items-center justify-center">
                <p className="text-center text-sm text-[--color-text-tertiary]">
                  Hover over cells in the heatmap to inspect attention patterns
                </p>
              </div>
            )}
          </motion.div>
        </div>

        {/* Understanding Section */}
        <div className="mt-12 grid md:grid-cols-3 gap-4">
          <div className="p-4 rounded-lg bg-[--bg-surface] border border-[--color-border]">
            <h3 className="font-semibold text-[--color-text-primary] mb-2">
              What is Attention?
            </h3>
            <p className="text-sm text-[--color-text-secondary]">
              Attention weights show which tokens the model looks at when processing each token.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-[--bg-surface] border border-[--color-border]">
            <h3 className="font-semibold text-[--color-text-primary] mb-2">
              Why Multiple Heads?
            </h3>
            <p className="text-sm text-[--color-text-secondary]">
              Different heads learn different patterns: some focus locally, others capture long-range relationships.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-[--bg-surface] border border-[--color-border]">
            <h3 className="font-semibold text-[--color-text-primary] mb-2">
              Reading the Matrix
            </h3>
            <p className="text-sm text-[--color-text-secondary]">
              Diagonal dominance (bright spots down the diagonal) means tokens attend to themselves—this is normal and expected.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
