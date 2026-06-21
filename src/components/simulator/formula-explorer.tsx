"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown } from "lucide-react";

interface FormulaExplorerProps {
  title: string;
  formula: string;
  variables: Array<{
    symbol: string;
    name: string;
    explanation: string;
  }>;
  example: {
    description: string;
    inputs: Record<string, number>;
    output: number;
  };
}

export function FormulaExplorer({
  title,
  formula,
  variables,
  example,
}: FormulaExplorerProps) {
  const [expanded, setExpanded] = useState(false);
  const [inputs, setInputs] = useState(example.inputs);
  const [result, setResult] = useState(example.output);

  const handleInputChange = (key: string, value: string) => {
    const num = parseFloat(value) || 0;
    const newInputs = { ...inputs, [key]: num };
    setInputs(newInputs);

    if (title.includes("Softmax")) {
      const sum = Object.values(newInputs).reduce((a, b) => a + Math.exp(b), 0);
      setResult(Math.exp(newInputs[key]) / sum);
    } else if (title.includes("Attention")) {
      setResult(Object.values(newInputs).reduce((a, b) => a * b, 1));
    } else {
      setResult(Object.values(newInputs).reduce((a, b) => a + b, 0));
    }
  };

  return (
    <motion.div className="rounded-lg border border-[--color-border] bg-[--bg-surface] overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 flex items-center justify-between hover:bg-[--bg-body] transition-colors"
      >
        <h3 className="font-semibold text-[--color-text-primary]">{title}</h3>
        <motion.div
          animate={{ rotate: expanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown className="w-5 h-5 text-[--color-text-tertiary]" />
        </motion.div>
      </button>

      {expanded && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          className="border-t border-[--color-border] p-4 space-y-4"
        >
          <div className="bg-[--bg-body] rounded p-4">
            <p className="text-xs text-[--color-text-tertiary] mb-2">Formula</p>
            <p className="text-sm font-mono text-[--accent-cyan]">{formula}</p>
          </div>

          <div className="space-y-2">
            <p className="text-xs font-semibold text-[--color-text-secondary]">
              Variables
            </p>
            {variables.map((v) => (
              <div key={v.symbol} className="text-xs">
                <div className="flex items-center gap-2 mb-1">
                  <code className="font-mono text-[--accent-primary]">
                    {v.symbol}
                  </code>
                  <span className="text-[--color-text-secondary]">{v.name}</span>
                </div>
                <p className="text-[--color-text-tertiary] ml-4">
                  {v.explanation}
                </p>
              </div>
            ))}
          </div>

          <div className="bg-[rgb(6,182,212,0.1)] rounded p-4">
            <p className="text-xs font-semibold text-[--color-text-secondary] mb-3">
              {example.description}
            </p>
            <div className="space-y-2 mb-3">
              {Object.entries(inputs).map(([key, value]) => (
                <div key={key}>
                  <label className="text-xs font-mono text-[--color-text-secondary]">
                    {key} =
                  </label>
                  <input
                    type="number"
                    value={value}
                    onChange={(e) => handleInputChange(key, e.target.value)}
                    className="w-full mt-1 px-2 py-1 rounded text-sm bg-[--bg-surface] border border-[--color-border] text-[--color-text-primary]"
                    step="0.1"
                  />
                </div>
              ))}
            </div>
            <div className="pt-3 border-t border-[--color-border]">
              <p className="text-xs text-[--color-text-secondary] mb-1">
                Result
              </p>
              <p className="text-lg font-mono text-[--accent-cyan]">
                {result.toFixed(4)}
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
