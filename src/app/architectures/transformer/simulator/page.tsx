"use client";

import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Play, Pause, RotateCcw } from "lucide-react";
import { Breadcrumb } from "@/components/breadcrumb";
import { PipelineStage } from "@/components/simulator/pipeline-stage";
import { TensorShapeCard } from "@/components/simulator/tensor-shape-card";

const STAGES = [
  {
    name: "Tokenization",
    description: "Convert text into tokens",
    inputShape: "(sequence_length,)",
    outputShape: "(sequence_length,)",
    explanation:
      "The input text is broken down into tokens using a vocabulary. Each word or subword becomes a token ID.",
    interview:
      "Why do we use subword tokenization instead of character-level or word-level?",
  },
  {
    name: "Embedding Lookup",
    description: "Convert token IDs to embeddings",
    inputShape: "(sequence_length,)",
    outputShape: "(sequence_length, embedding_dim=512)",
    explanation:
      "Each token ID is looked up in an embedding matrix to get its vector representation. This allows the model to work with continuous values.",
    interview: "What would happen if we used one-hot encoding instead of embeddings?",
  },
  {
    name: "Positional Encoding",
    description: "Add position information",
    inputShape: "(sequence_length, embedding_dim)",
    outputShape: "(sequence_length, embedding_dim)",
    explanation:
      "Positional encodings are added to embeddings so the model knows the order of tokens. Without this, the Transformer wouldn't understand word order.",
    interview: "Why use sinusoidal positional encodings instead of learned positions?",
  },
  {
    name: "Multi-Head Attention",
    description: "Compute attention across heads",
    inputShape: "(sequence_length, embedding_dim)",
    outputShape: "(sequence_length, embedding_dim)",
    explanation:
      "Attention computes relationships between all pairs of tokens. Multiple heads allow the model to attend to different aspects simultaneously.",
    interview: "What is the computational complexity of multi-head attention?",
  },
  {
    name: "Feed Forward",
    description: "Apply dense transformations",
    inputShape: "(sequence_length, embedding_dim)",
    outputShape: "(sequence_length, embedding_dim)",
    explanation:
      "A two-layer feed-forward network is applied position-wise. This adds non-linearity and processing power.",
    interview: "Why apply feed-forward separately to each position?",
  },
  {
    name: "Output Projection",
    description: "Project to vocabulary",
    inputShape: "(sequence_length, embedding_dim)",
    outputShape: "(sequence_length, vocab_size=30000)",
    explanation:
      "The final layer projects embeddings back to vocabulary size for token prediction at each position.",
    interview: "How does next-token prediction become autoregressive generation?",
  },
];

export default function TransformerSimulator() {
  const [inputText, setInputText] = useState("The cat sat on the mat");
  const [activeStage, setActiveStage] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [tokens, setTokens] = useState<string[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const tokenizeText = (text: string) => {
    return text
      .toLowerCase()
      .split(/\s+/)
      .filter((t) => t.length > 0);
  };

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Stop playback when isPlaying becomes false externally
  useEffect(() => {
    if (!isPlaying && intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, [isPlaying]);

  const handleReset = () => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setActiveStage(0);
    setIsPlaying(false);
  };

  const handleAutoPlay = () => {
    if (!isPlaying) {
      setIsPlaying(true);
      let currentStage = activeStage;
      intervalRef.current = setInterval(() => {
        if (currentStage < STAGES.length - 1) {
          currentStage++;
          setActiveStage(currentStage);
        } else {
          setIsPlaying(false);
          if (intervalRef.current !== null) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
        }
      }, 2000);
    } else {
      setIsPlaying(false);
    }
  };

  const handleInputChange = (text: string) => {
    setInputText(text);
    setTokens(tokenizeText(text));
  };

  if (tokens.length === 0 && inputText) {
    handleInputChange(inputText);
  }

  return (
    <div className="pb-20">
      {/* Header */}
      <div className="border-b border-[--color-border] bg-[--bg-surface] sticky top-14 z-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Breadcrumb
            items={[
              { label: "Architectures", href: "/architectures" },
              { label: "Transformer", href: "/architectures/transformer" },
              { label: "Simulator", current: true },
            ]}
          />
          <h1 className="text-3xl font-bold mb-2">Transformer Simulator</h1>
          <p className="text-[--color-text-secondary]">
            Follow your text through every stage of a Transformer model
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Input */}
        <div className="mb-8">
          <label className="block text-sm font-semibold text-[--color-text-primary] mb-2">
            Enter Text
          </label>
          <input
            type="text"
            value={inputText}
            onChange={(e) => handleInputChange(e.target.value)}
            className="w-full px-4 py-3 rounded-lg border border-[--color-border] bg-[--bg-surface] text-[--color-text-primary] placeholder-[--color-text-tertiary] focus:border-[--accent-primary] focus:outline-none transition-colors"
            placeholder="Type any sentence..."
          />
          <p className="text-xs text-[--color-text-tertiary] mt-2">
            {tokens.length} tokens: {tokens.map((t) => `"${t}"`).join(", ")}
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2 mb-8">
          <button
            onClick={handleAutoPlay}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[--accent-primary] text-white text-sm font-semibold hover:opacity-90 transition-opacity"
          >
            {isPlaying ? (
              <>
                <Pause className="w-4 h-4" /> Pause
              </>
            ) : (
              <>
                <Play className="w-4 h-4" /> Auto Play
              </>
            )}
          </button>
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 rounded-lg border border-[--color-border] text-[--color-text-secondary] text-sm font-semibold hover:bg-[--bg-surface] transition-colors"
          >
            <RotateCcw className="w-4 h-4" /> Reset
          </button>
        </div>

        {/* Pipeline */}
        <div className="space-y-4">
          {STAGES.map((stage, idx) => (
            <div key={idx}>
              <PipelineStage
                number={idx + 1}
                name={stage.name}
                description={stage.description}
                inputShape={stage.inputShape}
                outputShape={stage.outputShape}
                explanation={stage.explanation}
                interview={stage.interview}
                isActive={activeStage === idx}
                onClick={() => setActiveStage(idx)}
              />

              {idx < STAGES.length - 1 && (
                <div className="flex justify-center py-2">
                  <motion.div
                    animate={{
                      y: activeStage > idx ? [0, 8, 0] : 0,
                    }}
                    transition={{
                      duration: activeStage > idx ? 0.6 : 0,
                      repeat: Infinity,
                    }}
                  >
                    <ChevronDown className="w-5 h-5 text-[--accent-cyan]" />
                  </motion.div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Stage Details */}
        {activeStage < STAGES.length && (
          <motion.div
            key={activeStage}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="mt-12 grid md:grid-cols-2 gap-6"
          >
            <TensorShapeCard
              label="Input"
              shape={STAGES[activeStage].inputShape}
              description="Data entering this stage"
              value={
                activeStage === 0
                  ? tokens.length.toString()
                  : `(${tokens.length}, 512)`
              }
            />
            <TensorShapeCard
              label="Output"
              shape={STAGES[activeStage].outputShape}
              description="Data leaving this stage"
              value={
                activeStage === STAGES.length - 1
                  ? `(${tokens.length}, 30000)`
                  : `(${tokens.length}, 512)`
              }
            />
          </motion.div>
        )}

        {/* Help Text */}
        <div className="mt-12 p-6 rounded-lg bg-gradient-to-r from-[rgb(124,58,237,0.1)] to-[rgb(6,182,212,0.1)] border border-[--color-border]">
          <h3 className="font-semibold text-[--color-text-primary] mb-2">
            How to Use This Simulator
          </h3>
          <ul className="text-sm text-[--color-text-secondary] space-y-1">
            <li>
              • Type any sentence in the input box to see it tokenized into words
            </li>
            <li>• Click each stage to expand and see what happens at that step</li>
            <li>• Use &quot;Auto Play&quot; to watch the data flow through automatically</li>
            <li>• Check the interview question at each stage to test your understanding</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

function ChevronDown({ className }: { className: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M19 14l-7 7m0 0l-7-7m7 7V3"
      />
    </svg>
  );
}
