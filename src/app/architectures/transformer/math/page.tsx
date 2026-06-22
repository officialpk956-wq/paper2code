"use client";

import { motion } from "framer-motion";
import { Breadcrumb } from "@/components/breadcrumb";
import { FormulaExplorer } from "@/components/simulator/formula-explorer";

export default function TransformerMathPage() {
  return (
    <div className="pb-20">
      {/* Header */}
      <div className="border-b border-[--color-border] bg-[--bg-surface] sticky top-14 z-10">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Breadcrumb
            items={[
              { label: "Architectures", href: "/architectures" },
              { label: "Transformer", href: "/architectures/transformer" },
              { label: "Interactive Math", current: true },
            ]}
          />
          <h1 className="text-3xl font-bold mb-2">Transformer Mathematics</h1>
          <p className="text-[--color-text-secondary]">
            Explore the equations that power attention. Edit variables and see
            results update in real-time.
          </p>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ staggerChildren: 0.1, delayChildren: 0.2 }}
          className="space-y-6"
        >
          {/* Scaled Dot Product Attention */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
            <FormulaExplorer
              title="Scaled Dot-Product Attention"
              formula="Attention(Q,K,V) = softmax(QK^T / √d_k)V"
              variables={[
                {
                  symbol: "Q",
                  name: "Query Matrix",
                  explanation:
                    "Queries represent what the token is looking for. Shape: (sequence_length, d_k)",
                },
                {
                  symbol: "K",
                  name: "Key Matrix",
                  explanation:
                    "Keys represent what each token offers for matching. Shape: (sequence_length, d_k)",
                },
                {
                  symbol: "V",
                  name: "Value Matrix",
                  explanation:
                    "Values are what gets weighted and summed. Shape: (sequence_length, d_v)",
                },
                {
                  symbol: "d_k",
                  name: "Key Dimension",
                  explanation:
                    "The dimension of key vectors. Usually 64 in multi-head attention. Scaling by √d_k prevents explosion of dot products.",
                },
              ]}
              example={{
                description: "Example: Computing attention for a single query",
                inputs: {
                  Q: 1.5,
                  K: 2.3,
                  d_k: 64,
                },
                output: 0.0235,
              }}
            />
          </motion.div>

          {/* Softmax */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
            <FormulaExplorer
              title="Softmax Normalization"
              formula="softmax(x_i) = e^x_i / Σ(e^x_j)"
              variables={[
                {
                  symbol: "x_i",
                  name: "Input Score",
                  explanation:
                    "The scaled dot-product attention score for position i. Can be any value.",
                },
                {
                  symbol: "e",
                  name: "Euler's Number",
                  explanation:
                    "Mathematical constant ≈ 2.718. Used for exponential transformation.",
                },
                {
                  symbol: "Σ(e^x_j)",
                  name: "Normalization Sum",
                  explanation:
                    "Sum of exponentials across all positions. Ensures outputs sum to 1.",
                },
              ]}
              example={{
                description: "Example: Convert scores to probabilities",
                inputs: {
                  "score_1": 2.0,
                  "score_2": 1.0,
                  "score_3": 0.5,
                },
                output: 0.665,
              }}
            />
          </motion.div>

          {/* Positional Encoding */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
            <FormulaExplorer
              title="Positional Encoding (Sinusoidal)"
              formula="PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))"
              variables={[
                {
                  symbol: "pos",
                  name: "Position",
                  explanation:
                    "The position of the token in the sequence (0, 1, 2, ...).",
                },
                {
                  symbol: "i",
                  name: "Dimension Index",
                  explanation:
                    "Index of the embedding dimension (0 to d_model/2).",
                },
                {
                  symbol: "d_model",
                  name: "Model Dimension",
                  explanation:
                    "The embedding dimension, typically 512. Controls the frequency of oscillations.",
                },
              ]}
              example={{
                description: "Example: Encode position in the sequence",
                inputs: {
                  pos: 5,
                  i: 0,
                  d_model: 512,
                },
                output: 0.0876,
              }}
            />
          </motion.div>

          {/* Layer Normalization */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
            <FormulaExplorer
              title="Layer Normalization"
              formula="LayerNorm(x) = γ(x - μ) / √(σ² + ε) + β"
              variables={[
                {
                  symbol: "x",
                  name: "Input",
                  explanation: "The activation values to be normalized.",
                },
                {
                  symbol: "μ",
                  name: "Mean",
                  explanation: "Mean of x across the feature dimension.",
                },
                {
                  symbol: "σ²",
                  name: "Variance",
                  explanation: "Variance of x across the feature dimension.",
                },
                {
                  symbol: "ε",
                  name: "Epsilon",
                  explanation:
                    "Small constant (e.g., 1e-6) to prevent division by zero.",
                },
                {
                  symbol: "γ",
                  name: "Learnable Scale",
                  explanation:
                    "Trainable parameter that scales the normalized output.",
                },
                {
                  symbol: "β",
                  name: "Learnable Shift",
                  explanation:
                    "Trainable parameter that shifts the normalized output.",
                },
              ]}
              example={{
                description: "Example: Normalize activations",
                inputs: {
                  mean: 2.5,
                  variance: 1.2,
                  epsilon: 0.00001,
                },
                output: 0.0,
              }}
            />
          </motion.div>

          {/* Feed Forward */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
            <FormulaExplorer
              title="Feed Forward Network"
              formula="FFN(x) = max(0, xW₁ + b₁)W₂ + b₂"
              variables={[
                {
                  symbol: "x",
                  name: "Input",
                  explanation:
                    "The input to the feed-forward layer. Shape: (sequence_length, d_model)",
                },
                {
                  symbol: "W₁",
                  name: "First Weight Matrix",
                  explanation:
                    "Projects from d_model to d_ff (typically 2048). Shape: (d_model, d_ff)",
                },
                {
                  symbol: "W₂",
                  name: "Second Weight Matrix",
                  explanation:
                    "Projects back from d_ff to d_model. Shape: (d_ff, d_model)",
                },
                {
                  symbol: "max(0, ...)",
                  name: "ReLU Activation",
                  explanation:
                    "Non-linear activation that zeros out negative values.",
                },
              ]}
              example={{
                description: "Example: Apply feed-forward transformation",
                inputs: {
                  input: 1.5,
                  weight1: 0.8,
                  weight2: 0.6,
                },
                output: 0.72,
              }}
            />
          </motion.div>

          {/* Summary */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-12 p-6 rounded-lg bg-gradient-to-r from-[rgb(124,58,237,0.1)] to-[rgb(6,182,212,0.1)] border border-[--color-border]"
          >
            <h2 className="text-lg font-semibold text-[--color-text-primary] mb-3">
              How These Equations Work Together
            </h2>
            <p className="text-sm text-[--color-text-secondary] mb-3">
              In a Transformer layer:
            </p>
            <ol className="text-sm text-[--color-text-secondary] space-y-2 ml-4">
              <li>
                <span className="font-semibold text-[--color-text-primary]">
                  1. Scaled Dot-Product Attention
                </span>{" "}
                computes relationships between all token pairs
              </li>
              <li>
                <span className="font-semibold text-[--color-text-primary]">
                  2. Softmax
                </span>{" "}
                converts attention scores to normalized probabilities (0-1)
              </li>
              <li>
                <span className="font-semibold text-[--color-text-primary]">
                  3. Layer Normalization
                </span>{" "}
                stabilizes the activations before the next layer
              </li>
              <li>
                <span className="font-semibold text-[--color-text-primary]">
                  4. Feed Forward
                </span>{" "}
                applies additional non-linear transformations
              </li>
              <li>
                <span className="font-semibold text-[--color-text-primary]">
                  5. Positional Encoding
                </span>{" "}
                ensures the model understands token order
              </li>
            </ol>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
