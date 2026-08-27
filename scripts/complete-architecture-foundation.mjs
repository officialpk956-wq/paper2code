#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CONTENT = path.join(ROOT, "src/content/architectures");
const source = fs.readFileSync(path.join(ROOT, "src/data/content/architectures.ts"), "utf8");
const literal = source.match(/export const ARCHITECTURES[^=]*=\s*(\[[\s\S]*\]);/)?.[1];
if (!literal) throw new Error("Could not parse ARCHITECTURES");
const architectures = Function(`"use strict"; return (${literal});`)();

const aliases = {
  "lenet-5": "lenet",
  "googlenet-inception-v1": "googlenet",
  "swin-transformer": "swin",
  "u-net": "unet",
  "deeplab-v3-2": "deeplabv3plus",
};

const profiles = {
  "gemini-1-0": {
    intuition: "Gemini 1.0 was designed as a natively multimodal model family: text, images, audio, and video are treated as training signals for one coordinated model rather than being joined only after separate unimodal systems are finished.",
    flow: "multimodal examples -> modality encoders/tokenizers -> shared transformer -> task head or autoregressive decoder",
    math: String.raw`For paired modalities $x^{(a)}$ and $x^{(b)}$, alignment can be encouraged through a contrastive objective $\mathcal{L}=-\log\frac{\exp(s(z_a,z_b)/\tau)}{\sum_j\exp(s(z_a,z_j)/\tau)}$, while autoregressive training minimizes next-token negative log likelihood across mixed sequences.`,
    code: "multimodal-projector",
    references: ["Google DeepMind, Gemini: A Family of Highly Capable Multimodal Models (2023)", "Vaswani et al., Attention Is All You Need (2017)"],
  },
  "gemini-1-5": {
    intuition: "Gemini 1.5 extended the family toward very long context and sparse expert computation. The architectural story is not simply a larger context-number: the system must preserve useful retrieval and reasoning while attention state, routing, and serving cost grow.",
    flow: "long multimodal stream -> chunk/position representation -> sparse transformer layers -> long-context attention -> generated response",
    math: String.raw`Standard attention stores an $L\times L$ score matrix, giving $O(L^2)$ score work. Distributed or blockwise attention partitions the sequence while preserving exact normalization; MoE routing activates top-$k$ experts so expert compute scales with $k$ instead of the total expert count $E$.`,
    code: "long-context-mask",
    references: ["Google DeepMind, Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context (2024)", "Liu et al., Ring Attention with Blockwise Transformers (2023)"],
  },
  svd: {
    intuition: "SVD++ extends matrix factorization by representing not only explicit user-item interactions but also the set of items a user has implicitly touched. This helps when clicks, views, or purchases reveal preference even without ratings.",
    flow: "user id + implicit history + item id -> user vector augmented by history -> dot product + biases -> preference score",
    math: String.raw`The prediction is $\hat r_{ui}=\mu+b_u+b_i+q_i^\top\left(p_u+|N(u)|^{-1/2}\sum_{j\in N(u)}y_j\right)$, where $N(u)$ is the user's implicit-history set and $y_j$ is an implicit item vector.`,
    code: "recommender",
    references: ["Koren, Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model (2008)", "Koren, Bell, and Volinsky, Matrix Factorization Techniques for Recommender Systems (2009)"],
  },
  dien: {
    intuition: "DIEN models how a user's interests evolve over time. It extracts latent interest states from behavior sequences, then uses target-aware attention inside a second recurrent stage so the final representation emphasizes evolution relevant to the candidate item.",
    flow: "behavior sequence -> interest extractor GRU -> auxiliary supervision -> attention-aware evolution GRU -> candidate score",
    math: String.raw`An auxiliary next-behavior loss supervises hidden interest state $h_t$: $\mathcal{L}_{aux}=-\sum_t\log\sigma(h_t^\top e_{t+1})-\log\left(1-\sigma(h_t^\top e^-_{t+1})\right)$.`,
    code: "sequence-interest",
    references: ["Zhou et al., Deep Interest Evolution Network for Click-Through Rate Prediction (2019)"],
  },
  dcn: {
    intuition: "Deep & Cross Network learns explicit bounded-degree feature crosses without manually enumerating them, while a parallel deep network learns implicit nonlinear interactions. The two paths are combined before prediction.",
    flow: "dense + embedded features -> cross network || deep MLP -> concatenate -> prediction head",
    math: String.raw`A cross layer uses $x_{l+1}=x_0(x_l^\top w_l)+b_l+x_l$. Each layer raises the maximum polynomial interaction degree while adding only $O(d)$ parameters.`,
    code: "cross-network",
    references: ["Wang et al., Deep & Cross Network for Ad Click Predictions (2017)"],
  },
  "dcn-v2": {
    intuition: "DCN v2 strengthens the original cross network with matrix-valued transformations and mixtures of low-rank experts. It keeps explicit feature crossing while making interaction structure substantially more expressive.",
    flow: "embedded features -> mixture of low-rank cross experts -> gating -> stacked/parallel deep tower -> prediction",
    math: String.raw`A matrix cross layer can be written $x_{l+1}=x_0\odot(W_lx_l+b_l)+x_l$. Low-rank factorization $W_l=U_lV_l^\top$ reduces parameters and computation, while a gate combines several experts.`,
    code: "cross-network-v2",
    references: ["Wang et al., DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems (2020)"],
  },
  autoint: {
    intuition: "AutoInt treats embedded categorical fields as a short token sequence and uses multi-head self-attention to learn which feature fields interact. Residual connections allow several interaction layers without hand-crafted crosses.",
    flow: "feature fields -> embeddings -> interacting self-attention layers -> flatten/pool -> CTR prediction",
    math: String.raw`For field embeddings $E$, each head computes $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$. Attention weights select pairwise field interactions, and stacking layers composes higher-order interactions.`,
    code: "feature-attention",
    references: ["Song et al., AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks (2019)"],
  },
  chebnet: {
    intuition: "ChebNet turns spectral graph convolution into a localized polynomial of the graph Laplacian. Chebyshev recurrence avoids an expensive eigendecomposition and controls how many graph hops influence each output.",
    flow: "node features + adjacency -> normalized Laplacian -> Chebyshev polynomial messages -> weighted sum -> node representation",
    math: String.raw`With rescaled Laplacian $\tilde L$, the filter is $g_\theta*x=\sum_{k=0}^{K}\theta_kT_k(\tilde L)x$, where $T_0=1$, $T_1=z$, and $T_k(z)=2zT_{k-1}(z)-T_{k-2}(z)$.`,
    code: "chebyshev",
    references: ["Defferrard, Bresson, and Vandergheynst, Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (2016)"],
  },
};

const byName = new Map(architectures.map((architecture) => [architecture.name.toLowerCase(), architecture.slug]));

function difficulty(value) {
  return value === "Beginner" ? "beginner" : value === "Intermediate" ? "intermediate" : "advanced";
}

function relatedSlugs(architecture) {
  const values = [architecture.parent, architecture.derivedInto]
    .filter(Boolean)
    .flatMap((value) => value.split(","))
    .map((value) => value.trim().toLowerCase());
  return [...new Set(values.map((value) => byName.get(value)).filter(Boolean))];
}

function codeFor(kind) {
  if (kind === "recommender") return `\`\`\`python
import torch
from torch import nn

class SVDPlusPlus(nn.Module):
    def __init__(self, users, items, dim=32):
        super().__init__()
        self.user = nn.Embedding(users, dim)
        self.item = nn.Embedding(items, dim)
        self.implicit = nn.Embedding(items, dim)
        self.user_bias = nn.Embedding(users, 1)
        self.item_bias = nn.Embedding(items, 1)

    def forward(self, user_ids, item_ids, history_ids, history_mask):
        history = self.implicit(history_ids) * history_mask.unsqueeze(-1)
        count = history_mask.sum(1, keepdim=True).clamp_min(1).sqrt()
        user = self.user(user_ids) + history.sum(1) / count
        score = (user * self.item(item_ids)).sum(-1)
        return score + self.user_bias(user_ids).squeeze(-1) + self.item_bias(item_ids).squeeze(-1)

model = SVDPlusPlus(100, 500)
out = model(torch.tensor([1, 2]), torch.tensor([7, 8]), torch.tensor([[3, 4], [5, 0]]), torch.tensor([[1., 1.], [1., 0.]]))
assert out.shape == (2,)
\`\`\``;
  if (kind === "chebyshev") return `\`\`\`python
import torch

def chebyshev_features(x, scaled_laplacian, order):
    terms = [x]
    if order == 0:
        return terms
    terms.append(scaled_laplacian @ x)
    for _ in range(2, order + 1):
        terms.append(2 * scaled_laplacian @ terms[-1] - terms[-2])
    return terms

nodes, features = 5, 3
x = torch.randn(nodes, features)
laplacian = torch.eye(nodes)  # Replace with a properly rescaled graph Laplacian.
terms = chebyshev_features(x, laplacian, order=3)
weights = torch.randn(4, features, 8)
output = sum(term @ weight for term, weight in zip(terms, weights))
assert output.shape == (nodes, 8)
\`\`\``;
  if (kind === "long-context-mask") return `\`\`\`python
import torch

def block_causal_mask(length, block_size, device="cpu"):
    positions = torch.arange(length, device=device)
    causal = positions[:, None] >= positions[None, :]
    query_block = positions[:, None] // block_size
    key_block = positions[None, :] // block_size
    local_or_previous = key_block <= query_block
    return causal & local_or_previous

mask = block_causal_mask(length=16, block_size=4)
scores = torch.randn(2, 4, 16, 16)
scores = scores.masked_fill(~mask, float("-inf"))
attention = scores.softmax(dim=-1)
assert attention.shape == (2, 4, 16, 16)
assert torch.isfinite(attention).all()
\`\`\``;
  return `\`\`\`python
import torch
from torch import nn

class ArchitectureCore(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = torch.nn.functional.gelu(self.norm(self.input(x)))
        return self.output(hidden)

model = ArchitectureCore()
batch = torch.randn(4, 64)
result = model(batch)
assert result.shape == (4, 32)
print(result.shape)
\`\`\``;
}

function render(architecture, profile) {
  const parent = architecture.parent || "the methods that preceded it";
  const descendants = architecture.derivedInto || "later task-specific systems";
  return `# ${architecture.name}

## 1. Overview

${profile.intuition} Introduced in ${architecture.year} by ${architecture.authors}, its key innovation is ${architecture.keyInnovation}. The practical reason to study it is not historical trivia: the design exposes a reusable tradeoff between representation quality, computation, memory, and the structure of the input data.

## 2. Historical Context

${architecture.name} follows ${parent}. Earlier systems established the basic task, but left a concrete limitation in expressivity, efficiency, scale, or supervision. ${architecture.name} changes the computation rather than merely increasing parameter count. Its ideas influenced ${descendants}. When comparing papers, keep training data, evaluation protocol, and compute budget separate from the architecture itself; otherwise an apparent architectural gain may actually come from a stronger recipe.

## 3. Problem It Solves

The architecture addresses this problem: ${architecture.keyInnovation}. In production it has been associated with ${architecture.industryUsage}. Define the input contract, target output, training signal, and resource budget before implementation. A useful baseline should be simpler, use the same data split, and expose whether the innovation improves quality enough to justify additional complexity.

## 4. Architecture Diagram Data

\`\`\`text
${profile.flow}
       |                    |                    |
       +-- shapes/logs -----+-- validation ------+-- output
\`\`\`

Every arrow carries a tensor or structured record with a known shape, version, and owner. The forward path explains inference; the validation path explains how silent shape, routing, masking, or data errors are detected.

## 5. Layer-by-Layer Breakdown

Start with the input representation and identify what information is preserved. The middle transformation implements the architecture's defining operation. Normalization, residual paths, recurrent state, routing, or feature crossing stabilize information flow. The final head converts the shared representation into a task score. Inspect each boundary independently before tuning the full model.

## 6. Tensor Flow Walkthrough

Let batch size be $B$, sequence/history/node count be $L$, and hidden width be $D$. A typical path is **Input (B, L, D_in) -> projection (B, L, D) -> architecture core (B, L, D) -> pooling/head (B, D_out)**. Exact axes vary by domain, but no implementation should leave them implicit. Assert shapes at runtime and test empty, padded, and maximum-sized inputs.

## 7. Mathematical Foundations

${profile.math}

The equation identifies the model's inductive bias. Define every trainable object, normalization term, and sampled set before coding. Then test the formula on a hand-computable example; this catches sign, scaling, and broadcasting mistakes earlier than end-to-end training.

## 8. Training Procedure

Use a versioned dataset, deterministic evaluation split, and a baseline trained with the same budget. Log the main task metric, calibration or ranking quality where relevant, throughput, peak memory, and several failure slices. Begin with a tiny overfitting test, then a small representative run, and only then scale. Save optimizer state and configuration beside checkpoints so a result can be reproduced rather than merely demonstrated once.

## 9. PyTorch Implementation

${codeFor(profile.code)}

The snippet isolates the tensor contract. A full reproduction must add the architecture-specific loss, batching, negative sampling or masking, metrics, and data validation described by the original paper.

## 10. Strengths

The main strength is a clear inductive bias aligned with the task: ${architecture.keyInnovation}. This can improve data efficiency or make an otherwise expensive interaction tractable. The architecture also provides a useful conceptual unit that can be reused in larger systems and tested independently.

## 11. Weaknesses

The same inductive bias can become a limitation when the data or workload violates its assumptions. Additional state, routing, history, graph structure, or long-context computation increases implementation and serving complexity. Report sensitivity to sequence length, sparsity, cold-start cases, distribution shift, and hardware rather than relying on one aggregate benchmark.

## 12. Research Evolution

The direct lineage runs from ${parent} through ${architecture.name} toward ${descendants}. Later work typically improves optimization, scales the central operation, introduces sparsity or compression, or combines it with a stronger pretrained representation. Separate faithful extensions from architectures that merely reuse the name.

## 13. Interview Questions

**Q1: What is the defining innovation?** ${architecture.keyInnovation}. Explain the baseline first, then identify exactly which computation changes.

**Q2: What tensor contract is most important?** The core input and output shapes, including sequence, history, node, or field axes. Shape errors often produce valid-looking but incorrect computation.

**Q3: How would you evaluate it fairly?** Use the same data, budget, and metrics as a simple baseline, then add ablations and operational measurements.

**Q4: What is the main scaling risk?** The state or interaction dimension can increase memory, communication, or latency faster than expected. Estimate it before training.

**Q5: When should you avoid it?** Avoid it when a simpler baseline meets requirements or when the architecture's assumptions do not match the available data and serving constraints.

## 14. Related Papers

${profile.references.map((reference) => `- ${reference}`).join("\n")}

## 15. Further Reading

Read the primary paper first, then a maintained implementation and an independent reproduction. Compare equations with source code and write down every implementation choice not specified in the paper. Finally, inspect descendants listed in the architecture registry to understand which limitations later work attempted to remove.
`;
}

let copied = 0;
let generatedContent = 0;
let generatedMeta = 0;

for (const architecture of architectures) {
  const directory = path.join(CONTENT, architecture.slug);
  const contentFile = path.join(directory, "content.mdx");
  fs.mkdirSync(directory, { recursive: true });

  if (!fs.existsSync(contentFile) && aliases[architecture.slug]) {
    fs.copyFileSync(path.join(CONTENT, aliases[architecture.slug], "content.mdx"), contentFile);
    copied += 1;
  }
  if (!fs.existsSync(contentFile) && profiles[architecture.slug]) {
    fs.writeFileSync(contentFile, render(architecture, profiles[architecture.slug]), "utf8");
    generatedContent += 1;
  }
  if (!fs.existsSync(contentFile)) throw new Error(`No content source for ${architecture.slug}`);

  const metaFile = path.join(directory, "meta.json");
  if (!fs.existsSync(metaFile)) {
    const relationships = relatedSlugs(architecture);
    fs.writeFileSync(metaFile, `${JSON.stringify({
      type: "architecture",
      slug: architecture.slug,
      title: architecture.name,
      description: `${architecture.keyInnovation}. Used in ${architecture.industryUsage}.`,
      tags: ["architecture", architecture.category.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")],
      difficulty: difficulty(architecture.difficulty),
      year: architecture.year,
      relationships: relationships.length ? { architectures: relationships } : {},
    }, null, 2)}\n`, "utf8");
    generatedMeta += 1;
  }
}

console.log(`Copied ${copied} alias articles, generated ${generatedContent} articles, and generated ${generatedMeta} metadata files.`);
