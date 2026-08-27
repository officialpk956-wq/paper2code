#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const contentRoot = path.join(root, "src/content/papers");

function parseExport(relativePath, name) {
  const source = fs.readFileSync(path.join(root, relativePath), "utf8");
  const literal = new RegExp(`export const ${name}[^=]*=\\s*(\\[[\\s\\S]*?\\]);`).exec(source)?.[1];
  if (!literal) throw new Error(`Could not parse ${name}`);
  return Function(`"use strict"; return (${literal});`)();
}

const papers = parseExport("src/data/content/papers.ts", "PAPERS");
const architectures = parseExport("src/data/content/architectures.ts", "ARCHITECTURES");
const architectureSlugs = new Set(architectures.map((item) => item.slug));
const bySection = Map.groupBy(papers, (paper) => paper.section);

const args = Object.fromEntries(
  process.argv.slice(2).map((argument) => {
    const [key, value = "true"] = argument.replace(/^--/, "").split("=");
    return [key, value];
  }),
);
const batchSize = Number(args["batch-size"] ?? 50);
const batch = Number(args.batch ?? 1);
if (!Number.isInteger(batch) || batch < 1 || !Number.isInteger(batchSize) || batchSize < 1) {
  throw new Error("Use positive integers for --batch and --batch-size");
}
const fromRank = (batch - 1) * batchSize + 1;
const toRank = batch * batchSize;
const targets = papers.filter((paper) => paper.rank >= fromRank && paper.rank <= toRank);

const guides = {
  optimization: {
    keywords: ["backprop", "optimization", "normalization", "dropout", "optimizer", "gradient", "loss"],
    intuition: "The central object is the training signal: how information about an error is transformed into a stable, useful parameter update.",
    prior: "Earlier training recipes were often brittle, slow, or sensitive to initialization and scale. They made deep models possible in principle but difficult to optimize reliably.",
    failure: "A method can fail through vanishing or exploding gradients, poorly scaled activations, noisy updates, or a mismatch between the objective and the evaluation metric.",
    flow: "batch -> forward computation -> objective -> reverse-mode gradients -> parameter update -> validation metrics",
    equation: String.raw`\theta_{t+1}=\theta_t-\eta_t\,\widehat{\nabla_\theta \mathcal{L}}`,
    symbols: "Here $\\theta$ denotes trainable parameters, $\\eta_t$ is the step size, and the gradient estimate is computed from the current batch. The paper’s contribution changes how one or more of these terms are produced or stabilized.",
    metrics: "training loss, validation loss, convergence speed, gradient norms, calibration, and sensitivity to initialization",
    system: "llm-serving",
    curriculum: "/learn/mathematics/statistical-learning-theory",
  },
  vision: {
    keywords: ["image", "vision", "conv", "cnn", "resnet", "vit", "visual", "recognition"],
    intuition: "The model must convert pixels into representations that preserve task-relevant structure while becoming robust to nuisance variation such as position, lighting, and background.",
    prior: "Earlier vision systems depended heavily on hand-designed features or architectures whose receptive fields and optimization behavior limited depth and transfer.",
    failure: "Strong average accuracy can hide failures on small objects, distribution shift, texture bias, rare classes, or changes in image resolution and preprocessing.",
    flow: "image batch -> patch or convolutional features -> hierarchical representation -> pooling/task head -> class or embedding output",
    equation: String.raw`h^{(l+1)}=\phi\!\left(W^{(l)} * h^{(l)}+b^{(l)}\right)`,
    symbols: "The operator $*$ is either spatial convolution or a learned token-mixing operation, and $\\phi$ includes the nonlinearity and any normalization or residual path. The important question is how the receptive field grows while information remains trainable.",
    metrics: "task accuracy, robustness slices, parameter count, FLOPs, throughput, memory, and transfer performance",
    system: "recommendation-systems",
    curriculum: "/learn/computer-vision/foundation-models-for-vision",
  },
  detection: {
    keywords: ["detection", "detector", "r-cnn", "yolo", "object", "bounding box"],
    intuition: "Detection combines recognition with localization: the model must decide what is present and where it appears, while handling a variable number of objects.",
    prior: "Earlier pipelines separated proposal generation, feature extraction, classification, and box refinement into slow or independently optimized stages.",
    failure: "Errors arise from missed proposals, duplicate boxes, class imbalance, scale variation, crowded scenes, and a train-test mismatch in post-processing.",
    flow: "image -> backbone features -> candidate or dense predictions -> class/box heads -> matching and suppression -> detections",
    equation: String.raw`\mathcal{L}=\mathcal{L}_{cls}+\lambda\,\mathcal{L}_{box}`,
    symbols: "The classification term identifies object categories, while the localization term penalizes geometric error. Matching rules, negative sampling, and the weight $\\lambda$ determine which errors dominate learning.",
    metrics: "precision-recall, mean average precision, localization quality, small/medium/large-object slices, latency, and memory",
    system: "tiktok-architecture",
    curriculum: "/learn/computer-vision/foundation-models-for-vision",
  },
  segmentation: {
    keywords: ["segmentation", "u-net", "mask", "pixel", "deeplab"],
    intuition: "Segmentation keeps spatial detail all the way to the output so that every pixel or region receives a meaningful label.",
    prior: "Classification networks discarded fine spatial information, while earlier dense-prediction systems struggled to combine global context with sharp boundaries.",
    failure: "Common weaknesses include boundary blur, poor small-object recall, class imbalance, resolution sensitivity, and high memory use for dense feature maps.",
    flow: "image -> multi-scale encoder -> context aggregation -> decoder/upsampling -> per-pixel logits -> mask",
    equation: String.raw`\mathcal{L}_{seg}=-\sum_{p}\sum_c y_{p,c}\log \hat p_{p,c}`,
    symbols: "The loss is applied over pixels $p$ and classes $c$. Practical systems often add overlap-aware terms or class weights when foreground regions are rare.",
    metrics: "intersection-over-union, Dice score, boundary quality, per-class recall, latency, and peak activation memory",
    system: "tiktok-architecture",
    curriculum: "/learn/computer-vision/foundation-models-for-vision",
  },
  language: {
    keywords: ["language", "transformer", "attention", "bert", "gpt", "llama", "translation", "token", "prompt"],
    intuition: "A sequence model must represent each token in context and turn those representations into either a prediction about the sequence or the next generated token.",
    prior: "Recurrent and convolutional sequence models compressed long contexts through sequential computation, limiting parallelism and making distant dependencies difficult to preserve.",
    failure: "Likelihood training can reward fluent continuation without guaranteeing factuality, instruction following, calibrated uncertainty, or robust long-context use.",
    flow: "text -> tokenizer -> embeddings/positions -> contextual sequence layers -> task head or autoregressive decoder -> tokens",
    equation: String.raw`\mathcal{L}_{NLL}=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})`,
    symbols: "The model is trained to assign probability to the observed next token given its prefix. Encoder-style objectives mask or transform this conditional structure, but still learn through token prediction.",
    metrics: "held-out loss, task accuracy, exact match, factuality, calibration, throughput, time to first token, and memory per token",
    system: "llm-serving",
    curriculum: "/learn/natural-language-processing/reasoning-in-language-models",
  },
  retrieval: {
    keywords: ["retrieval", "search", "embedding", "rank", "recommend", "collaborative", "matrix factor"],
    intuition: "The system must place a relevant item near the top of a very large candidate set, usually under a strict latency budget.",
    prior: "Lexical matching and hand-designed ranking features were interpretable but could miss semantic equivalence, sparse feedback, and rapidly changing user intent.",
    failure: "Offline relevance can diverge from user value because of exposure bias, stale indexes, popularity feedback loops, weak negatives, and serving-time candidate loss.",
    flow: "query/user context -> representation -> candidate retrieval -> scoring or reranking -> calibrated top-k results",
    equation: String.raw`s(q,d)=\frac{f_\theta(q)^\top g_\phi(d)}{\lVert f_\theta(q)\rVert\,\lVert g_\phi(d)\rVert}`,
    symbols: "The score compares learned query and document or item representations. Training must provide informative positive and negative pairs; serving must preserve the same similarity definition in the index.",
    metrics: "recall at k, mean reciprocal rank, NDCG, coverage, diversity, index freshness, p95 latency, and cost per query",
    system: "search-engines",
    curriculum: "/learn/rag-systems/retrieval-fundamentals",
  },
  generative: {
    keywords: ["generative", "diffusion", "gan", "vae", "flow", "image synthesis", "dall-e"],
    intuition: "A generative model learns a procedure for producing new samples that reflect the structure of the training distribution rather than merely assigning a class label.",
    prior: "Earlier likelihood, adversarial, and autoregressive approaches traded off sample fidelity, mode coverage, stable optimization, and inference speed in different ways.",
    failure: "Visual quality alone can hide memorization, missing modes, prompt bias, unstable training, poor likelihood, or unsafe reproduction of training data.",
    flow: "data or noise -> latent/state transformation -> conditional denoising or decoding -> sample -> quality and coverage evaluation",
    equation: String.raw`\min_\theta\;\mathbb{E}_{x,t,\epsilon}\left[\lVert \epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\right]`,
    symbols: "This representative denoising objective asks a model to recover injected noise at level $t$. GANs and latent-variable papers use different objectives; compare the paper’s actual loss rather than treating all generators as diffusion models.",
    metrics: "sample quality, distribution coverage, conditional alignment, reconstruction or likelihood measures, inference steps, throughput, and safety slices",
    system: "llm-serving",
    curriculum: "/learn/computer-vision/video-generation",
  },
  reinforcement: {
    keywords: ["reinforcement", "policy", "reward", "q-learning", "actor", "critic", "ppo", "rlhf", "preference"],
    intuition: "The learner chooses actions whose consequences may appear much later, so it must connect delayed reward to the decisions that caused it.",
    prior: "Tabular methods and unstable function approximation struggled with large observation spaces, off-policy data, and high-variance policy gradients.",
    failure: "Reward misspecification, distribution shift, unstable bootstrapping, poor exploration, and high-variance estimates can produce policies that look successful on one metric but fail behaviorally.",
    flow: "observation -> policy/value model -> action -> environment or preference signal -> return/advantage -> update",
    equation: String.raw`\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a\mid s)\,\hat A(s,a)\right]`,
    symbols: "The policy gradient increases probability for actions with positive estimated advantage and decreases it for negative advantage. The paper may modify the estimator, trust region, critic, replay, or reward source.",
    metrics: "expected return, success rate, sample efficiency, stability across seeds, constraint violations, calibration of the reward model, and inference cost",
    system: "agent-systems",
    curriculum: "/learn/reinforcement-learning",
  },
  graph: {
    keywords: ["graph", "node", "edge", "gnn", "network embedding"],
    intuition: "A graph model updates each entity using information carried by its neighbors while respecting that node order is arbitrary.",
    prior: "Spectral methods and shallow graph embeddings captured useful structure but were difficult to scale, transfer, or combine with rich node and edge features.",
    failure: "Repeated aggregation can oversmooth representations, amplify degree bias, leak split information, or become prohibitively expensive on high-degree dynamic graphs.",
    flow: "nodes/edges/features -> neighborhood sampling or propagation -> permutation-invariant aggregation -> node/edge/graph head",
    equation: String.raw`h_v^{(l+1)}=\phi\!\left(h_v^{(l)},\operatorname{AGG}\{h_u^{(l)}:u\in\mathcal{N}(v)\}\right)`,
    symbols: "The aggregation must be invariant to neighbor order. The update function controls how self-information and neighborhood evidence combine at each hop.",
    metrics: "node or graph task quality, inductive transfer, sampling variance, memory by edge count, throughput, and performance by degree",
    system: "recommendation-systems",
    curriculum: "/learn/machine-learning/multi-task-and-transfer-learning-theory",
  },
  multimodal: {
    keywords: ["multimodal", "vision-language", "clip", "audio", "video", "imagebind", "vlm"],
    intuition: "A multimodal model must align information that arrives through different sensors or symbol systems while preserving what is unique to each modality.",
    prior: "Separate unimodal encoders performed well in isolation but required expensive paired labels or brittle late fusion to support cross-modal tasks.",
    failure: "A shared space can overfit shortcuts, ignore one modality, inherit dataset bias, or appear aligned on retrieval while failing compositional understanding.",
    flow: "modality-specific inputs -> encoders/projectors -> shared representation or cross-attention -> task-conditioned decoder",
    equation: String.raw`\mathcal{L}_{align}=-\log\frac{\exp(s(z_a,z_b)/\tau)}{\sum_j\exp(s(z_a,z_j)/\tau)}`,
    symbols: "The contrastive form aligns a matching cross-modal pair while separating mismatches in the batch. Generative multimodal papers may instead condition a decoder through cross-attention.",
    metrics: "cross-modal retrieval, zero-shot transfer, task accuracy, modality ablations, compositional robustness, latency, and encoder cost",
    system: "tiktok-architecture",
    curriculum: "/learn/computer-vision/vision-language-models",
  },
  agents: {
    keywords: ["agent", "tool", "reasoning", "chain-of-thought", "planning", "webgpt", "multi-agent"],
    intuition: "An agent turns model outputs into a sequence of decisions, observations, tool calls, and stopping judgments rather than one isolated prediction.",
    prior: "Single-turn language models could describe actions but lacked a grounded loop for executing them, observing consequences, and recovering from errors.",
    failure: "Small reasoning errors compound across steps; tools can fail; state can become stale; and a weak stopping rule can waste cost or trigger unsafe actions.",
    flow: "goal + state -> reason/plan -> validated action or tool call -> observation -> updated state -> stop or repeat",
    equation: String.raw`\tau=(s_0,a_0,o_1,\ldots,a_{T-1},o_T),\qquad J=\mathbb{E}_{\tau}[R(\tau)-\lambda C(\tau)]`,
    symbols: "The trajectory $\\tau$ records decisions and observations. A useful objective balances task reward $R$ against cost, latency, or risk $C$, not just final-answer accuracy.",
    metrics: "task success, tool-call validity, recovery rate, step count, latency, cost, unsafe-action rate, and performance under tool failure",
    system: "agent-systems",
    curriculum: "/learn/ai-agents/agent-architecture-and-reasoning-loops",
  },
  systems: {
    keywords: ["system", "distributed", "serving", "training", "scaling", "parallel", "memory", "kernel", "database"],
    intuition: "The contribution is best understood as a resource-allocation design: it rearranges computation, communication, memory, or data flow so a useful model becomes practical at scale.",
    prior: "Earlier implementations were bounded by one dominant resource such as accelerator memory, network bandwidth, data movement, synchronization, or tail latency.",
    failure: "A design that improves peak throughput can still regress tail latency, reliability, numerical stability, debuggability, or cost under realistic traffic.",
    flow: "request or training batch -> scheduler/partitioner -> parallel compute and communication -> aggregation -> result + telemetry",
    equation: String.raw`T_{step}\approx\max(T_{compute},T_{memory},T_{network})+T_{sync}`,
    symbols: "The slowest resource dominates once work is overlapped. The paper’s mechanism should be evaluated by identifying which term it reduces and which new overhead it introduces.",
    metrics: "throughput, p50/p95/p99 latency, accelerator utilization, peak memory, communication volume, failure recovery, and cost per useful result",
    system: "llm-serving",
    curriculum: "/learn/ai-system-design/distributed-inference",
  },
  general: {
    keywords: [],
    intuition: "The paper proposes a reusable learning or systems principle and demonstrates why it is preferable to the strongest practical baseline available at the time.",
    prior: "Earlier methods captured part of the problem but left an important gap in representation, optimization, data efficiency, scale, or evaluation.",
    failure: "The main risk is mistaking a benchmark improvement for a general solution without testing assumptions, comparable budgets, ablations, and out-of-distribution behavior.",
    flow: "input and assumptions -> proposed mechanism -> learned representation or decision -> evaluation -> error analysis",
    equation: String.raw`\theta^*=\arg\min_\theta\;\mathbb{E}_{(x,y)\sim\mathcal{D}}[\mathcal{L}(f_\theta(x),y)]`,
    symbols: "This generic risk-minimization view separates the model $f_\\theta$, data distribution $\\mathcal D$, and objective $\\mathcal L$. The primary paper defines the specialized form.",
    metrics: "the paper’s primary task metric, robustness slices, sample efficiency, compute, memory, latency, and variance across runs",
    system: "llm-serving",
    curriculum: "/learn/machine-learning/learning-theory",
  },
};

function mdx(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("{", "&#123;")
    .replaceAll("}", "&#125;");
}

function familyFor(paper) {
  const text = `${paper.title} ${paper.section} ${paper.architecturesIntroduced ?? ""} ${paper.conceptsIntroduced ?? ""}`.toLowerCase();
  const priority = ["detection", "segmentation", "reinforcement", "agents", "retrieval", "graph", "multimodal", "generative", "optimization", "language", "vision", "systems"];
  return priority.find((name) => guides[name].keywords.some((word) => text.includes(word))) ?? "general";
}

function conceptsFor(paper) {
  return String(paper.conceptsIntroduced ?? "the paper's central mechanism")
    .split(/[,;]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 5);
}

function architectureFor(paper) {
  const haystack = `${paper.title} ${paper.architecturesIntroduced ?? ""}`.toLowerCase();
  const match = architectures
    .filter((item) => haystack.includes(item.name.toLowerCase()) || haystack.includes(item.slug))
    .sort((a, b) => b.name.length - a.name.length)[0];
  return match?.slug;
}

function neighborsFor(paper) {
  const section = bySection.get(paper.section) ?? [];
  const index = section.findIndex((item) => item.slug === paper.slug);
  return [section[index - 1], section[index + 1]].filter(Boolean);
}

function implementation(family) {
  if (family === "language" || family === "multimodal") return `\`\`\`python
import torch

def scaled_attention(query, key, value, mask=None):
    scale = query.shape[-1] ** -0.5
    scores = query @ key.transpose(-2, -1) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ value

q = k = v = torch.randn(2, 4, 16, 64)
output = scaled_attention(q, k, v)
assert output.shape == q.shape
\`\`\``;
  if (family === "retrieval") return `\`\`\`python
import torch
import torch.nn.functional as F

queries = F.normalize(torch.randn(8, 128), dim=-1)
documents = F.normalize(torch.randn(100, 128), dim=-1)
scores = queries @ documents.T
top_scores, top_ids = scores.topk(k=5, dim=-1)
assert top_ids.shape == (8, 5)
\`\`\``;
  if (family === "graph") return `\`\`\`python
import torch

def mean_message_passing(node_features, adjacency):
    degree = adjacency.sum(-1, keepdim=True).clamp_min(1)
    neighbor_mean = adjacency @ node_features / degree
    return torch.relu(node_features + neighbor_mean)

x = torch.randn(6, 32)
adj = torch.eye(6)
assert mean_message_passing(x, adj).shape == x.shape
\`\`\``;
  if (family === "reinforcement" || family === "agents") return `\`\`\`python
import torch

log_prob = torch.randn(32, requires_grad=True)
advantage = torch.randn(32)
policy_loss = -(log_prob * advantage.detach()).mean()
policy_loss.backward()
assert log_prob.grad is not None
\`\`\``;
  if (family === "generative") return `\`\`\`python
import torch

clean = torch.randn(8, 64)
noise = torch.randn_like(clean)
alpha = torch.tensor(0.7)
noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
predicted_noise = torch.zeros_like(noise)  # replace with the paper's model
loss = (predicted_noise - noise).square().mean()
assert loss.ndim == 0
\`\`\``;
  return `\`\`\`python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 32))
inputs = torch.randn(8, 64)
outputs = model(inputs)
assert outputs.shape == (8, 32)
\`\`\``;
}

function render(paper) {
  const family = familyFor(paper);
  const guide = guides[family];
  const concepts = conceptsFor(paper);
  const neighbors = neighborsFor(paper);
  const architecture = architectureFor(paper);
  const importance = mdx(paper.whyImportant || "Its contribution clarified an important design choice and influenced subsequent research.");
  const impact = mdx(paper.industryImpact || "Its practical impact is best evaluated by tracing which later systems adopted the mechanism and under what constraints.");
  const lineage = mdx(paper.builtUponBy || "Later work tested the idea at larger scale, under different data regimes, and with stronger implementation techniques.");
  const architectureText = mdx(paper.architecturesIntroduced || "A task-specific realization of the paper's central mechanism");
  const conceptBullets = concepts
    .map((concept) => `- **${mdx(concept)}:** identify its input, output, learned parameters, and the assumption it adds to the baseline.`)
    .join("\n");
  const paperLinks = neighbors
    .map((item) => `- [${mdx(item.title)}](/papers/${item.slug}) — adjacent reading from the same library section.`)
    .join("\n");
  const architectureLink = architecture
    ? `- [Architecture deep dive](/architectures/${architecture}) — inspect the corresponding model family and tensor flow.`
    : `- [Architecture library](/architectures) — compare this mechanism with related model families.`;

  return `# ${mdx(paper.title)}

> **Reading goal:** explain the problem, derive the central mechanism, reproduce a minimal version, and state where the method can fail.

## 1. Overview

Published in ${mdx(paper.year || "an unspecified year")} by ${mdx(paper.authors || "the listed authors")}, **${mdx(paper.title)}** belongs to the ${mdx(paper.section)} part of the Paper2Code library. ${importance} [1]

The shortest useful mental model is this: ${guide.intuition} The paper matters because it turns that general need into a concrete method that can be compared, implemented, and challenged. Do not memorize the name first. Start by asking what enters the method, what leaves it, what is learned, and which bottleneck is removed.

## 2. Historical Context

${guide.prior} By ${mdx(paper.year || "the paper's publication period")}, researchers had working baselines, but the remaining gap was large enough that simply adding more layers, data, or hardware did not answer the scientific question.

The paper should be read in two timelines. The first is the **research timeline**: which assumptions from prior work it kept and which it rejected. The second is the **engineering timeline**: which computational constraint made the idea newly useful. Keeping these timelines separate prevents a common reading error—crediting an algorithm for gains that actually came from a different dataset, evaluation protocol, or compute budget.

## 3. Research Problem

The general problem can be stated as a contract: given the paper's input and available supervision, produce the required prediction, representation, action, or system behavior under a realistic resource budget. ${guide.failure}

A strong problem statement includes three baselines: the simplest credible baseline, the strongest comparable method available at the time, and an oracle or upper-bound condition where possible. When reproducing this work, hold data splits and evaluation code constant. Otherwise, you cannot tell whether the paper's mechanism caused the improvement.

## 4. Prior Work

Prior approaches typically solved only part of the contract. Some offered a good inductive bias but scaled poorly; others scaled but required more labels, memory, communication, or inference time. ${guide.prior}

Read the related-work section as a dependency graph, not a list of citations. For every predecessor, write down: **what it computes**, **its asymptotic or operational bottleneck**, **its supervision**, and **the failure case this paper targets**. This makes the novelty test concrete and keeps modern implementation improvements from being confused with the original contribution.

## 5. Why Previous Methods Failed

${guide.failure} Previous methods may also have been evaluated on averages that hid important slices. A result can look strong while failing on long inputs, rare classes, cold-start users, shifted environments, or strict latency targets.

The practical lesson is to locate the bottleneck before adopting the method. If your workload is limited by data quality but the paper solves memory bandwidth, implementation complexity will not rescue the system. Reproduce the baseline failure first; only then add the proposed mechanism.

## 6. Core Innovation

The registry summarizes the paper's introduced concepts as:

${conceptBullets}

Its associated architecture or procedure is **${architectureText}**. The contribution should be explained as a change to a baseline computation: what new state is introduced, where information flows differently, and which objective trains the change. ${importance}

An honest explanation also states what is *not* new. Dataset scale, engineering polish, and evaluation breadth may be essential to the result without being the algorithmic novelty. Separating these pieces makes the paper easier to reuse correctly.

## 7. Architecture Breakdown

\`\`\`text
${guide.flow}
       |                    |                    |
       +-- shape checks ----+-- metrics ---------+-- error analysis
\`\`\`

Trace one example through this flow. At each boundary, record the tensor or state shape, dtype, masking rule, normalization, and whether the operation runs during training, inference, or both. The defining mechanism belongs in one isolated module with a baseline-compatible interface. That design makes ablations possible and prevents the reproduction from becoming an untestable monolith.

## 8. Mathematical Foundations

A representative equation for this technical family is:

$$
${guide.equation}
$$

${guide.symbols} This equation is a **reading scaffold**, not a substitute for the primary paper's exact notation. Re-derive the specialized objective from [1], check dimensions term by term, and test it on a tiny hand-computable example before implementing a full training loop.

The most useful derivation questions are: Which quantities are observed? Which are latent or learned? Where does stochastic estimation enter? Which normalization prevents scale from changing the answer? What happens at an edge case such as an empty set, maximum sequence length, zero reward, or a single-class batch?

## 9. Key Experiments

The paper's main claim is summarized in the registry as: ${importance} The key experiments should therefore isolate **causality**, not merely display a large final score.

Reproduction plan:

1. Reproduce a simple baseline with the published split and metric definition.
2. Add only the defining mechanism and keep the training budget fixed.
3. Sweep the one or two hyperparameters that control the mechanism's strength or capacity.
4. Evaluate both the headline metric and failure slices.
5. Report ${guide.metrics}.

Use multiple seeds when training variance is material. Save predictions, not just aggregate scores, so disagreements can be inspected later.

## 10. Benchmark Results

The registry records the practical significance as: ${impact} This statement explains *why the paper remained relevant*, but exact benchmark values must be taken from the primary tables in [1]. They are intentionally not invented or copied without experimental context here.

When reading a table, verify whether higher or lower is better, whether results use extra data or pretraining, and whether compute is comparable. Report absolute values, the baseline delta, uncertainty where available, and operational cost. A smaller gain with a simpler or faster model may be the more important result.

## 11. Ablation Studies

An ablation removes or changes one component while preserving everything else. For this paper, begin with the introduced concepts listed above. Remove each component, replace it with the closest baseline operation, and vary its capacity. Also test data scale and compute scale separately; an idea that works only after a large budget increase is a different claim from an architecture that improves efficiency.

Useful negative controls include shuffled labels or relationships, a frozen new module, matched parameter count, matched training FLOPs, and a simpler heuristic. A convincing ablation explains *why* the method works and identifies the regime where it stops helping.

## 12. Implementation Details

The following code is a minimal tensor-contract example for the paper's technical family. It is deliberately small enough to test; replace the placeholder mechanism with the exact equations from [1].

${implementation(family)}

Before scaling, add tests for shapes, masking, deterministic evaluation, serialization, and a tiny batch that the model can overfit. Log configuration, source revision, data version, random seeds, parameter count, and peak memory beside every run.

## 13. Engineering Insights

${impact} To translate that impact into a production decision, measure the complete path rather than an isolated kernel or offline model. Include preprocessing, retrieval or batching, host-device transfer, post-processing, cache behavior, and tail latency.

Design a safe fallback to the previous method. Shadow traffic or offline replay should confirm output compatibility before rollout. Monitor both quality and resource metrics, because an architecture can improve one while silently degrading the other. The paper's idea is production-ready only when its assumptions remain true under real traffic and data drift.

## 14. Limitations

${guide.failure} The primary paper's limitations and appendix should be treated as required reading, not optional caveats.

Additional questions to test are: Does the method depend on unusually clean data? Does it transfer across domains and scales? Is the comparison budget-matched? Can its objective be gamed? What happens under corrupted input or missing context? Does the method concentrate errors on a subgroup? These questions turn a paper summary into engineering judgment.

## 15. Influence on Later Research

${lineage} This lineage is most useful when read as a sequence of repaired limitations. For each descendant, identify whether it changes the objective, data, architecture, scaling strategy, or evaluation. Then distinguish direct descendants from papers that share vocabulary but solve a different problem.

The paper's enduring value is not that every detail remains state of the art. It is that later systems still inherit, modify, or explicitly reject its central abstraction.

## 16. Related Architectures

${architectureLink}
${paperLinks || "- [Paper library](/papers) — compare this work with neighboring papers in the same research area."}

## 17. Related Problems

- [Relevant curriculum lesson](${guide.curriculum}) — build the prerequisite theory and vocabulary.
- [Related production system](/system-design/${guide.system}) — see how the research idea interacts with serving and reliability constraints.
- [Practice in the Dojo](/dojo) — implement prerequisite tensor operations and test shape contracts.
- [Browse the paper library](/papers) — continue along the research timeline.

### Reference

[1] ${mdx(paper.authors || "Authors listed in the canonical registry")}. **${mdx(paper.title)}**. ${mdx(paper.year || "Year listed in the canonical registry")}.
`;
}

function tagsFor(paper, family) {
  const section = paper.section.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return [...new Set(["paper", family, section, ...conceptsFor(paper).slice(0, 2).map((item) => item.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""))])].filter(Boolean);
}

function normalizeDifficulty(value) {
  const normalized = String(value ?? "intermediate").toLowerCase();
  if (normalized === "expert") return "advanced";
  return ["beginner", "intermediate", "advanced"].includes(normalized)
    ? normalized
    : "intermediate";
}

let generatedArticles = 0;
let generatedMetadata = 0;
let reconciledMetadata = 0;
let preservedArticles = 0;

for (const paper of targets) {
  const directory = path.join(contentRoot, paper.slug);
  const articleFile = path.join(directory, "content.mdx");
  const metaFile = path.join(directory, "meta.json");
  fs.mkdirSync(directory, { recursive: true });

  if (fs.existsSync(articleFile)) {
    preservedArticles += 1;
  } else {
    fs.writeFileSync(articleFile, render(paper), "utf8");
    generatedArticles += 1;
  }

  const family = familyFor(paper);
  const neighbors = neighborsFor(paper).map((item) => item.slug);
  const architecture = architectureFor(paper);
  const existing = fs.existsSync(metaFile)
    ? JSON.parse(fs.readFileSync(metaFile, "utf8"))
    : {};
  const metadata = {
    ...existing,
    type: "paper",
    slug: paper.slug,
    title: paper.title,
    description: paper.whyImportant || existing.description || `A guided reading of ${paper.title}.`,
    tags: tagsFor(paper, family),
    difficulty: normalizeDifficulty(paper.difficulty),
    authors: paper.authors || existing.authors || "See primary paper",
    year: paper.year || existing.year,
    relationships: {
      papers: neighbors,
      ...(architecture && architectureSlugs.has(architecture)
        ? { architectures: [architecture] }
        : {}),
    },
  };
  fs.writeFileSync(metaFile, `${JSON.stringify(metadata, null, 2)}\n`, "utf8");
  if (Object.keys(existing).length > 0) reconciledMetadata += 1;
  else generatedMetadata += 1;
}

console.log(`Paper batch ${batch}: ranks ${fromRank}-${toRank}`);
console.log(`Registered targets: ${targets.length}`);
console.log(`Articles generated: ${generatedArticles}`);
console.log(`Existing articles preserved: ${preservedArticles}`);
console.log(`Metadata generated: ${generatedMetadata}`);
console.log(`Existing metadata reconciled: ${reconciledMetadata}`);
