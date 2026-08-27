#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = fs.readFileSync(path.join(ROOT, "src/data/content/curriculum.ts"), "utf8");
const literal = source.match(/export const CURRICULUM[^=]*=\s*(\[[\s\S]*\]);/)?.[1];
if (!literal) throw new Error("Could not parse CURRICULUM");
const curriculum = Function(`"use strict"; return (${literal});`)();

const domainLinks = {
  mathematics: ["Architecture application", "/architectures/transformer"],
  "machine-learning": ["Architecture application", "/architectures/darts"],
  "deep-learning": ["Architecture application", "/architectures/transformer"],
  "natural-language-processing": ["Architecture application", "/architectures/bert"],
  "llm-engineering": ["Production system", "/system-design/llm-serving"],
  "rag-systems": ["Production system", "/system-design/rag-systems"],
  "ai-agents": ["Production system", "/system-design/agent-systems"],
  "reinforcement-learning": ["Architecture application", "/architectures/ppo"],
  "computer-vision": ["Architecture application", "/architectures/vit"],
  mlops: ["Production system", "/system-design/llm-serving"],
  "ai-system-design": ["System-design library", "/system-design/llm-serving"],
  "research-engineering": ["Research architecture library", "/architectures/transformer"],
};

function topicLink(domain, topic) {
  const key = `${domain.slug}/${topic.slug}`;
  const exact = {
    "mathematics/optimal-transport": ["Diffusion application", "/architectures/ddpm"],
    "mathematics/variational-inference-and-free-energy": ["VAE application", "/architectures/stable-diffusion"],
    "machine-learning/neural-architecture-search-nas": ["DARTS", "/architectures/darts"],
    "deep-learning/scaling-laws": ["Chinchilla", "/architectures/chinchilla"],
    "deep-learning/mixture-of-experts-moe": ["Mixtral 8x7B", "/architectures/mixtral-8x7b"],
    "deep-learning/mechanistic-interpretability": ["Transformer", "/architectures/transformer"],
    "natural-language-processing/reasoning-in-language-models": ["DeepSeek-V2", "/architectures/deepseek-v2"],
    "natural-language-processing/factuality-grounding-and-hallucination": ["RAG Systems", "/system-design/rag-systems"],
    "natural-language-processing/alignment": ["InstructGPT", "/architectures/instructgpt"],
    "llm-engineering/flash-attention-and-memory-efficient-attention": ["LLM Serving", "/system-design/llm-serving"],
    "llm-engineering/tensor-parallelism-pipeline-parallelism-and-fsdp": ["Distributed inference", "/system-design/llm-serving"],
    "rag-systems/multi-modal-rag": ["CLIP", "/architectures/clip"],
    "rag-systems/rag-for-code": ["Agent Systems", "/system-design/agent-systems"],
    "ai-agents/world-models-for-agents": ["Dreamer", "/architectures/dreamer"],
    "reinforcement-learning/world-models": ["Dreamer", "/architectures/dreamer"],
    "reinforcement-learning/alphazero-and-mcts-with-deep-rl": ["AlphaZero", "/architectures/alphazero"],
    "computer-vision/vision-language-models": ["CLIP", "/architectures/clip"],
    "computer-vision/foundation-models-for-vision": ["Segment Anything", "/architectures/sam"],
    "computer-vision/video-generation": ["Diffusion Transformer", "/architectures/dit"],
    "computer-vision/generative-video-editing-and-controlnet": ["ControlNet", "/architectures/controlnet"],
    "mlops/ml-platform-engineering": ["Uber Architecture", "/system-design/uber-architecture"],
    "mlops/real-time-inference-at-scale": ["LLM Serving", "/system-design/llm-serving"],
    "ai-system-design/system-design-for-retrieval-at-scale": ["Vector Databases", "/system-design/vector-databases"],
    "ai-system-design/llm-gateway-design": ["ChatGPT Architecture", "/system-design/chatgpt-architecture"],
    "ai-system-design/production-safety-and-alignment-systems": ["ChatGPT Architecture", "/system-design/chatgpt-architecture"],
    "ai-system-design/compound-ai-systems": ["Agent Systems", "/system-design/agent-systems"],
    "research-engineering/scaling-law-modeling": ["Chinchilla", "/architectures/chinchilla"],
    "research-engineering/mechanistic-interpretability-2": ["Transformer", "/architectures/transformer"],
    "research-engineering/alignment-research-methods": ["InstructGPT", "/architectures/instructgpt"],
  };
  return exact[key] ?? domainLinks[domain.slug];
}

let changed = 0;
for (const domain of curriculum) {
  for (const topic of domain.topics) {
    const file = path.join(ROOT, "src/content/curriculum", domain.slug, topic.slug, "content.mdx");
    let body = fs.readFileSync(file, "utf8");
    if (body.includes("## Related Platform Content")) continue;
    const [label, href] = topicLink(domain, topic);
    body = `${body.trim()}\n\n## Related Platform Content\n\n- [${domain.name} curriculum overview](/learn/${domain.slug})\n- [${label}](${href})\n- [Browse all system designs](/system-design)\n`;
    fs.writeFileSync(file, body, "utf8");
    changed += 1;
  }
}

console.log(`Added related-content links to ${changed} canonical curriculum lessons.`);
