import { ARCHITECTURES } from '@/data/content/architectures';
import { PAPERS } from '@/data/content/papers';
import { CURRICULUM } from '@/data/content/curriculum';
import { SD_SYSTEMS } from '@/data/content/systemDesign';

export const DOJO_SLUGS = [
  'numpy-array-creation',
  'ml-sigmoid',
  'ml-relu',
  'ml-mse',
  'numpy-dot-product',
  'ml-softmax',
  'ml-normalize',
  'ml-cross-entropy',
  'ml-gradient-descent',
  'ml-attention'
] as const;

export const WORKSPACE_PAPER_IDS = new Set<string>([
  'attention-is-all-you-need',
  'resnet',
  'bert',
  'vit',
  'lora',
  'flash-attention'
]);

// The static 200-paper Golden Library (src/data/content/papers.ts) auto-slugifies
// full paper titles, while the /papers/[id] workspace pages use short curated ids.
// This maps the 6 flagship library slugs to their workspace ids — use this (not a
// slug-identity check) anywhere a library entry links into a workspace.
export const LIBRARY_TO_WORKSPACE_ID: Record<string, string> = {
  'attention-is-all-you-need': 'attention-is-all-you-need',
  'deep-residual-learning-for-image-recognition': 'resnet',
  'bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding': 'bert',
  'an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale-vit': 'vit',
  'lora-low-rank-adaptation-of-large-language-models': 'lora',
  'flashattention-fast-and-memory-efficient-exact-attention-with-io-awareness': 'flash-attention',
};

export function findArch(name: string) {
  const trimmed = name.toLowerCase().trim();
  return ARCHITECTURES.find(a => a.name.toLowerCase().trim() === trimmed);
}

export function papersForArch(archName: string) {
  const trimmedArch = archName.toLowerCase().trim();
  return PAPERS.filter(p => {
    if (!p.architecturesIntroduced) return false;
    const tokens = p.architecturesIntroduced.split(',').map(s => s.toLowerCase().trim());
    return tokens.some(t => t === trimmedArch || t.includes(trimmedArch) || trimmedArch.includes(t));
  });
}

// Matches a raw comma-split token from a paper's `architecturesIntroduced`
// field (e.g. "ResNet-18/34/50/101/152", "BERT-Base") against the
// architecture library by substring containment. Deliberately narrower in
// scope than findArch: only ever called on that one controlled field, never
// on free-text (prerequisites, topic titles), where substring matching would
// risk false positives.
export function findArchLoose(token: string) {
  const trimmed = token.toLowerCase().trim();
  if (!trimmed) return undefined;
  return ARCHITECTURES.find(a => {
    const name = a.name.toLowerCase().trim();
    return name === trimmed || trimmed.includes(name) || name.includes(trimmed);
  });
}

export function findTopic(title: string) {
  const trimmed = title.toLowerCase().trim();
  for (const domain of CURRICULUM) {
    const topic = domain.topics.find(t => t.title.toLowerCase().trim() === trimmed);
    if (topic) {
      return { domainSlug: domain.slug, topicSlug: topic.slug, domainName: domain.name, topicTitle: topic.title };
    }
  }
  return undefined;
}

export type LearningReference = {
  href: string;
  kind: 'curriculum' | 'domain' | 'architecture' | 'system-design' | 'paper' | 'dojo';
};

const normalizeLearningReference = (value: string) => value
  .toLowerCase()
  .replace(/&/g, ' and ')
  .replace(/[^a-z0-9]+/g, ' ')
  .trim();

const LEARNING_REFERENCE_ALIASES: Record<string, LearningReference> = {
  'scaling law theory': { href: '/learn/deep-learning/scaling-laws', kind: 'curriculum' },
  'double descent': { href: '/learn/machine-learning/learning-theory', kind: 'curriculum' },
  'implicit regularization': { href: '/learn/machine-learning/learning-theory', kind: 'curriculum' },
  'flow matching': { href: '/learn/mathematics/optimal-transport', kind: 'curriculum' },
  'diffusion theory': { href: '/architectures/ddpm', kind: 'architecture' },
  'diffusion models theory': { href: '/architectures/ddpm', kind: 'architecture' },
  'vae': { href: '/learn/mathematics/variational-inference-and-free-energy', kind: 'curriculum' },
  'vaes': { href: '/learn/mathematics/variational-inference-and-free-energy', kind: 'curriculum' },
  'neural architecture search': { href: '/learn/machine-learning/neural-architecture-search-nas', kind: 'curriculum' },
  'automl': { href: '/learn/machine-learning/neural-architecture-search-nas', kind: 'curriculum' },
  'scaling law research': { href: '/learn/deep-learning/scaling-laws', kind: 'curriculum' },
  'pre training research': { href: '/learn/deep-learning/scaling-laws', kind: 'curriculum' },
  'mixture of experts': { href: '/learn/deep-learning/mixture-of-experts-moe', kind: 'curriculum' },
  'moe': { href: '/learn/deep-learning/mixture-of-experts-moe', kind: 'curriculum' },
  'sparse training': { href: '/learn/deep-learning/mixture-of-experts-moe', kind: 'curriculum' },
  'circuit analysis': { href: '/learn/deep-learning/mechanistic-interpretability', kind: 'curriculum' },
  'sparse autoencoders': { href: '/learn/research-engineering/mechanistic-interpretability-2', kind: 'curriculum' },
  'neural odes': { href: '/learn/deep-learning/neural-odes-and-continuous-depth-networks', kind: 'curriculum' },
  'normalizing flows': { href: '/learn/deep-learning/neural-odes-and-continuous-depth-networks', kind: 'curriculum' },
  'score based models': { href: '/architectures/score-sde', kind: 'architecture' },
  'chain of thought': { href: '/learn/natural-language-processing/reasoning-in-language-models', kind: 'curriculum' },
  'process reward models': { href: '/learn/natural-language-processing/reasoning-in-language-models', kind: 'curriculum' },
  'grpo': { href: '/learn/natural-language-processing/reasoning-in-language-models', kind: 'curriculum' },
  'rlhf': { href: '/architectures/instructgpt', kind: 'architecture' },
  'instruction tuning': { href: '/architectures/instructgpt', kind: 'architecture' },
  'constitutional ai': { href: '/learn/natural-language-processing/alignment', kind: 'curriculum' },
  'safety evals': { href: '/learn/natural-language-processing/alignment', kind: 'curriculum' },
  'rag': { href: '/system-design/rag-systems', kind: 'system-design' },
  'production rag': { href: '/system-design/rag-systems', kind: 'system-design' },
  'all rag components': { href: '/system-design/rag-systems', kind: 'system-design' },
  'rag infrastructure': { href: '/system-design/rag-systems', kind: 'system-design' },
  'faithful rag': { href: '/learn/natural-language-processing/factuality-grounding-and-hallucination', kind: 'curriculum' },
  'citation generation': { href: '/learn/natural-language-processing/factuality-grounding-and-hallucination', kind: 'curriculum' },
  'verification pipelines': { href: '/learn/natural-language-processing/factuality-grounding-and-hallucination', kind: 'curriculum' },
  'vlms': { href: '/learn/computer-vision/vision-language-models', kind: 'curriculum' },
  'multi modal rag': { href: '/learn/rag-systems/multi-modal-rag', kind: 'curriculum' },
  'vision language rag': { href: '/learn/rag-systems/multi-modal-rag', kind: 'curriculum' },
  'document ai': { href: '/learn/rag-systems/multi-modal-rag', kind: 'curriculum' },
  'continual learning': { href: '/learn/llm-engineering/continual-learning-and-catastrophic-forgetting', kind: 'curriculum' },
  'lifelong learning systems': { href: '/learn/llm-engineering/continual-learning-and-catastrophic-forgetting', kind: 'curriculum' },
  'model merging': { href: '/learn/llm-engineering/model-merging', kind: 'curriculum' },
  'lora': { href: '/papers/lora', kind: 'paper' },
  'qlora': { href: '/learn/research-engineering/efficient-fine-tuning-research', kind: 'curriculum' },
  'fine tuning': { href: '/learn/research-engineering/efficient-fine-tuning-research', kind: 'curriculum' },
  'distributed training': { href: '/learn/llm-engineering/tensor-parallelism-pipeline-parallelism-and-fsdp', kind: 'curriculum' },
  'tensor parallelism': { href: '/learn/llm-engineering/tensor-parallelism-pipeline-parallelism-and-fsdp', kind: 'curriculum' },
  'pipeline parallelism': { href: '/learn/llm-engineering/tensor-parallelism-pipeline-parallelism-and-fsdp', kind: 'curriculum' },
  'fsdp': { href: '/learn/llm-engineering/tensor-parallelism-pipeline-parallelism-and-fsdp', kind: 'curriculum' },
  'flash attention': { href: '/learn/llm-engineering/flash-attention-and-memory-efficient-attention', kind: 'curriculum' },
  'custom cuda kernels': { href: '/learn/ai-system-design/custom-cuda-kernels-for-ml', kind: 'curriculum' },
  'long context': { href: '/learn/llm-engineering/flash-attention-and-memory-efficient-attention', kind: 'curriculum' },
  'pagedattention': { href: '/system-design/llm-serving', kind: 'system-design' },
  'continuous batching': { href: '/system-design/llm-serving', kind: 'system-design' },
  'kv cache': { href: '/system-design/llm-serving', kind: 'system-design' },
  'llmops': { href: '/system-design/llm-serving', kind: 'system-design' },
  'agent architecture': { href: '/system-design/agent-systems', kind: 'system-design' },
  'memory systems': { href: '/system-design/agent-systems', kind: 'system-design' },
  'multi agent': { href: '/system-design/agent-systems', kind: 'system-design' },
  'mcts': { href: '/architectures/alphazero', kind: 'architecture' },
  'self play': { href: '/architectures/alphazero', kind: 'architecture' },
  'model based rl': { href: '/learn/reinforcement-learning/world-models', kind: 'curriculum' },
  'world models': { href: '/learn/reinforcement-learning/world-models', kind: 'curriculum' },
  'safe rl': { href: '/learn/reinforcement-learning/safe-rl-and-constrained-mdp', kind: 'curriculum' },
  'ai safety': { href: '/learn/natural-language-processing/alignment', kind: 'curriculum' },
  'diffusion': { href: '/architectures/ddpm', kind: 'architecture' },
  'latent diffusion': { href: '/architectures/stable-diffusion', kind: 'architecture' },
  'foundation models': { href: '/learn/computer-vision/foundation-models-for-vision', kind: 'curriculum' },
  'ml platform design': { href: '/learn/mlops/ml-platform-engineering', kind: 'curriculum' },
  'platform architecture': { href: '/learn/mlops/ml-platform-engineering', kind: 'curriculum' },
  'training infrastructure design': { href: '/learn/mlops/large-scale-training-infrastructure', kind: 'curriculum' },
  'inference infrastructure design': { href: '/learn/mlops/real-time-inference-at-scale', kind: 'curriculum' },
  'cost optimization at scale': { href: '/learn/mlops/cost-engineering-and-carbon-accounting-for-ml', kind: 'curriculum' },
  'model cards': { href: '/learn/mlops/ml-governance-and-compliance', kind: 'curriculum' },
  'monitoring': { href: '/learn/mlops/ml-platform-engineering', kind: 'curriculum' },
};

export function resolveLearningReference(value: string): LearningReference | undefined {
  const normalized = normalizeLearningReference(value);

  for (const domain of CURRICULUM) {
    const topic = domain.topics.find(item => normalizeLearningReference(item.title) === normalized);
    if (topic) return { href: `/learn/${domain.slug}/${topic.slug}`, kind: 'curriculum' };
    if (normalizeLearningReference(domain.name) === normalized) {
      return { href: `/learn/${domain.slug}`, kind: 'domain' };
    }
  }

  const architecture = ARCHITECTURES.find(item =>
    normalizeLearningReference(item.name) === normalized || normalizeLearningReference(item.slug) === normalized
  );
  if (architecture) return { href: `/architectures/${architecture.slug}`, kind: 'architecture' };

  const system = SD_SYSTEMS.find(item =>
    normalizeLearningReference(item.name) === normalized || normalizeLearningReference(item.slug) === normalized
  );
  if (system) return { href: `/system-design/${system.slug}`, kind: 'system-design' };

  const paper = PAPERS.find(item => normalizeLearningReference(item.title) === normalized);
  if (paper && LIBRARY_TO_WORKSPACE_ID[paper.slug]) {
    return { href: `/papers/${LIBRARY_TO_WORKSPACE_ID[paper.slug]}`, kind: 'paper' };
  }

  return LEARNING_REFERENCE_ALIASES[normalized];
}

export function dojoSlugFor(text: string): string | undefined {
  if (!text) return undefined;
  const lower = text.toLowerCase();
  if (lower.includes('attention')) return 'ml-attention';
  if (lower.includes('softmax')) return 'ml-softmax';
  if (lower.includes('gradient') || lower.includes('optimization')) return 'ml-gradient-descent';
  if (lower.includes('normalization') || lower.includes('norm')) return 'ml-normalize';
  if (lower.includes('cross-entropy') || lower.includes('cross entropy')) return 'ml-cross-entropy';
  if (lower.includes('sigmoid')) return 'ml-sigmoid';
  if (lower.includes('relu')) return 'ml-relu';
  if (lower.includes('mse') || lower.includes('squared error')) return 'ml-mse';
  if (lower.includes('dot product') || lower.includes('matrix mult') || lower.includes('linear')) return 'numpy-dot-product';
  if (lower.includes('initialization') || lower.includes('array')) return 'numpy-array-creation';
  return undefined;
}
