import { ARCHITECTURES } from '@/data/content/architectures';
import { PAPERS } from '@/data/content/papers';
import { CURRICULUM } from '@/data/content/curriculum';

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
