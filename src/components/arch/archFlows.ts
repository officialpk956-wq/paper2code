// Pure data + helpers for architecture/system diagrams.
// NOT a client module — safe to import from server components (paper &
// system-design detail pages) as well as the client ArchDiagram.

export type GenBlock = { label: string; sub?: string; accent?: string };

// Linear block-flow definitions for architectures / systems that don't have a
// hand-drawn diagram. Keep labels short so they fit the boxes.
export const GENERIC_FLOWS: Record<string, GenBlock[]> = {
  // ── architectures ─────────────────────────────────────────────
  cnn: [
    { label: 'Input', sub: 'Image', accent: '#A5B4FC' },
    { label: 'Conv', sub: '+ ReLU', accent: '#60A5FA' },
    { label: 'MaxPool', accent: '#60A5FA' },
    { label: 'Conv', sub: '+ ReLU', accent: '#60A5FA' },
    { label: 'MaxPool', accent: '#60A5FA' },
    { label: 'Flatten', accent: '#A78BFA' },
    { label: 'FC', accent: '#F472B6' },
    { label: 'Softmax', sub: 'Class', accent: '#34D399' },
  ],
  gan: [
    { label: 'Noise z', accent: '#A5B4FC' },
    { label: 'Generator', accent: '#A78BFA' },
    { label: 'Fake', sub: 'Image', accent: '#F472B6' },
    { label: 'Discrim.', accent: '#FB923C' },
    { label: 'Real / Fake', accent: '#34D399' },
  ],
  diffusion: [
    { label: 'Noise', sub: 'x_T', accent: '#A5B4FC' },
    { label: 'U-Net', sub: 'denoise ×T', accent: '#A78BFA' },
    { label: 'Latent', sub: 'x_0', accent: '#60A5FA' },
    { label: 'Decoder', accent: '#F472B6' },
    { label: 'Image', accent: '#34D399' },
  ],
  lstm: [
    { label: 'Input', accent: '#A5B4FC' },
    { label: 'Embedding', accent: '#60A5FA' },
    { label: 'LSTM', sub: '×T steps', accent: '#A78BFA' },
    { label: 'Hidden', sub: 'state', accent: '#60A5FA' },
    { label: 'Dense', accent: '#F472B6' },
    { label: 'Output', accent: '#34D399' },
  ],
  clip: [
    { label: 'Image', accent: '#A5B4FC' },
    { label: 'Image', sub: 'Encoder', accent: '#60A5FA' },
    { label: 'Contrastive', sub: 'align', accent: '#A78BFA' },
    { label: 'Text', sub: 'Encoder', accent: '#F472B6' },
    { label: 'Text', accent: '#FB923C' },
  ],
  autoencoder: [
    { label: 'Input', accent: '#A5B4FC' },
    { label: 'Encoder', accent: '#60A5FA' },
    { label: 'Latent z', accent: '#A78BFA' },
    { label: 'Decoder', accent: '#F472B6' },
    { label: 'Recon', accent: '#34D399' },
  ],
  seq2seq: [
    { label: 'Input', accent: '#A5B4FC' },
    { label: 'Encoder', sub: 'RNN', accent: '#60A5FA' },
    { label: 'Attention', accent: '#A78BFA' },
    { label: 'Decoder', sub: 'RNN', accent: '#F472B6' },
    { label: 'Output', accent: '#34D399' },
  ],
  // ── system-design pipelines ───────────────────────────────────
  rag: [
    { label: 'Query', accent: '#A5B4FC' },
    { label: 'Embed', accent: '#60A5FA' },
    { label: 'Vector', sub: 'Search', accent: '#A78BFA' },
    { label: 'Retrieve', sub: 'top-k', accent: '#F472B6' },
    { label: 'LLM', accent: '#FB923C' },
    { label: 'Answer', accent: '#34D399' },
  ],
  'vector-search': [
    { label: 'Docs', accent: '#A5B4FC' },
    { label: 'Embed', accent: '#60A5FA' },
    { label: 'Index', sub: 'HNSW', accent: '#A78BFA' },
    { label: 'ANN', sub: 'search', accent: '#F472B6' },
    { label: 'Results', accent: '#34D399' },
  ],
  'llm-serving': [
    { label: 'Request', accent: '#A5B4FC' },
    { label: 'Router', accent: '#60A5FA' },
    { label: 'KV Cache', accent: '#A78BFA' },
    { label: 'GPU', sub: 'batch', accent: '#F472B6' },
    { label: 'Stream', sub: 'tokens', accent: '#FB923C' },
    { label: 'Response', accent: '#34D399' },
  ],
  recommender: [
    { label: 'User', accent: '#A5B4FC' },
    { label: 'Features', accent: '#60A5FA' },
    { label: 'Candidates', accent: '#A78BFA' },
    { label: 'Ranking', accent: '#F472B6' },
    { label: 'Re-rank', accent: '#FB923C' },
    { label: 'Feed', accent: '#34D399' },
  ],
  'sd-default': [
    { label: 'Client', accent: '#A5B4FC' },
    { label: 'API', sub: 'Gateway', accent: '#60A5FA' },
    { label: 'Service', accent: '#A78BFA' },
    { label: 'Model', accent: '#F472B6' },
    { label: 'Store', accent: '#FB923C' },
    { label: 'Response', accent: '#34D399' },
  ],
};

/** Map a paper id/slug to a diagram slug via keyword matching. */
export function paperToArchSlug(id: string): string | null {
  const s = id.toLowerCase();
  if (s.includes('attention-is-all') || s === 'transformer') return 'transformer';
  if (s.includes('residual') || s.includes('resnet')) return 'resnet';
  if (s.includes('bert')) return 'bert';
  if (s.includes('image-is-worth') || s.includes('vision-transformer') || s === 'vit') return 'vit';
  if (s.includes('llama')) return 'llama';
  if (s.includes('instructgpt') || s.includes('rlhf') || s.includes('ppo') || s.includes('human-feedback')) return 'rlhf';
  if (s.includes('gpt') || s.includes('generative-pre') || s.includes('language-models-are')) return 'gpt';
  if (s.includes('mixture-of-expert') || s.includes('moe') || s.includes('mixtral') || s.includes('switch-transformer')) return 'moe';
  if (s.includes('mamba') || s.includes('state-space') || s.includes('ssm') || s === 's4') return 'mamba';
  if (s.includes('graph-neural') || s.includes('gnn') || s.includes('gcn') || s.includes('graph-attention') || s.includes('graph-convolut')) return 'gnn';
  if (s.includes('t5') || s.includes('bart') || s.includes('text-to-text') || s.includes('encoder-decoder')) return 'encdec';
  if (s.includes('u-net') || s.includes('unet') || s.includes('segment')) return 'unet';
  if (s.includes('vae') || s.includes('variational')) return 'vae';
  if (s.includes('palm') || s.includes('lora') || s.includes('flash') || s.includes('chinchilla') || s.includes('dpo')) return 'transformer';
  if (s.includes('alexnet') || s.includes('imagenet-classification') || s.includes('vgg') || s.includes('batch-normalization') || s.includes('convolutional')) return 'cnn';
  if (s.includes('generative-adversarial') || s === 'gan' || s.includes('-gan')) return 'gan';
  if (s.includes('diffusion') || s.includes('stable-diffusion')) return 'diffusion';
  if (s.includes('clip')) return 'clip';
  if (s.includes('bahdanau') || s.includes('seq2seq') || s.includes('sequence-to-sequence')) return 'seq2seq';
  if (s.includes('lstm') || s.includes('long-short')) return 'lstm';
  if (s.includes('gru') || s.includes('gated-recurrent') || s.includes('recurrent') || s.includes('rnn')) return 'gru';
  return null;
}

/** General alias — resolve any architecture slug/name to a diagram slug. */
export const toDiagramSlug = paperToArchSlug;

/** Map a system-design slug/name to a pipeline diagram slug. */
export function systemToFlowSlug(idOrName: string): string {
  const s = idOrName.toLowerCase();
  if (s.includes('rag') || s.includes('retrieval')) return 'rag';
  if (s.includes('vector') || s.includes('search') || s.includes('embedding')) return 'vector-search';
  if (s.includes('serving') || s.includes('inference') || s.includes('llm')) return 'llm-serving';
  if (s.includes('recommend') || s.includes('feed') || s.includes('ranking')) return 'recommender';
  return 'sd-default';
}
