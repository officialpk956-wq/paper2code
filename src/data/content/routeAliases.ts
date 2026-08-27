/**
 * Historical content routes retained after registry consolidation.
 *
 * Physical source folders are intentionally kept as migration evidence, while
 * requests are redirected to the canonical registered route. New content must
 * link directly to the canonical target rather than adding another alias.
 */
export const ARCHITECTURE_ROUTE_ALIASES: Record<string, string> = {
  deeplabv3plus: 'deeplab-v3-2',
  diffusion: 'ddpm',
  googlenet: 'googlenet-inception-v1',
  gpt: 'gpt-1',
  inceptionv3: 'inception-v2-v3',
  lenet: 'lenet-5',
  moe: 'mixtral-8x7b',
  swin: 'swin-transformer',
  unet: 'u-net',
  vgg16: 'vggnet',
  vgg19: 'vggnet',
};

export const SYSTEM_DESIGN_ROUTE_ALIASES: Record<string, string> = {
  'advanced-rag': 'rag-systems',
  'agentic-rag': 'agent-systems',
  'basic-rag': 'rag-systems',
  'chatgpt-system-design': 'chatgpt-architecture',
  'github-copilot': 'agent-systems',
  'multi-agent': 'agent-systems',
  'netflix-recommendation': 'netflix-architecture',
  perplexity: 'rag-systems',
  'recommendation-engine': 'recommendation-systems',
  'single-agent': 'agent-systems',
  'tiktok-recommendation': 'tiktok-architecture',
  'youtube-recommendation': 'youtube-architecture',
};

export const PAPER_ROUTE_ALIASES: Record<string, string> = {
  alexnet: 'imagenet-classification-with-deep-convolutional-neural-networks-alexnet',
  'bahdanau-attention': 'neural-machine-translation-by-jointly-learning-to-align-and-translate',
  'batch-normalization': 'batch-normalization-accelerating-deep-network-training',
  bert: 'bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding',
  chinchilla: 'training-compute-optimal-large-language-models-chinchilla',
  clip: 'learning-transferable-visual-models-from-natural-language-supervision-clip',
  'deep-residual-learning': 'deep-residual-learning-for-image-recognition',
  dpo: 'direct-preference-optimization-dpo',
  'flash-attention': 'flashattention-fast-and-memory-efficient-exact-attention-with-io-awareness',
  gan: 'generative-adversarial-nets',
  gpt: 'improving-language-understanding-by-generative-pre-training-gpt-1',
  'gpt-2': 'language-models-are-unsupervised-multitask-learners-gpt-2',
  'gpt-3': 'language-models-are-few-shot-learners-gpt-3',
  instructgpt: 'training-language-models-to-follow-instructions-with-human-feedback-instructgpt',
  'latent-diffusion-models': 'high-resolution-image-synthesis-with-latent-diffusion-models-stable-diffusion',
  llama: 'llama-open-and-efficient-foundation-language-models',
  lora: 'lora-low-rank-adaptation-of-large-language-models',
  palm: 'palm-scaling-language-modeling-with-pathways',
  'segment-anything': 'segment-anything-sam',
  'stable-diffusion': 'high-resolution-image-synthesis-with-latent-diffusion-models-stable-diffusion',
  'switch-transformer': 'switch-transformers-scaling-to-trillion-parameter-models-with-simple-and-efficient-sparsity',
  vgg: 'very-deep-convolutional-networks-for-large-scale-image-recognition-vggnet',
  'vision-transformer': 'an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale-vit',
};
