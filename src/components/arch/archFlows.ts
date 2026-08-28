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
  transformer: [
    { label: 'Tokens', sub: 'sequence', accent: '#A5B4FC' },
    { label: 'Embedding', sub: '+ position', accent: '#60A5FA' },
    { label: 'Self-Attn', sub: 'multi-head', accent: '#A78BFA' },
    { label: 'Add & Norm', accent: '#A78BFA' },
    { label: 'Feed Forward', sub: 'MLP', accent: '#F472B6' },
    { label: 'Add & Norm', accent: '#F472B6' },
    { label: 'Output', sub: 'contextual', accent: '#34D399' },
  ],
  resnet: [
    { label: 'Image', sub: 'H×W×3', accent: '#A5B4FC' },
    { label: 'Stem', sub: 'conv + pool', accent: '#60A5FA' },
    { label: 'Residual', sub: 'stage 1', accent: '#A78BFA' },
    { label: 'Residual', sub: 'stages 2–4', accent: '#A78BFA' },
    { label: 'Skip Paths', sub: 'identity', accent: '#FB923C' },
    { label: 'Global Pool', accent: '#F472B6' },
    { label: 'Classes', sub: 'softmax', accent: '#34D399' },
  ],
  bert: [
    { label: 'Tokens', sub: '[CLS] … [SEP]', accent: '#A5B4FC' },
    { label: 'Embeddings', sub: 'token + pos + seg', accent: '#60A5FA' },
    { label: 'Bi-Attn', sub: 'encoder ×L', accent: '#A78BFA' },
    { label: 'Feed Forward', sub: '+ residual', accent: '#A78BFA' },
    { label: 'Context', sub: 'all tokens', accent: '#F472B6' },
    { label: 'Task Head', sub: 'MLM / classify', accent: '#FB923C' },
    { label: 'Prediction', accent: '#34D399' },
  ],
  vit: [
    { label: 'Image', sub: 'H×W×3', accent: '#A5B4FC' },
    { label: 'Patches', sub: 'P×P', accent: '#60A5FA' },
    { label: 'Patch Embed', sub: '+ position', accent: '#60A5FA' },
    { label: 'Transformer', sub: 'encoder ×L', accent: '#A78BFA' },
    { label: 'CLS Token', sub: 'global feature', accent: '#F472B6' },
    { label: 'MLP Head', accent: '#FB923C' },
    { label: 'Classes', accent: '#34D399' },
  ],
  llama: [
    { label: 'Tokens', accent: '#A5B4FC' },
    { label: 'Embedding', accent: '#60A5FA' },
    { label: 'RMSNorm', accent: '#60A5FA' },
    { label: 'RoPE Attn', sub: 'causal', accent: '#A78BFA' },
    { label: 'SwiGLU', sub: 'feed forward', accent: '#F472B6' },
    { label: 'LM Head', accent: '#FB923C' },
    { label: 'Next Token', accent: '#34D399' },
  ],
  rlhf: [
    { label: 'Prompt', accent: '#A5B4FC' },
    { label: 'Policy LLM', sub: 'response', accent: '#60A5FA' },
    { label: 'Human Prefs', sub: 'rank pairs', accent: '#FB923C' },
    { label: 'Reward Model', sub: 'score', accent: '#A78BFA' },
    { label: 'PPO Update', sub: '+ KL penalty', accent: '#F472B6' },
    { label: 'Aligned LLM', accent: '#34D399' },
  ],
  gpt: [
    { label: 'Tokens', sub: 'prompt', accent: '#A5B4FC' },
    { label: 'Embedding', sub: '+ position', accent: '#60A5FA' },
    { label: 'Causal Attn', sub: 'decoder ×L', accent: '#A78BFA' },
    { label: 'Feed Forward', sub: '+ residual', accent: '#F472B6' },
    { label: 'LM Head', sub: 'vocab logits', accent: '#FB923C' },
    { label: 'Next Token', sub: 'autoregressive', accent: '#34D399' },
  ],
  moe: [
    { label: 'Tokens', accent: '#A5B4FC' },
    { label: 'Router', sub: 'top-k', accent: '#60A5FA' },
    { label: 'Experts', sub: 'sparse MLPs', accent: '#A78BFA' },
    { label: 'Weighted Mix', sub: 'gate scores', accent: '#F472B6' },
    { label: 'Residual', sub: '+ norm', accent: '#FB923C' },
    { label: 'Output', accent: '#34D399' },
  ],
  mamba: [
    { label: 'Tokens', accent: '#A5B4FC' },
    { label: 'Projection', sub: 'expand', accent: '#60A5FA' },
    { label: 'Conv1D', sub: 'local context', accent: '#60A5FA' },
    { label: 'Selective SSM', sub: 'state scan', accent: '#A78BFA' },
    { label: 'Gate', sub: 'input-aware', accent: '#F472B6' },
    { label: 'Projection', sub: '+ residual', accent: '#FB923C' },
    { label: 'Output', accent: '#34D399' },
  ],
  gnn: [
    { label: 'Graph', sub: 'nodes + edges', accent: '#A5B4FC' },
    { label: 'Node Embed', accent: '#60A5FA' },
    { label: 'Messages', sub: 'from neighbors', accent: '#A78BFA' },
    { label: 'Aggregate', sub: 'sum / attn', accent: '#A78BFA' },
    { label: 'Node Update', sub: 'MLP', accent: '#F472B6' },
    { label: 'Readout', sub: 'node / graph', accent: '#FB923C' },
    { label: 'Prediction', accent: '#34D399' },
  ],
  encdec: [
    { label: 'Input Tokens', accent: '#A5B4FC' },
    { label: 'Encoder', sub: 'bidirectional', accent: '#60A5FA' },
    { label: 'Memory', sub: 'context states', accent: '#A78BFA' },
    { label: 'Cross-Attn', sub: 'condition', accent: '#A78BFA' },
    { label: 'Decoder', sub: 'autoregressive', accent: '#F472B6' },
    { label: 'LM Head', accent: '#FB923C' },
    { label: 'Output Tokens', accent: '#34D399' },
  ],
  vae: [
    { label: 'Input', sub: 'x', accent: '#A5B4FC' },
    { label: 'Encoder', sub: 'q(z|x)', accent: '#60A5FA' },
    { label: 'μ and σ', sub: 'distribution', accent: '#A78BFA' },
    { label: 'Sample z', sub: 'reparameterize', accent: '#FB923C' },
    { label: 'Decoder', sub: 'p(x|z)', accent: '#F472B6' },
    { label: 'Reconstruction', accent: '#34D399' },
  ],
  unet: [
    { label: 'Image', accent: '#A5B4FC' },
    { label: 'Down Block', sub: 'encode', accent: '#60A5FA' },
    { label: 'Down Block', sub: 'encode', accent: '#60A5FA' },
    { label: 'Bottleneck', accent: '#A78BFA' },
    { label: 'Up Block', sub: '+ skip', accent: '#F472B6' },
    { label: 'Up Block', sub: '+ skip', accent: '#F472B6' },
    { label: 'Pixel Mask', accent: '#34D399' },
  ],
  gru: [
    { label: 'Sequence', sub: 'x₁…xₜ', accent: '#A5B4FC' },
    { label: 'Embedding', accent: '#60A5FA' },
    { label: 'Reset Gate', accent: '#A78BFA' },
    { label: 'Update Gate', accent: '#A78BFA' },
    { label: 'Hidden State', sub: 'hₜ', accent: '#F472B6' },
    { label: 'Dense Head', accent: '#FB923C' },
    { label: 'Output', accent: '#34D399' },
  ],
  'object-detection': [
    { label: 'Image', sub: 'H×W×3', accent: '#A5B4FC' },
    { label: 'Backbone', sub: 'CNN / ViT', accent: '#60A5FA' },
    { label: 'Neck', sub: 'FPN', accent: '#A78BFA' },
    { label: 'Detection Head', sub: 'class + box', accent: '#F472B6' },
    { label: 'NMS', sub: 'deduplicate', accent: '#FB923C' },
    { label: 'Boxes', sub: 'labels + scores', accent: '#34D399' },
  ],
  'rl-policy': [
    { label: 'State', sub: 'observation', accent: '#A5B4FC' },
    { label: 'Encoder', sub: 'features', accent: '#60A5FA' },
    { label: 'Policy', sub: 'action scores', accent: '#A78BFA' },
    { label: 'Environment', sub: 'transition', accent: '#F472B6' },
    { label: 'Reward', sub: 'learning signal', accent: '#FB923C' },
    { label: 'Update', sub: 'policy / value', accent: '#A78BFA' },
    { label: 'Action', accent: '#34D399' },
  ],
  vlm: [
    { label: 'Image', accent: '#A5B4FC' },
    { label: 'Vision Encoder', sub: 'visual tokens', accent: '#60A5FA' },
    { label: 'Projector', sub: 'align spaces', accent: '#A78BFA' },
    { label: 'Text Tokens', sub: 'prompt', accent: '#FB923C' },
    { label: 'Language Model', sub: 'cross-modal', accent: '#F472B6' },
    { label: 'Response', sub: 'text / action', accent: '#34D399' },
  ],
  embedding: [
    { label: 'Token', sub: 'word / subword', accent: '#A5B4FC' },
    { label: 'Context', sub: 'window / corpus', accent: '#60A5FA' },
    { label: 'Objective', sub: 'predict / co-occur', accent: '#A78BFA' },
    { label: 'Embedding Table', sub: 'dense vectors', accent: '#F472B6' },
    { label: 'Vector', sub: 'semantic space', accent: '#34D399' },
  ],
  speech: [
    { label: 'Audio', sub: 'waveform', accent: '#A5B4FC' },
    { label: 'Log-Mel', sub: 'spectrogram', accent: '#60A5FA' },
    { label: 'Audio Encoder', sub: 'Transformer', accent: '#A78BFA' },
    { label: 'Text Decoder', sub: 'autoregressive', accent: '#F472B6' },
    { label: 'Tokens', sub: 'transcript', accent: '#34D399' },
  ],
  agent: [
    { label: 'Goal', sub: 'user request', accent: '#A5B4FC' },
    { label: 'Planner', sub: 'reason + decide', accent: '#60A5FA' },
    { label: 'Tool Router', sub: 'select action', accent: '#A78BFA' },
    { label: 'Tool / Env', sub: 'execute', accent: '#F472B6' },
    { label: 'Observation', sub: 'result', accent: '#FB923C' },
    { label: 'Memory', sub: 'update context', accent: '#A78BFA' },
    { label: 'Response', accent: '#34D399' },
  ],
  'training-system': [
    { label: 'Dataset', sub: 'sharded batches', accent: '#A5B4FC' },
    { label: 'Workers', sub: 'CPU / GPU', accent: '#60A5FA' },
    { label: 'Forward', sub: 'model shards', accent: '#A78BFA' },
    { label: 'Backward', sub: 'gradients', accent: '#F472B6' },
    { label: 'Collective', sub: 'sync / reduce', accent: '#FB923C' },
    { label: 'Optimizer', sub: 'update weights', accent: '#A78BFA' },
    { label: 'Checkpoint', accent: '#34D399' },
  ],
  '3d-vision': [
    { label: 'Images', sub: 'multi-view', accent: '#A5B4FC' },
    { label: 'Camera Rays', sub: 'origin + direction', accent: '#60A5FA' },
    { label: 'Scene Model', sub: 'field / Gaussians', accent: '#A78BFA' },
    { label: 'Renderer', sub: 'project + blend', accent: '#F472B6' },
    { label: 'Image Loss', sub: 'optimize scene', accent: '#FB923C' },
    { label: 'Novel View', accent: '#34D399' },
  ],
  'paper-method': [
    { label: 'Problem', sub: 'research question', accent: '#A5B4FC' },
    { label: 'Data', sub: 'inputs', accent: '#60A5FA' },
    { label: 'Method', sub: 'core idea', accent: '#A78BFA' },
    { label: 'Objective', sub: 'learning signal', accent: '#F472B6' },
    { label: 'Training', sub: 'optimization', accent: '#FB923C' },
    { label: 'Evaluation', sub: 'metrics', accent: '#A78BFA' },
    { label: 'Result', sub: 'findings', accent: '#34D399' },
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
  const is = (...slugs: string[]) => slugs.includes(s);

  // Exact architecture families come first so hybrid names are not captured
  // by broader paper-title keyword rules below.
  if (is('r-cnn', 'fast-r-cnn', 'faster-r-cnn', 'mask-r-cnn', 'fpn', 'retinanet', 'ssd', 'yolo-v1', 'yolo-v2', 'yolo-v3', 'yolo-v4', 'yolo-v5', 'yolov8', 'detr', 'deformable-detr', 'rt-detr', 'fcos', 'centernet', 'cornernet', 'efficientdet')) return 'object-detection';
  if (is('dqn', 'double-dqn', 'dueling-dqn', 'rainbow', 'ddpg', 'td3', 'sac', 'ppo', 'trpo', 'a3c', 'dreamer', 'muzero', 'alphago', 'alphazero', 'decision-transformer', 'gato')) return 'rl-policy';
  if (is('blip', 'blip-2', 'coca', 'flamingo', 'cogvlm', 'llava', 'llava-1-5', 'minigpt-4', 'instructblip', 'gpt-4v', 'gpt-4o', 'gemini', 'gemini-1-0', 'gemini-1-5', 'seamlessm4t')) return 'vlm';
  if (is('word2vec', 'glove', 'fasttext')) return 'embedding';
  if (s === 'whisper') return 'speech';
  if (is('gcn', 'gat', 'gatv2', 'gin', 'graphsage', 'chebnet', 'mpnn', 'graphgps', 'graphormer', 'graph-transformer', 'diffpool', 'pinsage', 'r-gcn', 'deepwalk', 'node2vec', 'transe')) return 'gnn';
  if (is('vit', 'deit', 'beit', 'swin', 'swin-transformer', 'swin-v2', 'cvt', 'pvt', 'maxvit', 'mae', 'dino', 'dit')) return 'vit';
  if (is('unet', 'u-net', 'u-net-2', 'attention-u-net', 'fcn', 'segnet', 'deeplab-v1', 'deeplab-v2', 'deeplab-v3', 'deeplab-v3-2', 'deeplabv3plus', 'pspnet', 'segformer', 'sam', 'sam-2', 'mask2former', 'panoptic-fpn')) return 'unet';
  if (is('resnet', 'resnext', 'wideresnet')) return 'resnet';
  if (is('bert', 'albert', 'roberta', 'distilbert', 'deberta', 'electra', 'xlnet', 'xlm', 'xlm-roberta', 'elmo')) return 'bert';
  if (is('gpt', 'gpt-1', 'gpt-2', 'gpt-3', 'gpt-4', 'bloom', 'opt', 'gopher', 'codex', 'falcon', 'mistral-7b', 'phi-2', 'qwen', 'palm', 'palm-2', 'chinchilla', 'deepseek-v2', 'rwkv', 'transformer-xl')) return 'gpt';
  if (is('llama', 'llama-2', 'llama-3')) return 'llama';
  if (is('moe', 'mixtral-8x7b')) return 'moe';
  if (is('mamba', 's4')) return 'mamba';
  if (is('ae')) return 'autoencoder';
  if (is('vae')) return 'vae';
  if (is('seq2seq', 'attention-seq2seq')) return 'seq2seq';
  if (is('bart', 't5')) return 'encdec';
  if (s === 'lstm') return 'lstm';
  if (is('gru', 'rnn', 'bi-directional-rnn')) return 'gru';
  if (is('clip', 'align', 'imagebind')) return 'clip';
  if (is('ncf', 'deepfm', 'dlrm', 'din', 'dien', 'wide-and-deep', 'two-tower-model', 'sasrec', 'bert4rec', 'autoint', 'matrix-factorization')) return 'recommender';
  if (is('gan', 'dcgan', 'wgan', 'wgan-gp', 'conditional-gan', 'cyclegan', 'pix2pix', 'progan', 'stylegan', 'stylegan2', 'stylegan3', 'biggan', 'gaugan-spade', 'vqgan')) return 'gan';
  if (is('diffusion', 'ddpm', 'ddim', 'score-sde', 'ldm', 'stable-diffusion', 'stable-diffusion-3', 'sdxl', 'imagen', 'dall-e', 'dall-e-2', 'dall-e-3', 'flux', 'consistency-models', 'controlnet', 'svd')) return 'diffusion';
  if (is('alexnet', 'lenet', 'lenet-5', 'zfnet', 'vgg16', 'vgg19', 'vggnet', 'googlenet', 'googlenet-inception-v1', 'inception-v2-v3', 'inceptionv3', 'inception-v4-inception-resnet', 'xception', 'densenet', 'senet', 'squeezenet', 'shufflenet', 'shufflenet-v2', 'mobilenet-v1', 'mobilenet-v2', 'mobilenet-v3', 'efficientnet', 'efficientnetv2', 'regnet', 'nasnet', 'convnext', 'convnext-v2', 'darts', 'dcn', 'dcn-v2', 'wavenet', 'tcn')) return 'cnn';

  // Paper-title routing. These shared families keep standalone research pages
  // diagrammed without introducing bespoke data for individual papers.
  if (s.includes('retrieval') || s.includes('rag-') || s.includes('-rag') || s.includes('passage') || s.includes('nearest-neighbor') || s.includes('hnsw') || s.includes('faiss')) return 'rag';
  if (s.includes('agent') || s.includes('toolformer') || s.includes('toolbench') || s.includes('voyager') || s.includes('swe-bench') || s.includes('gorilla') || s.includes('codeact') || s.includes('tree-of-thoughts') || s.includes('chain-of-thought') || s.includes('self-consistency') || s.includes('react-synergizing') || s.includes('lats-language')) return 'agent';
  if (s.includes('distributed') || s.includes('parallelism') || s.includes('deepspeed') || s.includes('megatron') || s.includes('gpipe') || s.includes('pipedream') || s.includes('fully-sharded') || s.includes('zero-memory') || s.includes('tensorflow') || s.includes('pytorch') || s.includes('jax-') || s.includes('onnx') || s.includes('triton') || s.includes('cudnn')) return 'training-system';
  if (s.includes('serving') || s.includes('inference') || s.includes('pagedattention') || s.includes('tensorrt') || s.includes('orca-')) return 'llm-serving';
  if (s.includes('radiance-field') || s.includes('nerf') || s.includes('gaussian-splatting')) return '3d-vision';
  if (s.includes('object-detection') || s.includes('r-cnn') || s.includes('retinanet') || s.includes('you-only-look-once')) return 'object-detection';
  if (s.includes('reinforcement-learning') || s.includes('actor-critic') || s.includes('soft-actor-critic') || s.includes('alphago') || s.includes('alphazero') || s.includes('muzero') || s.includes('atari')) return 'rl-policy';
  if (s.includes('visual-language') || s.includes('language-image') || s.includes('multimodal') || s.includes('blip') || s.includes('flamingo') || s.includes('imagebind') || s.includes('llava') || s.includes('gemini')) return 'vlm';
  if (s.includes('contrastive-captioner') || s.includes('cogvlm')) return 'vlm';
  if (s.includes('word2vec') || s.includes('glove') || s.includes('word-representation') || s.includes('sentence-embedding')) return 'embedding';
  if (s.includes('speech-recognition') || s.includes('whisper')) return 'speech';
  if (s.includes('consistency-model') || s.includes('flow-matching') || s.includes('score-based-generative')) return 'diffusion';
  if (s.includes('dinov2') || s.includes('masked-autoencoder')) return 'vit';
  if (s.includes('dream-to-control') || s.includes('hindsight-experience')) return 'rl-policy';
  if (s.includes('electra') || s.includes('xlnet') || s.includes('longformer')) return 'bert';
  if (s.includes('inception-v1') || s.includes('mobilenet') || s.includes('squeeze-and-excitation')) return 'cnn';
  if (s.includes('neural-machine-translation')) return 'seq2seq';
  if (s.includes('ring-attention') || s.includes('roformer')) return 'transformer';
  if (s.includes('constitutional-ai') || s.includes('self-instruct') || s.includes('openai-o1')) return 'rlhf';
  if (s.includes('internet-augmented')) return 'rag';
  if (s.includes('scaling-laws') || s.includes('emerging-abilities')) return 'gpt';
  if (s.includes('adam-a-method') || s.includes('deep-double-descent') || s.includes('deep-learning-nature') || s.includes('dropout-a-simple-way') || s.includes('knowledge-distillation') || s.includes('layer-normalization') || s.includes('back-propagating-errors') || s.includes('lottery-ticket')) return 'paper-method';

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
