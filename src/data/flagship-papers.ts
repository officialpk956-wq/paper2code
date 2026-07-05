// Static content for the 6 flagship reference papers.
// The backend's Paper table holds user-uploaded PDFs keyed by numeric id —
// these curated slugs are served entirely from this file so the flagship
// workspace pages work regardless of backend state.

export type FlagshipMeta = {
  title: string; authors: string; year: number; abstract: string; color: string;
  archSlug?: string; archName?: string;
  contributions: string[];
};

export const FLAGSHIP_META: Record<string, FlagshipMeta> = {
  'attention-is-all-you-need': {
    title: 'Attention Is All You Need',
    authors: 'Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin',
    year: 2017, color: '#A78BFA',
    archSlug: 'transformer', archName: 'Transformer',
    abstract: 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable.',
    contributions: ['Self-attention replaces recurrence entirely', 'Multi-head attention for diverse representations', 'Sinusoidal positional encodings', 'Achieves 28.4 BLEU on WMT 2014 English-to-German'],
  },
  resnet: {
    title: 'Deep Residual Learning for Image Recognition',
    authors: 'He, Zhang, Ren, Sun',
    year: 2015, color: '#60A5FA',
    archSlug: 'resnet', archName: 'ResNet',
    abstract: 'Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.',
    contributions: ['Skip connections allow gradient flow through 100+ layers', 'Batch normalisation after each conv layer', 'Bottleneck blocks with 1×1 convolutions', 'Won ILSVRC 2015 with 3.57% top-5 error'],
  },
  bert: {
    title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
    authors: 'Devlin, Chang, Lee, Toutanova',
    year: 2018, color: '#A78BFA',
    archSlug: 'bert', archName: 'BERT',
    abstract: 'We introduce BERT: Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.',
    contributions: ['Bidirectional pre-training via Masked Language Modelling', 'Next-Sentence Prediction auxiliary task', 'Fine-tuning sets new state-of-the-art on 11 NLP benchmarks', 'WordPiece tokenisation with 30,000 vocabulary'],
  },
  vit: {
    title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale',
    authors: 'Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, et al.',
    year: 2020, color: '#A78BFA',
    archSlug: 'vit', archName: 'ViT',
    abstract: 'While the Transformer architecture has become the de-facto standard for NLP tasks, its applications to computer vision remain limited. We show that reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.',
    contributions: ['Pure Transformer for image classification — no CNN components', 'Image split into 16×16 patches treated as word tokens', '[CLS] token used for classification head', 'Outperforms CNNs at scale with far less compute'],
  },
  lora: {
    title: 'LoRA: Low-Rank Adaptation of Large Language Models',
    authors: 'Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen',
    year: 2021, color: '#A3E635',
    archName: 'LoRA',
    abstract: 'As we pre-train larger models, full fine-tuning becomes less feasible. We propose LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.',
    contributions: ['ΔW = BA: rank decomposition reduces trainable params by 10,000×', 'No inference latency — adapters merged at deploy time', 'Matches full fine-tuning quality on GPT-3 tasks', 'Can target any linear layer: Q, K, V, output projection'],
  },
  'flash-attention': {
    title: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
    authors: 'Dao, Fu, Ermon, Rudra, Ré',
    year: 2022, color: '#F472B6',
    archName: 'Flash Attention',
    abstract: 'We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. The bottleneck of attention is memory IO, not FLOPs — reducing IO is the key to speedup.',
    contributions: ['Tiled computation: never materialise the full N×N matrix in HBM', 'Online softmax: normalise on-the-fly using SRAM', 'Recomputation in backward pass trades FLOPs for memory', '3× end-to-end speedup, 10× memory reduction vs standard attention'],
  },
};

type GraphNode = { id: string; label: string; x: number; y: number; color: string };
type Graph = { nodes: GraphNode[]; edges: { source: string; target: string }[] };

export const FLAGSHIP_GRAPH: Record<string, Graph> = {
  'attention-is-all-you-need': {
    nodes: [
      { id: 'attn', label: 'Attention', x: 300, y: 140, color: '#A78BFA' },
      { id: 'mha', label: 'Multi-Head', x: 150, y: 270, color: '#60A5FA' },
      { id: 'ffn', label: 'Feed-Forward', x: 450, y: 270, color: '#60A5FA' },
      { id: 'enc', label: 'Encoder', x: 150, y: 400, color: '#A78BFA' },
      { id: 'dec', label: 'Decoder', x: 450, y: 400, color: '#A78BFA' },
      { id: 'pos', label: 'Pos. Encoding', x: 300, y: 400, color: '#A3E635' },
    ],
    edges: [
      { source: 'attn', target: 'mha' }, { source: 'attn', target: 'ffn' },
      { source: 'mha', target: 'enc' }, { source: 'ffn', target: 'dec' },
      { source: 'pos', target: 'enc' }, { source: 'pos', target: 'dec' },
    ],
  },
  resnet: {
    nodes: [
      { id: 'conv', label: 'Convolution', x: 300, y: 140, color: '#60A5FA' },
      { id: 'bn', label: 'BatchNorm', x: 150, y: 270, color: '#A78BFA' },
      { id: 'relu', label: 'ReLU', x: 450, y: 270, color: '#A78BFA' },
      { id: 'skip', label: 'Skip Connection', x: 300, y: 300, color: '#A78BFA' },
      { id: 'block', label: 'Residual Block', x: 300, y: 420, color: '#A78BFA' },
    ],
    edges: [
      { source: 'conv', target: 'bn' }, { source: 'conv', target: 'relu' },
      { source: 'bn', target: 'block' }, { source: 'relu', target: 'block' },
      { source: 'skip', target: 'block' },
    ],
  },
  bert: {
    nodes: [
      { id: 'wp', label: 'WordPiece', x: 150, y: 150, color: '#A3E635' },
      { id: 'mlm', label: 'Masked LM', x: 450, y: 150, color: '#A78BFA' },
      { id: 'nsp', label: 'NSP', x: 450, y: 300, color: '#A78BFA' },
      { id: 'enc', label: 'Encoder ×12', x: 300, y: 250, color: '#60A5FA' },
      { id: 'cls', label: '[CLS] Token', x: 150, y: 300, color: '#A78BFA' },
      { id: 'ft', label: 'Fine-tuning', x: 300, y: 420, color: '#F472B6' },
    ],
    edges: [
      { source: 'wp', target: 'enc' }, { source: 'mlm', target: 'enc' },
      { source: 'nsp', target: 'cls' }, { source: 'enc', target: 'ft' },
      { source: 'cls', target: 'ft' },
    ],
  },
  vit: {
    nodes: [
      { id: 'patch', label: 'Patches 16×16', x: 150, y: 150, color: '#A78BFA' },
      { id: 'proj', label: 'Linear Proj.', x: 400, y: 150, color: '#60A5FA' },
      { id: 'cls', label: '[CLS] Token', x: 150, y: 300, color: '#A78BFA' },
      { id: 'pos', label: 'Pos. Embed', x: 400, y: 300, color: '#A3E635' },
      { id: 'enc', label: 'Encoder', x: 275, y: 420, color: '#F472B6' },
    ],
    edges: [
      { source: 'patch', target: 'proj' }, { source: 'proj', target: 'enc' },
      { source: 'cls', target: 'enc' }, { source: 'pos', target: 'enc' },
    ],
  },
  lora: {
    nodes: [
      { id: 'w', label: 'Frozen W', x: 300, y: 140, color: '#525252' },
      { id: 'a', label: 'A (down)', x: 150, y: 280, color: '#A3E635' },
      { id: 'b', label: 'B (up)', x: 450, y: 280, color: '#A3E635' },
      { id: 'delta', label: 'ΔW = BA', x: 300, y: 300, color: '#A78BFA' },
      { id: 'merge', label: 'W + BA', x: 300, y: 420, color: '#A78BFA' },
    ],
    edges: [
      { source: 'a', target: 'delta' }, { source: 'b', target: 'delta' },
      { source: 'w', target: 'merge' }, { source: 'delta', target: 'merge' },
    ],
  },
  'flash-attention': {
    nodes: [
      { id: 'hbm', label: 'HBM (slow)', x: 150, y: 150, color: '#F87171' },
      { id: 'sram', label: 'SRAM (fast)', x: 450, y: 150, color: '#A78BFA' },
      { id: 'tile', label: 'Tiling', x: 300, y: 270, color: '#A78BFA' },
      { id: 'softmax', label: 'Online Softmax', x: 150, y: 400, color: '#60A5FA' },
      { id: 'recomp', label: 'Recompute', x: 450, y: 400, color: '#A3E635' },
    ],
    edges: [
      { source: 'hbm', target: 'tile' }, { source: 'sram', target: 'tile' },
      { source: 'tile', target: 'softmax' }, { source: 'tile', target: 'recomp' },
    ],
  },
};

export const FLAGSHIP_BLUEPRINT: Record<string, { components: { name: string; description: string }[] }> = {
  'attention-is-all-you-need': {
    components: [
      { name: 'Input Embeddings + Positional Encoding', description: 'Token lookup (d_model=512) with sinusoidal position signals added.' },
      { name: 'Encoder Stack ×6', description: 'Each layer: multi-head self-attention → Add&Norm → FFN (2048) → Add&Norm.' },
      { name: 'Decoder Stack ×6', description: 'Masked self-attention, then cross-attention over encoder output, then FFN.' },
      { name: 'Output Projection + Softmax', description: 'Linear to vocabulary size, softmax over next-token distribution.' },
    ],
  },
  resnet: {
    components: [
      { name: 'Stem: 7×7 Conv + MaxPool', description: 'Downsamples the 224×224 input to 56×56 with 64 channels.' },
      { name: 'Residual Stages ×4', description: 'Stacked blocks (3/4/6/3 in ResNet-50), each with identity skip connections.' },
      { name: 'Bottleneck Block', description: '1×1 reduce → 3×3 conv → 1×1 expand, with BatchNorm and ReLU.' },
      { name: 'Global Average Pool + FC', description: 'Pools 7×7 features and classifies over 1000 ImageNet classes.' },
    ],
  },
  bert: {
    components: [
      { name: 'WordPiece Embeddings', description: '30K vocabulary; token + segment + position embeddings summed.' },
      { name: 'Bidirectional Encoder ×12', description: 'Transformer encoder layers (768 hidden, 12 heads) attending both directions.' },
      { name: 'MLM Head', description: 'Predicts randomly masked tokens (15%) from context on both sides.' },
      { name: 'NSP Head', description: 'Binary classifier on [CLS] deciding if sentence B follows sentence A.' },
    ],
  },
  vit: {
    components: [
      { name: 'Patch Embedding', description: 'Splits image into 16×16 patches; each linearly projected to d_model.' },
      { name: '[CLS] Token + Position Embeddings', description: 'Learnable class token prepended; learned 1-D position embeddings added.' },
      { name: 'Transformer Encoder ×12', description: 'Standard encoder blocks — MHA + MLP with LayerNorm (pre-norm).' },
      { name: 'MLP Classification Head', description: 'Reads the final [CLS] representation to predict the image class.' },
    ],
  },
  lora: {
    components: [
      { name: 'Frozen Base Weights W', description: 'Pre-trained weight matrices remain untouched during fine-tuning.' },
      { name: 'Low-Rank Adapters A, B', description: 'Trainable matrices with rank r ≪ d; initialised so BA = 0 at start.' },
      { name: 'Forward Path', description: 'h = Wx + BAx·(α/r) — adapter output scaled and added to the frozen path.' },
      { name: 'Merge at Deployment', description: "W' = W + BA folded in offline, so inference cost is unchanged." },
    ],
  },
  'flash-attention': {
    components: [
      { name: 'Tiled Q/K/V Loading', description: 'Blocks of Q, K, V streamed from HBM into on-chip SRAM.' },
      { name: 'Online Softmax', description: 'Running max and normaliser updated per tile — no full N×N matrix ever stored.' },
      { name: 'Output Accumulation', description: 'Partial attention outputs rescaled and accumulated across tiles.' },
      { name: 'Backward Recomputation', description: 'Attention recomputed in the backward pass instead of stored — O(N) memory.' },
    ],
  },
};

const ATTENTION_CODE = `import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        out = torch.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)`;

export const FLAGSHIP_CODE: Record<string, { language: string; code: string }> = {
  'attention-is-all-you-need': { language: 'python', code: ATTENTION_CODE },
  resnet: {
    language: 'python',
    code: `import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x                      # the skip connection
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity              # residual addition
        return self.relu(out)`,
  },
  bert: {
    language: 'python',
    code: `import torch.nn as nn

class EncoderBlock(nn.Module):
    """One BERT encoder layer: bidirectional self-attention + FFN."""
    def __init__(self, d_model=768, num_heads=12, d_ff=3072, p=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=p, batch_first=True)
        self.ln1  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=pad_mask)  # no causal mask
        x = self.ln1(x + a)
        return self.ln2(x + self.ffn(x))`,
  },
  vit: {
    language: 'python',
    code: `import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Image → sequence of patch tokens (+ CLS token)."""
    def __init__(self, img=224, patch=16, in_ch=3, d_model=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)
        n_patches = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))

    def forward(self, x):                    # (B, 3, 224, 224)
        x = self.proj(x)                     # (B, D, 14, 14)
        x = x.flatten(2).transpose(1, 2)     # (B, 196, D)
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, x], dim=1) + self.pos`,
  },
  lora: {
    language: 'python',
    code: `import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Frozen W plus trainable low-rank update BA."""
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False          # freeze pre-trained weights
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale`,
  },
  'flash-attention': {
    language: 'python',
    code: `import torch, math

def tiled_attention(Q, K, V, tile=128):
    """Simplified FlashAttention idea: process K/V in tiles,
    keep a running softmax normaliser (never form full N×N)."""
    B, T, D = Q.shape
    out = torch.zeros_like(Q)
    m = torch.full((B, T, 1), -float('inf'))   # running max
    l = torch.zeros(B, T, 1)                   # running normaliser
    for s in range(0, T, tile):
        k, v = K[:, s:s+tile], V[:, s:s+tile]
        scores = Q @ k.transpose(-2, -1) / math.sqrt(D)
        m_new = torch.maximum(m, scores.max(-1, keepdim=True).values)
        p = torch.exp(scores - m_new)
        correction = torch.exp(m - m_new)
        l = l * correction + p.sum(-1, keepdim=True)
        out = out * correction + p @ v
        m = m_new
    return out / l`,
  },
};
