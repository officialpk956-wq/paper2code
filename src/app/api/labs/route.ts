import { NextResponse } from 'next/server';

const LABS_META = [
  {
    id: 'transformer',
    name: 'Transformer Lab',
    description: 'Explore attention, token length, and O(N²) complexity interactively.',
    icon: '⚡',
    params: [
      { key: 'd_model',    label: 'Model Dimension',  min: 64,  max: 1024, step: 64,  default: 512  },
      { key: 'num_heads',  label: 'Attention Heads',  min: 1,   max: 16,   step: 1,   default: 8    },
      { key: 'num_layers', label: 'Encoder Layers',   min: 1,   max: 24,   step: 1,   default: 6    },
      { key: 'seq_len',    label: 'Sequence Length',  min: 16,  max: 512,  step: 16,  default: 128  },
      { key: 'vocab_size', label: 'Vocabulary Size',  min: 1000,max: 50000,step: 1000,default: 10000},
    ],
    endpoint: '/api/labs/transformer',
  },
  {
    id: 'cnn',
    name: 'CNN Lab',
    description: 'See how depth, width, and kernel size affect FLOPs and spatial resolution.',
    icon: '🔭',
    params: [
      { key: 'base_channels',    label: 'Base Channels',     min: 8,   max: 256, step: 8,  default: 64  },
      { key: 'depth',            label: 'Network Depth',     min: 1,   max: 8,   step: 1,  default: 4   },
      { key: 'kernel_size',      label: 'Kernel Size',       min: 1,   max: 7,   step: 2,  default: 3   },
      { key: 'image_resolution', label: 'Input Resolution',  min: 32,  max: 512, step: 32, default: 224 },
    ],
    endpoint: '/api/labs/cnn',
  },
  {
    id: 'vit',
    name: 'ViT Lab',
    description: 'Trade patch granularity against token count and FLOPs in Vision Transformers.',
    icon: '👁️',
    params: [
      { key: 'image_size',  label: 'Image Size',   min: 32,  max: 512, step: 32, default: 224 },
      { key: 'patch_size',  label: 'Patch Size',   min: 4,   max: 32,  step: 4,  default: 16  },
      { key: 'hidden_dim',  label: 'Hidden Dim',   min: 64,  max: 1024,step: 64, default: 768 },
      { key: 'num_blocks',  label: 'Encoder Blocks',min: 1,  max: 24,  step: 1,  default: 12  },
    ],
    endpoint: '/api/labs/vit',
  },
  {
    id: 'diffusion',
    name: 'Diffusion Lab',
    description: 'Understand the FLOPs cost of T denoising steps in DDPM models.',
    icon: '🌊',
    params: [
      { key: 'latent_size',    label: 'Latent Spatial Size', min: 8,    max: 128, step: 8,   default: 32   },
      { key: 'channels',       label: 'Image Channels',      min: 1,    max: 4,   step: 1,   default: 3    },
      { key: 'diffusion_steps',label: 'Diffusion Steps (T)',  min: 100,  max: 2000,step: 100, default: 1000 },
    ],
    endpoint: '/api/labs/diffusion',
  },
];

export function GET() {
  return NextResponse.json({ labs: LABS_META });
}
