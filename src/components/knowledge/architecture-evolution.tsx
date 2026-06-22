'use client';

import { useState } from 'react';
import { ChevronDown } from 'lucide-react';

interface ArchNode {
  id: string;
  name: string;
  year: number;
  description: string;
  keyContribution: string;
  children?: string[];
}

const transformerFamily: ArchNode[] = [
  {
    id: 'seq2seq',
    name: 'Sequence-to-Sequence (2014)',
    year: 2014,
    description: 'Encoder-decoder with attention',
    keyContribution: 'Introduced attention mechanism',
    children: ['attention-is-all'],
  },
  {
    id: 'attention-is-all',
    name: 'Transformer (2017)',
    year: 2017,
    description: 'Pure attention-based architecture',
    keyContribution: 'Eliminated recurrence, enabled parallelization',
    children: ['bert', 'gpt', 'roberta'],
  },
  {
    id: 'bert',
    name: 'BERT (2018)',
    year: 2018,
    description: 'Bidirectional encoder',
    keyContribution: 'Masked language modeling for pre-training',
    children: ['roberta', 'albert'],
  },
  {
    id: 'gpt',
    name: 'GPT (2018)',
    year: 2018,
    description: 'Generative pre-trained transformer',
    keyContribution: 'Demonstrated scale benefits in language models',
    children: ['gpt2', 'gpt3'],
  },
  {
    id: 'roberta',
    name: 'RoBERTa (2019)',
    year: 2019,
    description: 'Improved BERT pre-training',
    keyContribution: 'Better training procedures',
    children: [],
  },
  {
    id: 'albert',
    name: 'ALBERT (2019)',
    year: 2019,
    description: 'Parameter-efficient BERT',
    keyContribution: 'Reduced model size',
    children: [],
  },
  {
    id: 'gpt2',
    name: 'GPT-2 (2019)',
    year: 2019,
    description: 'Scaled GPT with more data',
    keyContribution: 'Zero-shot learning capabilities',
    children: ['gpt3'],
  },
  {
    id: 'gpt3',
    name: 'GPT-3 (2020)',
    year: 2020,
    description: '175B parameter language model',
    keyContribution: 'Few-shot learning, in-context learning',
    children: [],
  },
];

const cnnFamily: ArchNode[] = [
  {
    id: 'lenet',
    name: 'LeNet (1998)',
    year: 1998,
    description: 'Early CNN for digit recognition',
    keyContribution: 'Pioneered convolutional networks',
    children: ['alexnet'],
  },
  {
    id: 'alexnet',
    name: 'AlexNet (2012)',
    year: 2012,
    description: 'Deep CNN with ReLU',
    keyContribution: 'Deep learning revolution with GPUs',
    children: ['vgg', 'googlenet'],
  },
  {
    id: 'vgg',
    name: 'VGG (2014)',
    year: 2014,
    description: '19-layer uniform architecture',
    keyContribution: 'Demonstrated depth importance',
    children: ['resnet'],
  },
  {
    id: 'googlenet',
    name: 'GoogLeNet (2014)',
    year: 2014,
    description: 'Inception modules',
    keyContribution: 'Multi-scale feature extraction',
    children: ['inception'],
  },
  {
    id: 'resnet',
    name: 'ResNet (2015)',
    year: 2015,
    description: 'Residual connections',
    keyContribution: 'Enabled very deep networks (152+ layers)',
    children: ['inception', 'densenet'],
  },
  {
    id: 'inception',
    name: 'Inception-v4 (2016)',
    year: 2016,
    description: 'Inception + residual connections',
    keyContribution: 'Improved Inception modules',
    children: [],
  },
  {
    id: 'densenet',
    name: 'DenseNet (2016)',
    year: 2016,
    description: 'Dense connections between layers',
    keyContribution: 'Improved gradient flow',
    children: [],
  },
];

const generativeFamily: ArchNode[] = [
  {
    id: 'vae',
    name: 'VAE (2013)',
    year: 2013,
    description: 'Variational autoencoder',
    keyContribution: 'Probabilistic generative model',
    children: ['diffusion'],
  },
  {
    id: 'gan',
    name: 'GAN (2014)',
    year: 2014,
    description: 'Generative adversarial network',
    keyContribution: 'Adversarial training framework',
    children: ['dcgan', 'stylegan'],
  },
  {
    id: 'dcgan',
    name: 'DCGAN (2015)',
    year: 2015,
    description: 'Deep convolutional GAN',
    keyContribution: 'Stable CNN-based GAN',
    children: ['stylegan'],
  },
  {
    id: 'stylegan',
    name: 'StyleGAN (2018)',
    year: 2018,
    description: 'Style-based generator',
    keyContribution: 'Fine-grained control over generation',
    children: [],
  },
  {
    id: 'diffusion',
    name: 'Diffusion Models (2020)',
    year: 2020,
    description: 'Denoising probabilistic models',
    keyContribution: 'High-quality image generation',
    children: ['ddpm', 'dalle2'],
  },
  {
    id: 'ddpm',
    name: 'DDPM (2020)',
    year: 2020,
    description: 'Denoising diffusion model',
    keyContribution: 'Theoretical foundation',
    children: ['dalle2'],
  },
  {
    id: 'dalle2',
    name: 'DALL-E 2 (2022)',
    year: 2022,
    description: 'Text-to-image diffusion',
    keyContribution: 'Bridged language and vision',
    children: [],
  },
];

interface ArchitectureEvolutionProps {
  family?: 'transformer' | 'cnn' | 'generative';
}

export function ArchitectureEvolution({ family = 'transformer' }: ArchitectureEvolutionProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  const familyData = {
    transformer: transformerFamily,
    cnn: cnnFamily,
    generative: generativeFamily,
  };

  const data = familyData[family];

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const rootNodes = data.filter((node) => !data.some((n) => n.children?.includes(node.id)));

  const getChildCount = (nodeId: string) => {
    const node = data.find((n) => n.id === nodeId);
    return node?.children?.length ?? 0;
  };

  const renderNode = (nodeId: string, depth: number = 0) => {
    const node = data.find((n) => n.id === nodeId);
    if (!node) return null;

    const isExpanded = expandedIds.has(nodeId);
    const childCount = getChildCount(nodeId);
    const isSelected = selectedId === nodeId;

    return (
      <div key={nodeId} className={`ml-${depth * 4}`}>
        <div
          onClick={() => {
            setSelectedId(nodeId);
            if (childCount > 0) {
              toggleExpand(nodeId);
            }
          }}
          className={`p-3 rounded-lg border-2 cursor-pointer transition-all mb-2 ${
            isSelected
              ? 'border-[--accent-primary] bg-[--accent-primary]/10'
              : 'border-[--color-border] bg-[--bg-body] hover:border-[--accent-primary]/50'
          }`}
        >
          <div className="flex items-start gap-2">
            {childCount > 0 && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  toggleExpand(nodeId);
                }}
                aria-label={isExpanded ? 'Collapse' : 'Expand'}
                className="p-0.5 rounded hover:bg-[--bg-surface] flex-shrink-0"
              >
                <ChevronDown
                  size={14}
                  className={`transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                />
              </button>
            )}
            {childCount === 0 && <div className="w-5" />}

            <div className="flex-1 min-w-0">
              <h4 className="text-sm font-semibold text-[--color-text-primary]">
                {node.name}
              </h4>
              <p className="text-xs text-[--color-text-secondary] mt-1">
                {node.description}
              </p>
              {isSelected && (
                <div className="mt-2 pt-2 border-t border-[--color-border]">
                  <p className="text-xs text-[--color-text-tertiary]">
                    <span className="font-medium">Key:</span> {node.keyContribution}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Children */}
        {isExpanded && node.children && (
          <div className="ml-4 border-l-2 border-[--color-border] pl-2">
            {node.children.map((childId) => renderNode(childId, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  const rootNode = rootNodes[0];

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border border-[--color-border] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-body]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-1">
          {family === 'transformer'
            ? 'Transformer Family Tree'
            : family === 'cnn'
              ? 'CNN Family Tree'
              : 'Generative Models Timeline'}
        </h3>
        <p className="text-xs text-[--color-text-secondary]">
          Evolution of architecture from inception to modern variants
        </p>
      </div>

      {/* Timeline */}
      <div className="flex-1 overflow-y-auto p-6">
        {rootNode && renderNode(rootNode.id)}
      </div>

      {/* Info Panel */}
      {selectedId && data.find((n) => n.id === selectedId) && (
        <div className="border-t border-[--color-border] bg-[--bg-body] p-4">
          {data.find((n) => n.id === selectedId) && (
            <>
              <h4 className="text-sm font-semibold text-[--color-text-primary]">
                {data.find((n) => n.id === selectedId)!.name}
              </h4>
              <p className="text-xs text-[--color-text-secondary] mt-1">
                {data.find((n) => n.id === selectedId)!.description}
              </p>
              <button className="mt-2 w-full px-3 py-1.5 rounded text-xs font-medium bg-[--accent-primary] hover:opacity-90 text-white transition-opacity">
                Learn More
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
