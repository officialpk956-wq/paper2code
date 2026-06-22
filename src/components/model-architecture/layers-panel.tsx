'use client';

import { ChevronDown, Plus, Trash2, Copy } from 'lucide-react';
import { useState } from 'react';

interface Layer {
  id: string;
  name: string;
  type: string;
  icon: string;
  params: number;
  outputShape: string;
  isExpanded?: boolean;
}

const layers: Layer[] = [
  {
    id: 'input',
    name: 'Input',
    type: 'Input',
    icon: '📥',
    params: 0,
    outputShape: '[B, 784]',
  },
  {
    id: 'embedding',
    name: 'Embedding',
    type: 'Embedding',
    icon: '🔤',
    params: 307200,
    outputShape: '[B, 512]',
  },
  {
    id: 'transformer-1',
    name: 'Transformer Block 1',
    type: 'Transformer',
    icon: '⚙️',
    params: 2097152,
    outputShape: '[B, 512]',
    isExpanded: true,
  },
  {
    id: 'transformer-2',
    name: 'Transformer Block 2',
    type: 'Transformer',
    icon: '⚙️',
    params: 2097152,
    outputShape: '[B, 512]',
  },
  {
    id: 'pool',
    name: 'Global Avg Pool',
    type: 'Pooling',
    icon: '📊',
    params: 0,
    outputShape: '[B, 512]',
  },
  {
    id: 'dense',
    name: 'Dense Layer',
    type: 'Dense',
    icon: '🧠',
    params: 262144,
    outputShape: '[B, 512]',
  },
  {
    id: 'output',
    name: 'Output Layer',
    type: 'Dense',
    icon: '📤',
    params: 5130,
    outputShape: '[B, 10]',
  },
];

interface LayersPanelProps {
  onSelectLayer?: (id: string) => void;
  selectedLayer?: string;
}

const formatParams = (params: number) => {
  if (params === 0) return '0';
  if (params >= 1000000) return `${(params / 1000000).toFixed(2)}M`;
  if (params >= 1000) return `${(params / 1000).toFixed(1)}K`;
  return params.toString();
};

export function LayersPanel({ onSelectLayer, selectedLayer }: LayersPanelProps) {
  const [expandedLayers, setExpandedLayers] = useState<string[]>(['transformer-1']);

  const toggleExpand = (id: string) => {
    setExpandedLayers((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-r border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-[--color-text-primary]">
            Model Layers
          </h3>
          <button className="p-1 rounded hover:bg-[--bg-surface] text-[--color-text-tertiary]">
            <Plus size={14} />
          </button>
        </div>
        <div className="text-xs text-[--color-text-tertiary]">
          {layers.length} layers • {formatParams(layers.reduce((sum, l) => sum + l.params, 0))} params
        </div>
      </div>

      {/* Layers List */}
      <div className="flex-1 overflow-y-auto">
        {layers.map((layer, idx) => (
          <div key={layer.id}>
            {/* Layer Item */}
            <div
              onClick={() => onSelectLayer?.(layer.id)}
              className={`group px-4 py-3 border-l-2 transition-all cursor-pointer ${
                selectedLayer === layer.id
                  ? 'bg-[--bg-surface] border-[--accent-primary]'
                  : 'border-transparent hover:bg-[--bg-surface]'
              }`}
            >
              <div className="flex items-start gap-2">
                {/* Step Number */}
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-[--accent-primary]/20 flex items-center justify-center mt-0.5">
                  <span className="text-xs font-bold text-[--accent-primary]">
                    {idx + 1}
                  </span>
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm">{layer.icon}</span>
                    <span className="text-xs font-semibold text-[--color-text-primary]">
                      {layer.name}
                    </span>
                    {layer.type !== 'Input' && (
                      <span className="px-1.5 py-0.5 rounded text-xs bg-[--accent-cyan]/20 text-[--accent-cyan] border border-[--accent-cyan]/30">
                        {formatParams(layer.params)}
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-[--color-text-tertiary] mt-1 font-mono">
                    {layer.outputShape}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  {layer.type === 'Transformer' && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleExpand(layer.id);
                      }}
                      className="p-1 hover:bg-[--bg-body] rounded"
                    >
                      <ChevronDown
                        size={14}
                        className={`text-[--color-text-tertiary] transition-transform ${
                          expandedLayers.includes(layer.id) ? 'rotate-180' : ''
                        }`}
                      />
                    </button>
                  )}
                  <button className="p-1 hover:bg-[--bg-body] rounded text-[--color-text-tertiary]">
                    <Copy size={12} />
                  </button>
                  <button className="p-1 hover:bg-red-500/20 rounded text-red-400">
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>

              {/* Expanded Details */}
              {expandedLayers.includes(layer.id) && layer.type === 'Transformer' && (
                <div className="mt-3 ml-8 space-y-2 pt-3 border-t border-[--color-border]">
                  <div className="text-xs text-[--color-text-secondary]">
                    <div className="flex justify-between mb-1">
                      <span>Multi-Head Attention</span>
                      <span className="text-[--accent-cyan]">1.3M</span>
                    </div>
                    <div className="flex justify-between mb-1">
                      <span>Feed Forward</span>
                      <span className="text-[--accent-cyan]">0.8M</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Connector */}
            {idx < layers.length - 1 && (
              <div className="pl-7 py-0.5">
                <div className="w-0.5 h-2 bg-[--color-border]" />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body]">
        <button className="w-full px-3 py-2 text-xs font-medium rounded bg-[--accent-primary] hover:opacity-90 text-white transition-opacity flex items-center justify-center gap-2">
          <Plus size={12} />
          Add Layer
        </button>
      </div>
    </div>
  );
}
