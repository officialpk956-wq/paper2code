'use client';

import { BarChart3, Zap, HardDrive, Activity } from 'lucide-react';
import { useState } from 'react';

interface ModelStatsProps {
  selectedLayer?: string;
}

export function ModelStats({ selectedLayer = 'transformer-1' }: ModelStatsProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'computation' | 'memory'>('overview');

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-l border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-3">
          Model Statistics
        </h3>
        <div className="flex gap-1 bg-[--bg-body] rounded p-1">
          {(['overview', 'computation', 'memory'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex-1 px-2 py-1.5 rounded text-xs font-medium transition-colors ${
                activeTab === tab
                  ? 'bg-[--accent-primary] text-white'
                  : 'text-[--color-text-secondary] hover:text-[--color-text-primary]'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {activeTab === 'overview' && (
          <>
            {/* Total Parameters */}
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 size={14} className="text-[--accent-primary]" />
                <div className="text-xs font-semibold text-[--color-text-primary]">
                  Total Parameters
                </div>
              </div>
              <div className="text-2xl font-bold text-[--accent-primary] mb-1">
                4.77M
              </div>
              <div className="text-xs text-[--color-text-tertiary]">
                Trainable: 4.77M (100%)
              </div>
            </div>

            {/* Parameter Distribution */}
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Parameter Distribution
              </div>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between mb-1 text-xs">
                    <span className="text-[--color-text-secondary]">Transformer Blocks</span>
                    <span className="text-[--accent-primary]">4.19M (87.8%)</span>
                  </div>
                  <div className="h-2 bg-[--bg-surface] rounded-full overflow-hidden">
                    <div className="h-full w-[87.8%] bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan]" />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1 text-xs">
                    <span className="text-[--color-text-secondary]">Embedding</span>
                    <span className="text-[--accent-cyan]">307K (6.4%)</span>
                  </div>
                  <div className="h-2 bg-[--bg-surface] rounded-full overflow-hidden">
                    <div className="h-full w-[6.4%] bg-[--accent-cyan]" />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1 text-xs">
                    <span className="text-[--color-text-secondary]">Output Layer</span>
                    <span className="text-green-400">267K (5.6%)</span>
                  </div>
                  <div className="h-2 bg-[--bg-surface] rounded-full overflow-hidden">
                    <div className="h-full w-[5.6%] bg-green-400" />
                  </div>
                </div>
              </div>
            </div>

            {/* Layer Details */}
            <div className="bg-[--accent-primary]/10 rounded-lg p-3 border border-[--accent-primary]/20">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Selected: {selectedLayer}
              </div>
              <div className="space-y-1 text-xs text-[--color-text-secondary]">
                <div className="flex justify-between">
                  <span>Parameters:</span>
                  <span className="text-[--accent-primary]">2.1M</span>
                </div>
                <div className="flex justify-between">
                  <span>Output Shape:</span>
                  <span className="text-[--accent-cyan]">[B, 512]</span>
                </div>
                <div className="flex justify-between">
                  <span>Type:</span>
                  <span>Transformer Block</span>
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'computation' && (
          <>
            {/* FLOPs */}
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="flex items-center gap-2 mb-2">
                <Zap size={14} className="text-yellow-400" />
                <div className="text-xs font-semibold text-[--color-text-primary]">
                  Total FLOPs
                </div>
              </div>
              <div className="text-2xl font-bold text-yellow-400 mb-1">
                8.2B
              </div>
              <div className="text-xs text-[--color-text-tertiary]">
                Per forward pass (batch size 1)
              </div>
            </div>

            {/* Computation Breakdown */}
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                FLOPs Breakdown
              </div>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between mb-1 text-xs">
                    <span className="text-[--color-text-secondary]">Attention (O(n²))</span>
                    <span className="text-yellow-400">4.2B</span>
                  </div>
                  <div className="h-2 bg-[--bg-surface] rounded-full overflow-hidden">
                    <div className="h-full w-[51%] bg-yellow-400" />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1 text-xs">
                    <span className="text-[--color-text-secondary]">Feed Forward</span>
                    <span className="text-yellow-400">3.1B</span>
                  </div>
                  <div className="h-2 bg-[--bg-surface] rounded-full overflow-hidden">
                    <div className="h-full w-[38%] bg-yellow-500" />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1 text-xs">
                    <span className="text-[--color-text-secondary]">Embedding/Output</span>
                    <span className="text-orange-400">0.9B</span>
                  </div>
                  <div className="h-2 bg-[--bg-surface] rounded-full overflow-hidden">
                    <div className="h-full w-[11%] bg-orange-400" />
                  </div>
                </div>
              </div>
            </div>

            {/* Latency Estimate */}
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Latency Estimate
              </div>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-[--color-text-secondary]">Single Batch (A100)</span>
                  <span className="text-[--accent-primary]">45ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[--color-text-secondary]">Throughput (GPU)</span>
                  <span className="text-[--accent-cyan]">22 samples/s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[--color-text-secondary]">CPU Inference</span>
                  <span className="text-orange-400">~2.5s</span>
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'memory' && (
          <>
            {/* Memory Usage */}
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="flex items-center gap-2 mb-2">
                <HardDrive size={14} className="text-[--accent-cyan]" />
                <div className="text-xs font-semibold text-[--color-text-primary]">
                  Memory Usage
                </div>
              </div>
              <div className="text-2xl font-bold text-[--accent-cyan] mb-1">
                18.9 MB
              </div>
              <div className="text-xs text-[--color-text-tertiary]">
                Model weights (FP32)
              </div>
            </div>

            {/* Memory Breakdown */}
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Memory by Component
              </div>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between mb-1 text-xs">
                    <span className="text-[--color-text-secondary]">FP32 Weights</span>
                    <span className="text-[--accent-cyan]">18.9 MB</span>
                  </div>
                  <div className="h-2 bg-[--bg-surface] rounded-full overflow-hidden">
                    <div className="h-full w-[100%] bg-[--accent-cyan]" />
                  </div>
                </div>
                <div className="flex justify-between text-xs pt-2 border-t border-[--color-border]">
                  <span className="text-[--color-text-secondary]">FP16 Weights (half precision)</span>
                  <span className="text-green-400">9.4 MB (-50%)</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-[--color-text-secondary]">INT8 Quantized</span>
                  <span className="text-green-400">4.7 MB (-75%)</span>
                </div>
              </div>
            </div>

            {/* Peak Memory */}
            <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
              <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
                Peak Memory (Inference)
              </div>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-[--color-text-secondary]">Batch Size 1</span>
                  <span className="text-[--accent-primary]">~50 MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[--color-text-secondary]">Batch Size 32</span>
                  <span className="text-[--accent-primary]">~1.2 GB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[--color-text-secondary]">Batch Size 128</span>
                  <span className="text-orange-400">~4.8 GB</span>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body]">
        <button className="w-full px-3 py-2 text-xs font-medium rounded bg-[--accent-primary] hover:opacity-90 text-white transition-opacity flex items-center justify-center gap-2">
          <Activity size={12} />
          Profile Model
        </button>
      </div>
    </div>
  );
}
