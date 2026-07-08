'use client';

import { useState, useEffect } from 'react';
import { apiGet, apiPost } from '@/lib/api';
import { Loader2, Zap, Network, Database, Server, AlertCircle, TrendingDown, TrendingUp, Minus } from 'lucide-react';
import { FadeIn } from '@/components/FadeIn';

type MutationDef = {
  label: string;
  icon: string;
  description: string;
  category: string;
};

type Metrics = {
  flops_score: number;
  params: number;
  depth: number;
  memory_mb: number;
};

type LabResponse = {
  before: { graph: string; metrics: Metrics };
  after: { graph: string; metrics: Metrics };
  diff: Record<string, string>;
};

type TrainingEstimate = {
  total_flops: number;
  training_time_hours: number;
  cost_estimate_usd: number;
  vram_required_gb: number;
};

const ARCHITECTURES = ['ResNet', 'Transformer', 'ViT', 'U-Net'];
const GPU_TYPES = ['A100', 'H100', 'V100', 'T4', 'RTX4090'];

export default function LabsPage() {
  // Global state
  const [architecture, setArchitecture] = useState('ResNet');
  const [error, setError] = useState('');

  // Mutations state
  const [mutationsDef, setMutationsDef] = useState<Record<string, MutationDef>>({});
  const [selectedMutations, setSelectedMutations] = useState<string[]>([]);
  const [labLoading, setLabLoading] = useState(false);
  const [labResult, setLabResult] = useState<LabResponse | null>(null);

  // Estimator state
  const [datasetSize, setDatasetSize] = useState(1000000);
  const [batchSize, setBatchSize] = useState(256);
  const [gpuType, setGpuType] = useState('A100');
  const [mixedPrecision, setMixedPrecision] = useState(true);
  const [gradientCheckpointing, setGradientCheckpointing] = useState(false);
  const [estimatorLoading, setEstimatorLoading] = useState(false);
  const [estimateResult, setEstimateResult] = useState<TrainingEstimate | null>(null);

  useEffect(() => {
    const fetchMutations = async () => {
      try {
        const res = await apiGet<{ mutations: Record<string, MutationDef> }>('/api/lab/mutations');
        setMutationsDef(res.mutations || {});
      } catch (err: any) {
        setError(err.message || 'Failed to load mutations');
      }
    };
    fetchMutations();
  }, []);

  const toggleMutation = (key: string) => {
    setSelectedMutations(prev =>
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    );
  };

  const applyMutations = async () => {
    if (selectedMutations.length === 0) return;
    setLabLoading(true);
    setError('');
    setLabResult(null);
    try {
      const res = await apiPost<LabResponse>('/api/lab/mutations', {
        architecture,
        mutations: selectedMutations.map(type => ({ type })),
        config: {},
      });
      setLabResult(res);
    } catch (err: any) {
      setError(err.message || 'Failed to apply mutations');
    } finally {
      setLabLoading(false);
    }
  };

  const runEstimator = async () => {
    setEstimatorLoading(true);
    setError('');
    setEstimateResult(null);
    try {
      const res = await apiPost<TrainingEstimate>('/api/training-estimator', {
        architecture,
        dataset_size: datasetSize,
        batch_size: batchSize,
        gpu_type: gpuType,
        epochs: 10,
        mixed_precision: mixedPrecision,
        gradient_checkpointing: gradientCheckpointing,
      });
      setEstimateResult(res);
    } catch (err: any) {
      setError(err.message || 'Failed to run estimator');
    } finally {
      setEstimatorLoading(false);
    }
  };

  const renderMetricDelta = (before: number, after: number, isLowerBetter = false) => {
    if (before === after) return <div className="flex items-center gap-1 text-[#525252]"><Minus size={14}/> 0%</div>;
    const diff = after - before;
    const pct = (diff / before) * 100;
    const isImprovement = isLowerBetter ? diff < 0 : diff > 0;
    const colorClass = isImprovement ? 'text-[#4ADE80]' : 'text-red-400';
    const Icon = diff > 0 ? TrendingUp : TrendingDown;
    return (
      <div className={`flex items-center gap-1 ${colorClass}`}>
        <Icon size={14} />
        {Math.abs(pct).toFixed(1)}%
      </div>
    );
  };

  const groupedMutations = Object.entries(mutationsDef).reduce((acc, [key, def]) => {
    if (!acc[def.category]) acc[def.category] = [];
    acc[def.category].push({ key, ...def });
    return acc;
  }, {} as Record<string, ({ key: string } & MutationDef)[]>);

  const inp = "w-full bg-[#111111] border border-[#262626] rounded-lg px-3 py-2 text-sm text-white outline-none focus:border-[#A78BFA] transition-colors";
  const selectInp = "bg-[#111111] border border-[#262626] rounded-lg px-3 py-2 text-sm text-white outline-none focus:border-[#A78BFA] transition-colors appearance-none";

  return (
    <div className="min-h-screen bg-transparent text-white pb-24">
      {/* HEADER */}
      <div className="border-b border-[#1A1A1A] bg-[#0A0A0A] px-8 py-6">
        <h1 className="text-[26px] font-bold text-white">Interactive Architecture Labs</h1>
        <p className="mt-1 text-[13px] text-[#525252]">
          Mutate architectures and see the impact on compute, memory, and training costs in real-time.
        </p>
      </div>

      <div className="mx-auto max-w-6xl px-8 py-8 space-y-12">
        {error && (
          <div className="flex items-center gap-3 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-400">
            <AlertCircle size={18} />
            {error}
          </div>
        )}

        {/* Global Architecture Selector */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-semibold text-[#A3A3A3]">Base Architecture:</span>
          <div className="flex gap-2">
            {ARCHITECTURES.map(arch => (
              <button key={arch} onClick={() => { setArchitecture(arch); setLabResult(null); setEstimateResult(null); }}
                className={`rounded-lg px-4 py-2 text-sm transition-colors ${
                  architecture === arch ? 'bg-[#A78BFA] text-black font-semibold' : 'bg-[#111111] border border-[#262626] text-[#A3A3A3] hover:border-[#A78BFA]/40 hover:text-white'
                }`}>
                {arch}
              </button>
            ))}
          </div>
        </div>

        {/* SECTION: ARCHITECTURE MUTATOR */}
        <section className="space-y-6">
          <div className="border-b border-[#1A1A1A] pb-4">
            <h2 className="text-xl font-bold flex items-center gap-2"><Network className="text-[#A78BFA]" size={20}/> Architecture Mutator</h2>
            <p className="text-sm text-[#A3A3A3] mt-1">Select structural changes to apply to {architecture}.</p>
          </div>

          {Object.keys(mutationsDef).length === 0 ? (
            <div className="py-8 flex justify-center"><Loader2 className="animate-spin text-[#525252]" /></div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(groupedMutations).map(([category, items]) => (
                <div key={category} className="space-y-3">
                  <div className="text-xs font-semibold uppercase tracking-wider text-[#525252]">{category}</div>
                  <div className="flex flex-col gap-2">
                    {items.map(item => {
                      const active = selectedMutations.includes(item.key);
                      return (
                        <button key={item.key} onClick={() => toggleMutation(item.key)}
                          className={`flex flex-col items-start p-3 rounded-xl border text-left transition-all ${
                            active ? 'bg-[#A78BFA]/10 border-[#A78BFA]/50 shadow-[0_0_15px_rgba(167,139,250,0.1)]' : 'bg-[#111111] border-[#262626] hover:border-[#333333]'
                          }`}>
                          <div className={`text-sm font-semibold ${active ? 'text-[#A78BFA]' : 'text-white'}`}>
                            {item.label}
                          </div>
                          <div className={`text-xs mt-1 ${active ? 'text-[#A78BFA]/70' : 'text-[#525252]'}`}>
                            {item.description}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="flex justify-end">
            <button onClick={applyMutations} disabled={labLoading || selectedMutations.length === 0}
              className="flex items-center gap-2 rounded-xl bg-[#A78BFA] px-6 py-2.5 text-sm font-bold text-black transition-colors hover:bg-[#C4B5FD] disabled:opacity-50">
              {labLoading ? <Loader2 size={16} className="animate-spin" /> : <Zap size={16} />}
              Apply Mutations & Analyze
            </button>
          </div>

          {labResult && (
            <FadeIn>
              <div className="mt-8 rounded-2xl border border-[#262626] bg-[#0A0A0A] overflow-hidden">
                <table className="w-full text-sm text-left">
                  <thead className="bg-[#111111] text-[#A3A3A3] text-xs uppercase tracking-wider">
                    <tr>
                      <th className="px-6 py-4 font-semibold border-b border-[#262626]">Metric</th>
                      <th className="px-6 py-4 font-semibold border-b border-[#262626]">Baseline</th>
                      <th className="px-6 py-4 font-semibold border-b border-[#262626]">Mutated</th>
                      <th className="px-6 py-4 font-semibold border-b border-[#262626]">Delta</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1A1A1A]">
                    <tr className="hover:bg-[#111111]/50 transition-colors">
                      <td className="px-6 py-4 text-white font-medium">Parameters</td>
                      <td className="px-6 py-4 text-[#A3A3A3]">{(labResult.before.metrics.params / 1e6).toFixed(1)}M</td>
                      <td className="px-6 py-4 text-white">{(labResult.after.metrics.params / 1e6).toFixed(1)}M</td>
                      <td className="px-6 py-4">{renderMetricDelta(labResult.before.metrics.params, labResult.after.metrics.params, true)}</td>
                    </tr>
                    <tr className="hover:bg-[#111111]/50 transition-colors">
                      <td className="px-6 py-4 text-white font-medium">FLOPs Score</td>
                      <td className="px-6 py-4 text-[#A3A3A3]">{labResult.before.metrics.flops_score.toFixed(1)}</td>
                      <td className="px-6 py-4 text-white">{labResult.after.metrics.flops_score.toFixed(1)}</td>
                      <td className="px-6 py-4">{renderMetricDelta(labResult.before.metrics.flops_score, labResult.after.metrics.flops_score, true)}</td>
                    </tr>
                    <tr className="hover:bg-[#111111]/50 transition-colors">
                      <td className="px-6 py-4 text-white font-medium">Peak Memory</td>
                      <td className="px-6 py-4 text-[#A3A3A3]">{labResult.before.metrics.memory_mb.toFixed(0)} MB</td>
                      <td className="px-6 py-4 text-white">{labResult.after.metrics.memory_mb.toFixed(0)} MB</td>
                      <td className="px-6 py-4">{renderMetricDelta(labResult.before.metrics.memory_mb, labResult.after.metrics.memory_mb, true)}</td>
                    </tr>
                    <tr className="hover:bg-[#111111]/50 transition-colors">
                      <td className="px-6 py-4 text-white font-medium">Network Depth</td>
                      <td className="px-6 py-4 text-[#A3A3A3]">{labResult.before.metrics.depth}</td>
                      <td className="px-6 py-4 text-white">{labResult.after.metrics.depth}</td>
                      <td className="px-6 py-4">{renderMetricDelta(labResult.before.metrics.depth, labResult.after.metrics.depth, false)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </FadeIn>
          )}
        </section>

        {/* SECTION: TRAINING COST ESTIMATOR */}
        <section className="space-y-6 pt-12 border-t border-[#1A1A1A]">
          <div className="border-b border-[#1A1A1A] pb-4">
            <h2 className="text-xl font-bold flex items-center gap-2"><Server className="text-[#A78BFA]" size={20}/> Training Cost Estimator</h2>
            <p className="text-sm text-[#A3A3A3] mt-1">Estimate wall-clock time and cost for training {architecture}.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="space-y-2">
              <label className="text-xs font-semibold text-[#A3A3A3] uppercase tracking-wider">Dataset Size</label>
              <input type="number" className={inp} value={datasetSize} onChange={e => setDatasetSize(Number(e.target.value))} min={1} />
            </div>
            <div className="space-y-2">
              <label className="text-xs font-semibold text-[#A3A3A3] uppercase tracking-wider">Batch Size</label>
              <input type="number" className={inp} value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} min={1} />
            </div>
            <div className="space-y-2">
              <label className="text-xs font-semibold text-[#A3A3A3] uppercase tracking-wider">GPU Instance</label>
              <div className="relative">
                <select className={`${selectInp} w-full`} value={gpuType} onChange={e => setGpuType(e.target.value)}>
                  {GPU_TYPES.map(g => <option key={g} value={g}>{g}</option>)}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-[#525252]">▼</div>
              </div>
            </div>
            <div className="space-y-3 pt-6">
              <label className="flex items-center gap-2 text-sm text-white cursor-pointer">
                <input type="checkbox" checked={mixedPrecision} onChange={e => setMixedPrecision(e.target.checked)} className="accent-[#A78BFA] w-4 h-4" />
                Mixed Precision (fp16)
              </label>
              <label className="flex items-center gap-2 text-sm text-white cursor-pointer">
                <input type="checkbox" checked={gradientCheckpointing} onChange={e => setGradientCheckpointing(e.target.checked)} className="accent-[#A78BFA] w-4 h-4" />
                Gradient Checkpointing
              </label>
            </div>
          </div>

          <div className="flex justify-end">
            <button onClick={runEstimator} disabled={estimatorLoading}
              className="flex items-center gap-2 rounded-xl border border-[#262626] bg-[#111111] px-6 py-2.5 text-sm font-bold text-white transition-colors hover:border-[#A78BFA]/50 hover:bg-[#1A1A1A] disabled:opacity-50">
              {estimatorLoading ? <Loader2 size={16} className="animate-spin" /> : <Database size={16} />}
              Estimate Training Cost
            </button>
          </div>

          {estimateResult && (
            <FadeIn>
              <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] p-5">
                  <div className="text-xs font-semibold uppercase tracking-wider text-[#A3A3A3]">Est. Time</div>
                  <div className="mt-2 flex items-baseline gap-1">
                    <span className="text-2xl font-bold text-white">{estimateResult.training_time_hours.toFixed(1)}</span>
                    <span className="text-sm text-[#525252]">hours</span>
                  </div>
                </div>
                <div className="rounded-xl border border-[#A78BFA]/30 bg-[#A78BFA]/5 p-5 relative overflow-hidden">
                  <div className="absolute -right-4 -top-4 w-16 h-16 bg-[#A78BFA]/10 rounded-full blur-xl pointer-events-none" />
                  <div className="text-xs font-semibold uppercase tracking-wider text-[#A78BFA]">Est. Cost</div>
                  <div className="mt-2 flex items-baseline gap-1">
                    <span className="text-2xl font-bold text-white">${estimateResult.cost_estimate_usd.toFixed(2)}</span>
                    <span className="text-sm text-[#A78BFA]/70">USD</span>
                  </div>
                </div>
                <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] p-5">
                  <div className="text-xs font-semibold uppercase tracking-wider text-[#A3A3A3]">Total Compute</div>
                  <div className="mt-2 flex items-baseline gap-1">
                    <span className="text-2xl font-bold text-white">{(estimateResult.total_flops / 1e15).toFixed(2)}</span>
                    <span className="text-sm text-[#525252]">PFLOPs</span>
                  </div>
                </div>
                <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] p-5">
                  <div className="text-xs font-semibold uppercase tracking-wider text-[#A3A3A3]">VRAM Required</div>
                  <div className="mt-2 flex items-baseline gap-1">
                    <span className="text-2xl font-bold text-white">{estimateResult.vram_required_gb.toFixed(1)}</span>
                    <span className="text-sm text-[#525252]">GB</span>
                  </div>
                </div>
              </div>
            </FadeIn>
          )}
        </section>
      </div>
    </div>
  );
}
