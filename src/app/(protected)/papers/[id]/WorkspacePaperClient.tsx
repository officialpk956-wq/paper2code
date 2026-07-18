'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { apiGet, apiPost } from '@/lib/api';
import { findArch } from '@/lib/crosslinks';
import { FLAGSHIP_META, FLAGSHIP_GRAPH, FLAGSHIP_BLUEPRINT, FLAGSHIP_CODE } from '@/data/flagship-papers';
import { loader } from '@monaco-editor/react';

loader.config({ paths: { vs: '/monaco-editor/vs' } });
const MonacoEditor = dynamic(() => import('@monaco-editor/react'), { ssr: false });

type Tab = 'summary' | 'graph' | 'blueprint' | 'executable' | 'challenges' | 'tutor' | 'implement';

const TAB_LIST: { id: Tab; label: string }[] = [
  { id: 'summary',    label: 'Summary' },
  { id: 'graph',      label: 'Knowledge Graph' },
  { id: 'blueprint',  label: 'Blueprint' },
  { id: 'executable', label: 'Executable' },
  { id: 'challenges', label: 'Challenges' },
  { id: 'tutor',      label: 'AI Tutor' },
  { id: 'implement',  label: 'Implement' },
];

type PaperMeta = {
  title: string; authors: string; year: number; abstract: string; color: string;
  archSlug?: string; archName?: string;
  contributions: string[];
  status?: string;
};

type Challenge = { num: string; title: string; difficulty: 'Easy' | 'Medium' | 'Hard'; href: string };

type GraphNode = { id: string; label?: string; x?: number; y?: number; color?: string };
type GraphEdge = { source: string; target: string };
type GraphData = { status?: string; nodes?: GraphNode[]; edges?: GraphEdge[] };

type BlueprintComponent = { name: string; description: string };
type BlueprintData = { status?: string; components?: BlueprintComponent[]; confidence_score?: number };

type ExecutableData = { status?: string; code?: string; language?: string };

type ImplData = {
  status: string;
  starter_code: string;
  shapes: Record<string, { input: number[]; output: number[] }>;
  layer_docs: Record<string, { summary: string; use_case?: string }>;
};

const PAPER_CHALLENGES: Record<string, Challenge[]> = {
  'attention-is-all-you-need': [
    { num: '#010', title: 'Scaled Dot-Product Attention', difficulty: 'Hard',   href: '/dojo/ml-attention' },
    { num: '#006', title: 'Softmax Function',             difficulty: 'Medium', href: '/dojo/ml-softmax' },
    { num: '#005', title: 'Dot Product',                  difficulty: 'Easy',   href: '/dojo/numpy-dot-product' },
    { num: '#009', title: 'Gradient Descent Step',        difficulty: 'Medium', href: '/dojo/ml-gradient-descent' },
  ],
  resnet: [
    { num: '#003', title: 'ReLU Activation',              difficulty: 'Easy',   href: '/dojo/ml-relu' },
    { num: '#004', title: 'Mean Squared Error',           difficulty: 'Medium', href: '/dojo/ml-mse' },
    { num: '#002', title: 'Sigmoid Activation',           difficulty: 'Easy',   href: '/dojo/ml-sigmoid' },
    { num: '#007', title: 'Min-Max Normalization',        difficulty: 'Medium', href: '/dojo/ml-normalize' },
  ],
  bert: [
    { num: '#010', title: 'Scaled Dot-Product Attention', difficulty: 'Hard',   href: '/dojo/ml-attention' },
    { num: '#006', title: 'Softmax Function',             difficulty: 'Medium', href: '/dojo/ml-softmax' },
    { num: '#008', title: 'Binary Cross-Entropy Loss',    difficulty: 'Medium', href: '/dojo/ml-cross-entropy' },
    { num: '#009', title: 'Gradient Descent Step',        difficulty: 'Medium', href: '/dojo/ml-gradient-descent' },
  ],
  vit: [
    { num: '#010', title: 'Scaled Dot-Product Attention', difficulty: 'Hard',   href: '/dojo/ml-attention' },
    { num: '#006', title: 'Softmax Function',             difficulty: 'Medium', href: '/dojo/ml-softmax' },
    { num: '#007', title: 'Min-Max Normalization',        difficulty: 'Medium', href: '/dojo/ml-normalize' },
    { num: '#005', title: 'Dot Product',                  difficulty: 'Easy',   href: '/dojo/numpy-dot-product' },
  ],
  lora: [
    { num: '#005', title: 'Dot Product',                  difficulty: 'Easy',   href: '/dojo/numpy-dot-product' },
    { num: '#009', title: 'Gradient Descent Step',        difficulty: 'Medium', href: '/dojo/ml-gradient-descent' },
    { num: '#010', title: 'Scaled Dot-Product Attention', difficulty: 'Hard',   href: '/dojo/ml-attention' },
    { num: '#001', title: 'Create a NumPy Array',         difficulty: 'Easy',   href: '/dojo/numpy-array-creation' },
  ],
  'flash-attention': [
    { num: '#010', title: 'Scaled Dot-Product Attention', difficulty: 'Hard',   href: '/dojo/ml-attention' },
    { num: '#006', title: 'Softmax Function',             difficulty: 'Medium', href: '/dojo/ml-softmax' },
    { num: '#005', title: 'Dot Product',                  difficulty: 'Easy',   href: '/dojo/numpy-dot-product' },
    { num: '#007', title: 'Min-Max Normalization',        difficulty: 'Medium', href: '/dojo/ml-normalize' },
  ],
};

const DEFAULT_CHALLENGES: Challenge[] = [
  { num: '#010', title: 'Scaled Dot-Product Attention', difficulty: 'Hard',   href: '/dojo/ml-attention' },
  { num: '#006', title: 'Softmax Function',             difficulty: 'Medium', href: '/dojo/ml-softmax' },
  { num: '#009', title: 'Gradient Descent Step',        difficulty: 'Medium', href: '/dojo/ml-gradient-descent' },
];

const DIFF_COLOR: Record<string, string> = {
  Easy: 'text-[#4ADE80]', Medium: 'text-[#FACC15]', Hard: 'text-[#F87171]',
};

const FLAGSHIP_TITLE_WORDS: Record<string, string> = {
  'attention-is-all-you-need': 'attention',
  'resnet': 'residual',
  'bert': 'bert',
  'vit': 'image is worth',
  'lora': 'lora',
  'flash-attention': 'flashattention',
};

export default function WorkspacePaperClient({ id }: { id: string }) {
  const [tab, setTab] = useState<Tab>('summary');
  
  // Data states
  const [meta, setMeta] = useState<PaperMeta | null>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [blueprintData, setBlueprintData] = useState<BlueprintData | null>(null);
  const [executableData, setExecutableData] = useState<ExecutableData | null>(null);
  const [implData, setImplData] = useState<ImplData | null>(null);
  const [implCode, setImplCode] = useState('');
  
  // UI states
  const [loading, setLoading] = useState(true);
  const [tabLoading, setTabLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Tutor chat state
  const [numericId, setNumericId] = useState<number | null>(null);
  const [chat, setChat] = useState<{ role: 'user' | 'assistant'; text: string }[]>([
    { role: 'assistant', text: 'Ask me anything about this paper. I can explain the architecture, compare it with related work, and discuss implementation details.' },
  ]);
  const [input, setInput] = useState('');
  const [tutorLoading, setTutorLoading] = useState(false);
  const [tutorSessionId, setTutorSessionId] = useState<string | null>(null);

  useEffect(() => {
    fetchSummary();
  }, [id]);

  useEffect(() => {
    if (tab === 'graph') fetchGraph();
    else if (tab === 'blueprint') fetchBlueprint();
    else if (tab === 'executable') fetchExecutable();
    else if (tab === 'implement') fetchImplement();
  }, [tab, id]);

  const fetchSummary = async () => {
    if (FLAGSHIP_META[id]) {
      setMeta({ ...FLAGSHIP_META[id] });
      setLoading(false);
      apiGet<{ papers: {id: number, title: string}[] }>('/api/papers')
        .then(data => {
          const found = data.papers?.find(p => p.id.toString() === id || 
            p.title.toLowerCase().includes(FLAGSHIP_TITLE_WORDS[id] ?? ''));
          if (found) setNumericId(found.id);
        })
        .catch(() => {});
      return;
    }
    try {
      const data = await apiGet<PaperMeta>(`/api/papers/${id}`);
      setMeta({ ...data, color: data.color || '#A78BFA' });
      setNumericId(parseInt(id, 10));
    } catch (err: unknown) {
      setError((err as Error).message || 'Failed to load summary');
    } finally {
      setLoading(false);
    }
  };

  const fetchGraph = async () => {
    if (graphData) return;
    if (FLAGSHIP_GRAPH[id]) { setGraphData(FLAGSHIP_GRAPH[id]); return; }
    setTabLoading(true);
    try {
      const data = await apiGet<GraphData>(`/api/papers/${id}/knowledge-graph`);
      setGraphData(data);
    } catch (err: unknown) {
      setError((err as Error).message || 'Failed to load graph');
    } finally {
      setTabLoading(false);
    }
  };

  const fetchBlueprint = async () => {
    if (blueprintData) return;
    if (FLAGSHIP_BLUEPRINT[id]) { setBlueprintData(FLAGSHIP_BLUEPRINT[id]); return; }
    setTabLoading(true);
    try {
      const data = await apiGet<BlueprintData>(`/api/papers/${id}/blueprint`);
      setBlueprintData(data);
    } catch (err: unknown) {
      setError((err as Error).message || 'Failed to load blueprint');
    } finally {
      setTabLoading(false);
    }
  };

  const fetchExecutable = async () => {
    if (executableData) return;
    if (FLAGSHIP_CODE[id]) { setExecutableData(FLAGSHIP_CODE[id]); return; }
    setTabLoading(true);
    try {
      const data = await apiGet<ExecutableData>(`/api/papers/${id}/executable-graph`);
      setExecutableData(data);
    } catch (err: unknown) {
      setError((err as Error).message || 'Failed to load executable code');
    } finally {
      setTabLoading(false);
    }
  };

  const fetchImplement = async () => {
    if (implData) return;
    setTabLoading(true);
    try {
      const data = await apiGet<ImplData>(`/api/papers/${id}/implement`);
      setImplData(data);
      setImplCode(data.starter_code);
    } catch (err: unknown) {
      setError((err as Error).message || 'Failed to load implementation');
    } finally {
      setTabLoading(false);
    }
  };

  // Poll for processing status
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | undefined;
    if (
      (tab === 'graph' && graphData?.status === 'processing') ||
      (tab === 'blueprint' && blueprintData?.status === 'processing') ||
      (tab === 'executable' && executableData?.status === 'processing')
    ) {
      interval = setInterval(() => {
        if (tab === 'graph') fetchGraph();
        if (tab === 'blueprint') fetchBlueprint();
        if (tab === 'executable') fetchExecutable();
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [tab, graphData, blueprintData, executableData]);

  const sendMessage = async () => {
    const userMsg = input.trim();
    if (!userMsg || tutorLoading) return;
    setChat(prev => [...prev, { role: 'user', text: userMsg }]);
    setInput('');
    setTutorLoading(true);
    
    const nid = numericId ?? parseInt(id, 10);
    
    try {
      if (numericId !== null || !isNaN(parseInt(id, 10))) {
        // Use paper-specific RAG endpoint
        const res = await apiPost<{ answer: string; referenced_papers: number[] }>(
          `/api/papers/${nid}/ask`,
          { question: userMsg }
        );
        setChat(prev => [...prev, { role: 'assistant', text: res.answer }]);
        // Show referenced paper IDs as a soft follow-up message if any
        if (res.referenced_papers?.length > 0) {
          setChat(prev => [...prev, { 
            role: 'assistant', 
            text: `See also: ${res.referenced_papers.map(pid => `Paper #${pid}`).join(', ')}`
          }]);
        }
      } else {
        // Fallback: general tutor (should rarely hit)
        const res = await apiPost<{ answer: string; session_id: string }>('/api/tutor/ask', {
          query: userMsg,
          session_id: tutorSessionId,
          context_type: 'paper',
          context_data: { title: meta?.title ?? '', paper_id: id },
        });
        setTutorSessionId(res.session_id);
        setChat(prev => [...prev, { role: 'assistant', text: res.answer }]);
      }
    } catch (err: unknown) {
      setChat(prev => [...prev, { 
        role: 'assistant', 
        text: (err as Error)?.message || 'Sorry, I could not answer that question.' 
      }]);
    } finally {
      setTutorLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col bg-transparent text-white p-8" style={{ height: 'calc(100vh - 56px)' }}>
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-[#262626] rounded w-1/4" />
          <div className="h-8 bg-[#262626] rounded w-1/2" />
          <div className="h-4 bg-[#262626] rounded w-1/3" />
        </div>
      </div>
    );
  }

  if (!meta) {
    return (
      <div className="flex flex-col bg-transparent text-white p-8" style={{ height: 'calc(100vh - 56px)' }}>
        <div className="text-[#A3A3A3]">Paper not found.</div>
      </div>
    );
  }

  const challenges = PAPER_CHALLENGES[id] ?? DEFAULT_CHALLENGES;

  return (
    <div className="flex flex-col bg-transparent text-white" style={{ height: 'calc(100vh - 56px)' }}>
      {/* HEADER */}
      <div className="border-b border-[#1A1A1A] bg-[#0A0A0A] px-6 py-4 flex-shrink-0">
        {error && (
          <div className="mb-4 bg-red-500/10 border border-red-500/20 px-4 py-2 text-xs text-red-500 rounded">
            Could not load data — retrying… ({error})
          </div>
        )}
        <Link href="/papers" className="inline-flex items-center gap-1 text-[11px] text-[#525252] hover:text-white mb-3">← Research Hub</Link>
        <div className="flex items-start justify-between">
          <div>
            <div className="h-0.5 w-8 rounded-full mb-2" style={{ backgroundColor: meta.color }} />
            <h1 className="text-[18px] font-bold text-white leading-tight max-w-2xl">{meta.title}</h1>
            <div className="text-[12px] text-[#525252] mt-1">{meta.authors} · {meta.year}</div>
            {(() => {
              const archNameMap: Record<string, string> = {
                'attention-is-all-you-need': 'Transformer',
                'resnet': 'ResNet',
                'bert': 'BERT',
                'vit': 'ViT',
                'lora': 'LoRA',
                'flash-attention': 'Flash Attention',
              };
              const archName = archNameMap[id];
              const archEntry = archName ? findArch(archName) : undefined;
              return (
                <div className="mt-3 flex gap-2">
                  <Link href="/papers?tab=library"
                    className="inline-flex items-center gap-1.5 text-[11px] font-medium rounded-full px-2.5 py-0.5 border border-[#2E4436] bg-[#1A1A1A] text-[#A3A3A3] hover:text-white hover:border-[#525252]">
                    View in Library →
                  </Link>
                  {archEntry && (
                    <Link href={`/architectures/${archEntry.slug}`}
                      className="inline-flex items-center gap-1.5 text-[11px] font-medium rounded-full px-2.5 py-0.5 border"
                      style={{ color: meta.color, borderColor: meta.color + '40', backgroundColor: meta.color + '10' }}>
                      Architecture: {archEntry.name} →
                    </Link>
                  )}
                </div>
              );
            })()}
          </div>
          <div className="flex gap-2">
            <button className="text-[12px] px-3 py-1.5 rounded-lg border border-[#262626] text-[#A3A3A3] hover:text-white">Export</button>
            <button className="text-[12px] px-3 py-1.5 rounded-lg bg-[#A78BFA] text-black font-semibold hover:brightness-110">Share</button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mt-4">
          {TAB_LIST.map(t => (
            <button key={t.id} type="button" onClick={() => setTab(t.id)}
              className={'px-3 py-1.5 text-[12px] rounded-md transition-colors ' +
                (tab === t.id ? 'bg-[#1A1A1A] text-white font-semibold' : 'text-[#525252] hover:text-white')}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* BODY */}
      <div className="flex-1 overflow-y-auto p-6 relative">
        {tabLoading && (
          <div className="absolute inset-0 bg-[#0A120D]/50 z-10 flex items-center justify-center">
            <div className="animate-pulse bg-[#111111] border border-[#262626] p-4 rounded-xl text-sm">Loading...</div>
          </div>
        )}

        {tab === 'summary' && (
          <div className="max-w-2xl space-y-6">
            <div className="rounded-xl border border-[#262626] bg-[#111111] p-5">
              <div className="text-[13px] font-semibold text-white mb-3">Abstract</div>
              <p className="text-[13px] leading-relaxed text-[#A3A3A3]">{meta.abstract}</p>
            </div>
            {meta.contributions && meta.contributions.length > 0 && (
              <div className="rounded-xl border border-[#262626] bg-[#111111] p-5">
                <div className="text-[13px] font-semibold text-white mb-3">Key Contributions</div>
                <ul className="space-y-2 text-[13px] text-[#A3A3A3]">
                  {meta.contributions.map((c: string) => (
                    <li key={c} className="flex gap-2">
                      <span style={{ color: meta.color }}>→</span> {c}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {tab === 'graph' && graphData?.status === 'processing' && (
          <div className="flex flex-col items-center justify-center h-full text-[#A3A3A3] text-sm">
            <div className="animate-spin mb-4 border-2 border-t-[#A78BFA] border-r-transparent border-b-transparent border-l-transparent rounded-full w-8 h-8" />
            Still processing...
            <div className="w-64 bg-[#111111] rounded-full h-2 mt-4 overflow-hidden">
              <div className="bg-[#A78BFA] h-full rounded-full animate-pulse w-1/2"></div>
            </div>
          </div>
        )}
        {tab === 'graph' && graphData?.status !== 'processing' && graphData?.nodes && (
          <div className="rounded-xl border border-[#262626] bg-[#111111] p-4" style={{ height: 520 }}>
            <div className="text-[13px] font-semibold text-white mb-4">Concept Graph</div>
            <svg viewBox="0 0 600 480" className="w-full h-[440px]">
              {graphData.edges?.map((edge: GraphEdge) => {
                const na = graphData.nodes?.find((n: GraphNode) => n.id === edge.source);
                const nb = graphData.nodes?.find((n: GraphNode) => n.id === edge.target);
                if (!na || !nb) return null;
                return <line key={edge.source + edge.target} x1={na.x || Math.random()*500} y1={na.y || Math.random()*400} x2={nb.x || Math.random()*500} y2={nb.y || Math.random()*400} stroke="#262626" strokeWidth={1.5} />;
              })}
              {graphData.nodes?.map((n: GraphNode) => {
                const nx = n.x || (Math.random() * 400 + 100);
                const ny = n.y || (Math.random() * 300 + 100);
                const color = n.color || '#A78BFA';
                return (
                  <g key={n.id}>
                    <circle cx={nx} cy={ny} r={36} fill={color + '18'} stroke={color + '60'} strokeWidth={1.5} />
                    <text x={nx} y={ny + 1} textAnchor="middle" dominantBaseline="middle" fill={color} fontSize={12} fontWeight={600}>
                      {n.label}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        )}

        {tab === 'blueprint' && blueprintData?.status === 'processing' && (
          <div className="flex flex-col items-center justify-center h-full text-[#A3A3A3] text-sm">
            <div className="animate-spin mb-4 border-2 border-t-[#A78BFA] border-r-transparent border-b-transparent border-l-transparent rounded-full w-8 h-8" />
            Still processing...
          </div>
        )}
        {tab === 'blueprint' && blueprintData?.status !== 'processing' && blueprintData?.components && (
          <div className="rounded-xl border border-[#262626] bg-[#111111] p-6">
            <div className="flex items-center justify-between border-b border-[#1A1A1A] pb-4 mb-4">
              <div className="text-[13px] font-semibold text-white">Architecture Blueprint</div>
              {blueprintData.confidence_score !== undefined && blueprintData.confidence_score < 0.70 && (
                <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#EF4444]/12 text-[#F87171] text-[11px] font-semibold border border-[#EF4444]/20">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#EF4444] animate-pulse" />
                  Approximate match ({Math.round(blueprintData.confidence_score * 100)}% confidence)
                </div>
              )}
            </div>
            <div className="text-sm text-[#A3A3A3]">
              {blueprintData.components?.map((c: BlueprintComponent) => (
                <div key={c.name} className="mb-2 p-2 bg-[#1A1A1A] rounded">
                  <span className="font-semibold">{c.name}</span>: {c.description}
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === 'executable' && executableData?.status === 'processing' && (
          <div className="flex flex-col items-center justify-center h-full text-[#A3A3A3] text-sm">
            <div className="animate-spin mb-4 border-2 border-t-[#A78BFA] border-r-transparent border-b-transparent border-l-transparent rounded-full w-8 h-8" />
            Still processing...
          </div>
        )}
        {tab === 'executable' && executableData?.status !== 'processing' && executableData?.code && (
          <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] overflow-hidden">
            <div className="flex items-center justify-between border-b border-[#1A1A1A] px-4 py-2.5">
              <div className="text-[13px] font-semibold text-white">implementation.{executableData.language === 'python' ? 'py' : 'txt'}</div>
              <button className="text-[12px] px-3 py-1 rounded-lg bg-[#A78BFA] text-black font-semibold hover:brightness-110">Run</button>
            </div>
            <pre className="p-4 text-[12px] text-[#A3A3A3] overflow-x-auto leading-relaxed font-mono whitespace-pre-wrap">
              {executableData.code}
            </pre>
          </div>
        )}

        {tab === 'challenges' && (
          <div className="space-y-3 max-w-2xl">
            <div className="text-[12px] text-[#525252] mb-4">
              Coding challenges derived from <span className="text-white">{meta.title}</span>
            </div>
            {challenges.map(p => (
              <div key={p.num} className="flex items-center justify-between rounded-xl border border-[#262626] bg-[#111111] px-4 py-3">
                <div>
                  <span className="text-[11px] text-[#525252] mr-2">{p.num}</span>
                  <span className="text-[14px] font-semibold text-white">{p.title}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className={DIFF_COLOR[p.difficulty] + ' text-[12px]'}>{p.difficulty}</span>
                  <Link href={p.href} className="text-[12px] px-3 py-1 rounded-lg bg-[#A78BFA] text-black font-semibold hover:brightness-110">
                    Solve →
                  </Link>
                </div>
              </div>
            ))}
          </div>
        )}

        {tab === 'tutor' && (
          <div className="flex flex-col gap-3 max-w-2xl h-[480px]">
            <div className="flex-1 overflow-y-auto space-y-3 rounded-xl border border-[#262626] bg-[#111111] p-4">
              {chat.map((msg, i) => (
                <div key={i} className={msg.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
                  <div className={'max-w-[80%] rounded-xl px-4 py-2.5 text-[13px] leading-relaxed ' +
                    (msg.role === 'user' ? 'bg-[#A78BFA] text-black font-medium' : 'bg-[#1A1A1A] text-[#A3A3A3]')}>
                    {msg.text}
                  </div>
                </div>
              ))}
              {tutorLoading && (
                <div className="flex justify-start">
                  <div className="max-w-[80%] rounded-xl px-4 py-2.5 text-[13px] bg-[#1A1A1A] text-[#A3A3A3] flex items-center gap-2">
                    <div className="w-1.5 h-1.5 bg-[#525252] rounded-full animate-bounce" />
                    <div className="w-1.5 h-1.5 bg-[#525252] rounded-full animate-bounce" style={{ animationDelay: '0.15s' }} />
                    <div className="w-1.5 h-1.5 bg-[#525252] rounded-full animate-bounce" style={{ animationDelay: '0.3s' }} />
                  </div>
                </div>
              )}
            </div>
            <div className="flex gap-2">
              <input value={input} onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && sendMessage()}
                placeholder="Ask about this paper..."
                disabled={tutorLoading}
                className="flex-1 bg-[#111111] border border-[#262626] rounded-xl px-4 py-2.5 text-[13px] text-white placeholder:text-[#525252] outline-none focus:border-[#A78BFA]" />
              <button onClick={sendMessage} disabled={tutorLoading}
                className="px-4 py-2.5 rounded-xl bg-[#A78BFA] text-black text-[13px] font-semibold hover:brightness-110 disabled:opacity-50">Send</button>
            </div>
          </div>
        )}

        {tab === 'implement' && (
          <div className="flex h-full gap-4">
            {/* Left Pane: Editor */}
            <div className="flex flex-col w-[60%] h-full rounded-xl border border-[#262626] bg-[#0A0A0A] overflow-hidden">
              <div className="flex flex-col border-b border-[#1A1A1A] px-4 py-3">
                <div className="text-[14px] font-bold text-white">Implement {meta.title}</div>
                <div className="text-[12px] text-[#A3A3A3] mt-1">Starter code generated from the architecture graph. Edit freely.</div>
                {implData?.status === "no_graph" && (
                  <div className="mt-2 bg-[#F59E0B]/10 border border-[#F59E0B]/20 text-[#F59E0B] px-3 py-2 text-[12px] rounded-lg">
                    No architecture graph available for this paper. Upload the paper PDF to extract architecture data.
                  </div>
                )}
              </div>
              <div className="flex-1 min-h-[400px]">
                <MonacoEditor
                  height="calc(100vh - 280px)"
                  language="python"
                  theme="vs-dark"
                  value={implCode}
                  onChange={v => setImplCode(v ?? '')}
                  options={{
                    fontSize: 13,
                    minimap: { enabled: false },
                    scrollBeyondLastLine: false,
                    fontFamily: 'JetBrains Mono, monospace',
                  }}
                />
              </div>
              <div className="border-t border-[#1A1A1A] p-3 flex items-center justify-between">
                <Link
                  href={`/papers/${id}/implement`}
                  className="text-[12px] px-4 py-2 rounded-lg bg-[#A78BFA] text-black font-semibold hover:brightness-110"
                >
                  Start Guided Implementation →
                </Link>
                <button
                  disabled
                  title="Coming soon"
                  className="text-[12px] px-4 py-2 rounded-lg bg-[#262626] text-[#A3A3A3] font-semibold cursor-not-allowed border border-[#1A1A1A]"
                >
                  Run in Sandbox
                </button>
              </div>
            </div>

            {/* Right Pane: Shape Hints & Layer Guide */}
            <div className="flex flex-col w-[40%] h-full overflow-y-auto space-y-4">
              <div className="rounded-xl border border-[#262626] bg-[#111111] p-5">
                <div className="text-[10px] uppercase font-bold text-[#A78BFA] tracking-wider mb-4">Tensor Shapes</div>
                {implData?.shapes && Object.keys(implData.shapes).length > 0 ? (
                  <div className="space-y-2">
                    {Object.entries(implData.shapes).map(([nodeId, shape]) => {
                      const inStr = Array.isArray(shape.input) ? `(${shape.input.join(', ')})` : String(shape.input);
                      const outStr = Array.isArray(shape.output) ? `(${shape.output.join(', ')})` : String(shape.output);
                      return (
                        <div key={nodeId} className="flex flex-col py-1 border-b border-[#1A1A1A] last:border-0">
                          <span className="text-[11px] font-semibold text-white mb-1">{nodeId}</span>
                          <span className="text-[11px] font-mono text-[#A3A3A3]">{inStr} → {outStr}</span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-[12px] text-[#525252]">No shape information available.</div>
                )}
              </div>

              <div className="rounded-xl border border-[#262626] bg-[#111111] p-5">
                <div className="text-[10px] uppercase font-bold text-[#A78BFA] tracking-wider mb-4">Layer Guide</div>
                {implData?.layer_docs && Object.keys(implData.layer_docs).length > 0 ? (
                  <div className="space-y-4">
                    {Object.entries(implData.layer_docs).map(([nodeId, doc]) => (
                      <div key={nodeId} className="flex flex-col pb-3 border-b border-[#1A1A1A] last:border-0 last:pb-0">
                        <span className="text-[12px] font-semibold text-[#D4D4D4] mb-1">{nodeId}</span>
                        <span className="text-[11px] leading-relaxed text-[#525252]">{doc.summary}</span>
                        {doc.use_case && <span className="text-[11px] leading-relaxed text-[#525252] italic mt-1">Use case: {doc.use_case}</span>}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-[12px] text-[#525252]">No layer documentation available.</div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
