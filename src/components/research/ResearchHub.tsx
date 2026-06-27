'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import { Upload, BookOpen, Code2, Layers3, ChevronRight, FileText, ExternalLink } from 'lucide-react';
import type { PaperMeta, ImplementationMeta } from '@/lib/content/schemas';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface UploadedPaper {
  id: number;
  title: string;
  architecture_type: string;
  module_count: number;
  parameter_count: number;
  flops: number;
  status: string;
  support_level: string;
}

type SerializedPaper = Pick<PaperMeta, 'slug' | 'title' | 'description' | 'authors' | 'year' | 'venue' | 'difficulty' | 'tags'>;
type SerializedImpl  = Pick<ImplementationMeta, 'slug' | 'title' | 'description' | 'difficulty' | 'tags' | 'paperSlug'>;

export interface ResearchHubProps {
  uploadedPapers: UploadedPaper[];
  contentPapers: SerializedPaper[];
  implementations: SerializedImpl[];
  archCount: number;
  implCount: number;
  sysDesignCount: number;
  paperLibCount: number;
}

type TabId = 'uploaded' | 'library' | 'implementations' | 'collections';

// ─── Research Collections (all slugs verified to exist in src/content/) ──────

const COLLECTIONS = [
  {
    id: 'transformers',
    name: 'Transformers',
    color: '#8B5CF6',
    description: 'The architecture that changed everything — from sequence models to multimodal AI.',
    items: [
      { title: 'Attention Is All You Need', type: 'paper', href: '/papers/attention-is-all-you-need' },
      { title: 'BERT', type: 'paper', href: '/papers/bert' },
      { title: 'Transformer Architecture', type: 'architecture', href: '/architectures/transformer' },
      { title: 'Build GPT from Scratch', type: 'implementation', href: '/paper-to-code/gpt' },
    ],
  },
  {
    id: 'vision',
    name: 'Vision Models',
    color: '#06B6D4',
    description: 'Transformers applied to images — ViT, CLIP, and visual foundation models.',
    items: [
      { title: 'Vision Transformer (ViT)', type: 'paper', href: '/papers/vision-transformer' },
      { title: 'CLIP', type: 'paper', href: '/papers/clip' },
      { title: 'ViT Architecture', type: 'architecture', href: '/architectures/vit' },
      { title: 'Build CLIP from Scratch', type: 'implementation', href: '/paper-to-code/clip' },
    ],
  },
  {
    id: 'diffusion',
    name: 'Diffusion Models',
    color: '#EC4899',
    description: 'Score-based generative modeling — from latent diffusion to Stable Diffusion.',
    items: [
      { title: 'Latent Diffusion Models', type: 'paper', href: '/papers/latent-diffusion-models' },
      { title: 'Stable Diffusion', type: 'paper', href: '/papers/stable-diffusion' },
      { title: 'U-Net Architecture', type: 'architecture', href: '/architectures/unet' },
      { title: 'Implement Stable Diffusion', type: 'implementation', href: '/paper-to-code/stable-diffusion' },
    ],
  },
  {
    id: 'llms',
    name: 'Large Language Models',
    color: '#F59E0B',
    description: 'GPT, LLaMA, and the scaling laws driving modern language models.',
    items: [
      { title: 'GPT-2: Language Models are Unsupervised Multitask Learners', type: 'paper', href: '/papers/gpt-2' },
      { title: 'LLaMA', type: 'paper', href: '/papers/llama' },
      { title: 'LLaMA Architecture', type: 'architecture', href: '/architectures/llama' },
      { title: 'Build LLaMA from Scratch', type: 'implementation', href: '/paper-to-code/llama' },
    ],
  },
  {
    id: 'generative',
    name: 'Generative Models',
    color: '#10B981',
    description: 'GANs, VAEs, and the full spectrum of generative architectures.',
    items: [
      { title: 'Generative Adversarial Networks', type: 'paper', href: '/papers/gan' },
      { title: 'GAN Architecture', type: 'architecture', href: '/architectures/gan' },
      { title: 'VAE Architecture', type: 'architecture', href: '/architectures/vae' },
      { title: 'Build GAN from Scratch', type: 'implementation', href: '/paper-to-code/gan' },
    ],
  },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

const DIFF_COLOR: Record<string, string> = {
  beginner: '#10B981',
  intermediate: '#F59E0B',
  advanced: '#EF4444',
};

const TYPE_COLOR: Record<string, string> = {
  paper: '#8B5CF6',
  architecture: '#06B6D4',
  implementation: '#10B981',
};

const STATUS_COLOR: Record<string, string> = {
  Published: '#10B981',
  Draft: '#F59E0B',
  Processing: '#06B6D4',
};

function firstAuthor(authors: string[]): string {
  if (!authors.length) return '';
  return authors.length === 1 ? authors[0] : `${authors[0]} et al.`;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function Badge({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wide"
      style={{ background: `${color}18`, color }}
    >
      {label}
    </span>
  );
}

function EmptyUploadState() {
  return (
    <div className="flex flex-col items-center justify-center py-20 px-6 text-center">
      <div
        className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
        style={{ background: 'rgba(249,115,22,0.12)', border: '1px solid rgba(249,115,22,0.25)' }}
      >
        <Upload size={24} style={{ color: '#F97316' }} />
      </div>
      <div className="text-[15px] font-bold text-white mb-2">No papers uploaded yet</div>
      <div className="text-[13px] text-slate-500 max-w-sm mb-6">
        Upload a PDF to extract the knowledge graph, architecture blueprint, and executable graph.
      </div>
      <Link
        href="/papers/upload"
        className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold text-white transition-all duration-200"
        style={{ background: 'linear-gradient(135deg, #F97316, #FB923C)', boxShadow: '0 0 20px rgba(249,115,22,0.3)' }}
      >
        <Upload size={14} />
        Upload Your First Paper
      </Link>
    </div>
  );
}

function UploadedPaperCard({ paper }: { paper: UploadedPaper }) {
  const statusColor = STATUS_COLOR[paper.status] ?? '#64748B';
  return (
    <Link
      href={`/papers/upload/${paper.id}`}
      className="block p-4 rounded-xl transition-all duration-200"
      style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
      onMouseEnter={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.background = 'rgba(249,115,22,0.06)';
        el.style.borderColor = 'rgba(249,115,22,0.25)';
        el.style.transform = 'translateY(-1px)';
      }}
      onMouseLeave={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.background = 'rgba(255,255,255,0.03)';
        el.style.borderColor = 'rgba(255,255,255,0.07)';
        el.style.transform = 'translateY(0)';
      }}
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="text-[13px] font-semibold text-white leading-snug line-clamp-2">{paper.title}</div>
        <ChevronRight size={14} className="flex-shrink-0 mt-0.5 text-slate-600" />
      </div>
      <div className="flex flex-wrap gap-2">
        {paper.architecture_type && (
          <Badge label={paper.architecture_type} color="#06B6D4" />
        )}
        <Badge label={paper.status} color={statusColor} />
        {paper.module_count > 0 && (
          <span className="text-[11px] text-slate-600">{paper.module_count} modules</span>
        )}
      </div>
    </Link>
  );
}

function ContentPaperCard({ paper }: { paper: SerializedPaper }) {
  return (
    <Link
      href={`/papers/${paper.slug}`}
      className="block p-4 rounded-xl transition-all duration-200"
      style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
      onMouseEnter={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.background = 'rgba(139,92,246,0.06)';
        el.style.borderColor = 'rgba(139,92,246,0.25)';
        el.style.transform = 'translateY(-1px)';
      }}
      onMouseLeave={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.background = 'rgba(255,255,255,0.03)';
        el.style.borderColor = 'rgba(255,255,255,0.07)';
        el.style.transform = 'translateY(0)';
      }}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <Badge label="Paper" color="#8B5CF6" />
        <Badge label={paper.difficulty} color={DIFF_COLOR[paper.difficulty] ?? '#64748B'} />
      </div>
      <div className="text-[13px] font-semibold text-white leading-snug mb-2 line-clamp-2">{paper.title}</div>
      <div className="text-[11px] text-slate-500">
        {firstAuthor(paper.authors)}{paper.year ? ` · ${paper.year}` : ''}{paper.venue ? ` · ${paper.venue}` : ''}
      </div>
    </Link>
  );
}

function ImplCard({ impl }: { impl: SerializedImpl }) {
  return (
    <Link
      href={`/paper-to-code/${impl.slug}`}
      className="block p-4 rounded-xl transition-all duration-200"
      style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
      onMouseEnter={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.background = 'rgba(16,185,129,0.06)';
        el.style.borderColor = 'rgba(16,185,129,0.25)';
        el.style.transform = 'translateY(-1px)';
      }}
      onMouseLeave={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.background = 'rgba(255,255,255,0.03)';
        el.style.borderColor = 'rgba(255,255,255,0.07)';
        el.style.transform = 'translateY(0)';
      }}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <Badge label="Implementation" color="#10B981" />
        <Badge label={impl.difficulty} color={DIFF_COLOR[impl.difficulty] ?? '#64748B'} />
      </div>
      <div className="text-[13px] font-semibold text-white leading-snug mb-2 line-clamp-2">{impl.title}</div>
      <div className="text-[11px] text-slate-500 line-clamp-2">{impl.description}</div>
    </Link>
  );
}

function CollectionSection({ col }: { col: typeof COLLECTIONS[number] }) {
  return (
    <div
      className="rounded-xl p-5"
      style={{ background: 'rgba(255,255,255,0.02)', border: `1px solid ${col.color}20` }}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="text-[14px] font-bold" style={{ color: col.color }}>{col.name}</span>
        <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full"
          style={{ background: `${col.color}15`, color: col.color }}>
          {col.items.length} items
        </span>
      </div>
      <p className="text-[12px] text-slate-500 mb-4 leading-relaxed">{col.description}</p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {col.items.map(item => (
          <Link
            key={item.href}
            href={item.href}
            className="flex items-center justify-between gap-2 px-3 py-2 rounded-lg transition-all duration-150"
            style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}
            onMouseEnter={e => {
              const el = e.currentTarget as HTMLElement;
              el.style.background = `${col.color}0c`;
              el.style.borderColor = `${col.color}30`;
            }}
            onMouseLeave={e => {
              const el = e.currentTarget as HTMLElement;
              el.style.background = 'rgba(255,255,255,0.03)';
              el.style.borderColor = 'rgba(255,255,255,0.06)';
            }}
          >
            <div className="flex items-center gap-2 min-w-0">
              <span
                className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                style={{ background: TYPE_COLOR[item.type] ?? '#64748B' }}
              />
              <span className="text-[12px] text-slate-300 truncate">{item.title}</span>
            </div>
            <ExternalLink size={11} className="flex-shrink-0 text-slate-600" />
          </Link>
        ))}
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function ResearchHub({
  uploadedPapers,
  contentPapers,
  implementations,
  archCount,
  implCount,
  paperLibCount,
}: ResearchHubProps) {
  const [activeTab, setActiveTab] = useState<TabId>('uploaded');
  const [search, setSearch] = useState('');

  const filteredUploaded = useMemo(() => {
    if (!search.trim()) return uploadedPapers;
    const q = search.toLowerCase();
    return uploadedPapers.filter(p => 
      p.title.toLowerCase().includes(q) ||
      (p.architecture_type && p.architecture_type.toLowerCase().includes(q)) ||
      p.status.toLowerCase().includes(q)
    );
  }, [uploadedPapers, search]);

  const filteredLibrary = useMemo(() => {
    if (!search.trim()) return contentPapers;
    const q = search.toLowerCase();
    return contentPapers.filter(p => 
      p.title.toLowerCase().includes(q) ||
      p.authors.some(a => a.toLowerCase().includes(q)) ||
      (p.tags && p.tags.some(t => t.toLowerCase().includes(q))) ||
      (p.description && p.description.toLowerCase().includes(q))
    );
  }, [contentPapers, search]);

  const filteredImpls = useMemo(() => {
    if (!search.trim()) return implementations;
    const q = search.toLowerCase();
    return implementations.filter(i => 
      i.title.toLowerCase().includes(q) ||
      (i.tags && i.tags.some(t => t.toLowerCase().includes(q))) ||
      (i.description && i.description.toLowerCase().includes(q))
    );
  }, [implementations, search]);

  const tabs: { id: TabId; label: string; count: number; icon: React.ReactNode }[] = [
    { id: 'uploaded', label: 'Uploaded Papers', count: uploadedPapers.length, icon: <Upload size={13} /> },
    { id: 'library',  label: 'Paper Library',   count: paperLibCount,          icon: <FileText size={13} /> },
    { id: 'implementations', label: 'Implementations', count: implCount,       icon: <Code2 size={13} /> },
    { id: 'collections', label: 'Research Collections', count: COLLECTIONS.length, icon: <Layers3 size={13} /> },
  ];

  const stats = [
    { label: 'Uploaded Papers',    value: String(uploadedPapers.length), color: '#F97316' },
    { label: 'Paper Library',      value: String(paperLibCount),          color: '#8B5CF6' },
    { label: 'Architectures',      value: String(archCount),              color: '#06B6D4' },
    { label: 'Implementations',    value: String(implCount),              color: '#10B981' },
  ];

  return (
    <div
      className="min-h-full overflow-y-auto"
      style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(139,92,246,0.2) transparent' }}
    >
      {/* Hero */}
      <div
        className="px-8 pt-8 pb-6"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}
      >
        <div className="max-w-5xl">
          <div className="flex items-start justify-between gap-6 flex-wrap">
            <div>
              <div className="flex items-center gap-2 mb-3">
                <div
                  className="w-7 h-7 rounded-lg flex items-center justify-center"
                  style={{ background: 'rgba(249,115,22,0.15)', border: '1px solid rgba(249,115,22,0.3)' }}
                >
                  <BookOpen size={14} style={{ color: '#F97316' }} />
                </div>
                <span className="text-[11px] font-bold uppercase tracking-[0.14em] text-slate-500">Research Hub</span>
              </div>
              <h1
                className="text-2xl font-black text-white mb-2"
                style={{ fontFamily: 'var(--font-heading)' }}
              >
                Research Hub
              </h1>
              <p className="text-[13px] text-slate-500 max-w-xl leading-relaxed">
                Explore uploaded papers, paper explainers, implementations, and architecture studies.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <Link
                href="/papers/upload"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-[13px] font-semibold text-white transition-all duration-200"
                style={{ background: 'linear-gradient(135deg, #F97316, #FB923C)', boxShadow: '0 0 16px rgba(249,115,22,0.25)' }}
                onMouseEnter={e => (e.currentTarget.style.boxShadow = '0 0 24px rgba(249,115,22,0.4)')}
                onMouseLeave={e => (e.currentTarget.style.boxShadow = '0 0 16px rgba(249,115,22,0.25)')}
              >
                <Upload size={13} />
                Upload Paper
              </Link>
              <Link
                href="#library"
                onClick={() => setActiveTab('library')}
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-[13px] font-semibold transition-all duration-200"
                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: '#CBD5E1' }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.09)';
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(139,92,246,0.4)';
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.05)';
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.1)';
                }}
              >
                <BookOpen size={13} />
                Browse Library
              </Link>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
            {stats.map(stat => (
              <div
                key={stat.label}
                className="rounded-xl px-4 py-3"
                style={{ background: 'rgba(255,255,255,0.03)', border: `1px solid ${stat.color}20` }}
              >
                <div
                  className="text-2xl font-black mb-0.5"
                  style={{ color: stat.color, fontFamily: 'var(--font-heading)' }}
                >
                  {stat.value}
                </div>
                <div className="text-[11px] text-slate-500">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Tabs & Search */}
      <div
        className="px-8 pt-4 pb-0 flex flex-wrap items-end justify-between gap-4"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}
      >
        <div className="flex gap-1">
          {tabs.map(tab => {
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className="flex items-center gap-1.5 px-4 py-2.5 text-[12px] font-semibold rounded-t-lg transition-all duration-150 relative"
                style={{
                  color: active ? '#A78BFA' : '#64748B',
                  background: active ? 'rgba(139,92,246,0.1)' : 'transparent',
                  borderBottom: active ? '2px solid #8B5CF6' : '2px solid transparent',
                }}
              >
                {tab.icon}
                {tab.label}
                <span
                  className="ml-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold"
                  style={{
                    background: active ? 'rgba(139,92,246,0.2)' : 'rgba(255,255,255,0.06)',
                    color: active ? '#A78BFA' : '#475569',
                  }}
                >
                  {tab.count}
                </span>
              </button>
            );
          })}
        </div>

        <div className="pb-2">
          <input
            type="text"
            placeholder="Search items..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="px-3 py-1.5 rounded-lg text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500"
            style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.1)',
              color: 'var(--color-text-primary)',
              width: '200px'
            }}
          />
        </div>
      </div>

      {/* Tab content */}
      <div className="px-8 py-6">

        {/* ── Uploaded Papers ── */}
        {activeTab === 'uploaded' && (
          uploadedPapers.length === 0 && !search
            ? <EmptyUploadState />
            : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredUploaded.map(p => <UploadedPaperCard key={p.id} paper={p} />)}
                {filteredUploaded.length === 0 && search && <div className="col-span-full text-center py-10 text-slate-500 text-sm">No uploaded papers found for &quot;{search}&quot;</div>}
              </div>
            )
        )}

        {/* ── Paper Library ── */}
        {activeTab === 'library' && (
          <div id="library" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredLibrary.map(p => <ContentPaperCard key={p.slug} paper={p} />)}
            {filteredLibrary.length === 0 && search && <div className="col-span-full text-center py-10 text-slate-500 text-sm">No library papers found for &quot;{search}&quot;</div>}
          </div>
        )}

        {/* ── Implementations ── */}
        {activeTab === 'implementations' && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredImpls.map(i => <ImplCard key={i.slug} impl={i} />)}
            {filteredImpls.length === 0 && search && <div className="col-span-full text-center py-10 text-slate-500 text-sm">No implementations found for &quot;{search}&quot;</div>}
          </div>
        )}

        {/* ── Research Collections ── */}
        {activeTab === 'collections' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {COLLECTIONS.map(col => <CollectionSection key={col.id} col={col} />)}
          </div>
        )}

      </div>
    </div>
  );
}
