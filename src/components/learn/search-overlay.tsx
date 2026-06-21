'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import { Search, X, BookOpen, Network, FileText, Server, Map, Code2, ArrowRight, Command } from 'lucide-react';

type Category = 'all' | 'lessons' | 'architectures' | 'papers' | 'system-design' | 'roadmaps' | 'projects';

const CATEGORIES: { id: Category; label: string; icon: React.ElementType }[] = [
  { id: 'all',          label: 'All',           icon: Search },
  { id: 'lessons',      label: 'Lessons',       icon: BookOpen },
  { id: 'architectures',label: 'Architectures', icon: Network },
  { id: 'papers',       label: 'Papers',        icon: FileText },
  { id: 'system-design',label: 'System Design', icon: Server },
  { id: 'roadmaps',     label: 'Roadmaps',      icon: Map },
  { id: 'projects',     label: 'Projects',      icon: Code2 },
];

const MOCK_RESULTS = [
  { id: '1', title: 'Multi-Head Attention',             cat: 'lessons',       domain: 'Deep Learning', color: '#06B6D4', href: '/learn/mha' },
  { id: '2', title: 'Transformer Architecture',         cat: 'architectures', domain: 'Deep Learning', color: '#8B5CF6', href: '/architectures/transformer' },
  { id: '3', title: 'Attention Is All You Need',        cat: 'papers',        domain: 'Research',      color: '#34D399', href: '/papers/attention' },
  { id: '4', title: 'LLM Serving System Design',        cat: 'system-design', domain: 'MLOps',         color: '#F59E0B', href: '/system-design/llm-serving' },
  { id: '5', title: 'Become an LLM Engineer',           cat: 'roadmaps',      domain: 'Career',        color: '#A78BFA', href: '/roadmaps/llm-engineer' },
  { id: '6', title: 'Scaled Dot-Product Attention',     cat: 'lessons',       domain: 'Deep Learning', color: '#06B6D4', href: '/learn/sdpa' },
  { id: '7', title: 'GPT-2 from Scratch',               cat: 'projects',      domain: 'LLMs',          color: '#EC4899', href: '/paper-to-code/gpt2' },
  { id: '8', title: 'BERT: Pre-training Deep Bidir.',   cat: 'papers',        domain: 'Research',      color: '#34D399', href: '/papers/bert' },
];

interface SearchOverlayProps {
  open: boolean;
  onClose: () => void;
}

export function SearchOverlay({ open, onClose }: SearchOverlayProps) {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState<Category>('all');
  const inputRef = useRef<HTMLInputElement>(null);

  /* Focus input when opened */
  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 60);
      setQuery('');
      setCategory('all');
    }
  }, [open]);

  /* Close on Escape */
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const filtered = MOCK_RESULTS.filter(r => {
    const matchQ = !query || r.title.toLowerCase().includes(query.toLowerCase());
    const matchC = category === 'all' || r.cat === category;
    return matchQ && matchC;
  });

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 z-50"
            style={{ background: 'rgba(5,8,22,0.85)', backdropFilter: 'blur(6px)' }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Panel */}
          <motion.div
            className="fixed top-[12vh] left-1/2 z-50 w-full max-w-xl -translate-x-1/2 rounded-2xl overflow-hidden"
            style={{
              background: 'rgba(10,15,30,0.98)',
              border: '1px solid rgba(139,92,246,0.3)',
              boxShadow: '0 0 60px rgba(139,92,246,0.2), 0 24px 64px rgba(0,0,0,0.6)',
            }}
            initial={{ opacity: 0, y: -20, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.97 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
          >
            {/* Search input */}
            <div className="flex items-center gap-3 px-4 py-3.5"
              style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
              <Search size={16} style={{ color: '#8B5CF6', flexShrink: 0 }} />
              <input
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Search lessons, papers, architectures…"
                className="flex-1 bg-transparent text-sm text-white placeholder-slate-600 outline-none"
                style={{ caretColor: '#8B5CF6' }}
              />
              {query && (
                <button onClick={() => setQuery('')}>
                  <X size={14} style={{ color: '#475569' }} />
                </button>
              )}
              <button onClick={onClose}
                className="flex items-center gap-1 px-2 py-1 rounded text-[10px] text-slate-600"
                style={{ background: 'rgba(255,255,255,0.06)' }}>
                <span>ESC</span>
              </button>
            </div>

            {/* Category tabs */}
            <div className="flex gap-1 px-3 py-2 overflow-x-auto"
              style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', scrollbarWidth: 'none' }}>
              {CATEGORIES.map(cat => {
                const Icon = cat.icon;
                const active = category === cat.id;
                return (
                  <button key={cat.id}
                    onClick={() => setCategory(cat.id)}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-semibold flex-none transition-all duration-150"
                    style={{
                      background: active ? 'rgba(139,92,246,0.18)' : 'transparent',
                      color: active ? '#A78BFA' : '#475569',
                      border: active ? '1px solid rgba(139,92,246,0.3)' : '1px solid transparent',
                    }}>
                    <Icon size={11} />
                    {cat.label}
                  </button>
                );
              })}
            </div>

            {/* Results */}
            <div className="max-h-[320px] overflow-y-auto" style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(139,92,246,0.2) transparent' }}>
              {filtered.length > 0 ? (
                <div className="py-1">
                  {filtered.map(r => (
                    <Link key={r.id} href={r.href} onClick={onClose}
                      className="flex items-center gap-3 px-4 py-2.5 transition-all duration-150"
                      style={{ cursor: 'pointer' }}
                      onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)'}
                      onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = 'transparent'}
                    >
                      <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
                        style={{ background: `${r.color}18`, color: r.color }}>
                        <Search size={11} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-[13px] font-semibold text-white truncate">{r.title}</div>
                        <div className="flex items-center gap-2 text-[10px] text-slate-600">
                          <span>{r.domain}</span>
                          <span>·</span>
                          <span className="capitalize">{r.cat.replace('-', ' ')}</span>
                        </div>
                      </div>
                      <ArrowRight size={12} style={{ color: '#334155', flexShrink: 0 }} />
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center py-10 text-slate-600">
                  <Search size={24} className="mb-2 opacity-30" />
                  <span className="text-sm">No results for &ldquo;{query}&rdquo;</span>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between px-4 py-2.5"
              style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
              <div className="flex items-center gap-3 text-[10px] text-slate-700">
                <span className="flex items-center gap-1">
                  <Command size={9} /><span>K</span> <span>to open</span>
                </span>
                <span>↑↓ to navigate</span>
                <span>↵ to select</span>
              </div>
              <span className="text-[10px] text-slate-700">{filtered.length} results</span>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
