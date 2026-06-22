'use client';

import { Copy, ExternalLink, Info, BookOpen } from 'lucide-react';
import { useState, useEffect } from 'react';
import { ARCHITECTURE_CATALOG } from '@/data/architecture-catalog';
import Link from 'next/link';

interface InspectorTab {
  id: string;
  label: string;
  icon: string;
}

const tabs: InspectorTab[] = [
  { id: 'math', label: 'Math', icon: '∑' },
  { id: 'code', label: 'Code', icon: '</>' },
  { id: 'paper', label: 'Paper', icon: '📄' },
];

interface InspectorProps {
  selectedSlug: string;
  selectedBlockId: string | null;
}

export function ArchitectureInspector({ selectedSlug, selectedBlockId }: InspectorProps) {
  const [activeTab, setActiveTab] = useState('math');
  const entry = ARCHITECTURE_CATALOG.find((e) => e.slug === selectedSlug);

  // If entry changes and active tab has no content, try to find one that does
  useEffect(() => {
    if (!entry) return;
    if (activeTab === 'math' && !entry.mathSnippet) setActiveTab('code');
    if (activeTab === 'code' && !entry.codeSnippet) setActiveTab('paper');
    if (activeTab === 'paper' && (!entry.papers || entry.papers.length === 0)) setActiveTab('math');
  }, [selectedSlug, entry, activeTab]);

  if (!entry) {
    return (
      <div className="flex flex-col h-full bg-[--bg-panel] border-l border-[--color-border] items-center justify-center p-6 text-center text-[--color-text-tertiary]">
        <Info size={24} className="mb-2" />
        <p className="text-sm">Select an architecture to inspect.</p>
      </div>
    );
  }

  const selectedBlock = entry.diagram?.find((b) => b.id === selectedBlockId);

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-l border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-1">
          {selectedBlock ? selectedBlock.label : entry.title}
        </h3>
        {selectedBlock?.sublabel && (
          <p className="text-[10px] text-[--color-text-tertiary] font-mono mb-3 uppercase tracking-wide">
            {selectedBlock.sublabel}
          </p>
        )}
        {!selectedBlock && entry.status === 'complete' && (
          <div className="flex gap-1 bg-[--bg-body] rounded p-1 mt-3">
          {tabs.map((tab) => {
            const isDisabled =
              (tab.id === 'math' && !entry.mathSnippet) ||
              (tab.id === 'code' && !entry.codeSnippet) ||
              (tab.id === 'paper' && (!entry.papers || entry.papers.length === 0));

            return (
              <button
                key={tab.id}
                disabled={isDisabled}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-[--accent-primary] text-white'
                    : isDisabled
                    ? 'text-[--color-text-tertiary] opacity-50 cursor-not-allowed'
                    : 'text-[--color-text-secondary] hover:text-[--color-text-primary]'
                }`}
              >
                <span className="mr-1">{tab.icon}</span>
                {tab.label}
              </button>
            );
          })}
        </div>
        )}
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-4">
        {selectedBlock ? (
          <div className="space-y-4">
            <div className="bg-gradient-to-br from-[--bg-body] to-[--bg-surface] rounded-lg p-4 border border-[--color-border]">
              <p className="text-sm text-[--color-text-primary] leading-relaxed">
                {selectedBlock.description}
              </p>
            </div>
            {selectedBlock.outputShape && (
              <div className="flex items-center gap-2 text-xs">
                <span className="text-[--color-text-tertiary]">Output Shape:</span>
                <span className="font-mono text-[--accent-cyan] bg-[rgb(6,182,212,0.1)] px-1.5 py-0.5 rounded">
                  {selectedBlock.outputShape}
                </span>
              </div>
            )}
          </div>
        ) : entry.status === 'coming-soon' ? (
          <div className="space-y-6">
            {entry.keyFacts && entry.keyFacts.length > 0 && (
              <div className="space-y-3">
                <h4 className="text-xs font-bold uppercase tracking-wider text-[--color-text-secondary]">
                  Key Facts
                </h4>
                <div className="space-y-2">
                  {entry.keyFacts.map((fact) => (
                    <div key={fact.label} className="bg-[--bg-body] border border-[--color-border] rounded p-2 text-xs">
                      <span className="text-[--color-text-tertiary] block mb-0.5">{fact.label}</span>
                      <span className="text-[--color-text-primary] font-medium">{fact.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div className="text-center p-4 rounded-lg bg-[--bg-body] border border-[--color-border]">
              <p className="text-xs text-[--color-text-secondary]">
                Implementation and math details are coming soon.
              </p>
            </div>
          </div>
        ) : (
          <>
            {activeTab === 'math' && entry.mathSnippet && (
              <div className="space-y-4">
                <h4 className="text-xs font-bold uppercase tracking-wider text-[--color-text-secondary]">
                  {entry.mathSnippet.title}
                </h4>
                <div className="bg-[--bg-body] rounded p-3 border border-[--color-border]">
                  <div className="text-[11px] font-mono text-[--color-text-primary] whitespace-pre-wrap leading-relaxed">
                    {entry.mathSnippet.tex}
                  </div>
                </div>
                {entry.mathSnippet.note && (
                  <div className="text-xs text-[--color-text-tertiary] leading-relaxed border-l-2 border-[--accent-primary] pl-3 py-1">
                    {entry.mathSnippet.note}
                  </div>
                )}
                {entry.keyFacts && (
                  <div className="mt-6 space-y-2">
                    {entry.keyFacts.map((fact) => (
                      <div key={fact.label} className="flex justify-between items-center text-xs pb-2 border-b border-[--color-border]">
                        <span className="text-[--color-text-tertiary]">{fact.label}</span>
                        <span className="text-[--color-text-primary] font-medium">{fact.value}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'code' && entry.codeSnippet && (
              <div className="space-y-3">
                <div className="bg-[--bg-body] rounded overflow-hidden border border-[--color-border]">
                  <div className="px-3 py-2 bg-[--bg-surface] border-b border-[--color-border] flex items-center justify-between">
                    <span className="text-xs font-semibold text-[--color-text-secondary]">{entry.codeSnippet.title}</span>
                    <button className="p-1 hover:bg-[--bg-body] rounded transition-colors" aria-label="Copy code">
                      <Copy size={14} className="text-[--color-text-tertiary]" />
                    </button>
                  </div>
                  <pre className="p-3 text-[10px] font-mono text-[--color-text-primary] overflow-x-auto whitespace-pre-wrap">
                    {entry.codeSnippet.code}
                  </pre>
                </div>
              </div>
            )}

            {activeTab === 'paper' && entry.papers && entry.papers.length > 0 && (
              <div className="space-y-4">
                <div className="bg-gradient-to-br from-[--accent-primary]/10 to-[--accent-cyan]/10 rounded-lg p-4 border border-[--color-border]">
                  <h4 className="font-semibold text-sm text-[--color-text-primary] mb-1">
                    {entry.papers[0].title}
                  </h4>
                  <p className="text-xs text-[--color-text-secondary] mb-3">
                    Published in {entry.papers[0].year}
                  </p>
                  <p className="text-xs text-[--color-text-primary] leading-relaxed">
                    {entry.description}
                  </p>
                </div>
                <Link
                  href={`/papers/${entry.papers[0].slug}`}
                  className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[--bg-body] hover:bg-[--bg-surface] border border-[--color-border] rounded text-xs font-medium text-[--accent-primary] transition-colors"
                >
                  <ExternalLink size={14} />
                  Read Full Paper
                </Link>
              </div>
            )}
          </>
        )}
      </div>

      {/* Footer CTA */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body] space-y-2">
        <h4 className="text-xs font-semibold text-[--color-text-primary] mb-2">Take the Next Step</h4>
        {entry.papers && entry.papers.length > 0 && (
          <Link
            href={`/papers/${entry.papers[0].slug}`}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[--bg-surface] hover:bg-[--bg-panel] border border-[--color-border] rounded text-xs font-medium text-[--color-text-primary] transition-colors"
          >
            <ExternalLink size={14} />
            Read the Paper
          </Link>
        )}
        <Link
            href="/learn"
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[--accent-primary] hover:opacity-90 rounded text-xs font-medium text-white transition-opacity"
          >
            <BookOpen size={14} />
            Learn the Theory
        </Link>
      </div>
    </div>
  );
}
