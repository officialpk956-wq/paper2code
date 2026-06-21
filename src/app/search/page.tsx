'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Search as SearchIcon, ArrowRight } from 'lucide-react';
import Link from 'next/link';
import {
  search,
  highlight,
  TYPE_LABELS,
  TYPE_COLORS,
  SearchResult,
  SearchResultType,
} from '@/lib/search/engine';

const ALL_TYPES: SearchResultType[] = [
  'problem',
  'architecture',
  'roadmap',
  'track',
  'topic',
  'navigation',
];

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [activeType, setActiveType] = useState<SearchResultType | 'all'>('all');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleQuery = useCallback((q: string) => {
    setQuery(q);
    if (q.trim().length === 0) {
      setResults([]);
      return;
    }
    const types = activeType === 'all' ? undefined : [activeType];
    setResults(search(q, { limit: 40, types }));
  }, [activeType]);

  // Re-run when filter changes with existing query
  useEffect(() => {
    if (query.trim().length > 0) {
      const types = activeType === 'all' ? undefined : [activeType];
      setResults(search(query, { limit: 40, types }));
    }
  }, [activeType, query]);

  const grouped = ALL_TYPES.reduce<Record<SearchResultType, SearchResult[]>>(
    (acc, t) => {
      acc[t] = results.filter((r) => r.type === t);
      return acc;
    },
    { problem: [], architecture: [], roadmap: [], track: [], topic: [], navigation: [] }
  );

  const hasResults = results.length > 0;

  return (
    <div className="flex flex-col h-[calc(100vh-56px)] bg-[--bg-body]" role="search" aria-label="Site search">
      {/* Search Header */}
      <div className="px-8 py-6 border-b border-[--color-border] bg-[--bg-panel]">
        <h1 className="text-2xl font-bold text-[--color-text-primary] mb-4">Search</h1>
        <div className="relative max-w-2xl">
          <SearchIcon
            size={18}
            className="absolute left-3 top-3 text-[--color-text-tertiary]"
            aria-hidden="true"
          />
          <input
            ref={inputRef}
            type="search"
            placeholder="Search papers, problems, architectures, roadmaps…"
            value={query}
            onChange={(e) => handleQuery(e.target.value)}
            aria-label="Search"
            className="w-full pl-10 pr-4 py-2.5 bg-[--bg-body] border border-[--color-border] rounded-lg
              text-sm text-[--color-text-primary] placeholder-[--color-text-tertiary]
              focus:border-[--accent-primary] focus:outline-none transition-colors"
          />
        </div>

        {/* Type filter pills */}
        {hasResults && (
          <div
            className="flex gap-2 mt-3 flex-wrap"
            role="group"
            aria-label="Filter by type"
          >
            <button
              onClick={() => setActiveType('all')}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                activeType === 'all'
                  ? 'bg-[--accent-primary] text-white'
                  : 'bg-[--bg-surface] text-[--color-text-secondary] hover:text-[--color-text-primary]'
              }`}
            >
              All ({results.length})
            </button>
            {ALL_TYPES.filter((t) => grouped[t].length > 0).map((t) => (
              <button
                key={t}
                onClick={() => setActiveType(t)}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  activeType === t ? 'text-white' : 'bg-[--bg-surface] text-[--color-text-secondary] hover:text-[--color-text-primary]'
                }`}
                style={{ backgroundColor: activeType === t ? TYPE_COLORS[t] : undefined }}
              >
                {TYPE_LABELS[t]} ({grouped[t].length})
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto">
        {!hasResults && query.trim().length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-8">
            <div className="w-16 h-16 rounded-full bg-[--bg-panel] border border-[--color-border] flex items-center justify-center">
              <SearchIcon size={28} className="text-[--color-text-tertiary]" aria-hidden="true" />
            </div>
            <h2 className="text-lg font-semibold text-[--color-text-primary]">Start searching</h2>
            <p className="text-sm text-[--color-text-secondary] max-w-md">
              Search across {/* approximate count */ 'problems, papers, architectures, roadmaps, and pages'}.
            </p>
          </div>
        )}

        {!hasResults && query.trim().length > 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-8">
            <div className="w-16 h-16 rounded-full bg-[--bg-panel] border border-[--color-border] flex items-center justify-center">
              <SearchIcon size={28} className="text-[--color-text-tertiary]" aria-hidden="true" />
            </div>
            <h2 className="text-lg font-semibold text-[--color-text-primary]">
              No results for &ldquo;{query}&rdquo;
            </h2>
            <p className="text-sm text-[--color-text-secondary] max-w-md">
              Try different keywords or browse content by category using the left navigation.
            </p>
          </div>
        )}

        {hasResults && (
          <div className="max-w-3xl mx-auto px-8 py-6 space-y-8">
            {(activeType === 'all' ? ALL_TYPES : [activeType]).map((type) => {
              const group = grouped[type];
              if (group.length === 0) return null;
              return (
                <section key={type} aria-label={`${TYPE_LABELS[type]} results`}>
                  <div className="flex items-center gap-2 mb-3">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: TYPE_COLORS[type] }}
                      aria-hidden="true"
                    />
                    <h2 className="text-xs font-semibold uppercase tracking-widest text-[--color-text-tertiary]">
                      {TYPE_LABELS[type]}
                    </h2>
                    <span className="text-xs text-[--color-text-tertiary]">({group.length})</span>
                  </div>
                  <ul className="space-y-2" role="list">
                    {group.map((result) => (
                      <li key={result.id}>
                        <Link
                          href={result.href}
                          className="group flex items-start justify-between gap-4 p-3 rounded-lg
                            border border-[--color-border] bg-[--bg-panel]
                            hover:border-[--accent-primary]/50 hover:bg-[--bg-hover]
                            transition-colors"
                        >
                          <div className="min-w-0 flex-1">
                            <p
                              className="text-sm font-semibold text-[--color-text-primary]
                                group-hover:text-[--accent-primary] transition-colors truncate"
                              /* Safe: highlight() escapes HTML before inserting <mark> */
                              dangerouslySetInnerHTML={{
                                __html: highlight(result.title, query),
                              }}
                            />
                            <p
                              className="text-xs text-[--color-text-secondary] mt-0.5 line-clamp-2"
                              dangerouslySetInnerHTML={{
                                __html: highlight(result.description, query),
                              }}
                            />
                            {result.tags && result.tags.length > 0 && (
                              <div className="flex gap-1 mt-1.5 flex-wrap">
                                {result.tags.slice(0, 3).map((tag) => (
                                  <span
                                    key={tag}
                                    className="px-1.5 py-0.5 rounded text-[10px] bg-[--bg-surface]
                                      text-[--color-text-tertiary] border border-[--color-border]"
                                  >
                                    {tag}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                          <ArrowRight
                            size={16}
                            className="text-[--color-text-tertiary] group-hover:text-[--accent-primary]
                              transition-colors flex-shrink-0 mt-0.5"
                            aria-hidden="true"
                          />
                        </Link>
                      </li>
                    ))}
                  </ul>
                </section>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
