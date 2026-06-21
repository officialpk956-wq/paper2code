'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { Search, ArrowRight } from 'lucide-react';
import { search, SearchResult, TYPE_COLORS, TYPE_LABELS } from '@/lib/search/engine';

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  // Open/close via CMD+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
      if (e.key === 'Escape') {
        setOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Focus input when opened
  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 10);
      setQuery('');
      setResults([]);
      setActiveIndex(0);
    }
  }, [open]);

  const handleQuery = useCallback((q: string) => {
    setQuery(q);
    setActiveIndex(0);
    if (q.trim().length === 0) {
      setResults([]);
      return;
    }
    setResults(search(q, { limit: 12 }));
  }, []);

  const handleSelect = useCallback((href: string) => {
    setOpen(false);
    setQuery('');
    setResults([]);
    router.push(href);
  }, [router]);

  // Keyboard navigation within results
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter' && results[activeIndex]) {
      handleSelect(results[activeIndex].href);
    }
  };

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/50 animate-in fade-in-0 duration-200"
        onClick={() => setOpen(false)}
        aria-hidden="true"
      />

      {/* Palette */}
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-2xl
          animate-in fade-in-0 slide-in-from-top-4 duration-300"
      >
        <div className="bg-[--bg-surface] border border-[--color-border] rounded-lg shadow-2xl overflow-hidden">
          {/* Search Input */}
          <div className="flex items-center gap-3 px-4 py-3 border-b border-[--color-border] bg-[--bg-body]">
            <Search className="w-5 h-5 text-[--color-text-tertiary]" aria-hidden="true" />
            <input
              ref={inputRef}
              type="search"
              placeholder="Search papers, problems, architectures…"
              value={query}
              onChange={(e) => handleQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              aria-label="Search"
              aria-autocomplete="list"
              aria-controls="cmd-palette-results"
              aria-activedescendant={
                results[activeIndex] ? `cmd-result-${results[activeIndex].id}` : undefined
              }
              className="flex-1 bg-transparent outline-none text-[--color-text-primary]
                placeholder-[--color-text-tertiary]"
            />
            <kbd
              className="text-xs text-[--color-text-tertiary] px-2 py-1 bg-[--bg-surface]
                rounded border border-[--color-border]"
              aria-label="Escape key to close"
            >
              ESC
            </kbd>
          </div>

          {/* Results */}
          <div
            id="cmd-palette-results"
            role="listbox"
            aria-label="Search results"
            className="max-h-96 overflow-y-auto"
          >
            {results.length === 0 && query.trim().length === 0 && (
              <div className="p-6 text-center">
                <p className="text-[--color-text-tertiary] text-sm">
                  Start typing to search across problems, papers, architectures, and more.
                </p>
              </div>
            )}

            {results.length === 0 && query.trim().length > 0 && (
              <div className="p-6 text-center">
                <p className="text-[--color-text-tertiary] text-sm">
                  No results for &ldquo;{query}&rdquo;
                </p>
              </div>
            )}

            {results.length > 0 && (
              <ul>
                {results.map((result, i) => (
                  <li key={result.id}>
                    <button
                      id={`cmd-result-${result.id}`}
                      role="option"
                      aria-selected={i === activeIndex}
                      onClick={() => handleSelect(result.href)}
                      className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors
                        ${i === activeIndex
                          ? 'bg-[--bg-active] text-[--color-text-primary]'
                          : 'hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary]'
                        }`}
                    >
                      {/* Type indicator */}
                      <span
                        className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                        style={{ backgroundColor: TYPE_COLORS[result.type] }}
                        aria-hidden="true"
                      />
                      <div className="flex-1 min-w-0">
                        <span className="text-sm font-medium truncate block">{result.title}</span>
                        <span className="text-xs text-[--color-text-tertiary] truncate block">
                          {TYPE_LABELS[result.type]} · {result.description.slice(0, 60)}
                          {result.description.length > 60 ? '…' : ''}
                        </span>
                      </div>
                      <ArrowRight
                        className="w-4 h-4 flex-shrink-0 opacity-50"
                        aria-hidden="true"
                      />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Footer */}
          <div className="px-4 py-2.5 border-t border-[--color-border] bg-[--bg-body] text-xs
            text-[--color-text-tertiary] flex items-center gap-4">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded bg-[--bg-surface] border border-[--color-border]">↑↓</kbd>
              navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded bg-[--bg-surface] border border-[--color-border]">↵</kbd>
              open
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded bg-[--bg-surface] border border-[--color-border]">ESC</kbd>
              close
            </span>
          </div>
        </div>
      </div>
    </>
  );
}
