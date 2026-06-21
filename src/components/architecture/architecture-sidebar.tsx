'use client';

import { Search, ChevronDown, Star } from 'lucide-react';
import { useState, useMemo } from 'react';
import { ARCHITECTURE_CATALOG, ARCH_CATEGORIES, CatalogEntry } from '@/data/architecture-catalog';

interface SidebarProps {
  selectedSlug: string;
  onSelect: (slug: string) => void;
}

export function ArchitectureSidebar({ selectedSlug, onSelect }: SidebarProps) {
  const [expandedSections, setExpandedSections] = useState<string[]>(ARCH_CATEGORIES);
  const [searchQuery, setSearchQuery] = useState('');

  const toggleSection = (category: string) => {
    setExpandedSections((prev) =>
      prev.includes(category) ? prev.filter((s) => s !== category) : [...prev, category]
    );
  };

  const groupedEntries = useMemo(() => {
    const filtered = ARCHITECTURE_CATALOG.filter(entry => {
      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase();
      return entry.title.toLowerCase().includes(q) || entry.tags.some(t => t.toLowerCase().includes(q));
    });

    const groups: Record<string, CatalogEntry[]> = {};
    for (const cat of ARCH_CATEGORIES) {
      groups[cat] = [];
    }
    for (const entry of filtered) {
      if (groups[entry.category]) {
        groups[entry.category].push(entry);
      }
    }
    return groups;
  }, [searchQuery]);

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-r border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-3">
          Explore Architecture
        </h3>
        <div className="relative">
          <Search size={16} className="absolute left-2 top-2.5 text-[--color-text-tertiary]" />
          <input
            type="text"
            placeholder="Search components..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-2 bg-[--bg-body] border border-[--color-border] rounded text-xs
                     text-[--color-text-primary] placeholder-[--color-text-tertiary]
                     focus:border-[--accent-primary] focus:outline-none transition-colors"
          />
        </div>
      </div>

      {/* Sections */}
      <div className="flex-1 overflow-y-auto">
        {ARCH_CATEGORIES.map((category) => {
          const items = groupedEntries[category];
          if (items.length === 0) return null;

          return (
            <div key={category} className="border-b border-[--color-border]">
              <button
                onClick={() => toggleSection(category)}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-[--bg-surface] transition-colors"
              >
                <span className="text-xs font-semibold text-[--color-text-secondary]">
                  {category}
                </span>
                <ChevronDown
                  size={16}
                  className={`text-[--color-text-tertiary] transition-transform ${
                    expandedSections.includes(category) ? 'rotate-180' : ''
                  }`}
                />
              </button>

              {expandedSections.includes(category) && (
                <div className="bg-[--bg-body] px-2 py-2 space-y-1">
                  {items.map((item) => {
                    const isSelected = item.slug === selectedSlug;
                    return (
                      <button
                        key={item.slug}
                        onClick={() => onSelect(item.slug)}
                        className={`w-full flex flex-col items-start gap-1 px-3 py-2 rounded text-xs transition-colors group ${
                          isSelected
                            ? 'bg-[--accent-primary]/10 text-[--accent-primary]'
                            : 'text-[--color-text-secondary] hover:bg-[--bg-surface] hover:text-[--color-text-primary]'
                        }`}
                      >
                        <div className="w-full flex items-center justify-between">
                          <div className="flex items-center gap-2 truncate">
                            <span className="w-4 h-4 rounded bg-gradient-to-br from-[--accent-primary] to-[--accent-cyan] flex-shrink-0" />
                            <span className="truncate font-medium">{item.title}</span>
                          </div>
                          {item.status === 'coming-soon' && (
                            <span className="ml-2 text-[9px] px-1.5 py-0.5 rounded border border-[--color-border] text-[--color-text-tertiary] uppercase font-bold tracking-wider whitespace-nowrap">
                              Soon
                            </span>
                          )}
                          {item.status === 'complete' && !isSelected && (
                            <Star
                              size={12}
                              className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity text-[--accent-primary] flex-shrink-0"
                            />
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body]">
        <p className="text-xs text-[--color-text-tertiary] leading-relaxed">
          Click any component to explore its implementation, math, and code.
        </p>
      </div>
    </div>
  );
}
