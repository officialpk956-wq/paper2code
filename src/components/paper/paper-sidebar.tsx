'use client';

import { Clock, Bookmark } from 'lucide-react';
import { useState } from 'react';
import Link from 'next/link';

interface PaperSection {
  id: string;
  title: string;
  icon: string;
  readTime: number;
}

const sections: PaperSection[] = [
  { id: 'abstract', title: 'Abstract', icon: '📋', readTime: 2 },
  { id: 'intro', title: 'Introduction', icon: '🎯', readTime: 8 },
  { id: 'related', title: 'Related Work', icon: '📚', readTime: 5 },
  { id: 'method', title: 'Methodology', icon: '🔧', readTime: 15 },
  { id: 'experiments', title: 'Experiments', icon: '📊', readTime: 10 },
  { id: 'results', title: 'Results & Discussion', icon: '📈', readTime: 8 },
  { id: 'conclusion', title: 'Conclusion', icon: '✅', readTime: 3 },
];

interface PaperSidebarProps {
  onSelectSection?: (id: string) => void;
  currentSection?: string;
}

export function PaperSidebar({ onSelectSection, currentSection = 'abstract' }: PaperSidebarProps) {
  const [bookmarked, setBookmarked] = useState<string[]>([]);

  const toggleBookmark = (id: string) => {
    setBookmarked((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const totalReadTime = sections.reduce((sum, s) => sum + s.readTime, 0);

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-r border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-3">
          Paper Outline
        </h3>
        <div className="flex items-center gap-2 px-3 py-2 bg-[--bg-body] rounded text-xs text-[--color-text-secondary]">
          <Clock size={14} />
          <span>{totalReadTime} min read</span>
        </div>
      </div>

      {/* Sections */}
      <div className="flex-1 overflow-y-auto">
        {sections.map((section, index) => (
          <div key={section.id}>
            {/* Progress line */}
            {index > 0 && (
              <div className="px-4 pt-2">
                <div className="h-4 flex items-center">
                  <div className="w-0.5 h-full bg-[--color-border]" />
                </div>
              </div>
            )}

            {/* Section button */}
            <button
              onClick={() => onSelectSection?.(section.id)}
              className={`w-full flex items-start gap-3 px-4 py-3 transition-colors ${
                currentSection === section.id
                  ? 'bg-[--bg-surface] border-l-2 border-[--accent-primary]'
                  : 'hover:bg-[--bg-surface] border-l-2 border-transparent'
              }`}
            >
              {/* Dot */}
              <div className={`flex-shrink-0 w-2.5 h-2.5 rounded-full mt-1 ${
                currentSection === section.id
                  ? 'bg-[--accent-primary]'
                  : 'bg-[--color-border]'
              }`} />

              {/* Content */}
              <div className="flex-1 text-left min-w-0">
                <div className="text-xs font-semibold text-[--color-text-primary] mb-0.5">
                  {section.icon} {section.title}
                </div>
                <div className="text-xs text-[--color-text-tertiary]">
                  {section.readTime} min
                </div>
              </div>

              {/* Bookmark button */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  toggleBookmark(section.id);
                }}
                className="flex-shrink-0 p-1 hover:bg-[--bg-body] rounded transition-colors"
              >
                <Bookmark
                  size={14}
                  className={bookmarked.includes(section.id) ? 'fill-[--accent-primary] text-[--accent-primary]' : 'text-[--color-text-tertiary]'}
                />
              </button>
            </button>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body]">
        <div className="space-y-2">
          <Link
            href="/papers/upload"
            className="block w-full rounded px-3 py-2 text-center text-xs font-medium text-[--accent-cyan] hover:bg-[--bg-surface] transition-colors"
          >
            Upload PDF
          </Link>
          <button className="w-full px-3 py-2 text-xs font-medium text-[--accent-primary] hover:bg-[--bg-surface] rounded transition-colors">
            Download PDF
          </button>
        </div>
      </div>
    </div>
  );
}
