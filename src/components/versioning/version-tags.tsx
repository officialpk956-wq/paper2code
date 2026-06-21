'use client';

import { Tag, Plus, Trash2, Star } from 'lucide-react';
import { useState } from 'react';

interface VersionTag {
  id: string;
  name: string;
  version: string;
  description: string;
  createdAt: string;
  author: string;
  isPinned: boolean;
  color: string;
}

const tags: VersionTag[] = [
  {
    id: 'tag-1',
    name: 'v1.0-release',
    version: 'v1.0 (24 commits)',
    description: 'First stable release with multi-head attention',
    createdAt: '5 days ago',
    author: 'You',
    isPinned: true,
    color: 'bg-green-500/20 border-green-500/30',
  },
  {
    id: 'tag-2',
    name: 'attention-optimized',
    version: 'v1.2 (current)',
    description: 'Optimized attention mechanism with bias=False',
    createdAt: '2 hours ago',
    author: 'You',
    isPinned: true,
    color: 'bg-blue-500/20 border-blue-500/30',
  },
  {
    id: 'tag-3',
    name: 'before-refactor',
    version: 'v0.9 (18 commits)',
    description: 'Backup before major refactoring',
    createdAt: '1 day ago',
    author: 'Sarah Chen',
    isPinned: false,
    color: 'bg-yellow-500/20 border-yellow-500/30',
  },
  {
    id: 'tag-4',
    name: 'scaling-experiment',
    version: 'v1.1 (22 commits)',
    description: 'Testing different model sizes (Small, Base, Large)',
    createdAt: '3 days ago',
    author: 'Alex Kumar',
    isPinned: false,
    color: 'bg-purple-500/20 border-purple-500/30',
  },
];

interface VersionTagsProps {
  workspaceName?: string;
}

export function VersionTags({ workspaceName = 'Transformer Architecture' }: VersionTagsProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [pinnedTags, setPinnedTags] = useState(
    tags.filter((t) => t.isPinned).map((t) => t.id)
  );

  const togglePin = (id: string) => {
    setPinnedTags((prev) =>
      prev.includes(id) ? prev.filter((t) => t !== id) : [...prev, id]
    );
  };

  const pinnedTagsList = tags.filter((t) => pinnedTags.includes(t.id));
  const unpinnedTagsList = tags.filter((t) => !pinnedTags.includes(t.id));

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-r border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-1">
          Version Tags
        </h3>
        <p className="text-xs text-[--color-text-secondary]">
          {workspaceName}
        </p>
      </div>

      {/* Tags List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Pinned Tags */}
        {pinnedTagsList.length > 0 && (
          <div>
            <div className="text-xs font-semibold text-[--color-text-tertiary] mb-2 px-2">
              PINNED ({pinnedTagsList.length})
            </div>
            <div className="space-y-2">
              {pinnedTagsList.map((tag) => (
                <div
                  key={tag.id}
                  onClick={() => setExpandedId(expandedId === tag.id ? null : tag.id)}
                  className={`group p-3 rounded-lg border-2 cursor-pointer transition-all ${tag.color}`}
                >
                  {/* Tag Header */}
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <Tag size={14} className="flex-shrink-0" />
                      <div className="min-w-0">
                        <div className="text-sm font-semibold text-[--color-text-primary] truncate">
                          {tag.name}
                        </div>
                        <div className="text-xs text-[--color-text-tertiary] truncate">
                          {tag.version}
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        togglePin(tag.id);
                      }}
                      className="p-1 rounded hover:bg-[--bg-body] text-yellow-400 flex-shrink-0"
                    >
                      <Star size={14} fill="currentColor" />
                    </button>
                  </div>

                  {/* Expanded Details */}
                  {expandedId === tag.id && (
                    <div className="mt-2 pt-2 border-t border-[--color-border] space-y-2">
                      <p className="text-xs text-[--color-text-secondary]">
                        {tag.description}
                      </p>
                      <div className="text-xs text-[--color-text-tertiary] space-y-1">
                        <div>Created {tag.createdAt} by {tag.author}</div>
                      </div>
                      <div className="flex gap-2 mt-2 pt-2 border-t border-[--color-border]">
                        <button className="flex-1 px-2 py-1 rounded text-xs font-medium bg-[--bg-body] hover:bg-[--bg-surface] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
                          View Version
                        </button>
                        <button className="px-2 py-1 rounded text-xs font-medium bg-red-500/10 hover:bg-red-500/20 text-red-400 transition-colors">
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Unpinned Tags */}
        {unpinnedTagsList.length > 0 && (
          <div>
            <div className="text-xs font-semibold text-[--color-text-tertiary] mb-2 px-2">
              ALL TAGS ({unpinnedTagsList.length})
            </div>
            <div className="space-y-2">
              {unpinnedTagsList.map((tag) => (
                <div
                  key={tag.id}
                  onClick={() => setExpandedId(expandedId === tag.id ? null : tag.id)}
                  className={`group p-3 rounded-lg border cursor-pointer transition-all ${tag.color}`}
                >
                  {/* Tag Header */}
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <Tag size={14} className="flex-shrink-0" />
                      <div className="min-w-0">
                        <div className="text-sm font-semibold text-[--color-text-primary] truncate">
                          {tag.name}
                        </div>
                        <div className="text-xs text-[--color-text-tertiary] truncate">
                          {tag.version}
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        togglePin(tag.id);
                      }}
                      className="p-1 rounded hover:bg-[--bg-body] text-[--color-text-tertiary] hover:text-yellow-400 transition-colors flex-shrink-0"
                    >
                      <Star size={14} />
                    </button>
                  </div>

                  {/* Expanded Details */}
                  {expandedId === tag.id && (
                    <div className="mt-2 pt-2 border-t border-[--color-border] space-y-2">
                      <p className="text-xs text-[--color-text-secondary]">
                        {tag.description}
                      </p>
                      <div className="text-xs text-[--color-text-tertiary]">
                        Created {tag.createdAt} by {tag.author}
                      </div>
                      <div className="flex gap-2 mt-2 pt-2 border-t border-[--color-border]">
                        <button className="flex-1 px-2 py-1 rounded text-xs font-medium bg-[--bg-body] hover:bg-[--bg-surface] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
                          View Version
                        </button>
                        <button className="px-2 py-1 rounded text-xs font-medium bg-red-500/10 hover:bg-red-500/20 text-red-400 transition-colors">
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body]">
        <button className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-medium bg-[--accent-primary] hover:opacity-90 text-white transition-opacity">
          <Plus size={12} />
          New Tag
        </button>
      </div>
    </div>
  );
}
