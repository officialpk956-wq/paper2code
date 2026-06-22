'use client';

import { Plus, Share2, Clock, Trash2, Download, Globe } from 'lucide-react';
import { useState } from 'react';

interface Workspace {
  id: string;
  name: string;
  type: string;
  lastModified: string;
  owner: string;
  shared: boolean;
  starred: boolean;
}

const workspaces: Workspace[] = [
  {
    id: 'transformer-design',
    name: 'Transformer Architecture',
    type: 'Model Design',
    lastModified: '2 hours ago',
    owner: 'You',
    shared: false,
    starred: true,
  },
  {
    id: 'system-design-v2',
    name: 'Microservices System',
    type: 'System Design',
    lastModified: '1 day ago',
    owner: 'You',
    shared: true,
    starred: false,
  },
  {
    id: 'paper-impl-gpt3',
    name: 'GPT-3 Implementation',
    type: 'Paper-to-Code',
    lastModified: '3 days ago',
    owner: 'You',
    shared: false,
    starred: false,
  },
  {
    id: 'database-design-v3',
    name: 'Production Database',
    type: 'Schema Design',
    lastModified: '1 week ago',
    owner: 'You',
    shared: true,
    starred: false,
  },
];

interface WorkspaceManagerProps {
  onSelectWorkspace?: (id: string) => void;
  selectedWorkspace?: string;
}

export function WorkspaceManager({ onSelectWorkspace, selectedWorkspace }: WorkspaceManagerProps) {
  const [filterType, setFilterType] = useState<string | null>(null);

  const types = [...new Set(workspaces.map((w) => w.type))];
  const filtered = workspaces.filter((w) => !filterType || w.type === filterType);

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-r border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-[--color-text-primary]">
            Your Workspaces
          </h3>
          <button className="p-1.5 rounded hover:bg-[--bg-surface] text-[--accent-primary]">
            <Plus size={16} />
          </button>
        </div>

        {/* Type Filter */}
        <div className="flex flex-wrap gap-1">
          <button
            onClick={() => setFilterType(null)}
            className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
              filterType === null
                ? 'bg-[--accent-primary] text-white'
                : 'bg-[--bg-body] text-[--color-text-secondary] hover:text-[--color-text-primary]'
            }`}
          >
            All
          </button>
          {types.map((type) => (
            <button
              key={type}
              onClick={() => setFilterType(type)}
              className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                filterType === type
                  ? 'bg-[--accent-primary] text-white'
                  : 'bg-[--bg-body] text-[--color-text-secondary] hover:text-[--color-text-primary]'
              }`}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Workspaces List */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-center">
            <div className="text-xs text-[--color-text-tertiary]">
              No workspaces found
            </div>
          </div>
        ) : (
          filtered.map((workspace) => (
            <div
              key={workspace.id}
              onClick={() => onSelectWorkspace?.(workspace.id)}
              className={`group px-4 py-3 border-l-2 cursor-pointer transition-all hover:bg-[--bg-surface] ${
                selectedWorkspace === workspace.id
                  ? 'bg-[--bg-surface] border-[--accent-primary]'
                  : 'border-transparent'
              }`}
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-[--color-text-primary] truncate">
                      {workspace.name}
                    </span>
                    {workspace.starred && <span className="text-yellow-400">★</span>}
                  </div>
                  <div className="text-xs text-[--color-text-tertiary] mt-0.5">
                    {workspace.type}
                  </div>
                </div>

                {workspace.shared && (
                  <Globe size={14} className="text-[--accent-cyan] flex-shrink-0" />
                )}
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1 text-xs text-[--color-text-tertiary]">
                  <Clock size={12} />
                  <span>{workspace.lastModified}</span>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button className="p-1 rounded hover:bg-[--bg-body] text-[--color-text-tertiary]">
                    <Share2 size={12} />
                  </button>
                  <button className="p-1 rounded hover:bg-[--bg-body] text-[--color-text-tertiary]">
                    <Download size={12} />
                  </button>
                  <button className="p-1 rounded hover:bg-red-500/20 text-red-400">
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body] space-y-2">
        <button className="w-full px-3 py-2 text-xs font-medium rounded bg-[--accent-primary] hover:opacity-90 text-white transition-opacity flex items-center justify-center gap-2">
          <Plus size={12} />
          New Workspace
        </button>
        <button className="w-full px-3 py-2 text-xs font-medium rounded bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          Browse Templates
        </button>
      </div>
    </div>
  );
}
