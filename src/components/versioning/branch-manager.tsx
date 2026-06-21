'use client';

import { GitBranch, Plus, Trash2, Merge, Copy } from 'lucide-react';
import { useState } from 'react';

interface Branch {
  id: string;
  name: string;
  from: string;
  owner: string;
  createdAt: string;
  lastModified: string;
  commits: number;
  isCurrent: boolean;
}

const branches: Branch[] = [
  {
    id: 'main',
    name: 'main',
    from: 'initial',
    owner: 'You',
    createdAt: '1 week ago',
    lastModified: '2 hours ago',
    commits: 24,
    isCurrent: true,
  },
  {
    id: 'feature/attention-optimization',
    name: 'feature/attention-optimization',
    from: 'main',
    owner: 'You',
    createdAt: '3 days ago',
    lastModified: '1 hour ago',
    commits: 8,
    isCurrent: false,
  },
  {
    id: 'feature/residual-connections',
    name: 'feature/residual-connections',
    from: 'main',
    owner: 'Sarah Chen',
    createdAt: '2 days ago',
    lastModified: '5 hours ago',
    commits: 5,
    isCurrent: false,
  },
  {
    id: 'experimental/scaling-laws',
    name: 'experimental/scaling-laws',
    from: 'main',
    owner: 'Alex Kumar',
    createdAt: '1 day ago',
    lastModified: '30 min ago',
    commits: 3,
    isCurrent: false,
  },
];

interface BranchManagerProps {
  workspaceName?: string;
}

export function BranchManager({ workspaceName = 'Transformer Architecture' }: BranchManagerProps) {
  const [, setSelectedBranch] = useState<string>('main');
  const [expandedBranch, setExpandedBranch] = useState<string | null>(null);

  const currentBranch = branches.find((b) => b.isCurrent);
  const otherBranches = branches.filter((b) => !b.isCurrent);

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-r border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-1">
          Branches
        </h3>
        <p className="text-xs text-[--color-text-secondary]">
          {workspaceName}
        </p>
      </div>

      {/* Branch List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {/* Current Branch */}
        {currentBranch && (
          <div className="mb-4">
            <div className="text-xs font-semibold text-[--color-text-tertiary] mb-2 px-2">
              CURRENT BRANCH
            </div>
            <div
              onClick={() => setSelectedBranch(currentBranch.id)}
              className="group p-3 rounded-lg bg-[--accent-primary]/10 border-2 border-[--accent-primary] cursor-pointer transition-all"
            >
              <div className="flex items-center gap-2 mb-1">
                <GitBranch size={14} className="text-[--accent-primary]" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-semibold text-[--color-text-primary] truncate">
                    {currentBranch.name}
                  </div>
                </div>
                <span className="text-xs bg-[--accent-primary] text-white px-2 py-0.5 rounded flex-shrink-0">
                  Active
                </span>
              </div>
              <div className="text-xs text-[--color-text-secondary] px-6 space-y-0.5">
                <div>Modified {currentBranch.lastModified}</div>
                <div>{currentBranch.commits} commits</div>
              </div>
            </div>
          </div>
        )}

        {/* Other Branches */}
        {otherBranches.length > 0 && (
          <div>
            <div className="text-xs font-semibold text-[--color-text-tertiary] mb-2 px-2">
              OTHER BRANCHES ({otherBranches.length})
            </div>
            <div className="space-y-2">
              {otherBranches.map((branch) => (
                <div key={branch.id} className="space-y-1">
                  <div
                    onClick={() => setExpandedBranch(expandedBranch === branch.id ? null : branch.id)}
                    className="group p-3 rounded-lg border border-[--color-border] bg-[--bg-body] hover:border-[--accent-primary]/50 cursor-pointer transition-all"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <GitBranch size={14} className="text-[--color-text-tertiary]" />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-semibold text-[--color-text-primary] truncate">
                          {branch.name}
                        </div>
                      </div>
                      <span className="text-xs text-[--color-text-tertiary] flex-shrink-0">
                        {branch.commits} commits
                      </span>
                    </div>
                    <div className="text-xs text-[--color-text-secondary] px-6 space-y-0.5">
                      <div>Modified {branch.lastModified}</div>
                      <div className="text-[--color-text-tertiary]">by {branch.owner}</div>
                    </div>
                  </div>

                  {/* Expanded Details */}
                  {expandedBranch === branch.id && (
                    <div className="ml-3 p-3 rounded-lg bg-[--bg-surface] border border-[--color-border] space-y-2">
                      <div className="text-xs">
                        <div className="text-[--color-text-secondary] mb-1">
                          <span className="font-semibold">Branched from:</span> {branch.from}
                        </div>
                        <div className="text-[--color-text-secondary] mb-3">
                          <span className="font-semibold">Created:</span> {branch.createdAt}
                        </div>
                      </div>

                      {/* Action Buttons */}
                      <div className="flex gap-2">
                        <button className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 rounded text-xs font-medium bg-[--accent-primary] hover:opacity-90 text-white transition-opacity">
                          <Merge size={12} />
                          Merge
                        </button>
                        <button className="px-2 py-1.5 rounded text-xs font-medium bg-[--bg-body] hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
                          <Copy size={12} />
                        </button>
                        <button className="px-2 py-1.5 rounded text-xs font-medium bg-red-500/10 hover:bg-red-500/20 text-red-400 transition-colors">
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
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body] space-y-2">
        <button className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-medium bg-[--accent-primary] hover:opacity-90 text-white transition-opacity">
          <Plus size={12} />
          New Branch
        </button>
        <button className="w-full px-3 py-2 rounded text-xs font-medium bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          Manage Branches
        </button>
      </div>
    </div>
  );
}
