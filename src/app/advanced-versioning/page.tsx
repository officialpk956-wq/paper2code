'use client';

import { useState } from 'react';
import { ThreeColumnLayout } from '@/components/layout/three-column-layout';
import { BranchManager } from '@/components/versioning/branch-manager';
import { VersionDiffViewer } from '@/components/versioning/version-diff-viewer';
import { MergeInterface } from '@/components/versioning/merge-interface';
import { VersionTags } from '@/components/versioning/version-tags';
import { ConflictResolver } from '@/components/versioning/conflict-resolver';
import { ComparisonTimeline } from '@/components/versioning/comparison-timeline';

export default function AdvancedVersioningPage() {
  const [selectedTab, setSelectedTab] = useState<'diff' | 'merge' | 'tags' | 'timeline'>('diff');

  const tabs = [
    { id: 'diff', label: 'Version Diff' },
    { id: 'merge', label: 'Merge & Conflicts' },
    { id: 'tags', label: 'Tags & Annotations' },
    { id: 'timeline', label: 'Comparison' },
  ] as const;

  const leftPanel = (
    <div className="flex flex-col h-full">
      <BranchManager workspaceName="Transformer Architecture" />
    </div>
  );

  const centerPanel = (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header with Tabs */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h1 className="text-2xl font-bold text-[--color-text-primary] mb-4">
          Advanced Versioning
        </h1>

        <div className="flex gap-2 border-b border-[--color-border]">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id as typeof selectedTab)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                selectedTab === tab.id
                  ? 'border-[--accent-primary] text-[--accent-primary]'
                  : 'border-transparent text-[--color-text-secondary] hover:text-[--color-text-primary]'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto">
        {selectedTab === 'diff' && (
          <div className="p-6 space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-[--color-text-primary] mb-3">
                Version Comparison
              </h2>
              <p className="text-sm text-[--color-text-secondary] mb-4">
                Review differences between versions with split or unified view. See exactly what changed with line-by-line diffs.
              </p>
              <div className="h-96">
                <VersionDiffViewer
                  fromVersion="v1.0 (2 days ago)"
                  toVersion="v1.2 (current)"
                  viewMode="split"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                  Split View
                </h3>
                <p className="text-xs text-[--color-text-tertiary]">
                  See old and new versions side-by-side for easy comparison
                </p>
              </div>
              <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                  Unified View
                </h3>
                <p className="text-xs text-[--color-text-tertiary]">
                  Compact view with additions/deletions/modifications highlighted
                </p>
              </div>
              <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                  Change Statistics
                </h3>
                <p className="text-xs text-[--color-text-tertiary]">
                  Quick summary of lines added, deleted, and modified
                </p>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'merge' && (
          <div className="p-6 space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-[--color-text-primary] mb-3">
                Merge with Conflict Detection
              </h2>
              <p className="text-sm text-[--color-text-secondary] mb-4">
                Automatically detect and resolve conflicts when merging branches. Choose between versions or edit manually.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
              <div className="h-96">
                <MergeInterface
                  sourceBranch="feature/attention-optimization"
                  targetBranch="main"
                />
              </div>

              <div className="h-96">
                <ConflictResolver
                  title="Conflict Resolution"
                  description="Review and resolve merge conflicts by file"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                  Resolution Strategies
                </h3>
                <ul className="text-xs text-[--color-text-secondary] space-y-1">
                  <li>✓ Keep ours (target branch)</li>
                  <li>✓ Take theirs (source branch)</li>
                  <li>✓ Manual merge editing</li>
                  <li>✓ Smart merge suggestions</li>
                </ul>
              </div>
              <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                  Severity Levels
                </h3>
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-xs">
                    <div className="w-2 h-2 rounded-full bg-red-500" />
                    <span className="text-[--color-text-secondary]">High - Function signatures</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <div className="w-2 h-2 rounded-full bg-yellow-500" />
                    <span className="text-[--color-text-secondary]">Medium - Variable changes</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <div className="w-2 h-2 rounded-full bg-blue-500" />
                    <span className="text-[--color-text-secondary]">Low - Comments, spacing</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'tags' && (
          <div className="p-6 space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-[--color-text-primary] mb-3">
                Version Tags & Annotations
              </h2>
              <p className="text-sm text-[--color-text-secondary] mb-4">
                Mark important versions with descriptive tags. Pin frequently-used versions for quick access.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="h-96">
                <VersionTags workspaceName="Transformer Architecture" />
              </div>

              <div className="space-y-4">
                <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                  <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                    Tag Management Features
                  </h3>
                  <ul className="text-xs text-[--color-text-secondary] space-y-1">
                    <li>✓ Create descriptive tags for important versions</li>
                    <li>✓ Pin tags for quick access</li>
                    <li>✓ Organize by purpose (releases, experiments, backups)</li>
                    <li>✓ Add annotations and notes</li>
                    <li>✓ View version from any tag</li>
                  </ul>
                </div>

                <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                  <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                    Common Tags
                  </h3>
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded bg-green-500/20 text-green-400">v1.0-release</span>
                      <span className="text-[--color-text-tertiary]">Release versions</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-400">stable</span>
                      <span className="text-[--color-text-tertiary]">Production-ready</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded bg-yellow-500/20 text-yellow-400">backup</span>
                      <span className="text-[--color-text-tertiary]">Before refactoring</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded bg-purple-500/20 text-purple-400">experiment</span>
                      <span className="text-[--color-text-tertiary]">Testing/exploration</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'timeline' && (
          <div className="p-6 space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-[--color-text-primary] mb-3">
                Version Timeline & Comparison
              </h2>
              <p className="text-sm text-[--color-text-secondary] mb-4">
                View the complete history of changes and select any two versions to compare in detail.
              </p>
            </div>

            <div className="h-96">
              <ComparisonTimeline
                workspaceName="Transformer Architecture"
                selectedVersions={['1', '3']}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                  Timeline View
                </h3>
                <ul className="text-xs text-[--color-text-secondary] space-y-1">
                  <li>✓ Visual chronological history</li>
                  <li>✓ Author and contributor info</li>
                  <li>✓ Change statistics per version</li>
                  <li>✓ Quick version comparison</li>
                </ul>
              </div>
              <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                  Change Metrics
                </h3>
                <div className="space-y-1 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded bg-green-500/20 text-green-400">+</span>
                    <span className="text-[--color-text-secondary]">Lines added</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-400">±</span>
                    <span className="text-[--color-text-secondary]">Lines modified</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded bg-red-500/20 text-red-400">−</span>
                    <span className="text-[--color-text-secondary]">Lines deleted</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const rightPanel = (
    <div className="flex flex-col h-full">
      <VersionTags workspaceName="Transformer Architecture" />
    </div>
  );

  return (
    <ThreeColumnLayout
      left={leftPanel}
      center={centerPanel}
      right={rightPanel}
    />
  );
}
