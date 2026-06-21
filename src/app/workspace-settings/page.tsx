'use client';

import { useState } from 'react';
import { WorkspaceManager } from '@/components/collaboration/workspace-manager';
import { CollaborationPanel } from '@/components/collaboration/collaboration-panel';
import { VersionHistory } from '@/components/collaboration/version-history';
import { TemplatesLibrary } from '@/components/collaboration/templates-library';

type TabType = 'workspaces' | 'collaboration' | 'history' | 'templates';

export default function WorkspaceSettingsPage() {
  const [activeTab, setActiveTab] = useState<TabType>('workspaces');
  const [selectedWorkspace, setSelectedWorkspace] = useState<string>();

  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h1 className="text-2xl font-bold text-[--color-text-primary]">
          Workspace Settings
        </h1>
        <p className="text-sm text-[--color-text-secondary] mt-1">
          Manage workspaces, collaboration, and version history
        </p>
      </div>

      {/* Tabs */}
      <div className="px-6 pt-4 border-b border-[--color-border] bg-[--bg-panel]">
        <div className="flex gap-4">
          {(['workspaces', 'collaboration', 'history', 'templates'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab
                  ? 'border-[--accent-primary] text-[--accent-primary]'
                  : 'border-transparent text-[--color-text-secondary] hover:text-[--color-text-primary]'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === 'workspaces' && (
          <div className="h-full">
            <WorkspaceManager
              onSelectWorkspace={setSelectedWorkspace}
              selectedWorkspace={selectedWorkspace}
            />
          </div>
        )}

        {activeTab === 'collaboration' && (
          <div className="h-full">
            <CollaborationPanel workspaceName={selectedWorkspace || 'Current Workspace'} />
          </div>
        )}

        {activeTab === 'history' && (
          <div className="h-full">
            <VersionHistory workspaceName={selectedWorkspace || 'Current Workspace'} />
          </div>
        )}

        {activeTab === 'templates' && (
          <div className="h-full">
            <TemplatesLibrary onSelectTemplate={() => undefined} />
          </div>
        )}
      </div>
    </div>
  );
}
