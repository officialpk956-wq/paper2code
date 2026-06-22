'use client';

import { useState } from 'react';
import { ThreeColumnLayout } from '@/components/layout/three-column-layout';
import { PresenceIndicator } from '@/components/collaboration/presence-indicator';
import { ActivityFeed } from '@/components/collaboration/activity-feed';
import { TypingIndicator } from '@/components/collaboration/typing-indicator';
import { LiveCursorTracker } from '@/components/collaboration/live-cursor-tracker';
import { NotificationToast } from '@/components/collaboration/notification-toast';

export default function RealTimeCollaborationPage() {
  const [selectedTab, setSelectedTab] = useState<'presence' | 'activity' | 'cursors'>('presence');

  const tabs = [
    { id: 'presence', label: 'Presence & Typing' },
    { id: 'activity', label: 'Activity Feed' },
    { id: 'cursors', label: 'Live Cursors' },
  ] as const;

  const leftPanel = (
    <div className="flex flex-col h-full bg-[--bg-panel]">
      <PresenceIndicator workspaceName="Transformer Architecture" showDetails={false} />
    </div>
  );

  const centerPanel = (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header with Tabs */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h1 className="text-2xl font-bold text-[--color-text-primary] mb-4">
          Real-time Collaboration
        </h1>

        <div className="flex gap-2 border-b border-[--color-border]">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id)}
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
        {selectedTab === 'presence' && (
          <div className="p-6 space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-[--color-text-primary] mb-3">
                Typing Indicators
              </h2>
              <p className="text-sm text-[--color-text-secondary] mb-4">
                Real-time indicators showing which collaborators are actively typing and where they&apos;re working.
              </p>
              <div className="bg-[--bg-panel] rounded-lg p-6 border border-[--color-border]">
                <TypingIndicator />
              </div>
            </div>

            <div>
              <h2 className="text-lg font-semibold text-[--color-text-primary] mb-3">
                Presence Status
              </h2>
              <p className="text-sm text-[--color-text-secondary] mb-4">
                Visual indicators for collaborator status: editing, viewing, or idle. See the Presence panel on the left for detailed user information.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-[--bg-panel] rounded-lg p-4 border border-green-400/30">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-3 h-3 rounded-full bg-green-500" />
                    <span className="text-sm font-medium text-[--color-text-primary]">Editing</span>
                  </div>
                  <p className="text-xs text-[--color-text-tertiary]">Actively modifying content</p>
                </div>
                <div className="bg-[--bg-panel] rounded-lg p-4 border border-blue-400/30">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500" />
                    <span className="text-sm font-medium text-[--color-text-primary]">Viewing</span>
                  </div>
                  <p className="text-xs text-[--color-text-tertiary]">Browsing content</p>
                </div>
                <div className="bg-[--bg-panel] rounded-lg p-4 border border-gray-400/30">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-3 h-3 rounded-full bg-gray-500" />
                    <span className="text-sm font-medium text-[--color-text-primary]">Idle</span>
                  </div>
                  <p className="text-xs text-[--color-text-tertiary]">Inactive in workspace</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'activity' && (
          <div className="p-6 space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-[--color-text-primary] mb-3">
                Activity Timeline
              </h2>
              <p className="text-sm text-[--color-text-secondary] mb-4">
                Complete history of workspace changes, with ability to undo modifications and see details about each action.
              </p>
              <div className="bg-[--bg-panel] rounded-lg border border-[--color-border] overflow-hidden">
                <ActivityFeed maxItems={10} />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3 h-3 rounded-full bg-green-400" />
                  <span className="text-sm font-medium text-[--color-text-primary]">Added</span>
                </div>
                <p className="text-xs text-[--color-text-tertiary]">New components or layers</p>
              </div>
              <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3 h-3 rounded-full bg-purple-400" />
                  <span className="text-sm font-medium text-[--color-text-primary]">Mentioned</span>
                </div>
                <p className="text-xs text-[--color-text-tertiary]">You were tagged in a comment</p>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'cursors' && (
          <div className="p-6 space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-[--color-text-primary] mb-3">
                Live Cursor Tracking
              </h2>
              <p className="text-sm text-[--color-text-secondary] mb-4">
                See where your collaborators are working in real-time with color-coded cursors and labels.
              </p>
            </div>

            <div className="h-96 bg-[--bg-panel] rounded-lg border border-[--color-border] overflow-hidden">
              <LiveCursorTracker canvasLabel="Transformer Architecture Canvas" />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                  Cursor Features
                </h3>
                <ul className="text-xs text-[--color-text-secondary] space-y-1">
                  <li>✓ Color-coded per user</li>
                  <li>✓ Real-time position sync</li>
                  <li>✓ User identification labels</li>
                  <li>✓ Smooth animation</li>
                </ul>
              </div>
              <div className="bg-[--bg-panel] rounded-lg p-4 border border-[--color-border]">
                <h3 className="text-sm font-semibold text-[--color-text-primary] mb-2">
                  Benefits
                </h3>
                <ul className="text-xs text-[--color-text-secondary] space-y-1">
                  <li>✓ Avoid duplicate work</li>
                  <li>✓ Follow collaborator focus</li>
                  <li>✓ Coordinate changes</li>
                  <li>✓ Improve communication</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const rightPanel = (
    <div className="flex flex-col h-full bg-[--bg-panel]">
      <ActivityFeed workspaceName="Transformer Architecture" maxItems={8} />
    </div>
  );

  return (
    <>
      <ThreeColumnLayout
        left={leftPanel}
        center={centerPanel}
        right={rightPanel}
      />
      {/* Notification toasts render fixed/overlaid — must live outside ThreeColumnLayout */}
      <NotificationToast position="top-right" maxToasts={3} />
    </>
  );
}
