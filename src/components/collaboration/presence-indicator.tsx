'use client';

import { LogOut } from 'lucide-react';
import { useState } from 'react';

interface Presence {
  id: string;
  name: string;
  email: string;
  avatar: string;
  status: 'viewing' | 'editing' | 'idle';
  joinedAt: string;
  lastSeen?: string;
}

const activePresences: Presence[] = [
  {
    id: '1',
    name: 'You',
    email: 'user@example.com',
    avatar: '👤',
    status: 'editing',
    joinedAt: 'now',
    lastSeen: 'now',
  },
  {
    id: '2',
    name: 'Sarah Chen',
    email: 'sarah@example.com',
    avatar: '👩',
    status: 'editing',
    joinedAt: '5 min ago',
    lastSeen: 'now',
  },
  {
    id: '3',
    name: 'Alex Kumar',
    email: 'alex@example.com',
    avatar: '👨',
    status: 'viewing',
    joinedAt: '2 min ago',
    lastSeen: '30s ago',
  },
];

interface PresenceIndicatorProps {
  workspaceName?: string;
  showDetails?: boolean;
}

export function PresenceIndicator({ workspaceName = 'Transformer Architecture', showDetails = false }: PresenceIndicatorProps) {
  const [hoveredUser, setHoveredUser] = useState<string | null>(null);

  const statusColor = {
    editing: 'bg-green-500',
    viewing: 'bg-blue-500',
    idle: 'bg-gray-500',
  };

  const statusText = {
    editing: 'Editing',
    viewing: 'Viewing',
    idle: 'Idle',
  };

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-r border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-1">
          Active Now
        </h3>
        <p className="text-xs text-[--color-text-secondary]">
          {workspaceName}
        </p>
      </div>

      {/* Presence List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {activePresences.map((presence) => (
          <div
            key={presence.id}
            className="group relative"
            onMouseEnter={() => setHoveredUser(presence.id)}
            onMouseLeave={() => setHoveredUser(null)}
          >
            {/* Compact View */}
            {!showDetails ? (
              <div className="flex items-center gap-3 p-2 rounded-lg hover:bg-[--bg-surface] transition-colors cursor-pointer">
                <div className="relative">
                  <div className="text-2xl">{presence.avatar}</div>
                  <div className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border border-[--bg-panel] ${statusColor[presence.status]}`} />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="text-sm font-semibold text-[--color-text-primary]">
                    {presence.name}
                  </div>
                  <div className="text-xs text-[--color-text-tertiary]">
                    {statusText[presence.status]} • {presence.lastSeen}
                  </div>
                </div>

                {/* Tooltip on hover */}
                {hoveredUser === presence.id && (
                  <div className="absolute left-0 top-full mt-1 w-48 bg-[--bg-surface] border border-[--color-border] rounded-lg p-3 z-50 shadow-xl">
                    <div className="text-sm font-semibold text-[--color-text-primary] mb-1">
                      {presence.name}
                    </div>
                    <div className="text-xs text-[--color-text-tertiary] space-y-1">
                      <div>Email: {presence.email}</div>
                      <div className="flex items-center gap-1">
                        <div className={`w-2 h-2 rounded-full ${statusColor[presence.status]}`} />
                        {statusText[presence.status]}
                      </div>
                      <div>Joined {presence.joinedAt}</div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              /* Detailed View */
              <div className="p-3 rounded-lg bg-[--bg-surface] border border-[--color-border]">
                <div className="flex items-start gap-3 mb-2">
                  <div className="relative">
                    <div className="text-3xl">{presence.avatar}</div>
                    <div className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-[--bg-surface] ${statusColor[presence.status]}`} />
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold text-[--color-text-primary]">
                      {presence.name}
                    </div>
                    <div className="text-xs text-[--color-text-tertiary]">
                      {presence.email}
                    </div>
                    <div className="flex items-center gap-1 mt-1">
                      <div className={`w-2 h-2 rounded-full ${statusColor[presence.status]}`} />
                      <span className="text-xs text-[--color-text-secondary]">
                        {statusText[presence.status]}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="text-xs text-[--color-text-tertiary] space-y-1 pl-11 mb-2">
                  <div>Joined {presence.joinedAt}</div>
                  {presence.lastSeen && (
                    <div>Last seen {presence.lastSeen}</div>
                  )}
                </div>

                {presence.id !== '1' && (
                  <button className="w-full px-2 py-1 rounded text-xs font-medium bg-[--bg-body] hover:bg-red-500/20 text-red-400 transition-colors flex items-center justify-center gap-1">
                    <LogOut size={12} />
                    Remove
                  </button>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Footer Stats */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body] text-xs text-[--color-text-secondary]">
        <div className="flex justify-between items-center">
          <span>Active Collaborators</span>
          <span className="text-[--accent-primary] font-semibold">{activePresences.length}</span>
        </div>
        <div className="flex justify-between items-center mt-2">
          <span>Editing Now</span>
          <span className="text-green-400 font-semibold">{activePresences.filter(p => p.status === 'editing').length}</span>
        </div>
      </div>
    </div>
  );
}
