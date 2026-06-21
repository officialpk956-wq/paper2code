'use client';

import { Share2, Users, Copy, Trash2 } from 'lucide-react';
import { useState } from 'react';

interface Collaborator {
  id: string;
  name: string;
  email: string;
  role: 'owner' | 'editor' | 'viewer';
  avatar: string;
  joinedAt: string;
}

const collaborators: Collaborator[] = [
  {
    id: '1',
    name: 'You',
    email: 'user@example.com',
    role: 'owner',
    avatar: '👤',
    joinedAt: 'owner',
  },
  {
    id: '2',
    name: 'Sarah Chen',
    email: 'sarah@example.com',
    role: 'editor',
    avatar: '👩',
    joinedAt: '2 weeks ago',
  },
  {
    id: '3',
    name: 'Alex Kumar',
    email: 'alex@example.com',
    role: 'viewer',
    avatar: '👨',
    joinedAt: '1 week ago',
  },
];

interface CollaborationPanelProps {
  workspaceName?: string;
}

export function CollaborationPanel({ workspaceName = 'Transformer Architecture' }: CollaborationPanelProps) {
  const [showLink, setShowLink] = useState(false);
  const [copied, setCopied] = useState(false);
  const shareLink = 'https://p2c.ai/ws/transformer-design?share=abc123';

  const handleCopyLink = () => {
    navigator.clipboard.writeText(shareLink);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-r border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-1">
          Collaboration
        </h3>
        <p className="text-xs text-[--color-text-secondary]">
          {workspaceName}
        </p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Share Section */}
        <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
          <div className="text-xs font-semibold text-[--color-text-primary] mb-3 flex items-center gap-2">
            <Share2 size={14} />
            Share Workspace
          </div>

          {/* Toggle Share Link */}
          <button
            onClick={() => setShowLink(!showLink)}
            className="w-full px-3 py-2 rounded text-xs font-medium bg-[--accent-primary] hover:opacity-90 text-white transition-opacity mb-3"
          >
            {showLink ? 'Hide Link' : 'Generate Share Link'}
          </button>

          {/* Share Link Display */}
          {showLink && (
            <div className="bg-[--bg-surface] rounded p-2 space-y-2">
              <div className="text-xs text-[--color-text-tertiary] mb-1">
                Share this link with collaborators:
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={shareLink}
                  readOnly
                  className="flex-1 px-2 py-1 rounded text-xs bg-[--bg-body] border border-[--color-border] text-[--color-text-primary]"
                />
                <button
                  onClick={handleCopyLink}
                  className={`p-1.5 rounded transition-colors ${
                    copied
                      ? 'bg-green-500 text-white'
                      : 'bg-[--bg-body] hover:bg-[--color-border] text-[--color-text-tertiary]'
                  }`}
                >
                  <Copy size={12} />
                </button>
              </div>
              {copied && <p className="text-xs text-green-400">Link copied!</p>}
            </div>
          )}
        </div>

        {/* Collaborators */}
        <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
          <div className="text-xs font-semibold text-[--color-text-primary] mb-3 flex items-center gap-2">
            <Users size={14} />
            Collaborators ({collaborators.length})
          </div>

          <div className="space-y-2">
            {collaborators.map((collab) => (
              <div
                key={collab.id}
                className="flex items-center justify-between p-2 rounded bg-[--bg-surface] hover:bg-[--bg-panel] transition-colors"
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <div className="text-lg flex-shrink-0">{collab.avatar}</div>
                  <div className="min-w-0">
                    <div className="text-xs font-semibold text-[--color-text-primary]">
                      {collab.name}
                    </div>
                    <div className="text-xs text-[--color-text-tertiary]">
                      {collab.email}
                    </div>
                  </div>
                </div>

                {/* Role Badge */}
                <div className="flex items-center gap-1 ml-2">
                  {collab.role === 'owner' && (
                    <span className="px-2 py-0.5 rounded text-xs bg-[--accent-primary] text-white font-medium">
                      Owner
                    </span>
                  )}
                  {collab.role === 'editor' && (
                    <span className="px-2 py-0.5 rounded text-xs bg-[--accent-cyan] text-white font-medium">
                      Editor
                    </span>
                  )}
                  {collab.role === 'viewer' && (
                    <span className="px-2 py-0.5 rounded text-xs bg-[--color-border] text-[--color-text-secondary] font-medium">
                      Viewer
                    </span>
                  )}

                  {collab.id !== '1' && (
                    <button className="p-1 rounded hover:bg-red-500/20 text-red-400">
                      <Trash2 size={12} />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Add Collaborator */}
          <button className="w-full mt-3 px-3 py-2 rounded text-xs font-medium bg-[--bg-body] hover:bg-[--bg-surface] border border-[--color-border] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors flex items-center justify-center gap-2">
            <Users size={12} />
            Add Collaborator
          </button>
        </div>

        {/* Permissions */}
        <div className="bg-[--bg-body] rounded-lg p-3 border border-[--color-border]">
          <div className="text-xs font-semibold text-[--color-text-primary] mb-3">
            Default Permissions
          </div>

          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="radio" name="perms" defaultChecked className="w-3 h-3" />
              <div className="flex-1">
                <div className="text-xs font-medium text-[--color-text-primary]">
                  Private
                </div>
                <div className="text-xs text-[--color-text-tertiary]">
                  Only you can access
                </div>
              </div>
            </label>

            <label className="flex items-center gap-2 cursor-pointer">
              <input type="radio" name="perms" className="w-3 h-3" />
              <div className="flex-1">
                <div className="text-xs font-medium text-[--color-text-primary]">
                  Shareable Link
                </div>
                <div className="text-xs text-[--color-text-tertiary]">
                  Anyone with link (read-only)
                </div>
              </div>
            </label>

            <label className="flex items-center gap-2 cursor-pointer">
              <input type="radio" name="perms" className="w-3 h-3" />
              <div className="flex-1">
                <div className="text-xs font-medium text-[--color-text-primary]">
                  Public
                </div>
                <div className="text-xs text-[--color-text-tertiary]">
                  Anyone can discover & view
                </div>
              </div>
            </label>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body]">
        <button className="w-full px-3 py-2 text-xs font-medium rounded bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          Share Settings
        </button>
      </div>
    </div>
  );
}
