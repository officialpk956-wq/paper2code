'use client';

import { Clock, Plus, Edit2, Trash2, MessageSquare, AlertCircle } from 'lucide-react';
import { useState } from 'react';

interface Activity {
  id: string;
  type: 'add' | 'modify' | 'delete' | 'comment' | 'mention';
  user: string;
  avatar: string;
  action: string;
  target: string;
  timestamp: string;
  details?: string;
}

const activities: Activity[] = [
  {
    id: '1',
    type: 'modify',
    user: 'You',
    avatar: '👤',
    action: 'Modified',
    target: 'Attention Head Visualization',
    timestamp: 'now',
    details: 'Updated styling parameters',
  },
  {
    id: '2',
    type: 'add',
    user: 'Sarah Chen',
    avatar: '👩',
    action: 'Added',
    target: 'Pooling Layer',
    timestamp: '2 min ago',
    details: 'New layer added to architecture',
  },
  {
    id: '3',
    type: 'mention',
    user: 'Alex Kumar',
    avatar: '👨',
    action: 'Mentioned you in',
    target: 'Transformer Block',
    timestamp: '5 min ago',
    details: '"Should we add residual connections?"',
  },
  {
    id: '4',
    type: 'comment',
    user: 'Sarah Chen',
    avatar: '👩',
    action: 'Commented on',
    target: 'Feed-Forward Network',
    timestamp: '8 min ago',
    details: 'Great implementation of the FFN layer',
  },
  {
    id: '5',
    type: 'modify',
    user: 'Alex Kumar',
    avatar: '👨',
    action: 'Updated',
    target: 'Model Configuration',
    timestamp: '12 min ago',
    details: 'Changed hidden dimensions to 768',
  },
];

const activityTypeConfig = {
  add: { icon: Plus, color: 'text-green-400', bgColor: 'bg-green-400/10' },
  modify: { icon: Edit2, color: 'text-blue-400', bgColor: 'bg-blue-400/10' },
  delete: { icon: Trash2, color: 'text-red-400', bgColor: 'bg-red-400/10' },
  comment: { icon: MessageSquare, color: 'text-yellow-400', bgColor: 'bg-yellow-400/10' },
  mention: { icon: AlertCircle, color: 'text-purple-400', bgColor: 'bg-purple-400/10' },
};

interface ActivityFeedProps {
  workspaceName?: string;
  maxItems?: number;
}

export function ActivityFeed({ workspaceName = 'Transformer Architecture', maxItems = 5 }: ActivityFeedProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const displayedActivities = activities.slice(0, maxItems);

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-r border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-1">
          Activity Feed
        </h3>
        <p className="text-xs text-[--color-text-secondary]">
          {workspaceName}
        </p>
      </div>

      {/* Activities Timeline */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-3">
          {displayedActivities.map((activity, idx) => {
            const config = activityTypeConfig[activity.type];
            const IconComponent = config.icon;
            const isExpanded = expandedId === activity.id;

            return (
              <div
                key={activity.id}
                className={`relative cursor-pointer transition-all rounded-lg border p-3 ${
                  isExpanded
                    ? `${config.bgColor} border-[--accent-primary]`
                    : 'border-[--color-border] bg-[--bg-body] hover:border-[--accent-primary]/50'
                }`}
                onClick={() => setExpandedId(isExpanded ? null : activity.id)}
              >
                {/* Timeline Connector */}
                {idx < displayedActivities.length - 1 && (
                  <div className="absolute left-8 top-14 w-0.5 h-3 bg-[--color-border]" />
                )}

                {/* Icon Circle */}
                <div className={`absolute -left-4 top-3 w-8 h-8 rounded-full border-2 border-[--color-border] flex items-center justify-center ${config.bgColor}`}>
                  <IconComponent size={14} className={config.color} />
                </div>

                {/* Content */}
                <div className="ml-6 space-y-2">
                  {/* Main Activity */}
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <span className="text-sm font-semibold text-[--color-text-primary]">
                          {activity.avatar} {activity.user}
                        </span>
                        <span className="text-xs text-[--color-text-secondary]">
                          {activity.action}
                        </span>
                      </div>
                      <div className="text-sm font-medium text-[--accent-primary]">
                        {activity.target}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 text-xs text-[--color-text-tertiary] flex-shrink-0">
                      <Clock size={12} />
                      {activity.timestamp}
                    </div>
                  </div>

                  {/* Expanded Details */}
                  {isExpanded && activity.details && (
                    <div className="bg-[--bg-surface] rounded px-2 py-2 border border-[--color-border] text-xs text-[--color-text-secondary] italic">
                      {activity.details}
                    </div>
                  )}

                  {/* Action Buttons */}
                  {isExpanded && (
                    <div className="flex items-center gap-2 mt-2 pt-2 border-t border-[--color-border]">
                      <button className="text-xs px-2 py-1 rounded bg-[--accent-primary]/10 hover:bg-[--accent-primary]/20 text-[--accent-primary] transition-colors">
                        View
                      </button>
                      <button className="text-xs px-2 py-1 rounded bg-[--bg-body] hover:bg-[--bg-surface] text-[--color-text-secondary] transition-colors">
                        Undo
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body]">
        <button className="w-full px-3 py-2 rounded text-xs font-medium bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          View Full Activity
        </button>
      </div>
    </div>
  );
}
