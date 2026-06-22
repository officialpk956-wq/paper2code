'use client';

import { Lightbulb, AlertCircle } from 'lucide-react';
import { useState } from 'react';

interface Recommendation {
  id: string;
  type: 'prerequisite' | 'next' | 'skill' | 'deep-dive';
  title: string;
  description: string;
  reason: string;
  icon: string;
  actionLabel: string;
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  estimatedTime?: number; // minutes
  blockedBy?: string;
}

const recommendations: Recommendation[] = [
  {
    id: 'prereq-1',
    type: 'prerequisite',
    title: 'Attention Mechanism Fundamentals',
    description: 'You need this foundation before tackling multi-head attention implementation',
    reason: 'Missing prerequisite for: Multi-Head Attention',
    icon: '📌',
    actionLabel: 'Learn First',
    difficulty: 'intermediate',
    estimatedTime: 35,
  },
  {
    id: 'next-1',
    type: 'next',
    title: 'Build Your First Transformer',
    description: 'You\'re ready to implement a complete transformer from scratch',
    reason: 'Based on: Attention mastery + 8/10 transformer understanding',
    icon: '🚀',
    actionLabel: 'Start Now',
    difficulty: 'advanced',
    estimatedTime: 120,
  },
  {
    id: 'skill-1',
    type: 'skill',
    title: 'Reinforcement Learning Fundamentals',
    description: 'Strengthen your RL foundation with policy gradient methods',
    reason: 'You\'re at 65% RL mastery - time to level up',
    icon: '⚡',
    actionLabel: 'Explore',
    difficulty: 'intermediate',
    estimatedTime: 60,
  },
  {
    id: 'deepdive-1',
    type: 'deep-dive',
    title: 'Advanced: Efficient Transformers',
    description: 'Explore sparse attention, linear attention, and state space models',
    reason: 'You\'ve mastered standard transformers - ready for optimization',
    icon: '🔬',
    actionLabel: 'Deep Dive',
    difficulty: 'advanced',
    estimatedTime: 90,
  },
];

interface AILearningCoachProps {
  userName?: string;
  onRecommendationClick?: (id: string) => void;
}

export function AILearningCoach({ userName = 'there', onRecommendationClick }: AILearningCoachProps) {
  const [expandedId, setExpandedId] = useState<string | null>('next-1');
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());

  const visibleRecommendations = recommendations.filter((r) => !dismissedIds.has(r.id));

  const dismiss = (id: string) => {
    setDismissedIds((prev) => new Set([...prev, id]));
  };

  const typeColors = {
    prerequisite: 'border-orange-500/30 bg-orange-500/5',
    next: 'border-green-500/30 bg-green-500/5',
    skill: 'border-blue-500/30 bg-blue-500/5',
    'deep-dive': 'border-purple-500/30 bg-purple-500/5',
  };

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-r border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <div className="flex items-center gap-2 mb-2">
          <Lightbulb size={16} className="text-yellow-400" />
          <h3 className="text-sm font-semibold text-[--color-text-primary]">
            Your Learning Coach
          </h3>
        </div>
        <p className="text-xs text-[--color-text-secondary]">
          Hi {userName}! Here&apos;s what I recommend next based on your learning journey.
        </p>
      </div>

      {/* Recommendations */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {visibleRecommendations.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-center">
            <div className="text-2xl mb-2">🎉</div>
            <p className="text-xs text-[--color-text-secondary]">
              You&apos;ve dismissed all recommendations. Explore more content or check back later!
            </p>
          </div>
        ) : (
          visibleRecommendations.map((rec) => {
            const isExpanded = expandedId === rec.id;

            return (
              <div
                key={rec.id}
                className={`rounded-lg border-2 transition-all cursor-pointer ${typeColors[rec.type]} ${
                  isExpanded ? 'ring-2 ring-[--accent-primary]' : ''
                }`}
                onClick={() => setExpandedId(isExpanded ? null : rec.id)}
              >
                {/* Header */}
                <div className="p-3">
                  <div className="flex items-start gap-3">
                    <div className="text-lg">{rec.icon}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <h4 className="text-sm font-semibold text-[--color-text-primary]">
                            {rec.title}
                          </h4>
                          <p className="text-xs text-[--color-text-secondary] mt-0.5">
                            {rec.description}
                          </p>
                        </div>
                        {rec.difficulty && (
                          <span
                            className={`px-1.5 py-0.5 rounded text-xs font-medium whitespace-nowrap flex-shrink-0 ${
                              rec.difficulty === 'beginner'
                                ? 'bg-green-500/20 text-green-400'
                                : rec.difficulty === 'intermediate'
                                  ? 'bg-yellow-500/20 text-yellow-400'
                                  : 'bg-red-500/20 text-red-400'
                            }`}
                          >
                            {rec.difficulty}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Reason Tag */}
                  <div className="flex items-center gap-1 mt-2 ml-9">
                    <AlertCircle size={12} className="text-[--color-text-tertiary]" />
                    <span className="text-xs text-[--color-text-tertiary]">{rec.reason}</span>
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="border-t border-current opacity-30 px-3 py-3 space-y-3">
                    {rec.estimatedTime && (
                      <div className="text-xs text-[--color-text-secondary]">
                        <div className="font-medium text-[--color-text-primary] mb-1">
                          Estimated Time
                        </div>
                        <div>{rec.estimatedTime} minutes of focused learning</div>
                      </div>
                    )}

                    {rec.type === 'prerequisite' && (
                      <div className="bg-orange-500/10 border border-orange-500/20 rounded p-2 text-xs text-orange-400">
                        ⚠️ Complete this first to unlock related content
                      </div>
                    )}

                    {rec.type === 'next' && (
                      <div className="bg-green-500/10 border border-green-500/20 rounded p-2 text-xs text-green-400">
                        ✓ You have the foundation to tackle this challenge
                      </div>
                    )}

                    {rec.type === 'deep-dive' && (
                      <div className="bg-purple-500/10 border border-purple-500/20 rounded p-2 text-xs text-purple-400">
                        🔬 Advanced topics building on your expertise
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex gap-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onRecommendationClick?.(rec.id);
                        }}
                        className="flex-1 px-2 py-1.5 rounded text-xs font-medium bg-[--accent-primary] hover:opacity-90 text-white transition-opacity"
                      >
                        {rec.actionLabel}
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          dismiss(rec.id);
                        }}
                        className="px-2 py-1.5 rounded text-xs font-medium bg-[--bg-body] hover:bg-[--bg-surface] text-[--color-text-secondary] transition-colors"
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Progress Summary */}
      <div className="border-t border-[--color-border] bg-[--bg-body] p-4 space-y-2">
        <div className="text-xs font-semibold text-[--color-text-primary] mb-2">
          Learning Path Progress
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs mb-1">
            <span className="text-[--color-text-secondary]">AI Fundamentals</span>
            <span className="font-semibold text-green-400">4/5</span>
          </div>
          <div className="bg-[--bg-surface] rounded h-1.5 overflow-hidden">
            <div className="bg-green-500 h-full" style={{ width: '80%' }} />
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs mb-1">
            <span className="text-[--color-text-secondary]">Advanced Topics</span>
            <span className="font-semibold text-blue-400">2/6</span>
          </div>
          <div className="bg-[--bg-surface] rounded h-1.5 overflow-hidden">
            <div className="bg-blue-500 h-full" style={{ width: '33%' }} />
          </div>
        </div>
      </div>
    </div>
  );
}
