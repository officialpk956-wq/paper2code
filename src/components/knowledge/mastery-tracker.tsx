'use client';

import { CheckCircle2, Lock, Star } from 'lucide-react';
import { useState } from 'react';

type MasteryLevel = 'not-started' | 'in-progress' | 'explored' | 'mastered' | 'expert';

interface MasteryItem {
  id: string;
  name: string;
  description: string;
  category: string;
  currentLevel: MasteryLevel;
  prerequisite?: string;
  relatedItems: string[];
}

const masteryItems: MasteryItem[] = [
  {
    id: 'attention',
    name: 'Attention Mechanism',
    description: 'Understanding scaled dot-product attention and how it works',
    category: 'Fundamentals',
    currentLevel: 'mastered',
    relatedItems: ['multi-head', 'transformer-arch'],
  },
  {
    id: 'multi-head',
    name: 'Multi-Head Attention',
    description: 'Implementing and understanding multiple attention heads',
    category: 'Fundamentals',
    currentLevel: 'expert',
    prerequisite: 'attention',
    relatedItems: ['attention', 'transformer-arch'],
  },
  {
    id: 'transformer-arch',
    name: 'Transformer Architecture',
    description: 'Full transformer encoder-decoder architecture',
    category: 'Architectures',
    currentLevel: 'mastered',
    prerequisite: 'multi-head',
    relatedItems: ['bert', 'gpt', 'multi-head'],
  },
  {
    id: 'bert',
    name: 'BERT Model',
    description: 'Bidirectional encoder for NLP tasks',
    category: 'Architectures',
    currentLevel: 'explored',
    prerequisite: 'transformer-arch',
    relatedItems: ['transformer-arch'],
  },
  {
    id: 'gpt',
    name: 'GPT Architecture',
    description: 'Generative pre-trained transformer for text generation',
    category: 'Architectures',
    currentLevel: 'in-progress',
    prerequisite: 'transformer-arch',
    relatedItems: ['transformer-arch'],
  },
  {
    id: 'diffusion',
    name: 'Diffusion Models',
    description: 'Denoising diffusion probabilistic models',
    category: 'Generative',
    currentLevel: 'not-started',
    relatedItems: [],
  },
  {
    id: 'reinforcement',
    name: 'Reinforcement Learning',
    description: 'Policy gradient methods and value-based learning',
    category: 'Learning Paradigms',
    currentLevel: 'explored',
    relatedItems: [],
  },
];

const masteryLevels: { level: MasteryLevel; label: string; description: string; color: string }[] = [
  {
    level: 'not-started',
    label: 'Not Started',
    description: 'Haven\'t begun learning this topic',
    color: 'bg-gray-500/10 border-gray-500/30 text-gray-400',
  },
  {
    level: 'in-progress',
    label: 'In Progress',
    description: 'Currently learning this topic',
    color: 'bg-blue-500/10 border-blue-500/30 text-blue-400',
  },
  {
    level: 'explored',
    label: 'Explored',
    description: 'Familiar with concepts and basics',
    color: 'bg-green-500/10 border-green-500/30 text-green-400',
  },
  {
    level: 'mastered',
    label: 'Mastered',
    description: 'Deep understanding, can implement',
    color: 'bg-purple-500/10 border-purple-500/30 text-purple-400',
  },
  {
    level: 'expert',
    label: 'Expert',
    description: 'Expert level knowledge and implementation',
    color: 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400',
  },
];

interface MasteryTrackerProps {
  onLevelChange?: (itemId: string, level: MasteryLevel) => void;
}

export function MasteryTracker({ onLevelChange }: MasteryTrackerProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [items, setItems] = useState<MasteryItem[]>(masteryItems);
  const [filterCategory, setFilterCategory] = useState<string | null>(null);

  const categories = [...new Set(items.map((i) => i.category))];
  const filteredItems = filterCategory
    ? items.filter((i) => i.category === filterCategory)
    : items;

  const handleLevelChange = (id: string, newLevel: MasteryLevel) => {
    setItems(items.map((i) => (i.id === id ? { ...i, currentLevel: newLevel } : i)));
    onLevelChange?.(id, newLevel);
  };

  const stats = {
    total: items.length,
    expert: items.filter((i) => i.currentLevel === 'expert').length,
    mastered: items.filter((i) => i.currentLevel === 'mastered').length,
    explored: items.filter((i) => i.currentLevel === 'explored').length,
    inProgress: items.filter((i) => i.currentLevel === 'in-progress').length,
  };

  const progressPercentage = ((stats.expert + stats.mastered + stats.explored) / stats.total) * 100;

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border border-[--color-border] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-body]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-3">
          Mastery Tracking
        </h3>

        {/* Progress Overview */}
        <div className="grid grid-cols-5 gap-2 mb-4">
          {masteryLevels.map((level) => {
            const count = stats[level.level === 'expert' ? 'expert' : level.level === 'mastered' ? 'mastered' : level.level === 'explored' ? 'explored' : 'inProgress'] || 0;
            return (
              <div
                key={level.level}
                className="text-center px-2 py-1.5 rounded border text-xs"
                style={{
                  backgroundColor: level.color.split(' ')[0],
                  borderColor: level.color.split(' ')[1],
                  color: level.color.split(' ')[2],
                }}
              >
                <div className="font-bold">{count}</div>
                <div className="text-[10px] opacity-75">{level.label}</div>
              </div>
            );
          })}
        </div>

        {/* Overall Progress */}
        <div>
          <div className="flex justify-between mb-1">
            <span className="text-xs font-medium text-[--color-text-secondary]">
              Overall Mastery
            </span>
            <span className="text-xs font-bold text-[--accent-primary]">
              {Math.round(progressPercentage)}%
            </span>
          </div>
          <div className="bg-[--bg-surface] rounded h-2 overflow-hidden">
            <div
              className="bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] h-full transition-all"
              style={{ width: `${progressPercentage}%` }}
            />
          </div>
        </div>
      </div>

      {/* Category Filter */}
      <div className="px-6 py-3 border-b border-[--color-border] bg-[--bg-body]">
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => setFilterCategory(null)}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              filterCategory === null
                ? 'bg-[--accent-primary] text-white'
                : 'bg-[--bg-surface] text-[--color-text-secondary] hover:text-[--color-text-primary]'
            }`}
          >
            All
          </button>
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setFilterCategory(cat)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                filterCategory === cat
                  ? 'bg-[--accent-primary] text-white'
                  : 'bg-[--bg-surface] text-[--color-text-secondary] hover:text-[--color-text-primary]'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Items List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {filteredItems.map((item) => {
          const isSelected = selectedId === item.id;
          const levelInfo = masteryLevels.find((l) => l.level === item.currentLevel);
          const prerequisiteItem = item.prerequisite
            ? items.find((i) => i.id === item.prerequisite)
            : null;
          const isLocked =
            prerequisiteItem && prerequisiteItem.currentLevel === 'not-started';

          return (
            <div
              key={item.id}
              onClick={() => setSelectedId(isSelected ? null : item.id)}
              className={`p-3 rounded-lg border-2 cursor-pointer transition-all ${
                isSelected
                  ? 'border-[--accent-primary] bg-[--accent-primary]/10'
                  : `border-[--color-border] bg-[--bg-body] hover:border-[--accent-primary]/50`
              } ${isLocked ? 'opacity-60' : ''}`}
            >
              <div className="flex items-start gap-3">
                {/* Status Icon */}
                <div className="flex-shrink-0 pt-0.5">
                  {isLocked ? (
                    <Lock size={14} className="text-[--color-text-tertiary]" />
                  ) : item.currentLevel === 'expert' ? (
                    <Star size={14} className="text-yellow-400" fill="currentColor" />
                  ) : item.currentLevel === 'mastered' ? (
                    <CheckCircle2 size={14} className="text-purple-400" />
                  ) : item.currentLevel === 'explored' ? (
                    <CheckCircle2 size={14} className="text-green-400" />
                  ) : item.currentLevel === 'in-progress' ? (
                    <CheckCircle2 size={14} className="text-blue-400" />
                  ) : null}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <h4 className="text-sm font-semibold text-[--color-text-primary]">
                      {item.name}
                    </h4>
                    <span
                      className={`px-2 py-0.5 rounded text-xs font-medium flex-shrink-0 border ${levelInfo?.color}`}
                    >
                      {levelInfo?.label}
                    </span>
                  </div>
                  <p className="text-xs text-[--color-text-secondary] mt-1">
                    {item.description}
                  </p>

                  {isSelected && (
                    <div className="mt-3 pt-3 border-t border-[--color-border] space-y-2">
                      {isLocked && (
                        <p className="text-xs text-yellow-400">
                          🔒 Unlock by mastering: {prerequisiteItem?.name}
                        </p>
                      )}

                      {!isLocked && (
                        <>
                          <p className="text-xs text-[--color-text-tertiary]">
                            Update your mastery level:
                          </p>
                          <div className="grid grid-cols-5 gap-1">
                            {masteryLevels.map((level) => (
                              <button
                                key={level.level}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleLevelChange(item.id, level.level);
                                }}
                                className={`px-2 py-1 rounded text-xs font-medium border transition-all ${
                                  item.currentLevel === level.level
                                    ? `${level.color} border-current`
                                    : 'bg-[--bg-surface] border-[--color-border] text-[--color-text-tertiary] hover:text-[--color-text-primary]'
                                }`}
                              >
                                {level.label.split(' ')[0]}
                              </button>
                            ))}
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="border-t border-[--color-border] bg-[--bg-body] p-4">
        <button className="w-full px-3 py-2 rounded text-xs font-medium bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          Export Mastery Report
        </button>
      </div>
    </div>
  );
}
