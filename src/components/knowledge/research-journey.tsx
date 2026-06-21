'use client';

import { Plus, Share2, Save, Trash2, GripVertical, CheckCircle } from 'lucide-react';
import { useState } from 'react';

interface JourneyStep {
  id: string;
  title: string;
  description: string;
  type: 'paper' | 'concept' | 'implementation' | 'project';
  status: 'not-started' | 'in-progress' | 'completed';
  order: number;
}

const sampleJourney: JourneyStep[] = [
  {
    id: 'step-1',
    title: 'Understand Transformer Architecture',
    description: 'Read "Attention Is All You Need" and grasp the fundamentals',
    type: 'paper',
    status: 'completed',
    order: 0,
  },
  {
    id: 'step-2',
    title: 'Master Multi-Head Attention',
    description: 'Deep dive into how attention heads work and interact',
    type: 'concept',
    status: 'completed',
    order: 1,
  },
  {
    id: 'step-3',
    title: 'Implement Transformer from Scratch',
    description: 'Build a working transformer in PyTorch with clear comments',
    type: 'implementation',
    status: 'in-progress',
    order: 2,
  },
  {
    id: 'step-4',
    title: 'Advanced: Efficient Transformers',
    description: 'Explore sparse attention, linear attention, and optimizations',
    type: 'paper',
    status: 'not-started',
    order: 3,
  },
  {
    id: 'step-5',
    title: 'Build NLP Application',
    description: 'Apply transformers to real-world NLP tasks',
    type: 'project',
    status: 'not-started',
    order: 4,
  },
];

interface ResearchJourneyProps {
  journeyName?: string;
  journeyDescription?: string;
  onSave?: (steps: JourneyStep[]) => void;
}

export function ResearchJourney({
  journeyName = 'Transformer Mastery Path',
  journeyDescription = 'A comprehensive journey from transformer basics to advanced optimizations',
  onSave,
}: ResearchJourneyProps) {
  const [steps, setSteps] = useState<JourneyStep[]>(sampleJourney);
  const [draggedId, setDraggedId] = useState<string | null>(null);
  const [, setIsEditing] = useState(false);

  const handleDragStart = (id: string) => {
    setDraggedId(id);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (targetId: string) => {
    if (!draggedId || draggedId === targetId) return;

    const draggedIndex = steps.findIndex((s) => s.id === draggedId);
    const targetIndex = steps.findIndex((s) => s.id === targetId);

    const newSteps = [...steps];
    const [draggedStep] = newSteps.splice(draggedIndex, 1);
    newSteps.splice(targetIndex, 0, draggedStep);

    // Reorder
    setSteps(
      newSteps.map((step, idx) => ({
        ...step,
        order: idx,
      }))
    );
    setDraggedId(null);
  };

  const updateStepStatus = (id: string, status: JourneyStep['status']) => {
    setSteps(
      steps.map((step) => (step.id === id ? { ...step, status } : step))
    );
  };

  const deleteStep = (id: string) => {
    setSteps(steps.filter((s) => s.id !== id));
  };

  const typeColors = {
    paper: 'bg-purple-500/10 border-purple-500/30 text-purple-400',
    concept: 'bg-blue-500/10 border-blue-500/30 text-blue-400',
    implementation: 'bg-green-500/10 border-green-500/30 text-green-400',
    project: 'bg-orange-500/10 border-orange-500/30 text-orange-400',
  };

  const typeIcons = {
    paper: '📄',
    concept: '💡',
    implementation: '🔧',
    project: '🚀',
  };

  const completionRate = (steps.filter((s) => s.status === 'completed').length / steps.length) * 100;

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border border-[--color-border] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-body]">
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className="text-sm font-semibold text-[--color-text-primary]">
              {journeyName}
            </h3>
            <p className="text-xs text-[--color-text-secondary] mt-1">
              {journeyDescription}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button className="p-1.5 rounded hover:bg-[--bg-surface] text-[--color-text-tertiary] hover:text-[--color-text-primary] transition-colors">
              <Share2 size={14} />
            </button>
            <button
              onClick={() => {
                onSave?.(steps);
                setIsEditing(false);
              }}
              className="p-1.5 rounded hover:bg-[--bg-surface] text-[--color-text-tertiary] hover:text-[--color-text-primary] transition-colors"
            >
              <Save size={14} />
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        <div>
          <div className="flex justify-between mb-1">
            <span className="text-xs font-medium text-[--color-text-secondary]">
              Progress
            </span>
            <span className="text-xs font-bold text-[--accent-primary]">
              {Math.round(completionRate)}%
            </span>
          </div>
          <div className="bg-[--bg-surface] rounded h-2 overflow-hidden">
            <div
              className="bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] h-full transition-all"
              style={{ width: `${completionRate}%` }}
            />
          </div>
        </div>
      </div>

      {/* Journey Steps */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-3">
          {steps.map((step, idx) => (
            <div
              key={step.id}
              draggable
              onDragStart={() => handleDragStart(step.id)}
              onDragOver={handleDragOver}
              onDrop={() => handleDrop(step.id)}
              className={`group p-4 rounded-lg border-2 transition-all cursor-move ${
                step.status === 'completed'
                  ? 'bg-green-500/5 border-green-500/30'
                  : step.status === 'in-progress'
                    ? 'bg-blue-500/5 border-blue-500/30'
                    : 'bg-[--bg-body] border-[--color-border]'
              } ${draggedId === step.id ? 'opacity-50' : ''}`}
            >
              {/* Step Header */}
              <div className="flex items-start gap-3 mb-2">
                {/* Drag Handle */}
                <div className="text-[--color-text-tertiary] opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 pt-1">
                  <GripVertical size={14} />
                </div>

                {/* Number Circle */}
                <div className="w-6 h-6 rounded-full bg-[--bg-surface] flex items-center justify-center text-xs font-bold text-[--color-text-primary] flex-shrink-0">
                  {idx + 1}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h4 className="text-sm font-semibold text-[--color-text-primary]">
                        {step.title}
                      </h4>
                      <p className="text-xs text-[--color-text-secondary] mt-0.5">
                        {step.description}
                      </p>
                    </div>
                    <div className={`px-2 py-1 rounded text-xs font-medium flex-shrink-0 border ${typeColors[step.type]}`}>
                      {typeIcons[step.type]} {step.type}
                    </div>
                  </div>
                </div>

                {/* Status Button */}
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  {step.status === 'completed' ? (
                    <button
                      onClick={() => updateStepStatus(step.id, 'not-started')}
                      className="p-1.5 rounded bg-green-500/20 hover:bg-green-500/30 text-green-400 transition-colors"
                    >
                      <CheckCircle size={14} />
                    </button>
                  ) : (
                    <button
                      onClick={() => updateStepStatus(step.id, 'completed')}
                      className="p-1.5 rounded hover:bg-[--bg-surface] text-[--color-text-tertiary] hover:text-green-400 transition-colors"
                    >
                      <CheckCircle size={14} />
                    </button>
                  )}
                  <button
                    onClick={() => deleteStep(step.id)}
                    className="p-1.5 rounded hover:bg-red-500/20 text-red-400 transition-colors"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>

              {/* Status Badge */}
              <div className="ml-9 flex items-center gap-2">
                {step.status === 'completed' && (
                  <span className="text-xs px-2 py-0.5 rounded bg-green-500/20 text-green-400 font-medium">
                    ✓ Completed
                  </span>
                )}
                {step.status === 'in-progress' && (
                  <span className="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 font-medium">
                    ⏳ In Progress
                  </span>
                )}
                {step.status === 'not-started' && (
                  <span className="text-xs px-2 py-0.5 rounded bg-[--bg-surface] text-[--color-text-tertiary]">
                    Not Started
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-[--color-border] bg-[--bg-body] p-4">
        <button className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-medium bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          <Plus size={12} />
          Add Step
        </button>
      </div>
    </div>
  );
}
