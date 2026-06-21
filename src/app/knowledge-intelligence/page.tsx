'use client';

import { useState } from 'react';
import { ThreeColumnLayout } from '@/components/layout/three-column-layout';
import { AILearningCoach } from '@/components/knowledge/ai-learning-coach';
import { KnowledgeGraph } from '@/components/knowledge/knowledge-graph';
import { LearningAnalytics } from '@/components/knowledge/learning-analytics';
import { ResearchJourney } from '@/components/knowledge/research-journey';
import { ArchitectureEvolution } from '@/components/knowledge/architecture-evolution';
import { MasteryTracker } from '@/components/knowledge/mastery-tracker';

export default function KnowledgeIntelligencePage() {
  const [selectedTab, setSelectedTab] = useState<'graph' | 'analytics' | 'journey' | 'evolution'>('graph');
  const [coachExpanded, setCoachExpanded] = useState(true);
  const [archFamily, setArchFamily] = useState<'transformer' | 'cnn' | 'generative'>('transformer');

  const tabs = [
    { id: 'graph', label: 'Knowledge Graph' },
    { id: 'analytics', label: 'Learning Analytics' },
    { id: 'journey', label: 'Research Journeys' },
    { id: 'evolution', label: 'Architecture Evolution' },
  ] as const;

  const leftPanel = (
    <div className="flex flex-col h-full">
      {coachExpanded ? (
        <AILearningCoach userName="Researcher" />
      ) : (
        <MasteryTracker />
      )}
    </div>
  );

  const centerPanel = (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header with Tabs */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold text-[--color-text-primary]">
            Knowledge Intelligence Layer
          </h1>
          <button
            onClick={() => setCoachExpanded(!coachExpanded)}
            className="px-3 py-1.5 text-xs font-medium rounded border border-[--color-border] text-[--color-text-secondary] hover:text-[--color-text-primary] hover:border-[--accent-primary]/50 transition-colors"
          >
            {coachExpanded ? 'Show Mastery' : 'Show Coach'}
          </button>
        </div>

        {/* Tab Navigation */}
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
      <div className="flex-1 overflow-hidden">
        {selectedTab === 'graph' && (
          <div className="h-full p-4">
            <KnowledgeGraph />
          </div>
        )}

        {selectedTab === 'analytics' && (
          <div className="h-full overflow-y-auto">
            <LearningAnalytics />
          </div>
        )}

        {selectedTab === 'journey' && (
          <div className="h-full p-4">
            <ResearchJourney
              journeyName="Transformer Mastery Path"
              journeyDescription="A comprehensive journey from transformer basics to advanced optimizations"
            />
          </div>
        )}

        {selectedTab === 'evolution' && (
          <div className="h-full p-4 flex flex-col gap-3">
            {/* Family selector */}
            <div className="flex gap-2 flex-shrink-0">
              {(['transformer', 'cnn', 'generative'] as const).map((fam) => (
                <button
                  key={fam}
                  onClick={() => setArchFamily(fam)}
                  className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                    archFamily === fam
                      ? 'bg-[--accent-primary] text-white'
                      : 'bg-[--bg-panel] border border-[--color-border] text-[--color-text-secondary] hover:text-[--color-text-primary]'
                  }`}
                >
                  {fam === 'transformer' ? 'Transformer' : fam === 'cnn' ? 'CNN' : 'Generative'}
                </button>
              ))}
            </div>
            <div className="flex-1 overflow-hidden">
              <ArchitectureEvolution family={archFamily} />
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const rightPanel = (
    <div className="flex flex-col h-full">
      {!coachExpanded ? (
        <AILearningCoach userName="Researcher" />
      ) : (
        <MasteryTracker />
      )}
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
