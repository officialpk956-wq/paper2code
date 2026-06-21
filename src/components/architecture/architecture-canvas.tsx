'use client';

import { ArchitectureDiagram } from '@/components/architecture-diagram';
import { ARCHITECTURE_CATALOG } from '@/data/architecture-catalog';
import { Construction } from 'lucide-react';

interface CanvasProps {
  selectedSlug: string;
  selectedBlockId: string | null;
  onSelectBlock: (id: string | null) => void;
}

export function ArchitectureCanvas({ selectedSlug, selectedBlockId, onSelectBlock }: CanvasProps) {
  const entry = ARCHITECTURE_CATALOG.find((e) => e.slug === selectedSlug);

  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h2 className="text-xl font-bold text-[--color-text-primary]">
          {entry ? entry.title : 'Architecture Explorer'}
        </h2>
        <p className="text-sm text-[--color-text-secondary] mt-1">
          Interactive visualization with tensor flow and layer dimensions
        </p>
      </div>

      {/* Canvas Area */}
      <div 
        className="flex-1 flex flex-col items-center justify-center overflow-auto p-6 relative"
        role="region"
        aria-label={`Architecture visualization for ${entry?.title || 'selected component'}`}
      >
        <div className="absolute inset-0 pointer-events-none opacity-50" style={{
          backgroundImage: 'radial-gradient(circle at 2px 2px, rgba(255,255,255,0.05) 1px, transparent 0)',
          backgroundSize: '24px 24px',
        }} />
        
        {entry?.status === 'complete' && entry.diagram ? (
          <div className="z-10 w-full max-w-2xl">
            <ArchitectureDiagram
              blocks={entry.diagram}
              selectedBlockId={selectedBlockId}
              onSelectBlock={onSelectBlock}
            />
          </div>
        ) : (
          <div className="z-10 flex flex-col items-center justify-center text-center max-w-sm p-8 rounded-2xl bg-[--bg-surface] border border-[--color-border]">
            <div className="w-16 h-16 rounded-full bg-[--bg-panel] border border-[--color-border] flex items-center justify-center mb-4">
              <Construction size={28} className="text-[--color-text-tertiary]" />
            </div>
            <h3 className="text-lg font-bold text-[--color-text-primary] mb-2">
              Interactive diagram coming soon
            </h3>
            <p className="text-sm text-[--color-text-secondary] leading-relaxed">
              We&apos;re currently modeling the computation graph and tensor flows for {entry?.title}.
            </p>
          </div>
        )}
      </div>

      {/* Info Footer */}
      <div className="px-6 py-4 border-t border-[--color-border] bg-[--bg-panel] text-xs text-[--color-text-secondary]">
        Hover over components to see details • Click to explore implementation
      </div>
    </div>
  );
}
