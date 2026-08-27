'use client';

import dynamic from 'next/dynamic';
import { useEffect, useMemo, useState } from 'react';

import ArchDiagram from './ArchDiagram';
import { GENERIC_FLOWS, toDiagramSlug } from './archFlows';

interface ArchDiagramViewProps {
  slug: string;
}

type DiagramMode = '2d' | '3d';

const STORAGE_KEY = 'paper2code:architecture-diagram-mode';

const ArchDiagram3D = dynamic(() => import('./ArchDiagram3D'), {
  ssr: false,
  loading: () => (
    <div
      aria-label="Loading 3D architecture diagram"
      className="h-[360px] w-full animate-pulse rounded-xl border border-[#262626] bg-gradient-to-br from-[#111111] to-[#0A0A0A]"
    />
  ),
});

export default function ArchDiagramView({ slug }: ArchDiagramViewProps) {
  const [mode, setMode] = useState<DiagramMode>('3d');
  const has3DFlow = useMemo(() => {
    const mappedSlug = toDiagramSlug(slug) ?? '';
    return (GENERIC_FLOWS[slug]?.length ?? 0) > 0 || (GENERIC_FLOWS[mappedSlug]?.length ?? 0) > 0;
  }, [slug]);

  useEffect(() => {
    try {
      const storedMode = window.localStorage.getItem(STORAGE_KEY);
      if (storedMode === '2d' || storedMode === '3d') setMode(storedMode);
    } catch {
      // Storage may be disabled; the in-memory preference still works.
    }
  }, []);

  const selectMode = (nextMode: DiagramMode) => {
    setMode(nextMode);
    try {
      window.localStorage.setItem(STORAGE_KEY, nextMode);
    } catch {
      // Storage may be disabled; keep the current session usable.
    }
  };

  const effectiveMode: DiagramMode = mode === '3d' && has3DFlow ? '3d' : '2d';

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="inline-flex rounded-lg border border-[#262626] bg-[#0A0A0A] p-1" aria-label="Diagram view">
          {(['2d', '3d'] as const).map((option) => {
            const active = effectiveMode === option;
            const unavailable = option === '3d' && !has3DFlow;
            return (
              <button
                key={option}
                type="button"
                disabled={unavailable}
                aria-pressed={active}
                title={unavailable ? '3D will be available when this flow is added to the shared registry' : undefined}
                onClick={() => selectMode(option)}
                className={
                  'rounded-md px-3 py-1 text-[11px] font-semibold uppercase tracking-wider transition-colors ' +
                  (active
                    ? 'bg-[#A78BFA] text-black'
                    : 'text-[#737373] hover:text-white disabled:cursor-not-allowed disabled:opacity-35')
                }
              >
                {option.toUpperCase()}
              </button>
            );
          })}
        </div>
        {!has3DFlow && (
          <span className="text-right text-[11px] text-[#525252]">Showing the available 2D blueprint</span>
        )}
      </div>

      {effectiveMode === '3d' ? <ArchDiagram3D slug={slug} /> : <ArchDiagram slug={slug} />}
    </div>
  );
}
