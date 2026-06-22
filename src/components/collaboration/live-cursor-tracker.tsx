'use client';

import { useState, useEffect, useRef } from 'react';

interface LiveCursor {
  id: string;
  name: string;
  avatar: string;
  color: string;
  x: number;
  y: number;
  timestamp: number;
}

// Simulate moving cursors
const generateCursorPositions = (id: string): { x: number; y: number } => {
  // Deterministic pseudo-random based on id for consistent movement
  const hash = id.charCodeAt(0) * 73;
  const baseX = (hash * 1234) % 800;
  const baseY = (hash * 5678) % 600;

  const timeOffset = Date.now() / 50;
  return {
    x: baseX + Math.sin(timeOffset / 10) * 100,
    y: baseY + Math.cos(timeOffset / 12) * 100,
  };
};

const liveCursors: LiveCursor[] = [
  {
    id: '2',
    name: 'Sarah',
    avatar: '👩',
    color: '#06B6D4',
    x: 250,
    y: 180,
    timestamp: 0,
  },
  {
    id: '3',
    name: 'Alex',
    avatar: '👨',
    color: '#7C3AED',
    x: 450,
    y: 350,
    timestamp: 0,
  },
];

interface LiveCursorTrackerProps {
  cursors?: LiveCursor[];
  canvasLabel?: string;
}

export function LiveCursorTracker({
  cursors = liveCursors,
  canvasLabel = 'Canvas'
}: LiveCursorTrackerProps) {
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }>>({});
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const tick = () => {
      const updated: Record<string, { x: number; y: number }> = {};
      cursors.forEach((cursor) => {
        updated[cursor.id] = generateCursorPositions(cursor.id);
      });
      setPositions(updated);
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [cursors]);

  return (
    <div className="relative w-full h-full rounded-lg border border-[--color-border] bg-[--bg-surface] overflow-hidden">
      {/* Canvas Label */}
      <div className="absolute top-0 left-0 right-0 px-4 py-3 bg-[--bg-panel] border-b border-[--color-border] flex items-center justify-between z-10">
        <h3 className="text-sm font-semibold text-[--color-text-primary]">
          {canvasLabel}
        </h3>
        <div className="flex items-center gap-2">
          {cursors.map((cursor) => (
            <div
              key={cursor.id}
              className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-[--bg-body] border border-[--color-border]"
              style={{ borderColor: cursor.color + '40' }}
            >
              <div className="text-sm">{cursor.avatar}</div>
              <span style={{ color: cursor.color }} className="font-medium">
                {cursor.name}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Canvas Area */}
      <div className="w-full h-full pt-16 relative">
        {/* Grid background */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ background: 'repeating-linear-gradient(0deg, transparent, transparent 39px, var(--color-border) 39px, var(--color-border) 40px), repeating-linear-gradient(90deg, transparent, transparent 39px, var(--color-border) 39px, var(--color-border) 40px)' }}
        />

        {/* Live Cursors */}
        {cursors.map((cursor) => {
          const pos = positions[cursor.id] || { x: cursor.x, y: cursor.y };
          return (
            <div
              key={cursor.id}
              className="absolute pointer-events-none transition-all duration-75"
              style={{
                left: `${(pos.x / 800) * 100}%`,
                top: `${(pos.y / 600) * 100 + 10}%`,
                transform: 'translate(-8px, -8px)',
              }}
            >
              {/* Cursor Arrow */}
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M0 0L0 14.4L5.2 9.2L10.4 16L14.4 15.2L9.2 8L14.4 2.8L0 0Z"
                  fill={cursor.color}
                />
              </svg>

              {/* User Label */}
              <div
                className="absolute left-5 top-5 px-2 py-1 rounded text-xs font-medium whitespace-nowrap pointer-events-none"
                style={{
                  background: cursor.color,
                  color: 'white',
                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
                }}
              >
                {cursor.avatar} {cursor.name}
              </div>
            </div>
          );
        })}

        {/* Center Info */}
        {cursors.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-3">🖱️</div>
              <div className="text-sm text-[--color-text-secondary]">
                No active cursors
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer Stats */}
      <div className="absolute bottom-0 left-0 right-0 px-4 py-3 bg-[--bg-panel] border-t border-[--color-border] flex justify-between text-xs text-[--color-text-tertiary]">
        <span>Active Cursors: {cursors.length}</span>
        <span>Last updated: just now</span>
      </div>
    </div>
  );
}
