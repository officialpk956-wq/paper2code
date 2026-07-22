'use client';

import type { ReactNode } from 'react';
import { usePrefersReducedMotion } from './usePrefersReducedMotion';

interface Props {
  items: ReactNode[];
  direction?: 'left' | 'right';
  speed?: number; // seconds per full loop
  className?: string;
}

/**
 * Pure-CSS infinite marquee. Duplicates content for a seamless loop.
 * Static (no animation) under reduced-motion.
 */
export function Marquee({ items, direction = 'left', speed = 40, className }: Props) {
  const reduced = usePrefersReducedMotion();
  const anim = reduced
    ? undefined
    : `${direction === 'left' ? 'p2c-marquee-l' : 'p2c-marquee-r'} ${speed}s linear infinite`;

  return (
    <div
      className={className}
      style={{
        overflow: 'hidden',
        maskImage: 'linear-gradient(to right, transparent, black 8%, black 92%, transparent)',
        WebkitMaskImage: 'linear-gradient(to right, transparent, black 8%, black 92%, transparent)',
      }}
    >
      <style>{`
        @keyframes p2c-marquee-l { from { transform: translateX(0);} to { transform: translateX(-50%);} }
        @keyframes p2c-marquee-r { from { transform: translateX(-50%);} to { transform: translateX(0);} }
        .p2c-marquee-track:hover { animation-play-state: paused; }
      `}</style>
      <div
        className="p2c-marquee-track"
        style={{ display: 'flex', gap: 12, width: 'max-content', animation: anim, willChange: 'transform' }}
      >
        {[...items, ...items].map((n, i) => (
          <div key={i} style={{ flex: '0 0 auto' }}>
            {n}
          </div>
        ))}
      </div>
    </div>
  );
}
