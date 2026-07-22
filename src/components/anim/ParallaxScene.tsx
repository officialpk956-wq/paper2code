'use client';

import { useEffect, useRef, type ReactNode } from 'react';
import { usePrefersReducedMotion } from './usePrefersReducedMotion';

interface Layer {
  children: ReactNode;
  /** Depth: 0 = far (moves least), 1 = near (moves most). */
  depth?: number;
  className?: string;
  /** Translate-Z in px (also applied for real 3D depth). */
  z?: number;
  /** Set to true if this layer holds interactive content. */
  interactive?: boolean;
}

interface Props {
  layers: Layer[];
  className?: string;
  pointerAmp?: number;
  scrollAmp?: number;
  perspective?: number;
}

/**
 * Perspective container with parallax depth layers. Each layer moves
 * proportional to its `depth` in response to pointer and scroll. Reduced
 * motion -> static.
 */
export function ParallaxScene({
  layers,
  className,
  pointerAmp = 24,
  scrollAmp = 90,
  perspective = 1200,
}: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const layerRefs = useRef<HTMLDivElement[]>([]);
  const rafRef = useRef<number | null>(null);
  const state = useRef({ px: 0, py: 0, tpx: 0, tpy: 0, sy: 0 });
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    if (reduced) return;
    const root = rootRef.current;
    if (!root) return;

    const onMove = (e: PointerEvent) => {
      const rect = root.getBoundingClientRect();
      state.current.tpx = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
      state.current.tpy = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
    };
    const onScroll = () => {
      const rect = root.getBoundingClientRect();
      state.current.sy = Math.max(0, -rect.top);
    };

    const tick = () => {
      state.current.px += (state.current.tpx - state.current.px) * 0.08;
      state.current.py += (state.current.tpy - state.current.py) * 0.08;
      layers.forEach((l, i) => {
        const el = layerRefs.current[i];
        if (!el) return;
        const d = l.depth ?? 0.5;
        const tx = -state.current.px * pointerAmp * d;
        const ty = -state.current.py * pointerAmp * d - state.current.sy * (scrollAmp / 800) * d;
        const tz = l.z ?? 0;
        el.style.transform = `translate3d(${tx.toFixed(2)}px, ${ty.toFixed(2)}px, ${tz}px)`;
      });
      rafRef.current = requestAnimationFrame(tick);
    };

    window.addEventListener('pointermove', onMove, { passive: true });
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('scroll', onScroll);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [layers, pointerAmp, scrollAmp, reduced]);

  return (
    <div
      ref={rootRef}
      className={className}
      style={{ position: 'relative', perspective: `${perspective}px`, transformStyle: 'preserve-3d' }}
    >
      {layers.map((l, i) => (
        <div
          key={i}
          ref={(el) => {
            if (el) layerRefs.current[i] = el;
          }}
          className={l.className}
          style={{
            position: 'absolute',
            inset: 0,
            willChange: 'transform',
            pointerEvents: l.interactive ? 'auto' : 'none',
            transformStyle: 'preserve-3d',
          }}
        >
          {l.children}
        </div>
      ))}
    </div>
  );
}
