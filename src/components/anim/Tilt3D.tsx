'use client';

import { useRef, type ReactNode, type CSSProperties } from 'react';
import { usePrefersReducedMotion } from './usePrefersReducedMotion';

interface Props {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  /** Max tilt in degrees. */
  max?: number;
  /** How far the card lifts on hover (px, translateZ). */
  lift?: number;
  /** Perspective in px on the wrapping container. */
  perspective?: number;
  /** Enable specular sheen following the cursor. */
  glare?: boolean;
  /** Ease-back duration on leave (ms). */
  ease?: number;
}

/**
 * 3D tilt wrapper. rotateX/rotateY based on pointer position, a small
 * translateZ lift, and an optional specular sheen highlight. Snaps back
 * smoothly on leave. Reduced motion -> no tilt, no glare.
 */
export function Tilt3D({
  children,
  className,
  style,
  max = 10,
  lift = 14,
  perspective = 900,
  glare = true,
  ease = 500,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const sheenRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number | null>(null);
  const reduced = usePrefersReducedMotion();

  const handleMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (reduced) return;
    const el = wrapRef.current;
    const inner = innerRef.current;
    if (!el || !inner) return;
    const rect = el.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width;
    const py = (e.clientY - rect.top) / rect.height;
    const rx = (0.5 - py) * (max * 2);
    const ry = (px - 0.5) * (max * 2);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      inner.style.transition = 'transform 90ms linear';
      inner.style.transform = `rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg) translateZ(${lift}px)`;
      if (sheenRef.current) {
        sheenRef.current.style.opacity = '1';
        sheenRef.current.style.background = `radial-gradient(circle at ${(px * 100).toFixed(1)}% ${(py * 100).toFixed(1)}%, rgba(255,255,255,0.16), rgba(255,255,255,0) 55%)`;
      }
    });
  };

  const handleLeave = () => {
    if (reduced) return;
    const inner = innerRef.current;
    if (!inner) return;
    inner.style.transition = `transform ${ease}ms cubic-bezier(.2,.9,.25,1)`;
    inner.style.transform = 'rotateX(0deg) rotateY(0deg) translateZ(0px)';
    if (sheenRef.current) sheenRef.current.style.opacity = '0';
  };

  return (
    <div
      ref={wrapRef}
      className={className}
      style={{ perspective: `${perspective}px`, transformStyle: 'preserve-3d', ...style }}
      onPointerMove={handleMove}
      onPointerLeave={handleLeave}
    >
      <div
        ref={innerRef}
        style={{ position: 'relative', transformStyle: 'preserve-3d', willChange: 'transform', height: '100%' }}
      >
        {children}
        {glare && !reduced && (
          <div
            ref={sheenRef}
            aria-hidden
            style={{
              position: 'absolute',
              inset: 0,
              opacity: 0,
              transition: 'opacity 250ms ease',
              pointerEvents: 'none',
              borderRadius: 'inherit',
              mixBlendMode: 'screen',
            }}
          />
        )}
      </div>
    </div>
  );
}
