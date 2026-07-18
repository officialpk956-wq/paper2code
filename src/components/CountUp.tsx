'use client';

import { useEffect, useRef, useState } from 'react';
import { useInView } from 'framer-motion';

interface CountUpProps {
  value: string;
  className?: string;
}

function parseNumeric(value: string): number {
  const match = value.match(/[\d.]+/);
  return match ? parseFloat(match[0]) : 0;
}

function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function CountUp({ value, className }: CountUpProps) {
  const end = parseNumeric(value);
  const [displayed, setDisplayed] = useState(prefersReducedMotion() ? end : 0);
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const hasAnimated = useRef(false);

  useEffect(() => {
    if (!inView || hasAnimated.current) return;
    if (prefersReducedMotion()) {
      setDisplayed(end);
      return;
    }

    hasAnimated.current = true;
    const duration = 1200;
    const startTime = performance.now();

    const tick = (now: number) => {
      const elapsed = Math.min(now - startTime, duration);
      const progress = 1 - Math.pow(1 - elapsed / duration, 3);
      setDisplayed(Math.round(progress * end));
      if (elapsed < duration) {
        requestAnimationFrame(tick);
      }
    };

    requestAnimationFrame(tick);
  }, [inView, end]);

  // Replace the numeric portion in the original value string with the animated count
  const suffix = value.replace(/[\d.]+/, '');

  return (
    <span ref={ref} className={className}>
      {displayed}
      {suffix}
    </span>
  );
}

export default CountUp;
