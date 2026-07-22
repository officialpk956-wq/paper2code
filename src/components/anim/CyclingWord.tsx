'use client';

import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useState, type CSSProperties } from 'react';
import { usePrefersReducedMotion } from './usePrefersReducedMotion';

interface Props {
  words: string[];
  interval?: number;
  className?: string;
  /** CSS gradient applied to the text (clipped). Defaults to cyan->violet. */
  gradient?: string;
}

const DEFAULT_GRADIENT = 'linear-gradient(120deg, #00E5FF, #7C5CFF)';

export function CyclingWord({ words, interval = 2400, className, gradient = DEFAULT_GRADIENT }: Props) {
  const [i, setI] = useState(0);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    if (reduced) return;
    const id = window.setInterval(() => setI((v) => (v + 1) % words.length), interval);
    return () => window.clearInterval(id);
  }, [words.length, interval, reduced]);

  const word = words[i];
  const gradStyle: CSSProperties = {
    backgroundImage: gradient,
    WebkitBackgroundClip: 'text',
    backgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    color: 'transparent',
  };

  if (reduced) {
    return (
      <span className={className} style={gradStyle}>
        {word}
      </span>
    );
  }

  return (
    <span className={className} style={{ position: 'relative', display: 'inline-block' }}>
      <span style={{ visibility: 'hidden', whiteSpace: 'nowrap' }} aria-hidden>
        {words.reduce((a, b) => (a.length >= b.length ? a : b))}
      </span>
      <span
        style={{
          position: 'absolute',
          inset: 0,
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          whiteSpace: 'nowrap',
        }}
      >
        <AnimatePresence mode="wait">
          <motion.span
            key={word}
            initial={{ opacity: 0, y: '0.4em', rotateX: -60 }}
            animate={{ opacity: 1, y: 0, rotateX: 0 }}
            exit={{ opacity: 0, y: '-0.4em', rotateX: 60 }}
            transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
            style={{ display: 'inline-block', transformStyle: 'preserve-3d', ...gradStyle }}
          >
            {word}
          </motion.span>
        </AnimatePresence>
      </span>
    </span>
  );
}
