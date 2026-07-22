'use client';

import { motion, type Variants } from 'framer-motion';
import type { ReactNode } from 'react';
import { usePrefersReducedMotion } from './usePrefersReducedMotion';

type Direction = 'up' | 'down' | 'left' | 'right' | 'none';

interface RevealProps {
  children: ReactNode;
  delay?: number;
  direction?: Direction;
  distance?: number;
  duration?: number;
  className?: string;
  once?: boolean;
}

const offsetFor = (d: Direction, dist: number) => {
  switch (d) {
    case 'up': return { y: dist };
    case 'down': return { y: -dist };
    case 'left': return { x: dist };
    case 'right': return { x: -dist };
    default: return {};
  }
};

export function Reveal({
  children,
  delay = 0,
  direction = 'up',
  distance = 16,
  duration = 0.6,
  className,
  once = true,
}: RevealProps) {
  const reduced = usePrefersReducedMotion();

  if (reduced) return <div className={className}>{children}</div>;

  const variants: Variants = {
    hidden: { opacity: 0, ...offsetFor(direction, distance) },
    visible: { opacity: 1, x: 0, y: 0, transition: { duration, delay, ease: [0.22, 1, 0.36, 1] } },
  };

  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="visible"
      viewport={{ once, amount: 0.2 }}
      variants={variants}
    >
      {children}
    </motion.div>
  );
}
