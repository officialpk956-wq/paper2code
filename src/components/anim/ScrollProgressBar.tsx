'use client';

import { motion, useScroll, useSpring } from 'framer-motion';

interface Props {
  className?: string;
  /** CSS gradient for the bar. Defaults to cyan->violet. */
  gradient?: string;
}

export function ScrollProgressBar({
  className,
  gradient = 'linear-gradient(90deg, #00E5FF, #7C5CFF)',
}: Props) {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 140, damping: 20, mass: 0.3 });

  return (
    <motion.div
      className={className}
      style={{
        scaleX,
        transformOrigin: '0% 50%',
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: 2,
        zIndex: 60,
        background: gradient,
        pointerEvents: 'none',
      }}
      aria-hidden
    />
  );
}
