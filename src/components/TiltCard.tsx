'use client';

import React, { useRef, useState } from 'react';
import { motion, useSpring } from 'framer-motion';

interface TiltCardProps {
  children: React.ReactNode;
  className?: string;
}

export function TiltCard({ children, className }: TiltCardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [glare, setGlare] = useState({ x: 50, y: 50, opacity: 0 });

  const rotateX = useSpring(0, { stiffness: 200, damping: 20 });
  const rotateY = useSpring(0, { stiffness: 200, damping: 20 });

  if (
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  ) {
    return <div className={className}>{children}</div>;
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (!el) return;

    const rect = el.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    const offsetX = e.clientX - centerX;
    const offsetY = e.clientY - centerY;

    const maxTilt = 8;
    const rx = -((offsetY / (rect.height / 2)) * maxTilt);
    const ry = (offsetX / (rect.width / 2)) * maxTilt;

    rotateX.set(Math.max(-maxTilt, Math.min(maxTilt, rx)));
    rotateY.set(Math.max(-maxTilt, Math.min(maxTilt, ry)));

    const glareX = ((offsetX / rect.width) * 100 + 50);
    const glareY = ((offsetY / rect.height) * 100 + 50);
    setGlare({ x: glareX, y: glareY, opacity: 1 });
  };

  const handleMouseLeave = () => {
    rotateX.set(0);
    rotateY.set(0);
    setGlare((g) => ({ ...g, opacity: 0 }));
  };

  return (
    <motion.div
      ref={ref}
      className={className}
      style={{
        rotateX,
        rotateY,
        transformPerspective: 800,
      }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      <div
        aria-hidden="true"
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          borderRadius: 'inherit',
          background: `radial-gradient(circle at ${glare.x}% ${glare.y}%, rgba(255,255,255,0.06), transparent 70%)`,
          opacity: glare.opacity,
          transition: 'opacity 0.3s ease',
        }}
      />
    </motion.div>
  );
}

export default TiltCard;
