'use client';

import React from 'react';

interface SectionLabelProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

export function SectionLabel({ children, className, style }: SectionLabelProps) {
  return (
    <div
      className={className}
      style={{
        fontSize: '10px',
        fontWeight: 700,
        letterSpacing: '0.12em',
        textTransform: 'uppercase',
        color: 'var(--color-text-muted)',
        ...style,
      }}
    >
      {children}
    </div>
  );
}
