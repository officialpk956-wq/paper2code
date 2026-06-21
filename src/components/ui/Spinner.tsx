'use client';

interface SpinnerProps {
  size?: number;
  color?: string;
  className?: string;
}

export function Spinner({
  size = 16,
  color = 'var(--accent-primary)',
  className = '',
}: SpinnerProps) {
  return (
    <span
      className={`inline-block rounded-full animate-spin ${className}`}
      style={{
        width: size,
        height: size,
        border: '2px solid currentColor',
        borderTopColor: 'transparent',
        color,
        flexShrink: 0,
      }}
    />
  );
}
