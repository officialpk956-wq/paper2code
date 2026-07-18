'use client';

import dynamic from 'next/dynamic';

const ArchNeural = dynamic(
  () => import('@/components/arch/ArchNeural').then(m => m.ArchNeural),
  { ssr: false },
);

interface ArchHeroBackgroundProps {
  color?: string;
}

/**
 * Thin client wrapper so the server component page can use ArchNeural
 * (which requires ssr: false) without violating Next.js 15 rules.
 */
export function ArchHeroBackground({ color = '#7C5CFF' }: ArchHeroBackgroundProps) {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden opacity-35">
      <ArchNeural color={color} />
    </div>
  );
}

export default ArchHeroBackground;
