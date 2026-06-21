'use client';

import { ThreeColumnLayout } from '@/components/layout/three-column-layout';
import { ArchitectureSidebar } from '@/components/architecture/architecture-sidebar';
import { ArchitectureCanvas } from '@/components/architecture/architecture-canvas';
import { ArchitectureInspector } from '@/components/architecture/architecture-inspector';
import { useState } from 'react';

export default function ArchitecturesPage() {
  const [selectedSlug, setSelectedSlug] = useState('transformer');
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null);

  const handleSelectSlug = (slug: string) => {
    setSelectedSlug(slug);
    setSelectedBlockId(null);
  };

  return (
    <ThreeColumnLayout
      left={<ArchitectureSidebar selectedSlug={selectedSlug} onSelect={handleSelectSlug} />}
      center={<ArchitectureCanvas selectedSlug={selectedSlug} selectedBlockId={selectedBlockId} onSelectBlock={setSelectedBlockId} />}
      right={<ArchitectureInspector selectedSlug={selectedSlug} selectedBlockId={selectedBlockId} />}
      leftWidth="w-64"
      rightWidth="w-80"
    />
  );
}
