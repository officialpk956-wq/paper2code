'use client';

import { ThreeColumnLayout } from '@/components/layout/three-column-layout';
import { LayersPanel } from '@/components/model-architecture/layers-panel';
import { ArchitectureViz } from '@/components/model-architecture/architecture-viz';
import { ModelStats } from '@/components/model-architecture/model-stats';
import { useState } from 'react';

export default function ModelArchitecturePage() {
  const [selectedLayer, setSelectedLayer] = useState<string>();

  return (
    <ThreeColumnLayout
      left={<LayersPanel onSelectLayer={setSelectedLayer} selectedLayer={selectedLayer} />}
      center={<ArchitectureViz selectedLayer={selectedLayer} />}
      right={<ModelStats selectedLayer={selectedLayer} />}
      leftWidth="w-72"
      rightWidth="w-80"
    />
  );
}
