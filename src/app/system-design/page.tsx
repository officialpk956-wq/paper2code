'use client';

import { ThreeColumnLayout } from '@/components/layout/three-column-layout';
import { PatternLibrary } from '@/components/system-design/pattern-library';
import { DesignCanvas } from '@/components/system-design/design-canvas';
import { DesignProperties } from '@/components/system-design/design-properties';
import { useState } from 'react';

export default function SystemDesignPage() {
  const [selectedPattern, setSelectedPattern] = useState<string>();

  return (
    <ThreeColumnLayout
      left={
        <PatternLibrary
          onSelectPattern={setSelectedPattern}
          onAddPattern={setSelectedPattern}
        />
      }
      center={<DesignCanvas selectedPattern={selectedPattern} />}
      right={<DesignProperties selectedPattern={selectedPattern} />}
      leftWidth="w-80"
      rightWidth="w-80"
    />
  );
}
