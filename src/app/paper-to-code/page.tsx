'use client';

import { ThreeColumnLayout } from '@/components/layout/three-column-layout';
import { PaperExcerpt } from '@/components/paper-to-code/paper-excerpt';
import { CodeEditor } from '@/components/paper-to-code/code-editor';
import { ImplementationMap } from '@/components/paper-to-code/implementation-map';
import { useState } from 'react';

export default function PaperToCodePage() {
  const [selectedExcerpt, setSelectedExcerpt] = useState<string>();

  return (
    <ThreeColumnLayout
      left={<PaperExcerpt onSelectExcerpt={setSelectedExcerpt} selectedExcerpt={selectedExcerpt} />}
      center={<CodeEditor selectedExcerpt={selectedExcerpt} />}
      right={<ImplementationMap selectedExcerpt={selectedExcerpt} />}
      leftWidth="w-80"
      rightWidth="w-80"
    />
  );
}
