'use client';

import { AttentionViz } from '@/components/model-architecture/attention-viz';

export default function AttentionVizPage() {
  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      <AttentionViz />
    </div>
  );
}
