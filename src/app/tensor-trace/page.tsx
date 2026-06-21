'use client';

import { TensorTrace } from '@/components/visualization/tensor-trace';

export default function TensorTracePage() {
  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      <TensorTrace />
    </div>
  );
}
