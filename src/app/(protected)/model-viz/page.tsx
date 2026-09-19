import type { Metadata } from 'next';
import { AuthGuard } from '@/components/AuthGuard';
import { PageSummary } from '@/components/PageSummary';
import ModelVizClient from './ModelVizClient';

export const metadata: Metadata = {
  title: 'Model Visualizer | paper2code',
  description: 'Upload an ONNX or PyTorch model and explore its computation graph, parameter counts, and FLOPs interactively.',
};

export default function ModelVizPage() {
  return (
    <>
      <PageSummary
        title="Model Visualizer"
        description="Upload an ONNX or PyTorch model and explore its computation graph, parameter counts, and FLOPs interactively."
      />
      <AuthGuard>
        <ModelVizClient />
      </AuthGuard>
    </>
  );
}
