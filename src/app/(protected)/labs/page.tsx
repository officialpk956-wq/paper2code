import type { Metadata } from 'next';
import { AuthGuard } from '@/components/AuthGuard';
import { PageSummary } from '@/components/PageSummary';
import LabsClient from './LabsClient';

export const metadata: Metadata = {
  title: 'Interactive Architecture Labs | paper2code',
  description: 'Mutate ML architectures (ResNet, Transformer, ViT, U-Net) and see the real-time impact on compute, memory, and training cost.',
};

export default function LabsPage() {
  return (
    <>
      <PageSummary
        title="Interactive Architecture Labs"
        description="Mutate architectures and see the impact on compute, memory, and training costs in real-time."
      />
      <AuthGuard>
        <LabsClient />
      </AuthGuard>
    </>
  );
}
