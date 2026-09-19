import type { Metadata } from 'next';
import { ARCHITECTURES } from '@/data/content/architectures';
import { AuthGuard } from '@/components/AuthGuard';
import { PageSummary } from '@/components/PageSummary';
import CompareClient from './CompareClient';

export const metadata: Metadata = {
  title: 'Compare Architectures | paper2code',
  description: `Compare any two of ${ARCHITECTURES.length} ML architectures side-by-side: structural differences and metric deltas.`,
};

export default function ArchitectureComparePage() {
  return (
    <>
      <PageSummary
        title="Compare Architectures"
        description="Analyze structural differences and metric deltas between any two ML architectures."
      />
      <AuthGuard>
        <CompareClient />
      </AuthGuard>
    </>
  );
}
