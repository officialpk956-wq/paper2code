import type { Metadata } from 'next';
import { PROBLEMS } from '@/data/problems';
import { AuthGuard } from '@/components/AuthGuard';
import { PageSummary } from '@/components/PageSummary';
import DojoListClient from './DojoListClient';

export const metadata: Metadata = {
  title: `Coding Dojo — ${PROBLEMS.length} ML Implementation Challenges | paper2code`,
  description: `Practice implementing ML algorithms and neural network components from scratch in the browser, across ${PROBLEMS.length} challenges from NumPy basics to attention mechanisms.`,
};

export default function DojoPage() {
  return (
    <>
      <PageSummary
        title="Coding Dojo"
        description={`Practice implementing ML algorithms and neural network components from scratch, across ${PROBLEMS.length} challenges.`}
      />
      <AuthGuard>
        <DojoListClient />
      </AuthGuard>
    </>
  );
}
