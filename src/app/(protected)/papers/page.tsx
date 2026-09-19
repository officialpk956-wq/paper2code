import type { Metadata } from 'next';
import { PAPERS } from '@/data/content/papers';
import { AuthGuard } from '@/components/AuthGuard';
import { PageSummary } from '@/components/PageSummary';
import PapersListClient from './PapersListClient';

export const metadata: Metadata = {
  title: `Paper Library — ${PAPERS.length} ML Papers to Implement | paper2code`,
  description: `Read and implement ${PAPERS.length} landmark machine learning papers, from backpropagation to LoRA, with guided code walkthroughs.`,
};

export default function PapersPage() {
  return (
    <>
      <PageSummary
        title="Paper Library"
        description={`Read and implement ${PAPERS.length} landmark machine learning papers, with guided code walkthroughs.`}
      />
      <AuthGuard>
        <PapersListClient />
      </AuthGuard>
    </>
  );
}
