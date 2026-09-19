import type { Metadata } from 'next';
import { ARCHITECTURES } from '@/data/content/architectures';
import { AuthGuard } from '@/components/AuthGuard';
import { PageSummary } from '@/components/PageSummary';
import ArchitecturesListClient from './ArchitecturesListClient';

export const metadata: Metadata = {
  title: `Architecture Library — ${ARCHITECTURES.length} ML Architectures | paper2code`,
  description: `Browse and study ${ARCHITECTURES.length} machine learning architectures, from Transformers to diffusion models, with lineage, industry usage, and implementation notes.`,
};

export default function ArchitecturesPage() {
  return (
    <>
      <PageSummary
        title="Architecture Library"
        description={`Browse ${ARCHITECTURES.length} machine learning architectures, from Transformers to diffusion models.`}
      />
      <AuthGuard>
        <ArchitecturesListClient />
      </AuthGuard>
    </>
  );
}
