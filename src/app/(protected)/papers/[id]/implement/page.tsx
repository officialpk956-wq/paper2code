import type { Metadata } from 'next';
import { FLAGSHIP_META } from '@/data/flagship-papers';
import { AuthGuard } from '@/components/AuthGuard';
import { PageSummary } from '@/components/PageSummary';
import ImplementClient from './ImplementClient';

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await params;
  const meta = FLAGSHIP_META[id];
  if (!meta) return { title: 'Guided Implementation | paper2code' };
  return {
    title: `Implement ${meta.title} | paper2code`,
    description: meta.abstract.slice(0, 155),
  };
}

export default async function GuidedImplementPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const meta = FLAGSHIP_META[id];

  return (
    <>
      <PageSummary
        eyebrow="Guided Implementation"
        title={meta ? `Implement ${meta.title}` : 'Guided Implementation'}
        description={meta?.abstract ?? 'Step through a guided, hands-on implementation of a landmark ML paper.'}
      />
      <AuthGuard>
        <ImplementClient id={id} />
      </AuthGuard>
    </>
  );
}
